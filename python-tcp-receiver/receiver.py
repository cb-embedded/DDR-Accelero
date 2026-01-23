#!/usr/bin/env python3
"""
DDR TCP Receiver - Real-time sensor data visualization

Receives accelerometer and gyroscope data from the DDR TCP Streamer Android app
and displays it in real-time using PyQtGraph.

Data format:
timestamp_ns,timestamp_ms,sensor_type,x,y,z

Where sensor_type is either "accel" or "gyro"
"""

import sys
import socket
import threading
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class SensorDataReceiver(QtCore.QObject):
    dataReceived = QtCore.pyqtSignal(str, float, float, float, float)
    connectionStatusChanged = QtCore.pyqtSignal(str)
    
    def __init__(self, port=5000):
        super().__init__()
        self.port = port
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.thread = None
        
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(1)
            
            self.connectionStatusChanged.emit(f"Listening on port {self.port}...")
            
            while self.running:
                try:
                    self.server_socket.settimeout(1.0)
                    self.client_socket, client_address = self.server_socket.accept()
                    self.connectionStatusChanged.emit(f"Connected to {client_address[0]}:{client_address[1]}")
                    
                    self._handle_client()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.connectionStatusChanged.emit(f"Connection error: {e}")
                    break
                    
        except Exception as e:
            self.connectionStatusChanged.emit(f"Server error: {e}")
        finally:
            self.connectionStatusChanged.emit("Disconnected")
    
    def _handle_client(self):
        buffer = ""
        try:
            while self.running:
                data = self.client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data.decode('utf-8', errors='ignore')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        self._parse_line(line)
                        
        except Exception as e:
            if self.running:
                print(f"Client error: {e}")
        finally:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
    
    def _parse_line(self, line):
        try:
            parts = line.split(',')
            if len(parts) >= 6:
                timestamp_ns = float(parts[0])
                timestamp_ms = float(parts[1])
                sensor_type = parts[2]
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                
                self.dataReceived.emit(sensor_type, timestamp_ms, x, y, z)
        except Exception as e:
            print(f"Parse error: {e} - Line: {line}")

class SensorPlotWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DDR TCP Receiver - Sensor Data Visualization")
        self.resize(1200, 800)
        
        self.maxDataPoints = 500
        
        self.accelTimeData = deque(maxlen=self.maxDataPoints)
        self.accelXData = deque(maxlen=self.maxDataPoints)
        self.accelYData = deque(maxlen=self.maxDataPoints)
        self.accelZData = deque(maxlen=self.maxDataPoints)
        
        self.gyroTimeData = deque(maxlen=self.maxDataPoints)
        self.gyroXData = deque(maxlen=self.maxDataPoints)
        self.gyroYData = deque(maxlen=self.maxDataPoints)
        self.gyroZData = deque(maxlen=self.maxDataPoints)
        
        self.startTime = None
        self.sampleCount = 0
        self.accelCount = 0
        self.gyroCount = 0
        
        self._setupUI()
        
        self.receiver = SensorDataReceiver(port=5000)
        self.receiver.dataReceived.connect(self.onDataReceived)
        self.receiver.connectionStatusChanged.connect(self.onConnectionStatusChanged)
        self.receiver.start()
        
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.updatePlots)
        self.updateTimer.start(50)
        
    def _setupUI(self):
        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)
        
        layout = QtWidgets.QVBoxLayout(centralWidget)
        
        self.statusLabel = QtWidgets.QLabel("Status: Starting...")
        self.statusLabel.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.statusLabel)
        
        self.statsLabel = QtWidgets.QLabel("Samples: 0 (Accel: 0, Gyro: 0)")
        self.statsLabel.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(self.statsLabel)
        
        graphicsLayoutWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(graphicsLayoutWidget)
        
        graphicsLayoutWidget.setBackground('w')
        
        self.accelPlot = graphicsLayoutWidget.addPlot(row=0, col=0, title="Accelerometer (m/s²)")
        self.accelPlot.setLabel('left', 'Acceleration', units='m/s²')
        self.accelPlot.setLabel('bottom', 'Time', units='s')
        self.accelPlot.addLegend()
        self.accelPlot.showGrid(x=True, y=True)
        
        self.accelXCurve = self.accelPlot.plot(pen=pg.mkPen('r', width=2), name='X')
        self.accelYCurve = self.accelPlot.plot(pen=pg.mkPen('g', width=2), name='Y')
        self.accelZCurve = self.accelPlot.plot(pen=pg.mkPen('b', width=2), name='Z')
        
        self.gyroPlot = graphicsLayoutWidget.addPlot(row=1, col=0, title="Gyroscope (rad/s)")
        self.gyroPlot.setLabel('left', 'Angular Velocity', units='rad/s')
        self.gyroPlot.setLabel('bottom', 'Time', units='s')
        self.gyroPlot.addLegend()
        self.gyroPlot.showGrid(x=True, y=True)
        
        self.gyroXCurve = self.gyroPlot.plot(pen=pg.mkPen('r', width=2), name='X')
        self.gyroYCurve = self.gyroPlot.plot(pen=pg.mkPen('g', width=2), name='Y')
        self.gyroZCurve = self.gyroPlot.plot(pen=pg.mkPen('b', width=2), name='Z')
    
    def onDataReceived(self, sensorType, timestamp, x, y, z):
        if self.startTime is None:
            self.startTime = timestamp
        
        relativeTime = (timestamp - self.startTime) / 1000.0
        
        if sensorType == "accel":
            self.accelTimeData.append(relativeTime)
            self.accelXData.append(x)
            self.accelYData.append(y)
            self.accelZData.append(z)
            self.accelCount += 1
        elif sensorType == "gyro":
            self.gyroTimeData.append(relativeTime)
            self.gyroXData.append(x)
            self.gyroYData.append(y)
            self.gyroZData.append(z)
            self.gyroCount += 1
        
        self.sampleCount += 1
    
    def onConnectionStatusChanged(self, status):
        self.statusLabel.setText(f"Status: {status}")
    
    def updatePlots(self):
        if len(self.accelTimeData) > 0:
            self.accelXCurve.setData(list(self.accelTimeData), list(self.accelXData))
            self.accelYCurve.setData(list(self.accelTimeData), list(self.accelYData))
            self.accelZCurve.setData(list(self.accelTimeData), list(self.accelZData))
        
        if len(self.gyroTimeData) > 0:
            self.gyroXCurve.setData(list(self.gyroTimeData), list(self.gyroXData))
            self.gyroYCurve.setData(list(self.gyroTimeData), list(self.gyroYData))
            self.gyroZCurve.setData(list(self.gyroTimeData), list(self.gyroZData))
        
        self.statsLabel.setText(f"Samples: {self.sampleCount} (Accel: {self.accelCount}, Gyro: {self.gyroCount})")
    
    def closeEvent(self, event):
        self.receiver.stop()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SensorPlotWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
