#!/usr/bin/env python3
"""
Test client that simulates the Android app sending sensor data.
Useful for testing the Python receiver without needing a real Android device.
"""

import socket
import time
import math
import sys

def main():
    host = 'localhost'
    port = 5000
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    print(f"Connecting to {host}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("Connected! Sending simulated sensor data...")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        sample_count = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            timestamp_ns = int(current_time * 1e9)
            timestamp_ms = int(current_time * 1000)
            
            accel_x = math.sin(elapsed * 2) * 2 + 0.1
            accel_y = math.cos(elapsed * 1.5) * 1.5 + 9.81
            accel_z = math.sin(elapsed * 3) * 0.5 - 0.2
            
            gyro_x = math.cos(elapsed * 2.5) * 0.3
            gyro_y = math.sin(elapsed * 2) * 0.4
            gyro_z = math.cos(elapsed * 1.8) * 0.2
            
            accel_line = f"{timestamp_ns},{timestamp_ms},accel,{accel_x:.6f},{accel_y:.6f},{accel_z:.6f}\n"
            gyro_line = f"{timestamp_ns},{timestamp_ms},gyro,{gyro_x:.6f},{gyro_y:.6f},{gyro_z:.6f}\n"
            
            sock.sendall(accel_line.encode('utf-8'))
            sock.sendall(gyro_line.encode('utf-8'))
            
            sample_count += 2
            
            if sample_count % 100 == 0:
                print(f"Sent {sample_count} samples")
            
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            sock.close()
        except:
            pass
        print(f"Total samples sent: {sample_count}")

if __name__ == '__main__':
    main()
