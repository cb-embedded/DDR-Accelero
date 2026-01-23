# DDR TCP Receiver - Real-time Sensor Data Visualization

Python application that receives accelerometer and gyroscope data from the DDR TCP Streamer Android app and displays it in real-time.

## Features

- **Real-time visualization**: Displays sensor data as it arrives
- **Dual plots**: Separate graphs for accelerometer and gyroscope
- **Color-coded axes**: X (red), Y (green), Z (blue)
- **Connection monitoring**: Shows connection status and sample counts
- **Automatic reconnection**: Waits for new connections after disconnect
- **Sliding window**: Shows last 500 data points for smooth visualization

## Requirements

- Python 3.7 or higher
- PyQt5
- PyQtGraph

## Installation

### Option 1: Using pip

```bash
cd python-tcp-receiver
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
cd python-tcp-receiver
conda install pyqt
pip install pyqtgraph
```

## Usage

1. Start the receiver:
```bash
python receiver.py
```

2. The receiver will start listening on port 5000

3. Find your computer's IP address:
   - On Linux/Mac: `ip addr` or `ifconfig`
   - On Windows: `ipconfig`

4. On your Android device:
   - Open the DDR TCP Streamer app
   - Enter your computer's IP address
   - Enter port 5000 (default)
   - Tap "Start Streaming"

5. Data will appear in real-time on the graphs

## Data Format

The receiver expects data in CSV format:
```
timestamp_ns,timestamp_ms,sensor_type,x,y,z
```

Where:
- `timestamp_ns`: Sensor timestamp in nanoseconds
- `timestamp_ms`: Wall clock timestamp in milliseconds
- `sensor_type`: Either "accel" or "gyro"
- `x`, `y`, `z`: Sensor values as floats

## Plots

### Accelerometer Plot (Top)
- Shows acceleration in m/s²
- X-axis (red): Left-right movement
- Y-axis (green): Forward-backward movement
- Z-axis (blue): Up-down movement

### Gyroscope Plot (Bottom)
- Shows angular velocity in rad/s
- X-axis (red): Pitch (rotation around X)
- Y-axis (green): Roll (rotation around Y)
- Z-axis (blue): Yaw (rotation around Z)

## Customization

You can modify the following parameters in `receiver.py`:

- `port`: Server port (default: 5000)
- `maxDataPoints`: Number of data points to display (default: 500)
- Update interval: Timer interval in milliseconds (default: 50ms)

## Troubleshooting

### No connection
- Ensure both devices are on the same network
- Check firewall settings (allow incoming connections on port 5000)
- Verify the IP address is correct
- Try pinging the computer from the phone

### Data not showing
- Check that the Android app shows "Connected" status
- Verify sensor permissions are granted on Android device
- Look for error messages in the terminal

### Poor performance
- Reduce `maxDataPoints` for lower memory usage
- Increase update interval for less CPU usage
- Close other applications using the network
