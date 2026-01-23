# DDR-Accelero

Machine Learning-powered Dance Dance Revolution arrow prediction from accelerometer data.

**GitHub Pages:** [https://cb-embedded.github.io/DDR-Accelero/](https://cb-embedded.github.io/DDR-Accelero/)

## Projects

This repository contains multiple components:

### 1. Android Data Collector (`android-collector/`)
Original Android app for collecting accelerometer and gyroscope data at maximum sampling rate. Data is saved to binary files (.ddrbin format) for later analysis.

- High-frequency sensor sampling (>200Hz on supported devices)
- Background recording with foreground service
- Binary data format for efficient storage
- Export and share functionality

[See detailed README](android-collector/README.md)

### 2. Android TCP Streamer (`android-tcp-streamer/`)
Variant of the data collector that streams sensor data in real-time over TCP instead of saving to files.

- Streams accelerometer and gyroscope data over TCP
- Configurable server IP address and port
- Simple CSV-like text format
- Background streaming with foreground service
- Maintains all background features (notification, service, battery wake lock)

[See detailed README](android-tcp-streamer/README.md)

### 3. Python TCP Receiver (`python-tcp-receiver/`)
Companion application for the TCP streamer that receives and visualizes sensor data in real-time.

- Real-time data visualization using PyQtGraph
- Separate plots for accelerometer and gyroscope
- Connection monitoring and statistics
- Sliding window display of last 500 samples

[See detailed README](python-tcp-receiver/README.md)

## Quick Start - TCP Streaming

1. **Install Python receiver on your computer:**
   ```bash
   cd python-tcp-receiver
   pip install -r requirements.txt
   python receiver.py
   ```

2. **Build and install Android app:**
   ```bash
   cd android-tcp-streamer
   ./gradlew installDebug
   ```

3. **Connect and stream:**
   - Find your computer's IP address
   - Open the Android app and enter the IP address
   - Tap "Start Streaming"
   - Watch real-time sensor data in the Python app

## Data Collection Only

If you just want to collect sensor data without streaming:

```bash
cd android-collector
./gradlew installDebug
```

Use the app to record data, then export files and analyze them using the provided Python script (`android-collector/read_ddrbin.py`).
