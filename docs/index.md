# DDR-Accelero

Machine Learning-powered Dance Dance Revolution arrow prediction from accelerometer data.

## Overview

This project explores using machine learning to predict Dance Dance Revolution (DDR) arrow patterns from accelerometer data collected while playing the game.

## Repository Components

### Android Data Collector
Original Android app for collecting accelerometer and gyroscope data at maximum sampling rate. Data is saved to binary files (.ddrbin format) for later analysis.

[View on GitHub](https://github.com/cb-embedded/DDR-Accelero/tree/main/android-collector)

### Android TCP Streamer
Variant of the data collector that streams sensor data in real-time over TCP instead of saving to files.

[View on GitHub](https://github.com/cb-embedded/DDR-Accelero/tree/main/android-tcp-streamer)

### Python TCP Receiver
Companion application for the TCP streamer that receives and visualizes sensor data in real-time.

[View on GitHub](https://github.com/cb-embedded/DDR-Accelero/tree/main/python-tcp-receiver)

### C Pad Recorder
C-based component for recording pad inputs.

[View on GitHub](https://github.com/cb-embedded/DDR-Accelero/tree/main/c-pad-recorder)

## Quick Start - TCP Streaming

1. **Install Python receiver on your computer:**
   ```bash
   cd python-tcp-receiver
   pip install -r requirements.txt
   python receiver.py
   ```

2. **Install and run the Android TCP Streamer app:**
   - Download the APK from the releases page
   - Configure your computer's IP address and port
   - Start streaming

## Getting Started

Visit the [main repository](https://github.com/cb-embedded/DDR-Accelero) for detailed documentation and source code.

## License

This project is open source. See the repository for license information.
