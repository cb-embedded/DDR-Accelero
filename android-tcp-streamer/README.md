# DDR TCP Streamer - Android Sensor Data Streaming App

Android app that streams accelerometer and gyroscope data over TCP in real-time.

## Features

- **High-frequency sensor sampling**: Uses `HIGH_SAMPLING_RATE_SENSORS` permission to exceed 200Hz
- **Background streaming**: Continue streaming even when app is in background using foreground service
- **Consistent sample rate**: Maintains stable sampling rate using wake lock
- **Real-time framerate display**: Shows current sampling rate in Hz
- **Simple text format**: CSV-like format for easy parsing
- **Configurable server**: Enter any IP address and port

## Sensors Streamed

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)

## Data Format

Data is streamed as plain text lines over TCP, one sample per line:

```
timestamp_ns,timestamp_ms,sensor_type,x,y,z
```

Where:
- `timestamp_ns`: Sensor event timestamp in nanoseconds (from system boot)
- `timestamp_ms`: Wall clock timestamp in milliseconds (epoch time)
- `sensor_type`: Either "accel" or "gyro"
- `x`, `y`, `z`: Sensor values as floats

Example:
```
12345678901234,1705678901234,accel,0.123,9.81,-0.456
12345678901235,1705678901234,gyro,0.001,-0.002,0.003
```

## Requirements

- Android 7.0 (API 24) or higher
- For high-frequency sampling (>200Hz): Android 12 (API 31) or higher
- For notifications: Android 13 (API 33) or higher (required for background streaming)

## Build

```bash
cd android-tcp-streamer
./gradlew assembleDebug
```

The APK will be generated at: `app/build/outputs/apk/debug/app-debug.apk`

## Install

```bash
cd android-tcp-streamer
./gradlew installDebug
```

Or manually:
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Usage

1. Start the Python receiver on your computer (see `../python-tcp-receiver/`)
2. Find your computer's IP address on the local network
3. Open the Android app
4. Enter your computer's IP address and port (default: 5000)
5. Tap "Start Streaming"
6. The app will connect and start streaming sensor data
7. The app will continue streaming even when in background
8. Tap "Stop Streaming" to disconnect

## Permissions

The app requires the following permissions:

- `HIGH_SAMPLING_RATE_SENSORS` (Android 12+): Allows sampling rates above 200Hz
- `WAKE_LOCK`: Prevents CPU from sleeping during background streaming
- `FOREGROUND_SERVICE`: Required for background streaming
- `FOREGROUND_SERVICE_SPECIAL_USE`: Special permission for sensor data collection service
- `POST_NOTIFICATIONS` (Android 13+): Required to show streaming notification
- `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` (Android 6+): Allows exemption from battery optimization
- `INTERNET`: Required for TCP communication

All permissions are requested at runtime when needed.
