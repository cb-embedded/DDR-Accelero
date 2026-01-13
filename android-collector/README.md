# DDR Collector - Android Sensor Data Collection App

Minimalistic Android app for collecting accelerometer and gyroscope data at maximum sampling rate.

## Features

- **High-frequency sensor sampling**: Uses `HIGH_SAMPLING_RATE_SENSORS` permission to exceed 200Hz
- **Real-time framerate display**: Shows current sampling rate in Hz
- **Binary data format**: Stores data in CBOR (Concise Binary Object Representation)
- **Export options**: 
  - Download button to save file locally
  - Share button using Android Share API

## Sensors Collected

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)

## Requirements

- Android 7.0 (API 24) or higher
- For high-frequency sampling (>200Hz): Android 12 (API 31) or higher

## Build

```bash
./gradlew assembleDebug
```

## Install

```bash
./gradlew installDebug
```

## Data Format

The app exports data in CBOR format with the following structure:

```
{
  "device": "Device model name",
  "timestamp": "2026-01-13_12-34-56",
  "sample_count": 1000,
  "timestamps": [nanosecond timestamps...],
  "accelerometer": [
    {"x": 0.1, "y": 0.2, "z": 9.8},
    ...
  ],
  "gyroscope": [
    {"x": 0.01, "y": 0.02, "z": 0.03},
    ...
  ]
}
```

## Usage

1. Open the app
2. Grant sensor permission if prompted (Android 12+)
3. Tap "Start Recording" to begin collecting sensor data
4. Framerate will be displayed in real-time
5. Tap "Stop Recording" to finish collection
6. Use "Download" to save the file locally
7. Use "Share" to share the file via any app (email, drive, etc.)
