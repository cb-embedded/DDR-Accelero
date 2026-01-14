# DDR Collector - Android Sensor Data Collection App

Minimalistic Android app for collecting accelerometer and gyroscope data at maximum sampling rate.

## Features

- **High-frequency sensor sampling**: Uses `HIGH_SAMPLING_RATE_SENSORS` permission to exceed 200Hz
- **Background recording**: Continue recording even when app is in background using foreground service
- **Consistent sample rate**: Maintains stable sampling rate using wake lock
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
- For notifications: Android 13 (API 33) or higher (required for background recording)

## Build

### Using Gradle Wrapper (Recommended)

```bash
cd android-collector
./gradlew assembleDebug
```

The APK will be generated at: `app/build/outputs/apk/debug/app-debug.apk`

### Using GitHub Actions

The repository includes a GitHub Actions workflow that automatically builds the APK when changes are pushed to the `android-collector` directory. The built APK is available as a downloadable artifact from the Actions tab.

### Prerequisites

- JDK 17 or higher
- Android SDK (automatically downloaded by Gradle)

If building locally and Android SDK is not automatically detected, create a `local.properties` file:

```
sdk.dir=/path/to/android/sdk
```

## Install

### Option 1: Build and Install

```bash
cd android-collector
./gradlew installDebug
```

### Option 2: Manual Install

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Option 3: Download from GitHub Actions

Download the pre-built APK from the [GitHub Actions artifacts](../../actions) and install it manually on your device.

## Data Format

The app exports data in CBOR format with the following structure:

```
{
  "device": "Device model name",
  "timestamp": "2026-01-13_12-34-56",
  "accelerometer_count": 500,
  "gyroscope_count": 500,
  "accelerometer": [
    {"timestamp": 1234567890, "x": 0.1, "y": 0.2, "z": 9.8},
    ...
  ],
  "gyroscope": [
    {"timestamp": 1234567891, "x": 0.01, "y": 0.02, "z": 0.03},
    ...
  ]
}
```

Each sensor reading includes its own nanosecond timestamp for precise synchronization.

## Usage

1. Open the app
2. Grant sensor permission if prompted (Android 12+)
3. Grant notification permission if prompted (Android 13+, required for background recording)
4. Tap "Start Recording" to begin collecting sensor data
5. The app will continue recording even when in background
6. Framerate will be displayed in real-time and shown in notification
7. Tap "Stop Recording" to finish collection
8. Use "Download" to save the file locally
9. Use "Share" to share the file via any app (email, drive, etc.)

## Permissions

The app requires the following permissions:

- `HIGH_SAMPLING_RATE_SENSORS` (Android 12+): Allows sampling rates above 200Hz
- `FOREGROUND_SERVICE`: Required for background recording
- `FOREGROUND_SERVICE_SPECIAL_USE`: Special permission for sensor data collection service
- `POST_NOTIFICATIONS` (Android 13+): Required to show recording notification

All permissions are requested at runtime when needed.
