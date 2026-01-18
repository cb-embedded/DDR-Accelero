# DDR Collector - Android Sensor Data Collection App

Minimalistic Android app for collecting accelerometer and gyroscope data at maximum sampling rate.

## Features

- **High-frequency sensor sampling**: Uses `HIGH_SAMPLING_RATE_SENSORS` permission to exceed 200Hz
- **Background recording**: Continue recording even when app is in background using foreground service
- **Consistent sample rate**: Maintains stable sampling rate using wake lock
- **Real-time framerate display**: Shows current sampling rate in Hz
- **Binary data format**: Stores data in a compact dedicated binary format
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

The app exports data in a custom binary format (`.ddrbin` files) with the following structure:

### File Header (32 bytes)

| Field           | Size (bytes) | Type      | Description                      |
|-----------------|--------------|-----------|----------------------------------|
| magic_number    | 4            | uint32    | 0x44444143 ('DDAC')              |
| version         | 1            | uint8     | Format version (currently 1)     |
| timestamp_start | 8            | uint64    | Recording start epoch (ns)       |
| reserved        | 19           | uint8[19] | Reserved for future use (zeros)  |

### Data Records (21 bytes each)

| Field        | Size | Type    | Description                        |
|--------------|------|---------|------------------------------------|
| sensor_type  | 1    | uint8   | 1 = accelerometer, 2 = gyroscope   |
| timestamp_ns | 8    | uint64  | Sample timestamp (nanoseconds)     |
| x            | 4    | float32 | X-axis value                       |
| y            | 4    | float32 | Y-axis value                       |
| z            | 4    | float32 | Z-axis value                       |

All values are stored in little-endian byte order. The file consists of a single 32-byte header followed by a stream of 21-byte data records.

### Python Reader Utility

To read and decode `.ddrbin` files, use this Python script:

```python
import struct
import sys

def read_ddrbin(filename):
    """Read and parse a .ddrbin sensor data file"""
    
    with open(filename, 'rb') as f:
        # Read header (32 bytes)
        header = f.read(32)
        
        # Parse header
        magic = struct.unpack('<I', header[0:4])[0]
        version = struct.unpack('<B', header[4:5])[0]
        timestamp_start = struct.unpack('<Q', header[5:13])[0]
        
        print(f"Magic: 0x{magic:08X} (expected: 0x44444143 'DDAC')")
        print(f"Version: {version}")
        print(f"Start timestamp: {timestamp_start} ns ({timestamp_start / 1e9:.3f} s)")
        print()
        
        if magic != 0x44444143:
            print("ERROR: Invalid magic number!")
            return
        
        # Read data records
        accel_count = 0
        gyro_count = 0
        
        while True:
            record = f.read(21)
            if len(record) < 21:
                break
            
            # Parse record
            sensor_type = struct.unpack('<B', record[0:1])[0]
            timestamp_ns = struct.unpack('<Q', record[1:9])[0]
            x = struct.unpack('<f', record[9:13])[0]
            y = struct.unpack('<f', record[13:17])[0]
            z = struct.unpack('<f', record[17:21])[0]
            
            sensor_name = "accel" if sensor_type == 1 else "gyro" if sensor_type == 2 else "unknown"
            
            if sensor_type == 1:
                accel_count += 1
            elif sensor_type == 2:
                gyro_count += 1
            
            # Print first few samples
            if accel_count + gyro_count <= 10:
                print(f"{sensor_name:8} t={timestamp_ns:20} x={x:8.4f} y={y:8.4f} z={z:8.4f}")
        
        print()
        print(f"Total samples: {accel_count} accelerometer, {gyro_count} gyroscope")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python read_ddrbin.py <file.ddrbin>")
        sys.exit(1)
    
    read_ddrbin(sys.argv[1])
```

Save this as `read_ddrbin.py` and run:
```bash
python read_ddrbin.py sensor_data_2026-01-18_12-34-56.ddrbin
```

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
- `WAKE_LOCK`: Prevents CPU from sleeping during background recording to maintain consistent sample rate. Automatically released after 1 hour or when recording stops to minimize battery impact.
- `FOREGROUND_SERVICE`: Required for background recording
- `FOREGROUND_SERVICE_SPECIAL_USE`: Special permission for sensor data collection service
- `POST_NOTIFICATIONS` (Android 13+): Required to show recording notification
- `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` (Android 6+): Allows the app to request exemption from battery optimization to prevent system throttling during background recording, which could reduce data quality and consistency

All permissions are requested at runtime when needed.
