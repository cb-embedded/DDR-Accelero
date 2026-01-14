# Sensor Capture Web App

Web-based accelerometer and gyroscope data capture directly from mobile phone browsers.

## Features

- **Maximum sampling rate**: Uses browser's `devicemotion` event for highest available sampling rate
- **Real-time display**: Shows current sensor values and sample rate
- **Metadata preservation**: Captures absolute timestamps for accurate reconstruction
- **Simple export**: Download data as JSON with one click
- **Mobile-optimized**: Responsive design for phone screens

## Usage

1. Open the app on your mobile device: https://cb-embedded.github.io/DDR-Accelero/sensor_capture/
2. Grant motion sensor permission when prompted (iOS requires this)
3. Tap "Start Recording" to begin capturing data
4. Perform your movements (e.g., dance to DDR)
5. Tap "Stop Recording" when done
6. Tap "Export Data" to download the JSON file

## Data Format

Exported JSON structure:

```json
{
  "metadata": {
    "device": "User agent string",
    "platform": "Device platform",
    "recorded_at": "ISO 8601 timestamp",
    "absolute_start_time": "ISO 8601 timestamp",
    "absolute_start_timestamp_ms": 1768392000000,
    "duration_seconds": 10.5,
    "sample_count": 2000,
    "accelerometer_count": 1000,
    "gyroscope_count": 1000,
    "average_sample_rate": 190.5
  },
  "accelerometer": [
    {
      "timestamp": 0.0,
      "x": 0.1,
      "y": 0.2,
      "z": 9.8
    }
  ],
  "gyroscope": [
    {
      "timestamp": 0.016,
      "x": 0.01,
      "y": 0.02,
      "z": 0.03
    }
  ]
}
```

### Metadata Fields

- `device`: Browser user agent for device identification
- `platform`: Device platform (iOS, Android, etc.)
- `recorded_at`: ISO 8601 timestamp of recording start
- `absolute_start_time`: Same as recorded_at (for compatibility)
- `absolute_start_timestamp_ms`: Unix timestamp in milliseconds for absolute time reconstruction
- `duration_seconds`: Total recording duration
- `sample_count`: Total number of samples (accel + gyro)
- `accelerometer_count`: Number of accelerometer samples
- `gyroscope_count`: Number of gyroscope samples
- `average_sample_rate`: Average samples per second

### Sensor Data

- **Timestamps**: Relative timestamps in seconds from recording start
- **Accelerometer**: Linear acceleration including gravity (m/s²)
  - `x`, `y`, `z`: Acceleration along each axis
- **Gyroscope**: Rotation rate (deg/s)
  - `x`: Rotation around X axis (beta)
  - `y`: Rotation around Y axis (gamma)
  - `z`: Rotation around Z axis (alpha)

## Browser Compatibility

- **iOS Safari**: Requires iOS 13+ and user permission
- **Android Chrome**: Works on Android 4.4+
- **Other browsers**: Most modern mobile browsers support DeviceMotionEvent

## Typical Sample Rates

- **Android**: 100-200 Hz (varies by device)
- **iOS**: 50-100 Hz (varies by device)
- **Desktop**: May not have motion sensors

## Notes

- Accelerometer data includes gravity (use `accelerationIncludingGravity`)
- Higher sampling rates drain battery faster
- Keep phone screen on during recording
- Export file size grows with recording duration (~1 KB per second)
