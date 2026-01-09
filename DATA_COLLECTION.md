# Data Collection Guide

## Overview
This guide explains how to collect data for training the DDR-Accelero machine learning model.

## Required Tools
1. **Android smartphone** with sensors (accelerometer, gyroscope, magnetometer)
2. **Sensor Logger app** - Download from Google Play Store
3. **DDR/Stepmania game** - For gameplay
4. **Video recording** (optional but recommended) - For label synchronization

## Data Collection Process

### Step 1: Install Sensor Logger
1. Install "Sensor Logger" app from Google Play Store
2. Open the app and grant necessary permissions
3. Configure sensors to record:
   - Accelerometer
   - Gyroscope
   - Magnetometer
4. Set recording rate to at least 100Hz (if available)

### Step 2: Prepare Recording Session
1. **Mount the phone securely** to your body:
   - Recommended: waist/hip position using belt clip or armband
   - Ensure phone orientation is consistent across sessions
   - Phone should not move independently during gameplay
2. Note the phone orientation (screen facing in/out, top/bottom)

### Step 3: Record a Session
1. Start Sensor Logger recording
2. **Wait 5 seconds** before starting the game (for synchronization)
3. Play the DDR/Stepmania song
4. Focus on clear, distinct movements for each arrow
5. **Wait 5 seconds** after finishing the song
6. Stop Sensor Logger recording

### Step 4: Export Sensor Data
1. In Sensor Logger, select the recorded session
2. Export as ZIP file
3. Transfer to your computer
4. Name the file descriptively: `session_songname_difficulty_YYYYMMDD.zip`

### Step 5: Create Label File
You need to create a CSV file with arrow press timestamps.

#### Method A: Manual Labeling (Most Accurate)
1. Watch video recording of gameplay
2. Note timestamp and arrow for each press
3. Create CSV file:
```csv
timestamp,arrow
1234567890.123,LEFT
1234567890.456,DOWN
1234567890.789,UP
1234567891.123,RIGHT
```

#### Method B: From Game Log/Replay (If Available)
1. Extract timing data from game replay/log files
2. Convert to same CSV format
3. Adjust timestamps to match sensor recording start time

#### Method C: From Chart File (Approximate)
1. Use the song's chart file (.sm or .ssc)
2. Calculate arrow timings from BPM and note positions
3. Add offset to match sensor recording start time

**Important**: Timestamps should be in Unix time (seconds since epoch) and match the format in Sensor Logger CSV files.

### Step 6: Organize Files
Place files in the following structure:
```
data/raw/
├── session1_song1_easy_20260109.zip
├── session1_song1_easy_20260109_labels.csv
├── session2_song2_hard_20260109.zip
└── session2_song2_hard_20260109_labels.csv
```

## Data Quality Tips

### Do's:
- ✅ Keep phone securely mounted in same position
- ✅ Use consistent phone orientation
- ✅ Make clear, distinct movements for each arrow
- ✅ Collect multiple sessions with different songs
- ✅ Record at highest available sensor rate
- ✅ Include variety of difficulties and patterns

### Don'ts:
- ❌ Don't let phone move/bounce independently
- ❌ Don't change phone position mid-session
- ❌ Don't rush movements (accuracy over speed)
- ❌ Don't mix different phone positions in same dataset
- ❌ Don't forget to label your data

## Example Label File Format

### labels.csv
```csv
timestamp,arrow
1704800010.123,LEFT
1704800010.523,DOWN
1704800010.923,LEFT
1704800011.323,RIGHT
1704800011.723,UP
1704800012.123,DOWN
1704800012.523,UP
1704800012.923,RIGHT
```

**Timestamp**: Unix timestamp in seconds (with millisecond precision)
**Arrow**: One of `LEFT`, `DOWN`, `UP`, `RIGHT`

## Processing Your Data

Once you have collected data:

```bash
# 1. Extract and validate sensor data
python scripts/extract_sensor_data.py data/raw/*.zip

# 2. Synchronize sensors and apply labels
python scripts/sync_and_label.py session1_song1_easy_20260109 --labels data/labels/session1_song1_easy_20260109_labels.csv

# 3. Prepare final dataset
python scripts/prepare_dataset.py
```

## Troubleshooting

### Issue: Timestamps don't match
- **Solution**: Sensor Logger timestamps are in milliseconds, not seconds. The scripts handle this automatically, but check if manual conversion is needed.

### Issue: Labels not aligned with movements
- **Solution**: Adjust the `label_tolerance_ms` in `config.yaml` to allow wider matching window.

### Issue: Poor sensor data quality
- **Solution**: Ensure phone is securely mounted, increase recording rate, check sensor calibration.

### Issue: Missing sensor data
- **Solution**: Verify all three sensors (acc, gyro, mag) are enabled in Sensor Logger settings.
