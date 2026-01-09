# DDR-Accelero Dataset Preparation Specification

## Overview
This specification defines the data preparation pipeline for training machine learning models to predict DDR (Dance Dance Revolution) / Stepmania arrow inputs from smartphone sensor data.

## Data Source
- **Input Format**: ZIP files from Android "Sensor Logger" app
- **Sensors Used**:
  - Accelerometer (3-axis: x, y, z)
  - Gyroscope (3-axis: x, y, z)
  - Magnetometer (3-axis: x, y, z)

## Expected Sensor Logger Data Format
The Sensor Logger app typically generates CSV files with the following structure:
- `Accelerometer.csv`: timestamp, x, y, z (m/s²)
- `Gyroscope.csv`: timestamp, x, y, z (rad/s)
- `Magnetometer.csv`: timestamp, x, y, z (μT)

Each file contains:
- Column 1: Timestamp (Unix time in seconds or milliseconds)
- Columns 2-4: Sensor readings for x, y, z axes

## DDR Arrow Mapping
DDR has 4 directional arrows:
- **LEFT**: Left arrow
- **DOWN**: Down arrow  
- **UP**: Up arrow
- **RIGHT**: Right arrow

## Data Collection Requirements
For supervised learning, we need:
1. **Sensor Data**: Raw accelerometer, gyroscope, magnetometer readings
2. **Labels**: Corresponding arrow presses with timestamps
3. **Metadata**: Session information (song, difficulty, player, device orientation)

## Dataset Preparation Pipeline

### 1. Data Loading
- Extract ZIP files from Sensor Logger
- Parse CSV files for each sensor type
- Validate data integrity (no missing timestamps, consistent sampling rates)

### 2. Data Synchronization
- Align timestamps across all sensor types
- Handle different sampling rates (interpolation if needed)
- Time window: Use sliding windows (e.g., 200ms) for feature extraction

### 3. Labeling Strategy
Since labels (arrow presses) need to be synchronized with sensor data:
- **Option A**: Manual labeling using video playback
- **Option B**: Automatic labeling using game logs/replay files
- **Option C**: Label timing file (CSV with timestamp and arrow)

**Recommended**: Create a simple label file format:
```csv
timestamp,arrow
1234567890.123,LEFT
1234567890.456,DOWN
1234567890.789,UP
```

### 4. Feature Engineering
Raw features per time window:
- Mean, standard deviation, min, max for each axis
- Magnitude: sqrt(x² + y² + z²)
- Frequency domain features (FFT)
- Angular velocity magnitude
- Jerk (rate of change of acceleration)

### 5. Data Cleaning
- Remove outliers (sensor noise)
- Handle missing values
- Normalize sensor readings

### 6. Dataset Structure
Output organized as:
```
data/
├── raw/              # Original ZIP files and extracted CSVs
├── processed/        # Cleaned and synchronized data
├── features/         # Extracted feature matrices
├── labels/           # Arrow labels with timestamps
└── splits/           # Train/validation/test splits
    ├── train/
    ├── val/
    └── test/
```

### 7. Train/Validation/Test Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

Split by session (not by individual samples) to avoid data leakage.

## Output Format
Final dataset format: NumPy arrays or CSV files
- **Features**: Shape (n_samples, n_features)
- **Labels**: Shape (n_samples,) - one-hot encoded or integer classes
- **Metadata**: JSON file with dataset statistics

## Minimal Implementation Requirements
1. **Script 1**: `extract_sensor_data.py` - Extract and validate sensor data from ZIP
2. **Script 2**: `sync_and_label.py` - Synchronize sensors and apply labels
3. **Script 3**: `prepare_dataset.py` - Feature extraction and train/test split
4. **Configuration**: `config.yaml` - Parameters (window size, overlap, features, etc.)
5. **Documentation**: Instructions for data collection and labeling

## Dependencies
- Python 3.8+
- pandas: Data manipulation
- numpy: Numerical operations
- scipy: Signal processing
- scikit-learn: Dataset splitting and preprocessing
- pyyaml: Configuration management

## Future Enhancements
- Real-time prediction pipeline
- Data augmentation techniques
- Support for multiple players/devices
- Advanced feature engineering (wavelet transforms, etc.)
