# DDR-Accelero
Virtual Dance Pad using smartphone accelerometer.

## Project Overview
DDR-Accelero uses machine learning to predict DDR (Dance Dance Revolution) / Stepmania arrow inputs from smartphone sensor data (accelerometer, gyroscope, magnetometer).

## Features
- Dataset preparation scripts for sensor data from Android "Sensor Logger" app
- Feature extraction from time-series sensor data
- Train/validation/test dataset splitting
- Support for custom labeling and data collection workflows

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Collect Data
Follow the [Data Collection Guide](DATA_COLLECTION.md) to record sensor data during DDR gameplay.

#### 2. Extract Sensor Data
```bash
python scripts/extract_sensor_data.py data/raw/your_session.zip
```

This extracts and validates sensor data from Sensor Logger ZIP files.

#### 3. Synchronize and Label
```bash
python scripts/sync_and_label.py your_session --labels data/labels/your_session_labels.csv
```

This synchronizes sensor timestamps and applies arrow labels.

#### 4. Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

This extracts features and creates train/val/test splits.

### Output
Final dataset is saved to `data/splits/dataset.npz` with:
- `X_train`, `y_train` - Training features and labels
- `X_val`, `y_val` - Validation features and labels
- `X_test`, `y_test` - Test features and labels

## Configuration
Edit `config.yaml` to customize:
- Sampling rate and window parameters
- Feature extraction methods
- Train/val/test split ratios
- Data cleaning and normalization options

## Project Structure
```
DDR-Accelero/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── SPEC.md                 # Technical specification
├── DATA_COLLECTION.md      # Data collection guide
├── scripts/
│   ├── extract_sensor_data.py   # Step 1: Extract from ZIP
│   ├── sync_and_label.py        # Step 2: Synchronize & label
│   └── prepare_dataset.py       # Step 3: Feature extraction & split
└── data/
    ├── raw/                # Original ZIP files
    ├── processed/          # Extracted sensor CSVs
    ├── features/           # Synchronized data
    ├── labels/             # Arrow label files
    └── splits/             # Final train/val/test datasets
```

## Documentation
- [SPEC.md](SPEC.md) - Detailed specification of data preparation pipeline
- [DATA_COLLECTION.md](DATA_COLLECTION.md) - Guide for collecting training data

## Next Steps
1. Collect training data (see DATA_COLLECTION.md)
2. Run dataset preparation pipeline
3. Train ML model on prepared dataset
4. Deploy real-time prediction system

## License
See LICENSE file for details.
