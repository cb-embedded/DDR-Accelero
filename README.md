# DDR-Accelero

Alignment tool for Dance Dance Revolution (DDR) accelerometer data with StepMania chart files.

## Current Status

**Working solution implemented** using biomechanical approach with clear correlation peaks.

### Results

Successfully achieved alignment with dominant correlation peaks:

| Recording | Peak Ratio | Z-score | Status |
|-----------|------------|---------|--------|
| Lucky Orb 5 Medium | **2.69** | **7.46** | ✓ SUCCESS |
| Decorator Medium 6 | **2.05** | **6.69** | ✓ SUCCESS |
| **Target** | **>2.0** | **>5.0** | |

## Usage

### Alignment Tool

```bash
python align_clean.py <capture_zip> <sm_file> <difficulty_level>
```

**Example:**
```bash
python align_clean.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5
```

**Arguments:**
- `capture_zip`: Path to sensor capture zip file (containing Gravity.csv)
- `sm_file`: Path to StepMania .sm chart file
- `difficulty_level`: Numeric difficulty level (e.g., 5 for Medium-5)

### Dataset Creation Tool

```bash
python create_dataset.py <capture_zip> <sm_file> <difficulty_level> <num_samples>
```

**Example:**
```bash
python create_dataset.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 10
```

**Arguments:**
- `capture_zip`: Path to sensor capture zip file (containing Gravity.csv, Gyroscope.csv, Magnetometer.csv)
- `sm_file`: Path to StepMania .sm chart file
- `difficulty_level`: Numeric difficulty level (e.g., 5 for Medium-5)
- `num_samples`: Number of sample visualizations to generate (e.g., 10)

**Output:**
- Console: Dataset statistics (number of samples, shapes)
- PNG files: Visualization of sample windows with sensor data and arrow labels
- Dataset in memory: X (sensor windows) and Y (arrow labels)

### Machine Learning Pipeline

```bash
python train_model.py <capture1_zip> <sm1_file> <diff1_level> [<capture2_zip> <sm2_file> <diff2_level> ...]
```

**Example:**
```bash
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6 \
  "raw_data/Charles_5_Medium-2026-01-10_09-22-48.zip" "sm_files/Charles.sm" 5
```

**Arguments:**
- Multiple capture sets (each set: `capture_zip sm_file difficulty_level`)
- Uses same format as dataset tool, but processes multiple captures at once

**Output:**
- Console: Training progress and evaluation metrics
- `artifacts/trained_model.pkl`: Trained model saved for later use

**Results (3 captures, 984 samples):**
- **Average Accuracy: 68.7%** vs Random Baseline: 59.6% (+9.1%)
- Per-arrow accuracy: Left 67%, Down 72%, Up 70%, Right 66%
- Model: Random Forest with statistical features (mean, std, min, max, percentiles per channel)

## Method

The solution uses a **biomechanical model**:
- Focuses on LEFT ARROW only (single direction = consistent pattern)
- Edge event detection (transitions via diff)
- Exponential decay kernel (tau=0.1s) models body dynamics
- FFT-based correlation
- Bandpass filter (0.5-8 Hz) isolates human movement frequencies

## Output

### Alignment Tool Output
Computes time offset between sensor recording and chart:
- Console: offset, peak ratio, z-score
- PNG: correlation plot showing clear peak

### Dataset Tool Output
Creates a labeled dataset from aligned sensor data:
- **X**: Sensor data windows [N × window_length × 9 channels]
  - 9 channels: accelerometer (x,y,z), gyroscope (x,y,z), magnetometer (x,y,z)
  - Window: randomly sampled with configurable size (default ±1s = 2s total)
- **Y**: Arrow labels [N × 4]
  - 4 arrows: [Left, Down, Up, Right] (binary: 1=pressed, 0=not pressed)
  - Label is the closest arrow combination to the window center
  - Supports single and double arrow events
- **Offsets**: Time offsets [N]
  - Offset in seconds of the label arrow from the window center
  - Positive = arrow after center, negative = arrow before center
- **Visualizations**: PNG files showing:
  - 9 sensor channel plots with blue vertical line at window center (t=0)
  - Red vertical line at the label arrow position
  - Arrow chronogram showing nearby arrow events
  - Highlighted label arrows in the legend
  - Offset information in the title

### ML Pipeline Output
Trains a multi-label classifier to predict arrow presses:
- **Model**: Random Forest with Binary Relevance (one classifier per arrow)
- **Features**: Statistical features extracted from sensor windows
  - Per channel: mean, std, min, max, 25th/75th percentiles (54 features total)
- **Evaluation**: Per-arrow accuracy, exact match accuracy, Hamming loss
- **Saved Model**: `artifacts/trained_model.pkl` for inference
  - Highlighted label arrows in the legend

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn

## Project Structure

```
DDR-Accelero/
├── align_clean.py          # Alignment solution (finds time offset)
├── create_dataset.py       # Dataset creation tool (creates labeled samples)
├── train_model.py          # ML pipeline (trains arrow prediction model)
├── raw_data/               # Sensor captures (.zip from Android Sensor Logger)
├── sm_files/               # StepMania charts (.sm files)
└── artifacts/              # Generated plots, visualizations, and trained models
```

## Key Insight

The biomechanical approach works by **matching the physics** of DDR gameplay:
- Single arrow = consistent sensor response
- Exponential kernel = body response dynamics (100ms decay)
- Edge events = transition detection (foot hitting pad)