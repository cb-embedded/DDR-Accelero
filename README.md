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
- `artifacts/trained_model.pth`: Trained CNN model saved in PyTorch format
- `artifacts/training_history.png`: Training and validation loss/accuracy curves
- `artifacts/prediction_sample_*.png`: Sample predictions on test data

**Note**: The model currently predicts only arrow labels, not offsets (see "Understanding the Dataset Design" section below for why offset prediction would be beneficial).

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

#### Dataset Structure (X, Y, Offsets)

**X: Raw Sensor Data Windows** [N × window_length × 9 channels]
- 9 channels: accelerometer (x,y,z), gyroscope (x,y,z), magnetometer (x,y,z)
- **IMPORTANT**: Windows are **randomly sampled** from the sensor timeline
- Window size: configurable (default ±1s = 2s total)
- The random sampling simulates real-world scenarios where the model sees arbitrary time windows

**Y: Arrow Labels** [N × 4]
- 4 arrows: [Left, Down, Up, Right] (binary: 1=pressed, 0=not pressed)
- Label is the **closest arrow combination** to the window center
- Supports single and double arrow events
- **Key Point**: Y represents which arrows are pressed near (but not necessarily at) the window center

**Offsets: Relative Time Offsets** [N]
- **CRUCIAL FOR ROBUST PREDICTION**: Offset in seconds of the label arrow from the window center
- Positive = arrow occurs after center, negative = arrow occurs before center
- Range: typically within ±3 seconds
- **Why This Matters**:
  - In real-life inference, the model will only see random windows without knowing arrow timing
  - The model must predict not only which arrows are pressed, but also estimate their relative position
  - Without offset information, the model cannot handle temporal uncertainty
  - This is what enables the model to work on unsynchronized sensor data

**Visualizations**: PNG files (`artifacts/dataset_sample_*.png`) showing:
- 9 sensor channel plots with blue vertical line at window center (t=0)
- Red vertical line at the label arrow position (shows the offset visually)
- Arrow chronogram showing nearby arrow events
- Highlighted label arrows in the legend
- Offset information in the title (e.g., "offset: +0.234s from center")

### ML Pipeline Output
Trains a CNN-based multi-label classifier to predict arrow presses:
- **Model Architecture**: 1D Convolutional Neural Network (CNN)
  - Input: Raw sensor time series [batch_size, 9 channels, time_steps]
  - 3 Conv1D layers with batch normalization and max pooling
  - Fully connected layers with dropout
  - Output: 4 sigmoid activations (one per arrow)
- **Current Implementation**: Predicts arrow labels (Y) only
  - **LIMITATION**: Does not currently predict offsets, only which arrows are near window center
  - **Future Enhancement**: Should be extended to predict both arrows AND offset for robust real-world inference
- **Training Data**: Uses randomly sampled windows with varying offsets (see Dataset section)
- **Evaluation Metrics**: 
  - Exact match accuracy (all 4 arrows must match)
  - Per-arrow accuracy
  - Hamming loss
- **Saved Model**: `artifacts/trained_model.pth` (PyTorch format)
- **Visualizations**: 
  - `artifacts/training_history.png` - Loss and accuracy curves
  - `artifacts/prediction_sample_*.png` - 10 random test predictions with sensor data

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, torch (PyTorch)

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

## Understanding the Dataset Design

### Why Random Windows?

The dataset uses **randomly sampled windows** (not centered on arrows) to simulate real-world inference scenarios:

**Training Scenario:**
- We know the precise timing of arrow events (from alignment with .sm files)
- But we deliberately sample windows at random positions
- Each window is labeled with its nearest arrow AND the offset from window center

**Real-World Inference Scenario:**
- The model receives a live sensor stream
- No knowledge of when arrows should be pressed
- The model must predict: "Which arrows?" AND "How far away are they?"

**Why Offset Prediction is Crucial:**
Without offset prediction, the model can only answer "What arrows are near this window?" but cannot tell you WHEN to press them. The offset tells you:
- Negative offset (-0.5s): "Arrow was 0.5s ago, you're late!"
- Near-zero offset (±0.1s): "Arrow is RIGHT NOW, press it!"
- Positive offset (+0.5s): "Arrow is 0.5s away, get ready!"

This design enables the model to work with arbitrary sensor windows, making it practical for real-time gameplay assistance.

### Current Model Limitation

⚠️ **Important Note**: The current `train_model.py` implementation trains a model that predicts only the arrow labels (Y), not the offsets. This means it can identify which arrows are near the window center, but cannot estimate their precise timing.

**To fully leverage the dataset design**, the model should be extended to:
1. Predict arrow labels: [Left, Down, Up, Right] (binary vector)
2. Predict offset: continuous value in seconds (regression output)

This would enable true real-world inference where the model can guide players on both WHAT to press and WHEN to press it.