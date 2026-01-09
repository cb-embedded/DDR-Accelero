# DDR-Accelero Experiments

This directory contains a series of independent experiments to explore the DDR arrow prediction problem using smartphone accelerometer data.

## Overview

The goal is to predict which DDR (Dance Dance Revolution) arrows are pressed on a dance pad using only smartphone accelerometer and gyroscope data.

## Setup

Install dependencies:
```bash
pip install -r ../requirements.txt
```

## Experiments

### Experiment 1: Visualize Sensor Data
**File:** `01_visualize_sensor_data.py`

**Purpose:** Extract and visualize raw sensor data from Android Sensor Logger captures to understand signal characteristics.

**Usage:**
```bash
python 01_visualize_sensor_data.py
```

**Output:**
- Plots of accelerometer and gyroscope signals
- Data statistics (duration, sample rate, etc.)
- Saved visualization in `output/` directory

**Key Insights:**
- Understand signal noise and amplitude
- Identify periodic patterns from dance movements
- Determine which sensor axes are most relevant

---

### Experiment 2: Parse StepMania Files
**File:** `02_parse_stepmania.py`

**Purpose:** Parse `.sm` files to extract note timing and arrow patterns.

**Usage:**
```bash
python 02_parse_stepmania.py
```

**Output:**
- List of charts and difficulties
- Note timing information
- Note density analysis

**Key Insights:**
- Extract BPM and offset for timing
- Understand note timing relative to music
- Prepare data for alignment with sensor data

---

### Experiment 3: Signal Alignment
**File:** `03_align_signals.py`

**Purpose:** Find the time offset between sensor data capture and StepMania chart using cross-correlation.

**Usage:**
```bash
python 03_align_signals.py
```

**Output:**
- Best time offset between signals
- Visualization of sensor signal, reference signal, and correlation
- Saved alignment plot in `output/` directory

**Key Insights:**
- Cross-correlation with physics-based modeling can accurately identify timing offset
- Proper alignment is crucial for supervised learning
- Causal impulse response (exponential decay) models biomechanical foot impact better than Gaussian smoothing
- Modeling damped body response produces 41% stronger correlation peaks on average

**Algorithm:**
1. Create reference signal from note timings using **causal impulse response**:
   - Model each foot press as damped inertial response: `h(t) = exp(-t/τ)`
   - Decay time τ = 150ms (typical human body response)
   - Apply same bandpass filter (0.5-10 Hz) as sensor signal
   - Creates "pseudo-acceleration" physically comparable to real signal
2. Preprocess sensor signal (filter, compute envelope)
3. Compute cross-correlation
4. Find peak correlation = best alignment

**Verification:**
See `verify_alignment.py` and `ALIGNMENT_VERIFICATION.md` for proof that the alignment works correctly across multiple captures.

---

### Alignment Verification Script
**File:** `verify_alignment.py`

**Purpose:** Verify the alignment algorithm works correctly by testing on multiple captures and generating correlation plots as proof.

**Usage:**
```bash
python verify_alignment.py
```

**Output:**
- Correlation plots for multiple song captures (saved in `experiments/` directory)
- Summary statistics showing alignment quality
- Verification report with correlation peak values

**Features:**
- Automatically matches sensor files with corresponding .sm files
- Tests alignment on 5 different songs
- Generates detailed correlation plots showing:
  - Processed sensor signal
  - Reference signal from chart
  - Cross-correlation with peak detection
- Provides quality metrics (offset, correlation peak, note count)

**Results:**
All tested captures show clear correlation peaks, proving the alignment algorithm works correctly. See correlation PNG files in experiments directory for visual proof.

---

### Experiment 4: Generate Dataset
**File:** `04_generate_dataset.py`

**Purpose:** Create a labeled dataset by extracting features from time windows around each arrow press.

**Usage:**
```bash
python 04_generate_dataset.py
```

**Output:**
- CSV file with features and labels
- Saved in `output/` directory

**Key Insights:**
- Extract statistical features from sensor windows (±250ms)
- Create both positive (arrow press) and negative (no press) samples
- Features include: mean, std, min, max, range, skewness, kurtosis, peak count, energy

**Features per sensor:**
- Per-axis statistics (X, Y, Z)
- Magnitude features
- Temporal patterns (peaks)

**Labels:**
- Binary labels for each arrow (left, down, up, right)
- Multi-label classification problem

---

### Experiment 5: Train Prediction Model
**File:** `05_train_model.py`

**Purpose:** Train a machine learning model to predict arrow presses from sensor features.

**Usage:**
```bash
python 05_train_model.py
```

**Note:** Run experiment 4 first to generate the dataset.

**Output:**
- Model evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Feature importance analysis

**Key Insights:**
- Random Forest can learn patterns from sensor data
- Performance depends on alignment quality and feature engineering
- Feature importance reveals which sensors/axes are most predictive

**Model:**
- Separate binary classifier for each arrow
- Random Forest with 100 trees
- 80/20 train-test split

---

## Workflow

Run experiments in order for best results:

1. **Visualize data** → Understand signal characteristics
2. **Parse StepMania** → Extract note timing
3. **Align signals** → Find time offset
4. **Generate dataset** → Create training data
5. **Train model** → Build predictor

## Results

Each experiment saves outputs to the `experiments/output/` directory:
- Visualizations (PNG files)
- Datasets (CSV files)
- Analysis results (printed to console)

## Future Improvements

1. **Better Alignment:**
   - Use DTW (Dynamic Time Warping) instead of cross-correlation
   - Incorporate music audio analysis
   - Manual annotation tool for ground truth

2. **Advanced Features:**
   - Frequency domain features (FFT, spectral features)
   - Temporal features (sliding windows, RNNs)
   - Sensor fusion (combine multiple sensors optimally)

3. **Better Models:**
   - Deep learning (LSTM, CNN, Transformers)
   - Multi-task learning (predict arrow combinations)
   - Sequence-to-sequence models
   - Real-time prediction (online learning)

4. **More Data:**
   - Multiple songs
   - Multiple users
   - Different difficulty levels
   - Different phone positions/orientations

5. **Evaluation:**
   - Cross-validation across songs
   - User-independent evaluation
   - Real-time latency testing
   - Error analysis (which arrows are confused?)

## Notes

- **Time alignment is critical:** Even small timing errors will degrade performance significantly
- **Feature engineering matters:** The right features make the difference between success and failure
- **Data quality:** Clean, consistent data collection is essential
- **Phone placement:** Sensor readings depend on where/how the phone is held or mounted

## Known Limitations

- Experiments assume sensor data contains at least accelerometer/gyroscope data
- Feature extraction requires sufficient data points in each window (>5 samples)
- Assumes sensor data format: Time, X, Y, Z columns
- Cross-correlation alignment may need adjustment for very short or noisy recordings

## References

- StepMania file format: https://github.com/stepmania/stepmania/wiki/sm
- Sensor Logger app for Android
- Human Activity Recognition (HAR) literature for inspiration
