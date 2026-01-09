# Experiment Proposals for DDR-Accelero

## Project Goal
Predict DDR (Dance Dance Revolution) arrow presses using smartphone accelerometer and gyroscope data.

## Data Available
- **Raw sensor captures** (10 files): Android Sensor Logger data containing accelerometer, gyroscope, and magnetometer readings
- **StepMania files** (81 files): `.sm` files with note timing and arrow patterns at various difficulty levels

## Proposed Experimental Pipeline

I've created 5 independent experiments that progressively build toward solving the problem:

### 1. Data Exploration (`01_visualize_sensor_data.py`)
**Goal:** Understand sensor data characteristics

**What it does:**
- Extracts accelerometer and gyroscope data from zip files
- Plots time-series signals for all axes (X, Y, Z)
- Analyzes sample rate, duration, and signal characteristics

**Key insights:**
- Visualize signal patterns during gameplay
- Identify which sensors/axes are most responsive to movements
- Understand noise levels and signal quality

### 2. Chart Parsing (`02_parse_stepmania.py`)
**Goal:** Extract timing information from StepMania charts

**What it does:**
- Parses `.sm` files to extract metadata (BPM, offset, artist, title)
- Extracts note timing and arrow patterns for each difficulty
- Calculates note density over time

**Key insights:**
- Converts chart data into timestamps
- Understands the relationship between beats and real time
- Identifies patterns in note sequences

### 3. Signal Alignment (`03_align_signals.py`)
**Goal:** Find temporal alignment between sensor data and chart

**What it does:**
- Creates a reference signal from note timings (impulse at each note)
- Preprocesses sensor signal (filtering, envelope detection)
- Uses cross-correlation to find the best time offset

**Key insights:**
- Determines when the chart starts in the sensor recording
- Critical for supervised learning (aligning labels with features)
- Visualizes alignment quality

**Algorithm:**
1. Create reference signal: spike at each note time
2. Compute sensor signal envelope (filtered acceleration magnitude)
3. Cross-correlate the two signals
4. Peak correlation = best time alignment

### 4. Dataset Generation (`04_generate_dataset.py`)
**Goal:** Create labeled training data

**What it does:**
- Extracts features from time windows (±250ms) around each note
- Computes statistical features: mean, std, min, max, range, skewness, kurtosis
- Creates positive samples (arrow presses) and negative samples (no presses)
- Saves as CSV with features and binary labels for each arrow

**Features extracted:**
- Per-axis statistics (X, Y, Z) for each sensor
- Magnitude features (energy, max, mean)
- Temporal features (peak count)
- ~30-40 features per sample

**Labels:**
- Binary label for each arrow: left, down, up, right
- Multi-label classification problem

### 5. Model Training (`05_train_model.py`)
**Goal:** Train a baseline predictive model

**What it does:**
- Loads the generated dataset
- Trains separate Random Forest classifiers for each arrow
- Evaluates performance with accuracy, precision, recall, F1
- Analyzes feature importance

**Key insights:**
- Demonstrates feasibility of prediction from sensor data
- Identifies which features are most predictive
- Provides baseline performance metrics

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments in order
cd experiments
python 01_visualize_sensor_data.py
python 02_parse_stepmania.py
python 03_align_signals.py
python 04_generate_dataset.py
python 05_train_model.py
```

## Expected Challenges

1. **Time Alignment:** Even small timing errors significantly degrade performance
   - Solution: Manual annotation tool or improved alignment algorithm (DTW)

2. **Phone Position:** Sensor readings depend on phone placement
   - Solution: Data augmentation or position-invariant features

3. **Limited Data:** Currently only 10 recordings
   - Solution: Collect more data across songs, users, difficulties

4. **Real-time Prediction:** Need low latency for practical use
   - Solution: Optimize model, use online learning, sliding windows

## Next Steps for Improvement

### Short-term
1. Run all experiments on the existing data
2. Analyze results and identify bottlenecks
3. Improve time alignment algorithm
4. Experiment with different feature sets

### Medium-term
1. Collect more diverse training data
2. Try deep learning models (LSTM, CNN, Transformers)
3. Implement real-time prediction pipeline
4. Add user study for evaluation

### Long-term
1. Build mobile app for real-time prediction
2. Explore transfer learning across songs/users
3. Investigate multi-modal fusion (audio + sensors)
4. Create public dataset for research community

## Additional Experiment Ideas

### Experiment 6: Time-Frequency Analysis
- Use wavelet transform or STFT to analyze frequency content
- Identify characteristic frequencies for each arrow direction
- May reveal patterns invisible in time domain

### Experiment 7: Sequence Modeling
- Use LSTM or GRU to capture temporal dependencies
- Predict arrow sequences rather than individual arrows
- Better for learning game patterns

### Experiment 8: User Calibration
- Collect calibration data for each user
- Learn user-specific patterns
- Improve personalized performance

### Experiment 9: Multi-Song Generalization
- Train on multiple songs
- Evaluate cross-song performance
- Test model generalization

### Experiment 10: Real-Time Evaluation
- Implement sliding window prediction
- Measure latency and throughput
- Test on actual gameplay

## Success Metrics

1. **Alignment Quality:** Can we correctly align sensor data with charts? (visual inspection, peak correlation value)

2. **Arrow Detection:** Per-arrow accuracy, precision, recall, F1 score

3. **Sequence Accuracy:** Percentage of correctly predicted arrow sequences

4. **Latency:** Time from sensor reading to prediction (for real-time use)

5. **Generalization:** Performance on unseen songs, users, difficulties

## Conclusion

The proposed experiments provide a comprehensive exploration of the DDR arrow prediction problem. They progress logically from data understanding to model training, with each experiment building on the previous ones. The modular structure allows for easy experimentation and iteration.

The key technical challenge is accurate time alignment between sensor data and charts. Once this is solved, the machine learning aspects are relatively straightforward, though there's significant room for improvement through better features, models, and data collection.
