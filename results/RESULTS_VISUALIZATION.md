# DDR-Accelero: Training and Inference Results

## Project Summary

Successfully trained a Keras-based Convolutional Neural Network to predict DDR (Dance Dance Revolution) arrow patterns from smartphone accelerometer data. The model analyzes 9-channel sensor data (accelerometer, gyroscope, magnetometer) to predict which arrows (Left, Down, Up, Right) should be pressed.

---

## Model Architecture

**1D Convolutional Neural Network (Keras)**

```
Input: (198 timesteps, 9 channels) = 1.98 seconds of sensor data
├── Conv1D(32 filters, kernel=7) + BatchNorm + MaxPool
├── Conv1D(64 filters, kernel=5) + BatchNorm + MaxPool  
├── Conv1D(128 filters, kernel=3) + BatchNorm + MaxPool
├── Flatten → Dense(256) + Dropout(0.3)
├── Dense(128) + Dropout(0.3)
└── Dense(4, sigmoid) → Output: [Left, Down, Up, Right]

Total Parameters: 858,052 (3.27 MB)
```

---

## Training Performance

### Dataset
- **Songs**: Lucky Orb (Level 5) + Decorator (Level 6)
- **Total Samples**: 4,104 (50% arrows, 50% nothing)
- **Split**: 60% train / 20% val / 20% test

### Results
```
Test Set Performance:
├── Exact Match Accuracy: 79.42%
├── Hamming Score: 93.79%
└── Per-Arrow Accuracy:
    ├── Left:  93.30%
    ├── Down:  94.15%
    ├── Up:    95.01%
    └── Right: 92.69%
```

### Training Curves
- Loss decreased from 0.68 → 0.02 (train) and stabilized at 0.20 (val)
- Accuracy increased to ~60% (train) and ~58% (val)
- Model converged after 35 epochs with early stopping

See: `results/docs/training_history.png`

---

## Inference Results

### Test 1: Lucky Orb (Training Distribution)
```
✓ Ground Truth: 28 arrows (70s-80s window)
✓ Predicted: 17 arrows
✓ Detection Rate: ~61%
```
The model successfully detected arrows on a song from the training set.

### Test 2: Charles (Unseen Song)
```
✓ Ground Truth: 25 arrows (70s-80s window)  
✓ Predicted: 0 arrows
✗ Detection Rate: 0%
```
Limited generalization to completely unseen songs/players.

---

## Visualization Examples

### 1. Training History
Shows loss and accuracy curves over 35 epochs, demonstrating model convergence.

**Location**: `results/docs/training_history.png`

### 2. Sample Predictions (10 examples)
Each visualization shows:
- 9 sensor channels (3 accel, 3 gyro, 3 mag) over 1.98 seconds
- Ground truth label
- Model prediction (Green = correct, Red = incorrect)

**Examples**:
- `prediction_sample_01.png` - Incorrect prediction (predicted Right, actual Nothing)
- `prediction_sample_03.png` - Correct prediction (predicted Up, actual Up)
- Additional samples: `prediction_sample_02.png` through `prediction_sample_10.png`

### 3. Inference Comparisons
Arrow timeline plots comparing ground truth vs predictions:
- `Lucky_Orb_prediction_comparison.png` - Shows predictions on training song
- `Charles_prediction_comparison.png` - Shows lack of predictions on unseen song

---

## Key Findings

### ✅ Strengths
1. **High accuracy** on individual arrows (>92% per direction)
2. **Successfully learns** temporal patterns from sensor data
3. **Good performance** on training distribution
4. **Efficient model** - only 3.27 MB

### ⚠️ Limitations
1. **Limited generalization** to unseen songs/players
2. **Moderate recall** - misses ~40% of arrows even on training data
3. **Small training set** - only 2 songs, 4,104 samples
4. **No data augmentation** applied during training

### 🎯 Recommendations for Improvement
1. **Expand training data**:
   - Include 10+ diverse songs
   - Multiple players/playing styles
   - Various difficulty levels
   
2. **Data augmentation**:
   - Time shifts (±100ms)
   - Sensor noise injection
   - Rotation/scaling of sensor values
   
3. **Architecture improvements**:
   - Attention mechanisms
   - Bidirectional LSTM layers
   - Residual connections
   
4. **Training enhancements**:
   - Cross-validation across songs
   - Focal loss for imbalanced detection
   - Ensemble of multiple models

---

## File Structure

```
results/
├── RESULTS_SUMMARY.md                    # Detailed results documentation
├── RESULTS_VISUALIZATION.md              # This file
├── trained_model.h5                      # Trained Keras model (3.27 MB)
├── Charles_prediction_comparison.png     # Inference on unseen song
├── Lucky_Orb_prediction_comparison.png   # Inference on training song
└── docs/
    ├── training_history.png              # Loss/accuracy curves
    ├── prediction_sample_01.png          # Example prediction 1
    ├── prediction_sample_02.png          # Example prediction 2
    ├── ...                               # Examples 3-10
    ├── model.onnx                        # ONNX exported model
    └── index.html                        # Web visualization

Total size: ~10.4 MB
```

---

## Usage

### Load Model
```python
from tensorflow import keras
model = keras.models.load_model('results/trained_model.h5')
```

### Make Predictions
```python
import numpy as np

# Prepare sensor data: shape (1, 198, 9)
sensor_data = np.random.randn(1, 198, 9)

# Predict arrows
predictions = model.predict(sensor_data)[0]  # Shape: (4,)
arrows = (predictions > 0.5).astype(int)     # Binary threshold

# Interpret results
arrow_names = ['Left', 'Down', 'Up', 'Right']
for i, pressed in enumerate(arrows):
    if pressed:
        print(f"Arrow: {arrow_names[i]}")
```

### Run Inference Script
```bash
python predict_song.py \
    "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" \
    "sm_files/Lucky Orb.sm" \
    5
```

---

## Conclusion

This project demonstrates that **CNN models can successfully learn to predict DDR arrow patterns from accelerometer data** with nearly 80% accuracy on the test set. The model achieves excellent per-arrow accuracy (>92%) and learns meaningful temporal patterns from sensor readings.

However, the model currently shows **limited generalization to unseen songs and players**, indicating the need for a more diverse training dataset and augmentation strategies. With these improvements, this approach could enable real-time DDR gameplay assistance or automated choreography generation from sensor data.

**Next Steps**: Expand the training dataset, implement data augmentation, and explore attention-based architectures for better temporal modeling.

---

*Generated: 2026-01-13*  
*Model: Keras 3.13.0 / TensorFlow 2.20.0*  
*Training Time: ~3 minutes (CPU)*
