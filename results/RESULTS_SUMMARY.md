# DDR-Accelero Training and Inference Results

## Overview
This folder contains the results from training a Keras CNN model on DDR accelerometer data and performing inference on test songs.

## Training Results

### Model Architecture
- **Type**: 1D Convolutional Neural Network (CNN) with Keras
- **Input**: 198 timesteps × 9 channels (accelerometer, gyroscope, magnetometer data)
- **Output**: 4-class multi-label prediction (Left, Down, Up, Right arrows)
- **Total Parameters**: 858,052 (3.27 MB)

### Training Configuration
- **Training Songs**: 
  - Lucky Orb (Medium, Level 5)
  - Decorator (Medium, Level 6)
- **Total Samples**: 4,104 (balanced 50/50 arrows vs. nothing)
- **Train/Val/Test Split**: 60/20/20
- **Epochs**: 35 (with early stopping)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary Crossentropy

### Performance Metrics

#### Test Set Results
- **Exact Match Accuracy**: 79.42%
- **Hamming Score**: 93.79%

#### Per-Arrow Accuracy
- Left: 93.30%
- Down: 94.15%
- Up: 95.01%
- Right: 92.69%

### Training History
The model converged well with:
- Training loss decreased from ~0.68 to ~0.02
- Validation loss stabilized around ~0.20
- Training accuracy reached ~60%
- Validation accuracy reached ~58%

See `docs/training_history.png` for visualization.

## Inference Results

### Test on Training Song (Lucky Orb)
- **Ground Truth Arrows**: 28
- **Predicted Arrows**: 17
- **Prediction Window**: 70.0s - 80.0s

The model successfully detected arrows on a song it was trained on, though with some missed detections (recall ~61%).

### Test on Unseen Song (Charles)
- **Ground Truth Arrows**: 25
- **Predicted Arrows**: 0
- **Prediction Window**: 70.0s - 80.0s

The model did not generalize well to completely unseen songs, suggesting the need for:
- More diverse training data
- Data augmentation
- More regularization

## Visualizations

### Training Progress
- `docs/training_history.png` - Loss and accuracy curves over epochs

### Sample Predictions
- `docs/prediction_sample_01.png` to `prediction_sample_10.png` - Individual prediction examples showing:
  - 9 sensor channels (3 accelerometer, 3 gyroscope, 3 magnetometer)
  - Ground truth labels
  - Model predictions (color-coded: green=correct, red=incorrect)

### Inference Comparisons
- `Lucky_Orb_prediction_comparison.png` - Ground truth vs predicted arrows over time
- `Charles_prediction_comparison.png` - Ground truth vs predicted arrows over time

## Model File
- `trained_model.h5` - Trained Keras model (858K parameters, ~3.3 MB)

## Key Findings

### Strengths
1. High per-arrow accuracy (>92% for all directions)
2. Model learns temporal patterns from accelerometer data
3. Good performance on training distribution

### Areas for Improvement
1. Generalization to unseen songs/players
2. Recall rate (many arrows missed)
3. Need more diverse training data
4. Could benefit from:
   - Cross-validation across multiple songs
   - Data augmentation (time shifts, noise injection)
   - Ensemble methods
   - Fine-tuning on target songs

## Conclusion

The Keras CNN model successfully learned to predict DDR arrow patterns from accelerometer data with ~79% exact match accuracy on the test set. The model shows promising results on songs within the training distribution but requires more diverse training data to generalize to unseen songs and players.
