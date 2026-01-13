# Training and Inference Results

This folder contains all results from training and testing the Keras CNN model on DDR accelerometer data.

## Quick Links

📊 **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** - Detailed performance metrics and findings

📈 **[RESULTS_VISUALIZATION.md](RESULTS_VISUALIZATION.md)** - Comprehensive visual guide with examples

## Contents

- `trained_model.h5` - Trained Keras model (858K parameters, 3.27 MB)
- `*_prediction_comparison.png` - Arrow timeline comparisons
- `docs/` - Training history and sample predictions

## Quick Stats

- **Test Accuracy**: 79.42% exact match, 93.79% Hamming score
- **Training Data**: 4,104 samples from 2 songs
- **Model Size**: 3.27 MB
- **Per-Arrow Accuracy**: 92-95% for all directions

## View Results

1. **Training Progress**: See `docs/training_history.png`
2. **Sample Predictions**: See `docs/prediction_sample_*.png`
3. **Inference Results**: See `*_prediction_comparison.png`

---

*Created: 2026-01-13*
