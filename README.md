# DDR-Accelero

Machine Learning-powered Dance Dance Revolution arrow prediction from accelerometer data.

## Web Application

**Try it online:** [https://cb-embedded.github.io/DDR-Accelero/](https://cb-embedded.github.io/DDR-Accelero/)

Upload sensor capture ZIP files and StepMania charts to run ML inference directly in your browser.

See [docs/README.md](docs/README.md) for web application documentation.

## API

### Alignment

Find time offset between sensor recording and chart:

```bash
python align_clean.py <capture_zip> <sm_file> <difficulty_level>
```

Example:
```bash
python align_clean.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5
```

### Training

Train a CNN model to predict arrows from sensor data:

```bash
python train_model.py <capture1_zip> <sm1_file> <diff1_level> [<capture2_zip> <sm2_file> <diff2_level> ...]
```

Example:
```bash
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6
```

Output: `artifacts/trained_model.pth`, `artifacts/training_history.png`, `artifacts/prediction_sample_*.png`

### Prediction

Generate predictions for a song and compare with ground truth:

```bash
python predict_song.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" \
  "sm_files/Lucky Orb.sm" 5 70.0 10.0
```

Arguments: `<capture_zip> <sm_file> <diff_level> [start_time] [duration]`
- `start_time`: Start time in seconds for prediction window (default: 70.0)
- `duration`: Duration of prediction window in seconds (default: 10.0)

### Visualization

Visualize arrow patterns from .sm files or predictions:

```python
from visualize_arrows import extract_sm_window, visualize_arrows

events = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', start_time=70.0, duration=10.0)
visualize_arrows(ground_truth, predictions, output_path='comparison.png')
```

### Model Export

Convert trained model to ONNX format for web inference:

```bash
python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, torch

## Project Structure

```
DDR-Accelero/
├── align_clean.py       # Alignment API
├── create_dataset.py    # Dataset creation
├── train_model.py       # Training API
├── predict_song.py      # Prediction API
├── visualize_arrows.py  # Visualization API
├── export_model_to_onnx.py  # Model conversion
├── docs/                # Web application
├── raw_data/            # Sensor captures
├── sm_files/            # StepMania charts
└── artifacts/           # Generated outputs
```

## Method

Biomechanical approach using:
- Exponential decay kernel (tau=0.1s) modeling body dynamics
- FFT-based correlation for alignment
- Bandpass filter (0.5-8 Hz) for human movement frequencies
- CNN with multi-task learning for arrow and timing prediction
