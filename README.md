# DDR-Accelero

Machine Learning-powered Dance Dance Revolution arrow prediction from accelerometer data.

**GitHub Pages:** [https://cb-embedded.github.io/DDR-Accelero/](https://cb-embedded.github.io/DDR-Accelero/)

Simple artifact gallery for visualizations (generated during training).

## Gamepad Recording

Two options are available for recording gamepad input:

### CLI Gamepad Recorder (Recommended for background capture)

The CLI recorder runs in the terminal and captures gamepad events even when the window loses focus. This is ideal for recording button presses while playing DDR games.

```bash
python pad_recorder_cli.py
```

**Features:**
- Works in background (no focus required)
- Continuous polling of gamepad state
- Exports to same CSV format as web recorder
- Includes YAML metadata (music name, difficulty)

**Requirements:** `pygame` (see requirements.txt)

### Web Gamepad Recorder

Browser-based recorder available at: [https://cb-embedded.github.io/DDR-Accelero/pad_recorder/](https://cb-embedded.github.io/DDR-Accelero/pad_recorder/)

**Note:** Due to browser security restrictions, the web recorder only captures events when the browser window has focus. It will stop recording if you switch to another window.

## API

### Alignment

Find time offset between sensor recording and chart:

```bash
python -m core.align <capture_zip> <sm_file> <difficulty_level>
```

Example:
```bash
python -m core.align "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5
```

### Training

Train a CNN model using Keras to predict arrows from sensor data:

```bash
python train_model.py <capture1_zip> <sm1_file> <diff1_level> [<capture2_zip> <sm2_file> <diff2_level> ...]
```

Example:
```bash
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6
```

Output: `artifacts/trained_model.h5`, `docs/training_history.png`, `docs/prediction_sample_*.png`

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
from utils.visualize import extract_sm_window, visualize_arrows

events = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', start_time=70.0, duration=10.0)
visualize_arrows(ground_truth, predictions, output_path='comparison.png')
```

### Model Export

Convert trained Keras model to ONNX format:

```bash
python -m utils.export_onnx --model-path artifacts/trained_model.h5 --output docs/model.onnx
```

Note: Requires `tf2onnx` package (`pip install tf2onnx`)

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, tensorflow, pygame

**Notes:** 
- Keras is included with TensorFlow 2.x
- pygame is only required for the CLI gamepad recorder

## Project Structure

```
DDR-Accelero/
├── core/
│   ├── align.py         # Alignment functionality (biomechanical approach)
│   └── dataset.py       # Dataset creation from captures and SM files
├── utils/
│   ├── visualize.py     # Visualization utilities
│   └── export_onnx.py   # Model export to ONNX
├── train_model.py       # Keras training script
├── predict_song.py      # Keras prediction script
├── docs/                # Artifacts (GitHub Pages) - generated during training
├── raw_data/            # Sensor captures
├── sm_files/            # StepMania charts
└── artifacts/           # Trained models (.h5 files)
```

## Method

Biomechanical approach using:
- Exponential decay kernel (tau=0.1s) modeling body dynamics
- FFT-based correlation for alignment
- Bandpass filter (0.5-8 Hz) for human movement frequencies
- Keras 1D CNN for arrow classification with 50ms threshold for "nothing" state
