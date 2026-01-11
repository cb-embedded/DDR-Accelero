# DDR Accelero Web Application

This directory contains the web-based inference system for DDR Accelero.

## Overview

The web application allows users to:
- Upload sensor capture ZIP files directly in the browser
- Upload StepMania (.sm) chart files for comparison
- Run ML inference in the browser (no server required)
- Visualize predictions vs ground truth with interactive scrollable charts

## Files

- `index.html` - Main application page
- `styles.css` - Application styling (simplified, minimal design)
- `app.js` - Main application logic and coordination
- `sm-parser.js` - StepMania file parser
- `zip-handler.js` - ZIP file extraction and sensor data parsing
- `inference.js` - ONNX-based inference engine
- `visualization.js` - SVG-based arrow visualization
- `model.onnx` - Trained ONNX model (generated, not in git)

## Architecture

### Data Flow

1. **Upload Phase**
   - User uploads sensor capture ZIP (containing Gravity.csv, Gyroscope.csv, Magnetometer.csv)
   - ZipHandler extracts and parses CSV files
   - User uploads .sm file
   - SMParser extracts arrow patterns and timing

2. **Inference Phase**
   - InferenceEngine processes sensor data using trained ONNX model
   - Uses sliding window approach (2-second windows with 0.5-second stride)
   - Model predicts both arrow combinations and timing offsets
   - Filters predictions based on confidence threshold

3. **Visualization Phase**
   - ArrowVisualizer creates side-by-side SVG visualizations
   - Left column: ML predictions
   - Right column: Ground truth from .sm file
   - Scrollable timeline with color-coded arrows

### Components

#### ZipHandler (`zip-handler.js`)
- Extracts CSV files from Android Sensor Logger ZIP files
- Parses sensor data (accelerometer, gyroscope, magnetometer)
- Provides unified data structure with timestamps

#### SMParser (`sm-parser.js`)
- Parses StepMania .sm files
- Extracts BPM and arrow patterns
- Filters arrows by difficulty level
- Provides time-windowed arrow queries

#### InferenceEngine (`inference.js`)
- Runs arrow prediction on sensor data using ONNX Runtime Web
- Loads trained ONNX model (ArrowCNN architecture)
- Uses sliding window approach for continuous prediction
- Preprocesses sensor data (resampling, normalization)
- Outputs arrow combinations with timing offsets and confidence scores

#### ArrowVisualizer (`visualization.js`)
- Creates SVG-based visualizations
- Displays arrows in StepMania style (vertical scrolling)
- Color-coded arrows: Left (Pink), Down (Cyan), Up (Yellow), Right (Orange)
- Side-by-side comparison of predictions and ground truth

## Model Setup

The web application requires a trained ONNX model to run inference. The model is not included in the repository due to its size but can be generated from a trained PyTorch model.

### Generate the ONNX Model

1. **Train a model** (if you don't have one already):
   ```bash
   python train_model.py \
     "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
     "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6 \
     "raw_data/Charles_5_Medium-2026-01-10_09-22-48.zip" "sm_files/Charles.sm" 5
   ```
   This creates `artifacts/trained_model.pth`

2. **Export to ONNX format:**
   ```bash
   python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx
   ```
   This creates `docs/model.onnx` and `docs/model.onnx.data`

3. **Deploy:** Copy both `model.onnx` and `model.onnx.data` to your web server or GitHub Pages deployment.

## Future Enhancements

## Additional Features to Consider

- **Real-time inference:** Stream sensor data and predict in real-time
- **Performance metrics:** Calculate precision, recall, F1 score
- **Export results:** Download predictions as JSON or CSV
- **Multiple difficulty support:** Compare predictions across difficulties
- **Audio playback:** Sync visualization with song audio

## Development

To run locally:

```bash
# Start a local web server
cd docs
python -m http.server 8080

# Open in browser
open http://localhost:8080
```

## Deployment

This application is deployed on GitHub Pages at:
https://cb-embedded.github.io/DDR-Accelero/

GitHub Pages automatically serves the `docs/` folder when enabled in repository settings.

## Browser Compatibility

- Chrome/Edge: ✓ Supported
- Firefox: ✓ Supported
- Safari: ✓ Supported
- Mobile: ✓ Responsive design

## Dependencies

All dependencies are loaded from CDNs:
- JSZip (3.10.1) - ZIP file handling
- ONNX Runtime Web (1.16.3) - ML inference (for future use)

No build step required - pure HTML/CSS/JavaScript!
