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
- `styles.css` - Application styling
- `app.js` - Main application logic and coordination
- `sm-parser.js` - StepMania file parser
- `zip-handler.js` - ZIP file extraction and sensor data parsing
- `inference.js` - Inference engine (currently uses demo algorithm)
- `visualization.js` - SVG-based arrow visualization

## Architecture

### Data Flow

1. **Upload Phase**
   - User uploads sensor capture ZIP (containing Gravity.csv, Gyroscope.csv, Magnetometer.csv)
   - ZipHandler extracts and parses CSV files
   - User uploads .sm file
   - SMParser extracts arrow patterns and timing

2. **Inference Phase**
   - InferenceEngine processes sensor data
   - Currently uses a demo algorithm based on accelerometer peak detection
   - TODO: Replace with ONNX model for actual ML inference

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
- Runs arrow prediction on sensor data
- **Current Implementation:** Demo algorithm using accelerometer peak detection
- **Future Implementation:** ONNX model with trained CNN

#### ArrowVisualizer (`visualization.js`)
- Creates SVG-based visualizations
- Displays arrows in StepMania style (vertical scrolling)
- Color-coded arrows: Left (Pink), Down (Cyan), Up (Yellow), Right (Orange)
- Side-by-side comparison of predictions and ground truth

## Future Enhancements

### Model Integration

To integrate a trained PyTorch model:

1. **Export to ONNX:**
   ```python
   import torch
   from train_model import ArrowCNN
   
   # Load trained model
   model = ArrowCNN(input_channels=9, seq_length=198)
   checkpoint = torch.load('artifacts/trained_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   
   # Export to ONNX
   dummy_input = torch.randn(1, 9, 198)
   torch.onnx.export(
       model,
       dummy_input,
       'docs/model.onnx',
       input_names=['input'],
       output_names=['arrows', 'offset'],
       dynamic_axes={'input': {0: 'batch_size'}}
   )
   ```

2. **Update inference.js:**
   - Load ONNX model using ONNX Runtime Web
   - Replace demo algorithm with model inference
   - Apply same preprocessing as Python version

### Additional Features

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
