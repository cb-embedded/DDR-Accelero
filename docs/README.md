# DDR Accelero Web Application

Browser-based inference for DDR arrow prediction from sensor data.

## Features

- Upload sensor capture ZIP files (from Android Sensor Logger)
- Upload StepMania (.sm) chart files for comparison
- Run ML inference in browser (no server required)
- Visualize predictions vs ground truth

## Files

- `index.html` - Main application
- `app.js` - Application logic
- `sm-parser.js` - StepMania file parser
- `zip-handler.js` - ZIP extraction and sensor data parsing
- `inference.js` - ONNX-based inference engine
- `visualization.js` - SVG-based arrow visualization
- `styles.css` - Minimal styling
- `model.onnx` - Trained model (generated separately)

## Model Setup

Generate ONNX model from trained PyTorch model:

```bash
# Train model (if needed)
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6

# Export to ONNX
python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx
```

## Local Development

```bash
cd docs
python -m http.server 8080
```

Then open http://localhost:8080

## Deployment

Deployed on GitHub Pages: https://cb-embedded.github.io/DDR-Accelero/

## Dependencies

Loaded from CDNs:
- JSZip (3.10.1) - ZIP file handling
- ONNX Runtime Web (1.16.3) - ML inference
