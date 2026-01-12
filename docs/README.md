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
- `model.onnx` - Trained model with embedded weights
- `lib/` - Local JavaScript libraries (ONNX Runtime Web, JSZip)

## Model Setup

Generate ONNX model from trained PyTorch model:

```bash
# Train model (if needed)
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6

# Export to ONNX (opset 18, IR version 10)
python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx
```

**Important**: The model must be in embedded format (all weights in single file, no external .data file) and should use ONNX opset 18 or lower for compatibility with ONNX Runtime Web 1.18.0.

## Local Development

```bash
cd docs
python -m http.server 8080
```

Then open http://localhost:8080

## Deployment

Deployed on GitHub Pages: https://cb-embedded.github.io/DDR-Accelero/

## Dependencies

All dependencies are bundled locally in the `lib/` directory:
- JSZip (3.10.1) - ZIP file handling
- ONNX Runtime Web (1.18.0) - ML inference

**Note**: Libraries are stored locally because CDNs may be blocked by network policies. ONNX Runtime Web 1.18.0 is required for proper support of ONNX opset 18 models.

## Troubleshooting

### Model Loading Issues

If you see errors like "8493520" or "11855136", this typically means:
1. ONNX Runtime Web version is incompatible with the model's opset version
2. The model uses external data format that isn't properly accessible

**Solution**: Ensure the model is in embedded format (use `onnx.save()` with all weights included) and use ONNX Runtime Web 1.18.0 or later.

### WASM Loading Issues

If you see "Failed to resolve module specifier", ensure:
1. All `.wasm` files are in the `lib/` directory
2. The `ort.env.wasm.wasmPaths` is set to `'lib/'` in inference.js
3. You're using a compatible version of ONNX Runtime Web (version 1.18.0 is recommended)
