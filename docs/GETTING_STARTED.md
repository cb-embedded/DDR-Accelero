# Getting Started with DDR Accelero Web App

This guide will help you get started with the web-based inference system for DDR Accelero.

## 🚀 Quick Start

### Option 1: Use GitHub Pages (Recommended)

Once GitHub Pages is enabled, simply visit:
**https://cb-embedded.github.io/DDR-Accelero/**

No installation required! Everything runs in your browser.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/cb-embedded/DDR-Accelero.git
cd DDR-Accelero/docs

# Start a local web server
python3 -m http.server 8080

# Open in your browser
open http://localhost:8080
```

## 📖 How to Use

### Step 1: Prepare Your Files

You'll need two files:

1. **Sensor Capture (ZIP file)**
   - Recorded using Android Sensor Logger app
   - Must contain: `Gravity.csv`, `Gyroscope.csv`, `Magnetometer.csv`
   - Example: `Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip`

2. **Chart File (.sm file)**
   - StepMania chart file for the same song
   - Example: `Lucky Orb.sm`

### Step 2: Upload Files

1. Click "Choose File" under **Sensor Capture**
2. Select your ZIP file
3. Wait for the green checkmark
4. Click "Choose File" under **Chart File (.sm)**
5. Select your .sm file
6. Wait for the green checkmark

### Step 3: Select Difficulty

Choose the difficulty level that matches your gameplay recording (e.g., 5 for Medium-5).

### Step 4: Run Inference

Click the **"Predict Arrows"** button and wait for the results.

### Step 5: View Results

Scroll down to see:
- Statistics (duration, ground truth count, predictions count)
- Side-by-side visualization:
  - **Left column**: ML predictions from sensor data
  - **Right column**: Ground truth from .sm file
- Color-coded arrows:
  - 🩷 Pink = Left
  - 🩵 Cyan = Down
  - 💛 Yellow = Up
  - 🧡 Orange = Right

## 🎮 Example Files

You can find example files in the repository:

- **Sensor Captures**: `raw_data/` directory
- **Chart Files**: `sm_files/` directory

Try these combinations:
```
Lucky Orb:
- Capture: raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip
- Chart: sm_files/Lucky Orb.sm
- Difficulty: 5

Decorator:
- Capture: raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip
- Chart: sm_files/DECORATOR.sm
- Difficulty: 6

Charles:
- Capture: raw_data/Charles_5_Medium-2026-01-10_09-22-48.zip
- Chart: sm_files/Charles.sm
- Difficulty: 5
```

## 🔬 Current Implementation

The web app currently uses a **demo inference algorithm** based on accelerometer peak detection. This is a simplified algorithm that:

1. Calculates accelerometer magnitude
2. Detects peaks (local maxima)
3. Estimates arrow direction from sensor axes
4. Generates predictions with timestamps

### Future: ML Model Integration

To use the actual trained ML model:

1. Train a model using Python:
   ```bash
   python train_model.py \
     "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
     "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6
   ```

2. Export to ONNX format:
   ```bash
   python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx
   ```

3. Update `docs/inference.js` to load the ONNX model

4. Refresh the web app

## 🐛 Troubleshooting

### Files not uploading?
- Check file format: ZIP files must be unencrypted
- Check file size: Large files (>50MB) may be slow to process

### No predictions generated?
- Check sensor data quality
- Ensure sufficient duration (>5 seconds recommended)
- Try adjusting the difficulty level

### Visualization not showing?
- Check browser console for errors (F12)
- Try refreshing the page
- Ensure both files were uploaded successfully

### Incorrect BPM or timing?
- Verify the .sm file is properly formatted
- Check that difficulty level matches the chart

## 💡 Tips

1. **Best Results**: Use recordings with clear, deliberate movements
2. **Timing**: The demo algorithm works best with medium-paced songs
3. **Comparison**: The visualization helps identify where predictions differ from ground truth
4. **Scrolling**: Use your mouse wheel to scroll through the entire chart

## 🔗 Links

- **GitHub Repository**: https://github.com/cb-embedded/DDR-Accelero
- **Python Tools**: See main README.md for Python-based alignment and training tools
- **Component Tests**: Open `test.html` to verify all components are working

## 📝 Feedback

Found a bug or have suggestions? Please:
1. Open an issue on GitHub
2. Include:
   - What you were trying to do
   - What happened vs what you expected
   - Browser and OS information
   - Screenshots if possible

## 🎯 Next Steps

After testing the web app:

1. **Collect more data**: Record more gameplay sessions
2. **Train better models**: Use more diverse training data
3. **Improve preprocessing**: Experiment with different sensor processing techniques
4. **Share results**: Post your visualizations and findings

Happy DDR playing! 🎮🎵
