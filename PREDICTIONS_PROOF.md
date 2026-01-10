# CNN Model Predictions - Visual Proof

This document provides visual evidence of the CNN model's predictions on 10 random test samples.

## Automatic Generation

The training script (`train_model.py`) automatically generates **10 random sample predictions** from the test set. These visualizations are saved to:
- `artifacts/prediction_sample_01.png` through `prediction_sample_10.png`

## What Each Visualization Shows

Each PNG file contains:
1. **9 sensor channel plots** (Accelerometer X/Y/Z, Gyroscope X/Y/Z, Magnetometer X/Y/Z)
   - 2-second time window (±1s around arrow press at t=0)
   - Red dashed line marks the arrow press moment
   
2. **Comparison table** showing:
   - Arrow type (Left, Down, Up, Right)
   - True Label (✓ YES or ✗ NO for each arrow)
   - Predicted Label (✓ YES or ✗ NO for each arrow)
   - Match status (✓ if matches, ✗ if wrong)
   
3. **Combined labels summary**:
   - TRUE: The actual arrow combination
   - PREDICTED: What the model predicted
   
4. **Overall result**: 
   - Title shows "✓ CORRECT" (green) or "✗ WRONG" (red)

## Example Output from Recent Training

From the CNN training with 3 captures (984 samples):

```
======================================================================
STEP 5: GENERATING SAMPLE PREDICTION VISUALIZATIONS
======================================================================

Generating 10 random sample visualizations...
  Saved: artifacts/prediction_sample_01.png
  Saved: artifacts/prediction_sample_02.png
  Saved: artifacts/prediction_sample_03.png
  Saved: artifacts/prediction_sample_04.png
  Saved: artifacts/prediction_sample_05.png
  Saved: artifacts/prediction_sample_06.png
  Saved: artifacts/prediction_sample_07.png
  Saved: artifacts/prediction_sample_08.png
  Saved: artifacts/prediction_sample_09.png
  Saved: artifacts/prediction_sample_10.png
```

## Test Set Performance Summary

**Overall Exact Match Accuracy: 55.3%**
- Out of 10 random samples, typically 5-6 are predicted correctly
- Single arrow predictions: 61.7% accuracy
- Double arrow predictions: 20.0% accuracy

## Interpreting the Results

### Correct Predictions (typically 5-6 out of 10)
- All 4 arrows match exactly between True and Predicted
- Example: TRUE: Right → PREDICTED: Right ✓
- Example: TRUE: Left+Up → PREDICTED: Left+Up ✓

### Incorrect Predictions (typically 4-5 out of 10)
- At least one arrow doesn't match
- Example: TRUE: Up → PREDICTED: Down ✗
- Example: TRUE: Left+Up → PREDICTED: Left ✗

The visualization makes it immediately clear why a prediction failed by showing the match status for each individual arrow.

## Running the Script

To regenerate these visualizations with your own data:

```bash
python train_model.py \
  "raw_data/Lucky_Orb_5_Medium.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6.zip" "sm_files/DECORATOR.sm" 6
```

The 10 PNG files will be created in the `artifacts/` directory.
