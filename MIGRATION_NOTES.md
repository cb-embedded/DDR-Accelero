# Migration Notes: Removing Offset-Based Approach

## Overview

This document describes the changes made to simplify the DDR-Accelero dataset construction and training by removing the offset-based prediction approach.

## Motivation

The previous approach predicted both:
1. **Arrow combinations** (classification)
2. **Temporal offset** (regression) - timing of arrow relative to window center

The new approach simplifies this to:
- **Arrow combinations only** (classification)
- Uses a **50ms threshold** to determine if arrows should be predicted or labeled as "nothing"

This creates a cleaner classification problem where the model learns:
- **When to predict**: arrows within 50ms of window center
- **When not to predict**: no arrows within 50ms (labeled as [0,0,0,0])

## Key Changes

### 1. Dataset Construction (`create_dataset.py`)

**Before:**
- Every window was labeled with the closest arrow and its offset
- Returned: `X, Y, offsets, t_centers`

**After:**
- Windows with closest arrow ≤ 50ms: labeled with arrow combination
- Windows with closest arrow > 50ms: labeled as [0,0,0,0] (nothing)
- Returned: `X, Y, t_centers` (no offsets)

**Code changes:**
```python
# New threshold constant
OFFSET_THRESHOLD = 0.050  # 50ms

# Labeling logic
if closest_offset <= OFFSET_THRESHOLD:
    Y.append(arrows[closest_idx])  # Label with arrow
else:
    Y.append(np.array([0, 0, 0, 0]))  # Label as nothing
```

### 2. Model Architecture (`train_model.py`)

**Before:**
- Multi-task model with two heads:
  - Arrow classification head (4 outputs)
  - Offset regression head (1 output)
- Combined loss: `weight_arrows * loss_arrows + weight_offset * loss_offset`

**After:**
- Single-task model:
  - Arrow classification head only (4 outputs)
- Simple loss: `BCELoss` for multi-label classification

**Model changes:**
```python
class ArrowCNN(nn.Module):
    def forward(self, x):
        # ... conv and fc layers ...
        arrows = self.sigmoid(self.fc_arrows(x))
        return arrows  # Only arrows, no offset
```

### 3. Training Process

**Before:**
- Tracked: train/val loss (arrows + offset), exact match, offset MAE
- Two separate losses to balance

**After:**
- Tracked: train/val loss (arrows only), exact match
- Simpler training loop with single loss

### 4. Prediction (`predict_song.py`)

**Before:**
```python
arrows_out, offset_out = model(X)
chart_time = pred_time - offset + pred_offset  # Used predicted offset
```

**After:**
```python
arrows_out = model(X)
chart_time = pred_time - offset  # No predicted offset
```

### 5. Web Inference (`docs/inference.js`)

**Before:**
- Expected model output: `{arrows: [4], offset: [1]}`
- Adjusted prediction time using offset

**After:**
- Expected model output: `{arrows: [4]}`
- No offset adjustment needed

## Benefits

1. **Simpler model**: Single task instead of multi-task
2. **Clearer semantics**: Model explicitly learns "nothing" state
3. **Easier training**: Single loss function, no weight balancing
4. **More realistic**: Model learns when NOT to predict
5. **Better evaluation**: "Nothing" samples included in metrics

## Migration Steps

To use the new approach:

1. **Re-create datasets** with updated `create_dataset.py`
2. **Re-train model** with updated `train_model.py`
3. **Re-export ONNX** with updated `export_model_to_onnx.py`
4. **Test predictions** with updated `predict_song.py`
5. **Deploy web app** with updated `docs/inference.js`

## Example Commands

```bash
# Create dataset (automatically uses new approach)
python create_dataset.py capture.zip song.sm 5 10

# Train model (automatically uses new architecture)
python train_model.py capture1.zip song1.sm 5 capture2.zip song2.sm 6

# Export to ONNX (automatically uses new output format)
python export_model_to_onnx.py --model-path artifacts/trained_model.pth --output docs/model.onnx

# Test predictions (automatically uses new inference)
python predict_song.py capture.zip song.sm 5 70.0 10.0
```

## Expected Results

### Dataset Statistics
You should now see output like:
```
Samples with arrows: 450 (75.0%)
Samples with nothing: 150 (25.0%)
```

### Training Output
Simpler metrics:
```
Epoch [5/100] - Train Loss: 0.2341, Val Loss: 0.2567, Val Exact Match: 68.3%
```

### Evaluation Metrics
Now includes "nothing" in accuracy:
```
Distribution of samples by number of simultaneous arrows:
  0 arrows: 120 samples (24.0%)  <- New!
  1 arrows: 280 samples (56.0%)
  2 arrows: 100 samples (20.0%)
```

## Backward Compatibility

**Not compatible with old models!**
- Old `.pth` files have 2 outputs (arrows + offset)
- New `.pth` files have 1 output (arrows only)
- Must re-train from scratch

**ONNX models:**
- Old: outputs `{arrows: Tensor, offset: Tensor}`
- New: outputs `{arrows: Tensor}`

## Testing

After migrating, verify:
1. ✓ Dataset creates "nothing" samples (Y = [0,0,0,0])
2. ✓ Model trains without offset loss
3. ✓ ONNX export produces single output
4. ✓ Web inference works without offset
5. ✓ Predictions match expected arrow patterns

## Questions?

If you encounter issues:
- Check that all files are updated to the new version
- Verify no old model files are being loaded
- Ensure ONNX export matches inference.js expectations
