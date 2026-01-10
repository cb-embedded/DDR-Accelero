# Before and After Comparison

## Documentation Improvements for Dataset Structure

### Section: "What is X (Raw Data Windows)"

#### BEFORE
```
- **X**: Sensor data windows [N × window_length × 9 channels]
  - 9 channels: accelerometer (x,y,z), gyroscope (x,y,z), magnetometer (x,y,z)
  - Window: randomly sampled with configurable size (default ±1s = 2s total)
```

#### AFTER
```
**X: Raw Sensor Data Windows** [N × window_length × 9 channels]
- 9 channels: accelerometer (x,y,z), gyroscope (x,y,z), magnetometer (x,y,z)
- **IMPORTANT**: Windows are **randomly sampled** from the sensor timeline
- Window size: configurable (default ±1s = 2s total)
- The random sampling simulates real-world scenarios where the model sees arbitrary time windows
```

**What Changed:**
- ✅ Emphasized "randomly sampled" with bold text and "IMPORTANT" marker
- ✅ Explained WHY random sampling matters (real-world scenarios)
- ✅ Made it clear this is a deliberate design choice, not a bug

---

### Section: "What is Y (Arrow Labels)"

#### BEFORE
```
- **Y**: Arrow labels [N × 4]
  - 4 arrows: [Left, Down, Up, Right] (binary: 1=pressed, 0=not pressed)
  - Label is the closest arrow combination to the window center
  - Supports single and double arrow events
```

#### AFTER
```
**Y: Arrow Labels** [N × 4]
- 4 arrows: [Left, Down, Up, Right] (binary: 1=pressed, 0=not pressed)
- Label is the **closest arrow combination** to the window center
- Supports single and double arrow events
- **Key Point**: Y represents which arrows are pressed near (but not necessarily at) the window center
```

**What Changed:**
- ✅ Emphasized "closest arrow combination" with bold
- ✅ Added explicit clarification about "near but not necessarily at"
- ✅ Helps readers understand Y is about proximity, not exact timing

---

### Section: "What is Offset (Relative Time)"

#### BEFORE
```
- **Offsets**: Time offsets [N]
  - Offset in seconds of the label arrow from the window center
  - Positive = arrow after center, negative = arrow before center
```

#### AFTER
```
**Offsets: Relative Time Offsets** [N]
- **CRUCIAL FOR ROBUST PREDICTION**: Offset in seconds of the label arrow from the window center
- Positive = arrow occurs after center, negative = arrow occurs before center
- Range: typically within ±3 seconds
- **Why This Matters**:
  - In real-life inference, the model will only see random windows without knowing arrow timing
  - The model must predict not only which arrows are pressed, but also estimate their relative position
  - Without offset information, the model cannot handle temporal uncertainty
  - This is what enables the model to work on unsynchronized sensor data
```

**What Changed:**
- ✅ Added prominent "CRUCIAL FOR ROBUST PREDICTION" marker
- ✅ Explained the range (±3 seconds)
- ✅ Added comprehensive "Why This Matters" section with 4 key points
- ✅ Connected offset to real-world inference scenarios
- ✅ Made it crystal clear why this is not optional

---

### New Section: "Understanding the Dataset Design"

#### BEFORE
(Did not exist)

#### AFTER
```markdown
## Understanding the Dataset Design

### Why Random Windows?

The dataset uses **randomly sampled windows** (not centered on arrows) to simulate real-world inference scenarios:

**Training Scenario:**
- We know the precise timing of arrow events (from alignment with .sm files)
- But we deliberately sample windows at random positions
- Each window is labeled with its nearest arrow AND the offset from window center

**Real-World Inference Scenario:**
- The model receives a live sensor stream
- No knowledge of when arrows should be pressed
- The model must predict: "Which arrows?" AND "How far away are they?"

**Why Offset Prediction is Crucial:**
Without offset prediction, the model can only answer "What arrows are near this window?" but cannot tell you WHEN to press them. The offset tells you:
- Negative offset (-0.5s): "Arrow was 0.5s ago, you're late!"
- Near-zero offset (±0.1s): "Arrow is RIGHT NOW, press it!"
- Positive offset (+0.5s): "Arrow is 0.5s away, get ready!"

This design enables the model to work with arbitrary sensor windows, making it practical for real-time gameplay assistance.

### Current Model Limitation

⚠️ **Important Note**: The current `train_model.py` implementation trains a model that predicts only the arrow labels (Y), not the offsets. This means it can identify which arrows are near the window center, but cannot estimate their precise timing.

**To fully leverage the dataset design**, the model should be extended to:
1. Predict arrow labels: [Left, Down, Up, Right] (binary vector)
2. Predict offset: continuous value in seconds (regression output)
```

**What Changed:**
- ✅ Entirely new section explaining the design philosophy
- ✅ Contrasts training vs real-world scenarios
- ✅ Provides concrete examples of offset values and their meanings
- ✅ Documents current limitation (model doesn't predict offsets)
- ✅ Provides clear recommendation for future enhancement

---

### New Section: ML Pipeline Updated

#### BEFORE
```
### ML Pipeline Output
Trains a multi-label classifier to predict arrow presses:
- **Model**: Random Forest with Binary Relevance (one classifier per arrow)
- **Features**: Statistical features extracted from sensor windows
  - Per channel: mean, std, min, max, 25th/75th percentiles (54 features total)
- **Evaluation**: Per-arrow accuracy, exact match accuracy, Hamming loss
- **Saved Model**: `artifacts/trained_model.pkl` for inference
```

#### AFTER
```
### ML Pipeline Output
Trains a CNN-based multi-label classifier to predict arrow presses:
- **Model Architecture**: 1D Convolutional Neural Network (CNN)
  - Input: Raw sensor time series [batch_size, 9 channels, time_steps]
  - 3 Conv1D layers with batch normalization and max pooling
  - Fully connected layers with dropout
  - Output: 4 sigmoid activations (one per arrow)
- **Current Implementation**: Predicts arrow labels (Y) only
  - **LIMITATION**: Does not currently predict offsets, only which arrows are near window center
  - **Future Enhancement**: Should be extended to predict both arrows AND offset for robust real-world inference
- **Training Data**: Uses randomly sampled windows with varying offsets (see Dataset section)
- **Evaluation Metrics**: 
  - Exact match accuracy (all 4 arrows must match)
  - Per-arrow accuracy
  - Hamming loss
- **Saved Model**: `artifacts/trained_model.pth` (PyTorch format)
- **Visualizations**: 
  - `artifacts/training_history.png` - Loss and accuracy curves
  - `artifacts/prediction_sample_*.png` - 10 random test predictions with sensor data
```

**What Changed:**
- ✅ Updated to reflect actual CNN implementation (not Random Forest)
- ✅ Added architectural details
- ✅ Clearly documented the limitation (no offset prediction)
- ✅ Connected to the dataset design explanation
- ✅ Listed all outputs and visualizations
- ✅ Corrected file format (.pth not .pkl)

---

## Summary of Improvements

### Clarity ✅
- Made dataset structure crystal clear with bold headers
- Explained each component (X, Y, Offsets) in detail
- Used concrete examples

### Emphasis ✅
- Highlighted critical concepts: "CRUCIAL FOR ROBUST PREDICTION"
- Used visual markers (⚠️) for important notes
- Bold text for key terms

### Completeness ✅
- Added "Why This Matters" explanations
- Documented current limitations
- Provided recommendations for future work
- Created dedicated section explaining design philosophy

### Accuracy ✅
- Updated model type (CNN not Random Forest)
- Corrected file formats (.pth not .pkl)
- Reflected actual code implementation
- Added all outputs and visualizations

### Real-World Context ✅
- Explained training vs inference scenarios
- Showed how offset enables practical use
- Provided concrete examples (-0.5s, +0.5s)
- Connected design choices to real-world needs

The documentation now provides a complete picture of what X, Y, and offsets are, and why the offset is crucial for robust prediction in real-world scenarios.
