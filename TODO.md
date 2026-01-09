# TODO: Alignment Ambiguity Diagnosis

## Current Status
✓ Step 0: Input discovery completed (see INPUT_DISCOVERY.md)
✓ Implemented minimal alignment script: `align_x_axis.py`
✓ Generated PNG artifacts in `artifacts/` directory
✓ Tested kernel shapes (impulse, bipolar, gaussian) - no improvement
✓ Tested all axes (x, y, z, magnitude) - all weak
⚠️ **PROBLEM**: Weak correlation peak across ALL axes and kernel types

## Results from Lucky Orb Medium Test

### All Axes Tested (bipolar kernel):
| Axis | Peak Ratio | Z-score | Offset | Status |
|------|------------|---------|--------|--------|
| x | 1.16 | 4.33 | 30.420s | ⚠ WEAK |
| y | 1.26 | 4.87 | 30.410s | ⚠ WEAK |
| z | 1.15 | 5.36 | 30.550s | ⚠ WEAK |
| magnitude | 1.00 | 4.48 | 44.680s | ⚠ WEAK |

**Best axis**: Y (peak ratio: 1.26, still below 2.0 target)
**Assessment**: ALL axes show weak correlation - problem is more fundamental

## Diagnosis: Why is correlation weak across ALL axes?

Since kernel shape and axis selection don't help, the problem is more fundamental. Possible root causes:

### Hypothesis 1: Expected signal timing is wrong ⭐ MOST LIKELY
**Evidence**:
- All offsets found are around 30s (x: 30.42s, y: 30.41s, z: 30.55s)
- These are suspiciously consistent across axes
- But correlation is still weak even at these offsets

**Root cause possibilities**:
a) **BPM from .sm file doesn't match actual gameplay tempo**
   - Player may have used speed modifiers
   - .sm file BPM might be incorrect
   
b) **OFFSET field in .sm is not the actual recording start offset**
   - OFFSET (1.764s) is for music file, not sensor recording
   - Sensor recording likely started BEFORE music began
   - The 30s offset we detect is probably: time_before_music_starts + .sm_offset

c) **Player reaction time delay**
   - Notes in .sm file represent when to press
   - Sensor captures actual foot movement (later than visual cue)
   - Typical DDR reaction time: 100-300ms

**Proposed solution - MANUAL VERIFICATION** (Option 3):
1. Visually inspect raw signal around detected offset (~41s = 30.4s + 11.3s first note)
2. Look for periodic patterns matching note density
3. Count peaks in a known high-density region
4. Compare to expected note count from .sm

### Hypothesis 2: Single-axis signals are too noisy
**Evidence**:
- Magnitude didn't help (actually worse: ratio=1.00)
- Individual axes all weak
- Suggests signal is genuinely noisy/variable

**Proposed solution**:
- Try derivative of signal (velocity → acceleration response)
- Try bandpass filtered version (remove drift + noise)
- BUT: violates "minimal preprocessing" principle

### Hypothesis 3: Phone orientation/placement varies
**Evidence**:
- Unknown where phone was placed during capture
- Axis directions relative to body movement unknown
- Could explain why no single axis dominates

**Proposed solution**:
- Need metadata about phone placement
- Or: Use PCA to find dominant movement axis
- Or: Accept this is a limitation and use all axes

### Hypothesis 4: Not all notes create equal sensor response
**Evidence**:
- Different arrow directions (left/down/up/right) create different movements
- Expected signal treats all notes equally
- Reality: some movements are subtle, others are strong

**Proposed solution**:
- Weight expected signal by arrow type
- BUT: requires assumption about which arrows are "stronger"
- Violates minimal assumptions principle

## RECOMMENDED NEXT STEP: Manual Visual Verification

Before adding more heuristics, we need to understand if the detected offset (~30.4s) is approximately correct.

### Create a diagnostic visualization script:

```python
# Create: verify_alignment_visual.py
# Shows raw signal, expected signal, and zoomed windows around:
# 1. Detected offset + first note time (~41s)
# 2. High-density note region (find from .sm)
# 3. Low-density region (for comparison)
```

### Questions to answer from visual inspection:

1. **At ~41s (detected first note time), is there visible activity in sensor data?**
   - YES → Offset roughly correct, but correlation is weak for other reasons
   - NO → Offset is wrong, need different alignment method

2. **Do high-activity regions in sensor data align with high-note-density regions from .sm?**
   - YES → Timing is approximately correct, notes just don't correlate 1:1 with sensor peaks
   - NO → Timing is systematically wrong (BPM error, speed modifier, etc.)

3. **Can you visually count peaks in a 10-second window and compare to note count?**
   - If counts match roughly → alignment is good, correlation is just weak due to noise/variability
   - If counts don't match → fundamental timing problem

4. **Is there periodicity in the sensor data matching the expected BPM (126)?**
   - Beat period = 60/126 = 0.476s
   - Check if peaks occur roughly every 0.5s during dense sections

### Decision point after visual verification:

**IF visual inspection confirms alignment is approximately correct:**
→ Problem is: individual notes don't create distinct sensor peaks (too noisy, too variable)
→ Solution: Accept weak correlation as reality, use alignment anyway
→ OR: Try different features (envelope, spectral, etc.) - but violates minimalism

**IF visual inspection shows timing is wrong:**
→ Problem is: .sm timing doesn't match actual gameplay
→ Solution: Need to find correct BPM/offset
→ Method: Manual annotation of a few known notes, or audio sync

## Implementation Request

Please run Option 3 (Manual Verification) by creating the diagnostic visualization script.

After seeing the visual evidence, we can make an informed decision about whether:
- The current alignment is "good enough" despite weak correlation
- We need a completely different approach
- There's a systematic timing error to fix
