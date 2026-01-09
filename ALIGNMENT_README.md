# DDR-Accelero Alignment: Phase 1 Implementation

## Overview

This implementation follows a **strict step-by-step, evidence-driven approach** to achieve unambiguous alignment between raw accelerometer data and StepMania `.sm` files using cross-correlation.

## Goal

Align captured accelerometer data with `.sm` chart to enable future supervised learning for arrow prediction. **Phase 1 focus**: Achieve a **dominant correlation peak** (peak_ratio > 2.0, z_score > 5.0) that unambiguously identifies the time offset.

## Repository Structure

```
DDR-Accelero/
├── raw_data/          # ZIP files from Android Sensor Logger
├── sm_files/          # StepMania .sm chart files
├── artifacts/         # Generated PNG evidence files
├── align_x_axis.py    # Core alignment script
├── test_all_axes.py   # Test all axes for best correlation
├── verify_alignment_visual.py  # Visual diagnostic tool
├── INPUT_DISCOVERY.md # Data format documentation
└── TODO.md            # Current status and next steps
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Align with default settings (bipolar kernel)
python align_x_axis.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" \
  "sm_files/Lucky Orb.sm" \
  Medium

# Try different kernel shapes
python align_x_axis.py capture.zip song.sm Medium impulse
python align_x_axis.py capture.zip song.sm Medium bipolar
python align_x_axis.py capture.zip song.sm Medium gaussian

# Test all axes to find best correlation
python test_all_axes.py capture.zip song.sm Medium bipolar

# Visual verification of alignment
python verify_alignment_visual.py capture.zip song.sm Medium y
```

## Implementation Philosophy

### 1. **Minimal, Synthetic Code**
- One function = one operation
- No premature abstraction
- Only generate what's necessary for current step

### 2. **Evidence as PNG**
Every script outputs PNG artifacts to `artifacts/`:
- `*_raw_x.png` - Raw accelerometer signal
- `*_expected_*.png` - Expected signal from .sm notes
- `*_correlation_*.png` - Cross-correlation curve with metrics
- `*_alignment_*.png` - Overlay of aligned signals
- `*_visual_verification_*.png` - Detailed diagnostic plots

### 3. **Stop and Diagnose**
When results are ambiguous, **stop and create TODO** rather than piling on heuristics.

### 4. **Reproducible and Inspectable**
All scripts print metadata (sample rate, duration, peak metrics) and produce deterministic output.

## Current Status

### ✅ Completed

1. **Input Discovery** (see INPUT_DISCOVERY.md)
   - Capture format: `Gravity.csv` with columns: time, seconds_elapsed, z, y, x
   - Sample rate: ~200 Hz (computed dynamically)
   - .sm format: BPM, OFFSET, note patterns

2. **Core Alignment Script** (`align_x_axis.py`)
   - Loads raw accelerometer data
   - Parses .sm file for note timestamps
   - Creates expected signal with configurable kernel (impulse/bipolar/gaussian)
   - Computes cross-correlation
   - Calculates dominance metrics: peak_ratio, z_score
   - Generates 4 PNG artifacts

3. **Multi-Axis Testing** (`test_all_axes.py`)
   - Tests x, y, z, and magnitude simultaneously
   - Identifies best axis for correlation
   - Generates comparison plot

4. **Visual Verification** (`verify_alignment_visual.py`)
   - Detailed diagnostic visualization
   - Marks predicted first note time
   - Highlights high-density regions
   - Periodicity check
   - Checklist for manual assessment

### ⚠️ Current Issue: Weak Correlation

**Test case**: Lucky Orb Medium

| Axis | Peak Ratio | Z-score | Status |
|------|------------|---------|--------|
| x | 1.16 | 4.33 | ⚠ WEAK |
| y | 1.26 | 4.87 | ⚠ WEAK |
| z | 1.15 | 5.36 | ⚠ WEAK |
| magnitude | 1.00 | 4.48 | ⚠ WEAK |

**Best**: Y-axis with peak_ratio=1.26, still below target of 2.0

**Conclusion**: Weak correlation persists across ALL axes and kernel types, suggesting a fundamental issue rather than a signal processing choice.

## Dominance Metrics

### Peak Ratio
```
peak_ratio = highest_peak / second_highest_peak
```
Measures how much the main peak dominates over competing peaks.
- **Target**: > 2.0 (peak is at least 2x stronger than runner-up)
- **Current**: 1.0-1.3 (peaks are similar magnitude)

### Z-Score
```
z_score = (peak_value - mean(correlation)) / std(correlation)
```
Measures how many standard deviations the peak is above the noise floor.
- **Target**: > 5.0 (peak clearly above background)
- **Current**: 3.5-5.4 (marginal significance)

## Next Steps

See `TODO.md` for detailed analysis and options. Summary:

1. **Visual Verification** (RECOMMENDED NEXT)
   - Inspect `*_visual_verification_y.png`
   - Answer checklist: Is there activity at predicted first note time?
   - Determine if alignment is approximately correct despite weak metrics

2. **If alignment looks correct visually**:
   - Accept that individual notes don't create distinct peaks
   - Weak correlation may be reality due to noise/variability
   - Proceed with dataset generation using best available alignment

3. **If timing appears wrong**:
   - Investigate BPM mismatch or player speed modifiers
   - Check if OFFSET in .sm matches recording start
   - Consider audio-based sync as alternative

## Technical Details

### Expected Signal Generation

Three kernel options to model physical response:

1. **Impulse**: Delta function at each note time
   ```
   signal[t] = 1.0 if note at t else 0.0
   ```

2. **Bipolar**: Acceleration + deceleration (50ms each phase)
   ```
   [+1, +1, ...] followed by [-1, -1, ...]
   ```

3. **Gaussian**: Smooth pulse (100ms width)
   ```
   exp(-(t-t_note)²/2σ²)
   ```

### Cross-Correlation

```python
# Normalize signals
x_norm = (x - mean(x)) / std(x)
exp_norm = (exp - mean(exp)) / std(exp)

# Correlate
corr = correlate(x_norm, exp_norm, mode='full')

# Find peak
peak_idx = argmax(corr)
offset = lags[peak_idx]
```

### Sampling

- Raw data: ~200 Hz (varies per capture)
- Resampled to: 100 Hz (for consistency)
- Method: Linear interpolation

## Files Not to Commit

The `.gitignore` excludes:
- `__pycache__/` - Python bytecode
- IDE configuration files
- OS-specific files (`.DS_Store`, etc.)

## References

- **StepMania Format**: [StepMania File Formats](https://github.com/stepmania/stepmania/wiki/sm)
- **Android Sensor Logger**: Captures from phone accelerometer/gyroscope
- **Cross-correlation**: Standard signal processing technique for alignment

## Questions?

See `TODO.md` for current diagnostic questions and decision points.

---

**Last Updated**: 2026-01-09
**Status**: Phase 1 - Awaiting visual verification to inform next steps
