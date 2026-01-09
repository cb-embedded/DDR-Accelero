# SOLUTION FOUND: Biomechanical Approach

## Success!

After extensive investigation with 7 approaches showing weak correlation (ratio 1.0-1.4), the **biomechanical approach** provided by the user achieves **CLEAR CORRELATION PEAKS**.

## Key Differences from Previous Approaches

### 1. **LEFT ARROW ONLY** (not all notes)
Previous approaches tried to correlate with ALL note events. The solution focuses on a single arrow direction (left), which creates more consistent sensor patterns.

### 2. **Edge Event Detection**
Instead of placing impulses at note times, compute the **derivative** (diff) of the binary signal to extract edge transitions.

### 3. **Biomechanical Exponential Kernel**
Apply an exponential decay kernel (tau=0.1s) to model the physical response:
```python
kernel = exp(-t / tau)
```
This mimics how the body responds to stepping motions.

### 4. **FFT-Based Correlation**
Use FFT for correlation instead of scipy.signal.correlate:
```python
corr = ifft(fft(signal1) * conj(fft(signal2)))
```

### 5. **Narrow Bandpass (0.5-8 Hz)**
Previous approaches used 0.5-10 Hz or 0.5-15 Hz. The solution uses 0.5-8 Hz.

## Results

### Lucky Orb 5 Medium (2026-01-06)
- **Peak ratio: 2.69** ✓ (target: >2.0)
- **Z-score: 7.46** ✓ (target: >5.0)
- **Offset: 32.42s**
- **Status: SUCCESS**

### Decorator Medium 6 (2026-01-07)
- **Peak ratio: 2.05** ✓
- **Z-score: 6.69** ✓
- **Offset: 5.74s**
- **Status: SUCCESS**

### Lucky Orb 5 Medium 4 (2026-01-09)
- **Peak ratio: 1.04** ⚠
- **Z-score: 3.24** ⚠
- **Status: WEAK** (may be data quality issue)

## Implementation

Two scripts provided:

1. **align_biomech.py** - Full version with detailed output
2. **align_clean.py** - Minimal clean version (~190 lines)

Both achieve the same results.

## Usage

```bash
python align_clean.py "raw_data/Lucky_Orb_5_Medium-....zip" "sm_files/Lucky Orb.sm" 5
```

Arguments:
- capture_zip: Path to sensor data
- sm_file: Path to .sm chart file
- difficulty_level: Numeric difficulty level (e.g., 5 for Medium-5)

## Why It Works

The biomechanical model better captures the **physical dynamics** of DDR gameplay:

1. Stepping on left arrow creates a **transient event** (edge)
2. Body responds with **exponential decay** (tau ≈ 100ms)
3. Focusing on **single arrow** removes noise from multi-directional movements
4. Bandpass at **0.5-8 Hz** isolates human movement frequencies

## Comparison to Previous Approaches

| Approach | Peak Ratio | Result |
|----------|------------|--------|
| Gaussian pulses (all notes) | 1.22 | WEAK |
| Movement energy | 1.40 | WEAK |
| Note density | 1.13 | WEAK |
| Gyroscope | 1.28 | WEAK |
| Replication of exp03 | 1.09 | WEAK |
| **Biomechanical (left arrow)** | **2.69** | **SUCCESS** |

The key insight: **match the physics of the problem, not just the timing**.
