# DDR-Accelero

Alignment tool for Dance Dance Revolution (DDR) accelerometer data with StepMania chart files.

## Current Status

**Working solution implemented** using biomechanical approach with clear correlation peaks.

### Results

Successfully achieved alignment with dominant correlation peaks:

| Recording | Peak Ratio | Z-score | Status |
|-----------|------------|---------|--------|
| Lucky Orb 5 Medium | **2.69** | **7.46** | ✓ SUCCESS |
| Decorator Medium 6 | **2.05** | **6.69** | ✓ SUCCESS |
| **Target** | **>2.0** | **>5.0** | |

## Usage

```bash
python align_clean.py <capture_zip> <sm_file> <difficulty_level>
```

**Example:**
```bash
python align_clean.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5
```

**Arguments:**
- `capture_zip`: Path to sensor capture zip file (containing Gravity.csv)
- `sm_file`: Path to StepMania .sm chart file
- `difficulty_level`: Numeric difficulty level (e.g., 5 for Medium-5)

## Method

The solution uses a **biomechanical model**:
- Focuses on LEFT ARROW only (single direction = consistent pattern)
- Edge event detection (transitions via diff)
- Exponential decay kernel (tau=0.1s) models body dynamics
- FFT-based correlation
- Bandpass filter (0.5-8 Hz) isolates human movement frequencies

## Output

Computes time offset between sensor recording and chart:
- Console: offset, peak ratio, z-score
- PNG: correlation plot showing clear peak

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib

## Project Structure

```
DDR-Accelero/
├── align_clean.py          # Working alignment solution
├── raw_data/               # Sensor captures (.zip from Android Sensor Logger)
├── sm_files/               # StepMania charts (.sm files)
├── artifacts/              # Generated correlation plots (2 examples)
└── experiments/            # Original experiment scripts (reference)
```

## Key Insight

The biomechanical approach works by **matching the physics** of DDR gameplay:
- Single arrow = consistent sensor response
- Exponential kernel = body response dynamics (100ms decay)
- Edge events = transition detection (foot hitting pad)