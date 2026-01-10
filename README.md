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

### Alignment Tool

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

### Dataset Creation Tool

```bash
python create_dataset.py <capture_zip> <sm_file> <difficulty_level> <num_samples>
```

**Example:**
```bash
python create_dataset.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 10
```

**Arguments:**
- `capture_zip`: Path to sensor capture zip file (containing Gravity.csv, Gyroscope.csv, Magnetometer.csv)
- `sm_file`: Path to StepMania .sm chart file
- `difficulty_level`: Numeric difficulty level (e.g., 5 for Medium-5)
- `num_samples`: Number of sample visualizations to generate (e.g., 10)

**Output:**
- Console: Dataset statistics (number of samples, shapes)
- PNG files: Visualization of sample windows with sensor data and arrow labels
- Dataset in memory: X (sensor windows) and Y (arrow labels)

## Method

The solution uses a **biomechanical model**:
- Focuses on LEFT ARROW only (single direction = consistent pattern)
- Edge event detection (transitions via diff)
- Exponential decay kernel (tau=0.1s) models body dynamics
- FFT-based correlation
- Bandpass filter (0.5-8 Hz) isolates human movement frequencies

## Output

### Alignment Tool Output
Computes time offset between sensor recording and chart:
- Console: offset, peak ratio, z-score
- PNG: correlation plot showing clear peak

### Dataset Tool Output
Creates a labeled dataset from aligned sensor data:
- **X**: Sensor data windows [N × window_length × 9 channels]
  - 9 channels: accelerometer (x,y,z), gyroscope (x,y,z), magnetometer (x,y,z)
  - Window: randomly sampled with configurable size (default ±1s = 2s total)
- **Y**: Arrow labels [N × 4]
  - 4 arrows: [Left, Down, Up, Right] (binary: 1=pressed, 0=not pressed)
  - Label is the closest arrow combination to the window center
  - Supports single and double arrow events
- **Offsets**: Time offsets [N]
  - Offset in seconds of the label arrow from the window center
  - Positive = arrow after center, negative = arrow before center
- **Visualizations**: PNG files showing:
  - 9 sensor channel plots with blue vertical line at window center (t=0)
  - Red vertical line at the label arrow position
  - Arrow chronogram showing nearby arrow events
  - Highlighted label arrows in the legend
  - Offset information in the title

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib

## Project Structure

```
DDR-Accelero/
├── align_clean.py          # Alignment solution (finds time offset)
├── create_dataset.py       # Dataset creation tool (creates labeled samples)
├── raw_data/               # Sensor captures (.zip from Android Sensor Logger)
├── sm_files/               # StepMania charts (.sm files)
└── artifacts/              # Generated plots and visualizations
```

## Key Insight

The biomechanical approach works by **matching the physics** of DDR gameplay:
- Single arrow = consistent sensor response
- Exponential kernel = body response dynamics (100ms decay)
- Edge events = transition detection (foot hitting pad)