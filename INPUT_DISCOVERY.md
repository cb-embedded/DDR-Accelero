# Input Discovery - DDR-Accelero Alignment

## Capture File Format

**Location**: `raw_data/*.zip`

**Contents**: ZIP archives from Android Sensor Logger app containing:
- `Gravity.csv` - Accelerometer/gravity sensor data (primary signal)
- `Gyroscope.csv` - Gyroscope data
- `Magnetometer.csv` - Magnetometer data
- `Metadata.csv` - Recording metadata

**Gravity.csv Format**:
```
time,seconds_elapsed,z,y,x
1767725101070056200,0.19005615234375,8.289701461791992,5.236734867095947,-0.1668400913476944
```
- Column 0: `time` - Unix nanosecond timestamp
- Column 1: `seconds_elapsed` - Seconds since recording start
- Column 2: `z` - Z-axis acceleration (m/s²)
- Column 3: `y` - Y-axis acceleration (m/s²)
- Column 4: `x` - X-axis acceleration (m/s²)

**Time Base**: `seconds_elapsed` column provides time in seconds from start of recording

**Sampling Rate Inference**: 
- Calculate from time differences: `1 / mean(diff(seconds_elapsed))`
- Typical rate: ~200 Hz (varies slightly)
- Will be computed dynamically per capture

## StepMania (.sm) File Format

**Location**: `sm_files/*.sm`

**Key Fields**:
- `#TITLE:` - Song title
- `#ARTIST:` - Artist name
- `#OFFSET:` - Initial delay in seconds before first beat
- `#BPMS:` - Beats per minute (format: `beat=bpm`)
- `#NOTES:` - Chart data blocks

**Chart Structure**:
```
#NOTES:
     dance-single:
     [Author]:
     [Difficulty]:
     [Meter]:
     [Radar]:
     [Note data]:
```

**Note Data Format**:
- Measures separated by commas
- Each measure = 4 beats
- Each line in measure = 4 characters (left, down, up, right)
- `0` = no arrow, `1` = tap, `2` = hold start, `3` = hold end

**Timing Calculation**:
- `beat_duration = 60.0 / bpm`
- `note_time = offset + (beat_number * beat_duration)`

## Chart Selection

**Available difficulties per .sm file**: Easy, Medium, Hard, Expert, Edit (varies by song)

**For Lucky Orb example**: 
- File: `sm_files/Lucky Orb.sm`
- Capture: `raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip`
- Suggested chart: **Medium** (indicated in filename)
- OFFSET: 1.764s
- BPM: 126

## Existing Scripts Assumptions (Problems Identified)

**03_align_signals.py issues**:
1. **Over-processing**: Applies bandpass filter (0.5-10 Hz), envelope detection, smoothing
   - Problem: Loses correlation with raw impulse pattern from notes
2. **Magnitude instead of x-axis**: Uses `sqrt(x² + y² + z²)`
   - Problem: Mixes all axes, dilutes x-axis signal
3. **Heavy smoothing of reference**: Gaussian convolution with 100ms window
   - Problem: Spreads impulses, reduces peak sharpness
4. **No dominance metric**: Only reports best offset, doesn't measure peak quality
5. **Arbitrary parameters**: Window sizes and filter cutoffs not justified

## Required Clarifications

None at this stage - format is clear from inspection. Will proceed with:
- X-axis only (column 4 from Gravity.csv)
- Raw data (minimal normalization only)
- Simple impulse train for expected signal
- Dominant peak metric computation
