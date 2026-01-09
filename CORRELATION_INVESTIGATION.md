# Correlation Investigation Results

## User Request
"Generate an intermediate signal from the arrows (.sm). Check and display correlation for the whole length of signals. Use lucky orb 5 medium. If done correctly, there should be a clear correlation peak."

## Approaches Tested

### 1. **align_minimal.py** - Basic Gaussian pulses at note times
- X-axis: peak_ratio=1.22, z_score=4.08, lag=30.27s
- Y-axis: peak_ratio=1.17, z_score=3.63, lag=30.30s  
- Z-axis: peak_ratio=1.01, z_score=3.81, lag=40.47s
- Magnitude: peak_ratio=1.21, z_score=4.07, lag=31.15s
- **Result**: WEAK across all axes

### 2. **align_energy.py** - High-pass filtered movement energy
- Peak_ratio=1.40, z_score=4.32, lag=30.32s
- **Result**: WEAK

### 3. **align_density.py** - Note density signal (2s window) vs activity
- Peak_ratio=1.06, z_score=2.39, lag=30.23s
- **Result**: WEAK

### 4. **align_gyro.py** - Gyroscope magnitude envelope
- Peak_ratio=1.28, z_score=3.79, lag=30.48s
- **Result**: WEAK

### 5. **align_final.py** - Ultra-minimal bandpass filtered axes
- X: ratio=1.13, z=4.00, lag=30.27s
- Y: ratio=1.10, z=3.51, lag=30.27s
- Z: ratio=1.01, z=3.63, lag=30.45s
- **Result**: WEAK across all

### 6. **align_replication.py** - Exact replication of experiment 03
- Lucky Orb 5 Medium: ratio=1.09, z=5.11, lag=-5.94s
- Lucky Orb 5 Medium 4: ratio=1.01, z=4.25, lag=26.43s
- Decorator Medium 6: ratio=1.07, z=4.71, lag=-7.92s
- **Result**: WEAK (note: negative lags suggest timing issues)

### 7. **align_solution.py** - Sliding window note density
- Peak_ratio=1.13, z_score=4.11, lag=-0.69s
- **Result**: WEAK

## Consistent Findings

1. **Peak ratio consistently 1.0-1.4** (target: >2.0)
2. **Offset consistently around 30s** (positive lags) or near 0/-6s (for some approaches)
3. **Z-scores range 2.4-5.1**, sometimes meeting threshold but ratio always fails
4. **All axes show similar weak correlation**
5. **Different sensor types (accel, gyro) show same pattern**

## Possible Explanations

1. **Data Quality**: The sensor data may not have strong enough signals for individual notes
2. **Timing Mismatch**: BPM or offset in .sm may not match actual gameplay
3. **Phone Placement**: Variable phone position affects sensor readings
4. **Player Variability**: Not all notes create equal sensor responses
5. **Fundamental Mismatch**: Individual DDR steps may not create distinct accelerometer peaks

## Files Generated

All correlation plots saved to `artifacts/` directory:
- `*_minimal_*.png` - Individual axis tests
- `*_energy.png` - Energy-based correlation
- `*_density.png` - Density-based correlation  
- `*_gyro.png` - Gyroscope-based correlation
- `*_FINAL.png` - Final minimal approach
- `*_REPLICATION.png` - Experiment 03 replication
- `*_SOLUTION.png` - Density solution attempt

## Recommendation

Despite extensive testing with multiple approaches, a clear dominant correlation peak (ratio >2.0) was not achieved. The consistent offset around 30s suggests approximate alignment may be correct, but the weak peak ratio indicates:

1. Either the data inherently doesn't support strong correlation, OR
2. A different approach is needed (e.g., manual annotation, audio-based sync, or different feature extraction)

The user stated "there SHOULD be a clear peak if done correctly" - this investigation has tested 7 different approaches without achieving that result.
