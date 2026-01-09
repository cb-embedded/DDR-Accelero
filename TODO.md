# TODO: Alignment Ambiguity Diagnosis

## Current Status
✓ Step 0: Input discovery completed (see INPUT_DISCOVERY.md)
✓ Implemented minimal alignment script: `align_x_axis.py`
✓ Generated PNG artifacts in `artifacts/` directory
⚠️ **PROBLEM**: Weak correlation peak detected

## Results from Lucky Orb Medium Test
- **Peak ratio**: 1.15 (target: > 2.0 for dominance)
- **Z-score**: 3.63 (target: > 5.0)
- **Assessment**: Alignment is ambiguous - multiple peaks of similar magnitude

## Diagnosis: Why is the peak weak?

The cross-correlation between raw x-axis and impulse train is showing weak/ambiguous alignment. Possible causes:

### Hypothesis 1: Simple impulse train is too crude
**Evidence**: 
- Expected signal is just delta functions (single-sample impulses)
- Real accelerometer response to a step is not instantaneous
- Each foot step likely produces a bipolar pulse (acceleration + deceleration)

**Proposed solution**: Apply a minimal kernel to impulses to mimic physical response
- Option A: Derivative-like bipolar pulse (+ followed by -)
- Option B: Short Gaussian pulse (~50-100ms width)
- Option C: Measured impulse response from a known aligned region

### Hypothesis 2: X-axis may not be the dominant axis for steps
**Evidence**:
- Need to inspect which axis shows strongest correlation with steps
- Phone orientation during gameplay is unknown

**Proposed solution**: 
- Try Y-axis and Z-axis separately
- Try magnitude (as original script did)
- Document phone placement/orientation

### Hypothesis 3: Timing assumptions are incorrect
**Evidence**:
- BPM or offset from .sm file might not match actual gameplay
- Player may have started late/early
- Sensor recording may have started before/after song

**Proposed solution**:
- Manually inspect raw signal around expected first note time
- Check if visual patterns in raw signal match note density from .sm
- Verify BPM by measuring periodicity in raw signal during high-density regions

### Hypothesis 4: Raw signal needs minimal preprocessing
**Evidence**:
- DC offset and slow drift visible in raw x signal
- High-frequency noise may be masking correlation

**Proposed solution** (only if justified):
- Remove DC offset (mean subtraction) - ALREADY DONE in correlation
- High-pass filter to remove drift (>0.5 Hz)
- Low-pass filter to remove sensor noise (<20 Hz)

## Decision Required

Before proceeding, please specify which approach to try:

### Option 1: Improve expected signal shape (RECOMMENDED FIRST)
Add a minimal physical kernel to impulses to better match accelerometer response.

**Command to run after modification**:
```bash
python align_x_axis.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" Medium
```

### Option 2: Test other axes
Modify script to allow axis selection (x, y, z, or magnitude).

**Commands to run**:
```bash
python align_x_axis.py --axis y "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" Medium
python align_x_axis.py --axis z "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" Medium
python align_x_axis.py --axis magnitude "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" Medium
```

### Option 3: Manual verification of timing
Inspect alignment overlay more carefully and check if patterns are visible by eye.

**Questions to answer**:
1. Can you see periodic patterns in raw x signal during gameplay (20-60s in overlay)?
2. Do high-density note regions (visible in expected signal) correspond to high activity in raw signal?
3. Should we zoom into specific time windows for verification?

### Option 4: Add minimal preprocessing
Add justified preprocessing (e.g., high-pass filter for drift, derivative of signal).

**Trade-off**: More processing = less "raw", but may reveal clearer patterns

## Recommendation

Start with **Option 1** (improve expected signal shape) because:
1. It's the most likely cause - impulses are too simplistic
2. It stays true to "minimal" approach
3. It's a justified physical assumption (steps create acceleration/deceleration)
4. Easy to test and compare results

Specifically, try a **derivative-like bipolar kernel**: each step creates both acceleration (positive) and deceleration (negative) in quick succession (~100ms apart).

Please confirm or specify an alternative approach.
