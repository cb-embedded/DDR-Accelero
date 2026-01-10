#!/usr/bin/env python3
"""
SOLUTION: Biomechanical approach for DDR alignment.

Key insight: Use LEFT ARROW ONLY + biomechanical model (exponential decay kernel).

This achieves clear correlation peaks (ratio >2.0, z-score >5.0).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import zipfile


def align_capture(capture_path, sm_path, diff_level, diff_type='medium', verbose=True):
    """
    Align accelerometer capture with StepMania chart.
    
    Args:
        capture_path: Path to capture .zip file
        sm_path: Path to .sm chart file
        diff_level: Difficulty level (int)
        diff_type: Difficulty type (e.g., 'easy', 'medium', 'hard')
        verbose: Print progress messages
        
    Returns:
        dict with keys:
            - offset: time offset in seconds
            - ratio: peak ratio metric
            - z_score: z-score metric
            - t_acc: accelerometer time array
            - ax: accelerometer signal (normalized)
            - t_chart: chart time array
            - p: processed chart signal (biomechanical model)
            - lags: correlation lag array
            - corr: correlation values
    """
    capture_path = Path(capture_path)
    sm_path = Path(sm_path)
    
    # ================== PARAMETERS ==================
    DT = 0.01          # 100 Hz sampling
    TAU = 0.10         # biomechanical time constant (s)
    BP_LOW = 0.5       # bandpass low (Hz)
    BP_HIGH = 8.0      # bandpass high (Hz)
    
    if verbose:
        print("="*70)
        print("DDR ALIGNMENT - BIOMECHANICAL APPROACH")
        print("="*70)
    
    # ================== LOAD ACCELEROMETER ==================
    if verbose:
        print("\n[1/3] Loading accelerometer X-axis...")
    with zipfile.ZipFile(capture_path, 'r') as zf:
        with zf.open('Gravity.csv') as f:
            acc = pd.read_csv(f)
    
    t_acc = acc["seconds_elapsed"].to_numpy()
    t_i = np.arange(t_acc.min(), t_acc.max(), DT)
    ax = np.interp(t_i, t_acc, acc["x"].to_numpy())
    ax = (ax - ax.mean()) / ax.std()
    
    if verbose:
        print(f"  Duration: {t_acc[-1] - t_acc[0]:.1f}s, Samples: {len(t_i)}")
    
    # ================== PARSE STEPMANIA (LEFT ARROW ONLY) ==================
    if verbose:
        print("\n[2/3] Parsing .sm for LEFT ARROW...")
    with open(sm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Extract BPM (use first BPM value if multiple)
    bpm_line = [l for l in text.splitlines() if l.startswith("#BPMS:")][0]
    bpm_str = bpm_line.split(":")[1].split(";")[0].split("=")[1].split(",")[0]
    bpm = float(bpm_str)
    sec_per_beat = 60.0 / bpm
    
    # Find chart at specified difficulty type and level
    chart = None
    blocks = text.split("#NOTES:")[1:]
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 6:
            if lines[2].strip(":").lower() == diff_type.lower() and int(lines[3].strip(":")) == diff_level:
                chart = lines[5:]
                break
    
    if chart is None:
        raise ValueError(f"Chart not found: {diff_type} level {diff_level}")
    
    # Parse measures
    measures, cur = [], []
    for l in chart:
        if l == ";":
            if cur: measures.append(cur)
            break
        if l == ",":
            measures.append(cur)
            cur = []
        else:
            cur.append(l)
    
    # Extract LEFT ARROW times (first position = left)
    times, s = [], []
    t = 0.0
    for m in measures:
        n = len(m)
        for r in m:
            times.append(t)
            r = r.ljust(4, "0")
            s.append(1 if r[0] == "1" else 0)  # Left arrow
            t += (4 * sec_per_beat) / n
    
    times = np.array(times)
    s = np.array(s)
    
    if verbose:
        print(f"  BPM: {bpm}, Left arrows: {np.sum(s)}/{len(s)} events")
    
    # Resample to uniform grid
    t_chart = np.arange(times.min(), times.max(), DT)
    s_i = np.interp(t_chart, times, s)
    
    # ================== BIOMECHANICAL TRANSFORMATION ==================
    if verbose:
        print("\n[3/3] Applying biomechanical model...")
    
    # 1) Edge events
    e = np.diff(s_i, prepend=0)
    
    # 2) Exponential decay kernel (causal)
    k_t = np.arange(0, 0.5, DT)
    kernel = np.exp(-k_t / TAU)
    kernel /= kernel.sum()
    
    # 3) Convolution
    p = np.convolve(e, kernel, mode="same")
    
    # 4) Bandpass filter
    fs = 1.0 / DT
    b, a = butter(2, [BP_LOW/(fs/2), BP_HIGH/(fs/2)], btype="band")
    p = filtfilt(b, a, p)
    
    p = (p - p.mean()) / p.std()
    
    # ================== FFT CORRELATION ==================
    n = min(len(ax), len(p))
    ax_corr = ax[:n]
    p_corr = p[:n]
    
    corr = np.fft.ifft(np.fft.fft(ax_corr) * np.conj(np.fft.fft(p_corr))).real
    lags = np.arange(n) * DT
    
    peak_idx = np.argmax(corr)
    offset = lags[peak_idx]
    
    # Compute metrics
    exclude = int(2.0 / DT)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx-exclude):min(len(corr), peak_idx+exclude)] = False
    
    if np.any(mask):
        second = np.max(corr[mask])
        ratio = corr[peak_idx] / second if second > 1e-10 else 999
    else:
        ratio = 999
    
    z = (corr[peak_idx] - np.mean(corr)) / np.std(corr)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Offset: {offset:.3f}s")
        print(f"Peak ratio: {ratio:.2f} {'✓ GOOD' if ratio > 2.0 else '⚠ WEAK'}")
        print(f"Z-score: {z:.2f} {'✓ GOOD' if z > 5.0 else '⚠ WEAK'}")
        
        if ratio > 2.0 and z > 5.0:
            print("\n✓✓✓ CLEAR CORRELATION PEAK DETECTED!")
    
    return {
        'offset': offset,
        'ratio': ratio,
        'z_score': z,
        't_acc': t_i,
        'ax': ax,
        't_chart': t_chart,
        'p': p,
        'lags': lags,
        'corr': corr,
        'peak_idx': peak_idx
    }


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        print("\nUsage: python align_clean.py <capture.zip> <song.sm> <difficulty_level>")
        print("Example: python align_clean.py 'raw_data/Lucky_Orb_5_Medium-....zip' 'sm_files/Lucky Orb.sm' 5")
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    diff_level = int(sys.argv[3])
    
    result = align_capture(capture_path, sm_path, diff_level, verbose=True)
    
    # ================== PLOT ==================
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    
    axes[0].plot(result['t_acc'], result['ax'], linewidth=0.6)
    axes[0].set_title('Accelerometer X-axis (normalized)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(result['t_chart'], result['p'], linewidth=0.8, color='orange')
    axes[1].set_title('Biomechanical Model (LEFT ARROW: edges + exp kernel + bandpass)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(result['lags'], result['corr'], linewidth=0.8)
    axes[2].axvline(result['offset'], color='r', linestyle='--', linewidth=2)
    axes[2].set_title(f'FFT Correlation: ratio={result["ratio"]:.2f}, z={result["z_score"]:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_xlim([0, min(200, result['lags'][-1])])
    axes[2].legend([f'Peak: {result["offset"]:.2f}s'])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'{capture_path.stem}_CLEAN.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_file}")
    print("="*70)


if __name__ == '__main__':
    main()
