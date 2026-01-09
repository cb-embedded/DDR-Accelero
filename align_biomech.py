#!/usr/bin/env python3
"""
Alternative approach from user: Biomechanical transformation with exponential kernel.

Key differences:
1. Extract ONLY left arrow events (not all notes)
2. Use diff to get edge events (transitions)
3. Apply exponential decay kernel (biomechanical model)
4. FFT-based correlation
5. Bandpass filter 0.5-8 Hz

Testing on Lucky Orb 5 Medium.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import zipfile
import re


def load_accel_data(zip_path):
    """Load accelerometer X-axis data."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('Gravity.csv') as f:
            acc = pd.read_csv(f)
    
    t_acc = acc["seconds_elapsed"].to_numpy()
    x_axis = acc["x"].to_numpy()
    
    return t_acc, x_axis


def parse_sm_left_arrow(sm_path, difficulty_name, difficulty_level):
    """Parse .sm file and extract ONLY left arrow (first position) events.
    
    Args:
        sm_path: Path to .sm file
        difficulty_name: e.g., "Medium"
        difficulty_level: e.g., 5
    """
    with open(sm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Extract BPM
    bpm_line = [l for l in text.splitlines() if l.startswith("#BPMS:")][0]
    bpm = float(bpm_line.split(":")[1].split(";")[0].split("=")[1])
    sec_per_beat = 60.0 / bpm
    
    # Find matching chart
    blocks = text.split("#NOTES:")[1:]
    chart_lines = None
    
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) < 6:
            continue
        
        # Check difficulty name and level
        diff_name = lines[2].strip(":").lower()
        try:
            diff_level = int(lines[3].strip(":"))
        except:
            continue
        
        if diff_name == difficulty_name.lower() and diff_level == difficulty_level:
            chart_lines = lines[5:]
            break
    
    if chart_lines is None:
        raise ValueError(f"Chart not found for {difficulty_name} level {difficulty_level}")
    
    # Parse measures
    measures = []
    cur = []
    for l in chart_lines:
        if l == ";":
            if cur:
                measures.append(cur)
            break
        if l == ",":
            measures.append(cur)
            cur = []
        else:
            cur.append(l)
    
    # Extract left arrow times (first position in 4-char string)
    times = []
    left_arrows = []
    t = 0.0
    
    for m in measures:
        n = len(m)
        for r in m:
            times.append(t)
            r = r.ljust(4, "0")
            # Left arrow is first character: 0=none, 1=tap
            left_arrows.append(1 if r[0] == "1" else 0)
            t += (4 * sec_per_beat) / n
    
    return np.array(times), np.array(left_arrows), bpm


def create_biomechanical_signal(times, arrows, dt=0.01, tau=0.10, bp_low=0.5, bp_high=8.0):
    """Create biomechanical model signal from arrow events.
    
    Args:
        times: Array of time points
        arrows: Binary array (1=arrow, 0=no arrow)
        dt: Time step (100 Hz)
        tau: Biomechanical time constant (s)
        bp_low: Bandpass low cutoff (Hz)
        bp_high: Bandpass high cutoff (Hz)
    
    Returns:
        t_chart: Time array
        p: Processed signal
    """
    # Resample to uniform grid
    t_chart = np.arange(times.min(), times.max(), dt)
    s_i = np.interp(t_chart, times, arrows)
    
    # 1) Extract edge events (transitions)
    e = np.diff(s_i, prepend=0)
    
    # 2) Create causal exponential decay kernel
    k_t = np.arange(0, 0.5, dt)
    kernel = np.exp(-k_t / tau)
    kernel /= kernel.sum()
    
    # 3) Convolve events with kernel
    p = np.convolve(e, kernel, mode="same")
    
    # 4) Bandpass filter
    fs = 1.0 / dt
    b, a = butter(2, [bp_low/(fs/2), bp_high/(fs/2)], btype="band")
    p = filtfilt(b, a, p)
    
    # Normalize
    p = (p - p.mean()) / p.std()
    
    return t_chart, p


def process_accelerometer(t_acc, ax, dt=0.01):
    """Process accelerometer signal.
    
    Args:
        t_acc: Time array (original sampling)
        ax: X-axis acceleration
        dt: Target time step (100 Hz)
    
    Returns:
        t_i: Resampled time array
        ax_norm: Normalized signal
    """
    # Resample to uniform 100 Hz
    t_i = np.arange(t_acc.min(), t_acc.max(), dt)
    ax_interp = np.interp(t_i, t_acc, ax)
    
    # Normalize
    ax_norm = (ax_interp - ax_interp.mean()) / ax_interp.std()
    
    return t_i, ax_norm


def fft_correlation(signal1, signal2, dt):
    """Compute correlation using FFT.
    
    Args:
        signal1: First signal (accelerometer)
        signal2: Second signal (biomechanical model)
        dt: Time step
    
    Returns:
        lags: Time lags
        corr: Correlation values
        offset: Detected offset (s)
    """
    # Make same length
    n = min(len(signal1), len(signal2))
    signal1 = signal1[:n]
    signal2 = signal2[:n]
    
    # FFT-based correlation
    corr = np.fft.ifft(np.fft.fft(signal1) * np.conj(np.fft.fft(signal2))).real
    lags = np.arange(n) * dt
    
    # Find peak
    peak_idx = np.argmax(corr)
    offset = lags[peak_idx]
    
    # Compute metrics
    # Peak ratio (exclude ±2s around peak)
    exclude_samples = int(2.0 / dt)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx - exclude_samples):min(len(corr), peak_idx + exclude_samples)] = False
    
    if np.any(mask):
        second_peak = np.max(corr[mask])
        peak_ratio = corr[peak_idx] / second_peak if second_peak > 1e-10 else 999
    else:
        peak_ratio = 999
    
    z_score = (corr[peak_idx] - np.mean(corr)) / np.std(corr)
    
    return lags, corr, offset, peak_ratio, z_score


def main():
    if len(sys.argv) != 4:
        print("Usage: python align_biomech.py <capture.zip> <song.sm> <difficulty_level>")
        print("Example: python align_biomech.py raw_data/Lucky_Orb_5_Medium-....zip 'sm_files/Lucky Orb.sm' 5")
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    diff_level = int(sys.argv[3])
    
    print("="*70)
    print("BIOMECHANICAL APPROACH (Alternative from User)")
    print("="*70)
    print(f"\nCapture: {capture_path.name}")
    print(f"SM: {sm_path.name}")
    print(f"Difficulty: Medium level {diff_level}")
    print(f"\nKey features:")
    print("  - LEFT ARROW ONLY (not all notes)")
    print("  - Edge events (diff)")
    print("  - Exponential decay kernel (tau=0.1s)")
    print("  - FFT correlation")
    print("  - Bandpass 0.5-8 Hz")
    
    # Parameters
    DT = 0.01  # 100 Hz
    TAU = 0.10  # biomechanical constant
    BP_LOW = 0.5  # Hz
    BP_HIGH = 8.0  # Hz
    
    # Load accelerometer
    print("\n[1/4] Loading accelerometer X-axis...")
    t_acc, ax = load_accel_data(capture_path)
    print(f"  Duration: {t_acc[-1] - t_acc[0]:.1f}s")
    print(f"  Samples: {len(ax)}")
    
    # Parse .sm for LEFT ARROW only
    print("\n[2/4] Parsing .sm for LEFT ARROW events...")
    times, left_arrows, bpm = parse_sm_left_arrow(sm_path, "Medium", diff_level)
    print(f"  BPM: {bpm}")
    print(f"  Total arrow events: {len(times)}")
    print(f"  Left arrows pressed: {np.sum(left_arrows)}")
    
    # Process signals
    print("\n[3/4] Processing signals...")
    print("  a) Resampling accelerometer to 100 Hz...")
    t_i, ax_norm = process_accelerometer(t_acc, ax, DT)
    
    print("  b) Creating biomechanical model signal...")
    t_chart, bio_signal = create_biomechanical_signal(
        times, left_arrows, dt=DT, tau=TAU, bp_low=BP_LOW, bp_high=BP_HIGH)
    
    # Correlation
    print("\n[4/4] Computing FFT correlation...")
    lags, corr, offset, peak_ratio, z_score = fft_correlation(ax_norm, bio_signal, DT)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Detected offset: {offset:.3f}s")
    print(f"Peak ratio: {peak_ratio:.2f} {'✓✓✓ GOOD' if peak_ratio > 2.0 else '⚠ WEAK'}")
    print(f"Z-score: {z_score:.2f} {'✓✓✓ GOOD' if z_score > 5.0 else '⚠ WEAK'}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Plot 1: Accelerometer
    axes[0].plot(t_i, ax_norm, linewidth=0.6, alpha=0.8)
    axes[0].set_title('Accelerometer X-axis (normalized, 100 Hz)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Normalized acceleration')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Biomechanical signal
    axes[1].plot(t_chart, bio_signal, linewidth=0.8, color='orange')
    axes[1].set_title('Biomechanical Model Signal (LEFT ARROW only, edge events + exp kernel + bandpass)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Model signal')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation
    axes[2].plot(lags, corr, linewidth=0.8)
    axes[2].axvline(offset, color='r', linestyle='--', linewidth=2.5,
                    label=f'Peak: {offset:.2f}s')
    axes[2].set_title(f'FFT Correlation: Peak ratio={peak_ratio:.2f}, Z-score={z_score:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_ylabel('Correlation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, min(200, lags[-1])])  # Show first 200s
    
    plt.tight_layout()
    
    # Save
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'{capture_path.stem}_BIOMECH.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_file}")
    
    if peak_ratio > 2.0 and z_score > 5.0:
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS: CLEAR CORRELATION PEAK!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ Still showing weak correlation")
        print("="*70)
    
    print()


if __name__ == '__main__':
    main()
