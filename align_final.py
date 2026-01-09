#!/usr/bin/env python3
"""
Ultra-minimal alignment: absolute basics with full correlation display.

Based on user feedback: there SHOULD be a clear peak if done correctly.
Let's go back to basics and check everything carefully.
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt


def load_data(zip_path):
    """Load accelerometer (gravity) data."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    time = df['seconds_elapsed'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    return time, x, y, z


def parse_sm(sm_path, difficulty):
    """Parse .sm file to get note times."""
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Get BPM and offset
    offset = float(re.search(r'#OFFSET:([^;]+);', content).group(1))
    bpm_str = re.search(r'#BPMS:([^;]+);', content).group(1).strip()
    bpm = float(bpm_str.split('=')[1].split(',')[0] if '=' in bpm_str else bpm_str.split(',')[0])
    
    # Find chart
    chart_pattern = r'#NOTES:\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^;]+);'
    for match in re.finditer(chart_pattern, content, re.MULTILINE | re.DOTALL):
        if difficulty.lower() in match.group(3).lower():
            notes_data = match.group(6).strip()
            break
    
    # Parse note times
    note_times = []
    beat_duration = 60.0 / bpm
    current_beat = 0.0
    
    for measure in notes_data.split(','):
        lines = [l.strip() for l in measure.strip().split('\n') if l.strip() and len(l.strip()) == 4]
        if lines:
            beats_per_line = 4.0 / len(lines)
            for i, line in enumerate(lines):
                if any(c != '0' for c in line):
                    note_times.append(offset + (current_beat + i * beats_per_line) * beat_duration)
            current_beat += 4.0
    
    return np.array(note_times), bpm, offset


def resample(time, signal, rate=100):
    """Resample to fixed rate."""
    duration = time[-1] - time[0]
    new_time = np.linspace(0, duration, int(duration * rate))
    new_signal = np.interp(new_time, time - time[0], signal)
    return new_time, new_signal


def create_reference(note_times, duration, rate):
    """Create reference signal from note times."""
    n = int(duration * rate)
    ref = np.zeros(n)
    
    # Place Gaussian at each note
    sigma = 0.075 * rate  # 75ms width
    w = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-w, w+1) / sigma) ** 2)
    
    for t in note_times:
        idx = int(t * rate)
        if 0 <= idx < n:
            left, right = max(0, idx-w), min(n, idx+w+1)
            k_left = w - (idx - left)
            k_right = k_left + (right - left)
            ref[left:right] += kernel[k_left:k_right]
    
    return ref


def correlate_signals(sensor, reference, rate):
    """Compute full cross-correlation."""
    # Normalize
    s1 = (sensor - np.mean(sensor)) / np.std(sensor)
    s2 = (reference - np.mean(reference)) / np.std(reference)
    
    # Full correlation
    corr = correlate(s1, s2, mode='full')
    lags_samples = np.arange(-len(s2)+1, len(s1))
    lags_time = lags_samples / rate
    
    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = lags_time[peak_idx]
    peak_val = corr[peak_idx]
    
    # Compute peak ratio (exclude ±2s around peak)
    exclude_width = int(2 * rate)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx-exclude_width):min(len(corr), peak_idx+exclude_width)] = False
    
    if np.sum(mask) > 0:
        second_val = np.max(corr[mask])
        ratio = peak_val / second_val if second_val > 1e-10 else 999
    else:
        ratio = 999
    
    # Z-score
    z = (peak_val - np.mean(corr)) / np.std(corr)
    
    return lags_time, corr, peak_lag, ratio, z


def main():
    if len(sys.argv) != 4:
        print("Usage: python align_final.py <capture.zip> <song.sm> <difficulty>")
        sys.exit(1)
    
    cap_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    diff = sys.argv[3]
    
    print("="*70)
    print("FINAL MINIMAL ALIGNMENT CHECK")
    print("="*70)
    
    # Load sensor
    print("\n[1/5] Loading sensor...")
    time, x, y, z = load_data(cap_path)
    print(f"  {len(time)} samples, {time[-1]-time[0]:.1f}s")
    
    # Parse .sm
    print("\n[2/5] Parsing .sm...")
    notes, bpm, ofs = parse_sm(sm_path, diff)
    print(f"  {len(notes)} notes, BPM={bpm}, offset={ofs}s")
    print(f"  First note: {notes[0]:.2f}s, Last: {notes[-1]:.2f}s")
    
    # Test each axis
    print("\n[3/5] Testing axes...")
    rate = 100
    results = {}
    
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        # Resample
        t_r, sig_r = resample(time, axis_data, rate)
        
        # Bandpass filter (optional - keep movements, remove DC and high freq noise)
        b, a = butter(3, [1, 15], btype='band', fs=rate)
        sig_filt = filtfilt(b, a, sig_r)
        
        # Create reference
        duration = max(t_r[-1], notes[-1] if len(notes) > 0 else 0)
        ref = create_reference(notes, duration, rate)
        
        # Match lengths
        min_len = min(len(sig_filt), len(ref))
        sig_filt = sig_filt[:min_len]
        ref = ref[:min_len]
        
        # Correlate
        lags, corr, peak_lag, ratio, z = correlate_signals(sig_filt, ref, rate)
        
        results[axis_name] = {
            'signal': sig_filt,
            'reference': ref,
            'lags': lags,
            'corr': corr,
            'peak_lag': peak_lag,
            'ratio': ratio,
            'z': z
        }
        
        status = "✓✓✓" if ratio > 2.0 and z > 5.0 else "⚠"
        print(f"  {axis_name}: lag={peak_lag:6.2f}s, ratio={ratio:5.2f}, z={z:5.2f} {status}")
    
    # Best axis
    best = max(results.keys(), key=lambda k: results[k]['ratio'])
    r = results[best]
    
    print("\n[4/5] BEST:")
    print(f"  Axis: {best}")
    print(f"  Peak lag: {r['peak_lag']:.3f}s")
    print(f"  Peak ratio: {r['ratio']:.2f} {'✓ GOOD' if r['ratio'] > 2.0 else '⚠ WEAK'}")
    print(f"  Z-score: {r['z']:.2f} {'✓ GOOD' if r['z'] > 5.0 else '⚠ WEAK'}")
    
    # Plot
    print("\n[5/5] Plotting...")
    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    
    # Plot each axis correlation
    for idx, axis_name in enumerate(['x', 'y', 'z']):
        ax = axes[idx]
        r = results[axis_name]
        ax.plot(r['lags'], r['corr'], linewidth=0.7, alpha=0.8)
        ax.axvline(r['peak_lag'], color='r', linestyle='--', linewidth=1.5, 
                   label=f"Peak: {r['peak_lag']:.2f}s")
        ax.set_title(f"{axis_name.upper()}-axis: ratio={r['ratio']:.2f}, z={r['z']:.2f}")
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Correlation')
        ax.set_xlim([-50, 50])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Best axis - full range
    ax = axes[3]
    r = results[best]
    ax.plot(r['lags'], r['corr'], linewidth=0.8)
    ax.axvline(r['peak_lag'], color='r', linestyle='--', linewidth=2)
    ax.set_title(f"BEST ({best}): FULL LAG RANGE")
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('Correlation')
    ax.legend([f'Peak at {r["peak_lag"]:.2f}s'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'{cap_path.stem}_FINAL.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_file}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if r['ratio'] > 2.0 and r['z'] > 5.0:
        print("✓✓✓ CLEAR CORRELATION PEAK FOUND!")
        print(f"    First note ({notes[0]:.2f}s in .sm) → {r['peak_lag'] + notes[0]:.2f}s in recording")
    else:
        print("⚠⚠⚠ Correlation is still WEAK")
        print("    This suggests:")
        print("    1. Individual notes don't create distinct sensor peaks")
        print("    2. Timing/BPM mismatch")
        print("    3. Phone placement variability")
        print("    4. Or another fundamental issue")
    
    print("="*70)


if __name__ == '__main__':
    main()
