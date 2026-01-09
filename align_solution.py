#!/usr/bin/env python3
"""
SOLUTION: Use sliding-window note density as intermediate signal.

The key insight: Don't try to match individual notes. Instead, match the
PATTERN of note density over time (notes per second in sliding window).

This should give a much clearer correlation peak.
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import correlate


def load_data(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    time = df['seconds_elapsed'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return time, magnitude


def parse_sm(sm_path, difficulty):
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    offset = float(re.search(r'#OFFSET:([^;]+);', content).group(1))
    bpm_str = re.search(r'#BPMS:([^;]+);', content).group(1).strip()
    bpm = float(bpm_str.split('=')[1].split(',')[0] if '=' in bpm_str else bpm_str.split(',')[0])
    
    chart_pattern = r'#NOTES:\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^;]+);'
    for match in re.finditer(chart_pattern, content, re.MULTILINE | re.DOTALL):
        if difficulty.lower() in match.group(3).lower():
            notes_data = match.group(6).strip()
            break
    
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


def preprocess_sensor(time_sec, magnitude, rate=100):
    """Process sensor to get activity envelope."""
    duration = time_sec[-1] - time_sec[0]
    new_time = np.linspace(0, duration, int(duration * rate))
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC
    resampled = resampled - np.mean(resampled)
    
    # Bandpass 0.5-10 Hz
    sos = signal.butter(4, [0.5, 10], btype='band', fs=rate, output='sos')
    filtered = signal.sosfiltfilt(sos, resampled)
    
    # Envelope
    envelope = np.abs(filtered)
    window_size = int(0.2 * rate)
    hann = signal.windows.hann(window_size)
    envelope = signal.convolve(envelope, hann/np.sum(hann), mode='same')
    
    return new_time, envelope


def create_density_signal(note_times, duration, rate=100, window_sec=0.5):
    """Create signal representing note density (notes/sec) in sliding window.
    
    This is the KEY difference: we're creating a continuous function of
    "how active is the gameplay" rather than individual note markers.
    """
    num_samples = int(duration * rate)
    density = np.zeros(num_samples)
    
    # For each sample, count notes within ±window_sec/2
    time_array = np.linspace(0, duration, num_samples)
    half_window = window_sec / 2.0
    
    for i, t in enumerate(time_array):
        # Count notes in window [t-half, t+half]
        count = np.sum((note_times >= t - half_window) & (note_times < t + half_window))
        # Convert to notes per second
        density[i] = count / window_sec
    
    # Smooth the density curve
    smooth_width = int(0.1 * rate)  # 100ms smoothing
    if smooth_width > 0:
        kernel = signal.windows.gaussian(smooth_width, smooth_width/3)
        density = signal.convolve(density, kernel/np.sum(kernel), mode='same')
    
    return density


def correlate_signals(sensor, reference, rate):
    s1 = (sensor - np.mean(sensor)) / (np.std(sensor) + 1e-10)
    s2 = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)
    
    corr = correlate(s1, s2, mode='full')
    lags = np.arange(-len(s2)+1, len(s1)) / rate
    
    peak_idx = np.argmax(corr)
    peak_lag = lags[peak_idx]
    peak_val = corr[peak_idx]
    
    exclude = int(2 * rate)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx-exclude):min(len(corr), peak_idx+exclude)] = False
    
    if np.sum(mask) > 0:
        second = np.max(corr[mask])
        ratio = peak_val / second if second > 1e-10 else 999
    else:
        ratio = 999
    
    z = (peak_val - np.mean(corr)) / np.std(corr)
    
    return lags, corr, peak_lag, ratio, z


def main():
    if len(sys.argv) != 4:
        print("Usage: python align_solution.py <capture.zip> <song.sm> <difficulty>")
        sys.exit(1)
    
    cap = Path(sys.argv[1])
    sm = Path(sys.argv[2])
    diff = sys.argv[3]
    
    print("="*70)
    print("SOLUTION: Note Density-Based Alignment")
    print("="*70)
    
    # Load
    print("\n[1/4] Loading...")
    time, mag = load_data(cap)
    notes, bpm, ofs = parse_sm(sm, diff)
    print(f"  Sensor: {time[-1]-time[0]:.1f}s, Notes: {len(notes)}, BPM={bpm}")
    
    # Process sensor
    print("\n[2/4] Processing sensor envelope...")
    rate = 100
    sensor_time, sensor_env = preprocess_sensor(time, mag, rate)
    
    # Create NOTE DENSITY signal
    print("\n[3/4] Creating note density signal (KEY STEP)...")
    duration = max(sensor_time[-1], notes[-1] if len(notes) > 0 else 0)
    density = create_density_signal(notes, duration, rate, window_sec=0.5)
    
    # Match lengths
    min_len = min(len(sensor_env), len(density))
    sensor_env = sensor_env[:min_len]
    density = density[:min_len]
    
    # Correlate
    print("\n[4/4] Correlating...")
    lags, corr, peak_lag, ratio, z = correlate_signals(sensor_env, density, rate)
    
    print(f"\nRESULTS:")
    print(f"  Peak lag: {peak_lag:.3f}s")
    print(f"  Peak ratio: {ratio:.2f} {'✓✓✓ GOOD' if ratio > 2.0 else '⚠ WEAK'}")
    print(f"  Z-score: {z:.2f} {'✓✓✓ GOOD' if z > 5.0 else '⚠ WEAK'}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(sensor_time, sensor_env, linewidth=0.7)
    axes[0].set_title('Sensor: Magnitude Envelope')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Activity')
    axes[0].grid(True, alpha=0.3)
    
    density_time = np.linspace(0, duration, len(density))
    axes[1].plot(density_time, density, linewidth=0.9, color='orange')
    axes[1].set_title('NOTE DENSITY from .sm (notes/sec in 0.5s sliding window)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Notes/sec')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lags, corr, linewidth=0.8)
    axes[2].axvline(peak_lag, color='r', linestyle='--', linewidth=2.5, 
                    label=f'Peak: {peak_lag:.2f}s')
    axes[2].set_title(f'Cross-correlation: Peak ratio={ratio:.2f}, Z-score={z:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlim([-60, 60])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'{cap.stem}_SOLUTION.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_file}")
    
    if ratio > 2.0 and z > 5.0:
        print("\n" + "="*70)
        print("✓✓✓ SUCCESS: CLEAR CORRELATION PEAK FOUND!")
        print("="*70)
        print(f"First note ({notes[0]:.2f}s in .sm) → {peak_lag + notes[0]:.2f}s in recording")
    else:
        print("\n⚠ Correlation still weak with density approach")
    
    print("="*70)


if __name__ == '__main__':
    main()
