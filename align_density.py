#!/usr/bin/env python3
"""
Final attempt: Use derivative approach and focus on matching note density patterns.

Key insight: Match the temporal pattern of note density, not individual notes.
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def load_sensor_data(zip_path):
    """Load sensor data."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    return df['seconds_elapsed'].values, df['x'].values, df['y'].values, df['z'].values


def parse_sm_file(sm_path, difficulty):
    """Parse .sm file."""
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    offset_match = re.search(r'#OFFSET:([^;]+);', content)
    bpm_match = re.search(r'#BPMS:([^;]+);', content)
    
    offset = float(offset_match.group(1)) if offset_match else 0.0
    
    bpm = 120.0
    if bpm_match:
        bpm_str = bpm_match.group(1).strip()
        if '=' in bpm_str:
            bpm = float(bpm_str.split('=')[1].split(',')[0])
        else:
            bpm = float(bpm_str.split(',')[0])
    
    chart_pattern = r'#NOTES:\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^;]+);'
    
    for match in re.finditer(chart_pattern, content, re.MULTILINE | re.DOTALL):
        chart_type = match.group(1).strip()
        chart_difficulty = match.group(3).strip()
        notes_data = match.group(6).strip()
        
        if 'dance-single' not in chart_type:
            continue
        
        if difficulty.lower() not in chart_difficulty.lower():
            continue
        
        note_times = parse_note_times(notes_data, bpm, offset)
        return note_times, bpm, offset
    
    raise ValueError(f"No chart found")


def parse_note_times(notes_data, bpm, offset):
    """Extract note times."""
    note_times = []
    measures = notes_data.split(',')
    beat_duration = 60.0 / bpm
    current_beat = 0.0
    
    for measure in measures:
        lines = [line.strip() for line in measure.strip().split('\n') 
                 if line.strip() and len(line.strip()) == 4]
        
        if not lines:
            continue
        
        beats_per_line = 4.0 / len(lines)
        
        for i, line in enumerate(lines):
            if any(c != '0' for c in line):
                beat = current_beat + (i * beats_per_line)
                time_sec = offset + (beat * beat_duration)
                note_times.append(time_sec)
        
        current_beat += 4.0
    
    return np.array(note_times)


def compute_activity_signal(x, y, z, time_sec, target_rate=100):
    """Compute activity signal using multiple methods and combine."""
    # Resample
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_rate)
    new_time = np.linspace(0, duration, num_samples)
    
    x_r = np.interp(new_time, time_sec - time_sec[0], x)
    y_r = np.interp(new_time, time_sec - time_sec[0], y)
    z_r = np.interp(new_time, time_sec - time_sec[0], z)
    
    # Method 1: High-pass filtered magnitude
    b, a = butter(3, 0.5, btype='high', fs=target_rate)
    x_hp = filtfilt(b, a, x_r)
    y_hp = filtfilt(b, a, y_r)
    z_hp = filtfilt(b, a, z_r)
    mag_hp = np.sqrt(x_hp**2 + y_hp**2 + z_hp**2)
    
    # Method 2: Derivative (jerk) magnitude
    dt = 1.0 / target_rate
    dx = np.gradient(x_r, dt)
    dy = np.gradient(y_r, dt)
    dz = np.gradient(z_r, dt)
    jerk_mag = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Smooth jerk
    jerk_smooth = gaussian_filter1d(jerk_mag, sigma=target_rate*0.05)  # 50ms smoothing
    
    # Combine both signals
    # Normalize each
    mag_hp_norm = (mag_hp - np.mean(mag_hp)) / (np.std(mag_hp) + 1e-10)
    jerk_norm = (jerk_smooth - np.mean(jerk_smooth)) / (np.std(jerk_smooth) + 1e-10)
    
    # Average
    activity = (mag_hp_norm + jerk_norm) / 2.0
    
    return new_time, activity


def create_note_density_signal(note_times, duration, sample_rate, window_sec=1.0):
    """Create signal based on note density (notes per unit time)."""
    num_samples = int(duration * sample_rate)
    density = np.zeros(num_samples)
    
    # For each time point, count notes in a window
    time_array = np.linspace(0, duration, num_samples)
    window_samples = int(window_sec * sample_rate)
    
    for i, t in enumerate(time_array):
        # Count notes within window_sec of this time
        notes_in_window = np.sum((note_times >= t - window_sec/2) & (note_times < t + window_sec/2))
        density[i] = notes_in_window
    
    # Smooth
    density_smooth = gaussian_filter1d(density, sigma=sample_rate*0.2)  # 200ms smoothing
    
    return density_smooth


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    
    print("="*70)
    print("Note Density Correlation Approach")
    print("="*70)
    
    # Load
    print("\n[1/4] Loading data...")
    time_sec, x, y, z = load_sensor_data(capture_path)
    note_times, bpm, offset = parse_sm_file(sm_path, difficulty)
    print(f"  Duration: {time_sec[-1]-time_sec[0]:.1f}s, Notes: {len(note_times)}")
    
    # Compute activity
    print("\n[2/4] Computing activity signal...")
    target_rate = 100
    activity_time, activity = compute_activity_signal(x, y, z, time_sec, target_rate)
    
    # Create note density signal
    print("\n[3/4] Creating note density signal...")
    max_dur = max(activity_time[-1], note_times[-1] if len(note_times) > 0 else 0)
    density = create_note_density_signal(note_times, max_dur, target_rate, window_sec=2.0)
    
    # Trim to same length
    min_len = min(len(activity), len(density))
    activity = activity[:min_len]
    density = density[:min_len]
    
    # Correlate
    print("\n[4/4] Computing correlation...")
    activity_norm = (activity - np.mean(activity)) / (np.std(activity) + 1e-10)
    density_norm = (density - np.mean(density)) / (np.std(density) + 1e-10)
    
    corr = correlate(activity_norm, density_norm, mode='full')
    lags = np.arange(-len(density) + 1, len(activity))
    time_lags = lags / target_rate
    
    peak_idx = np.argmax(corr)
    peak_lag = time_lags[peak_idx]
    peak_val = corr[peak_idx]
    
    # Metrics
    window = int(2 * target_rate)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx - window):min(len(corr), peak_idx + window)] = False
    
    if np.any(mask):
        second_peak = np.max(corr[mask])
        peak_ratio = peak_val / second_peak if second_peak > 0 else np.inf
    else:
        peak_ratio = np.inf
    
    z_score = (peak_val - np.mean(corr)) / (np.std(corr) + 1e-10)
    
    print(f"\nResults:")
    print(f"  Peak lag: {peak_lag:.3f}s")
    print(f"  Peak ratio: {peak_ratio:.2f} (target: >2.0)")
    print(f"  Z-score: {z_score:.2f} (target: >5.0)")
    
    if peak_ratio > 2.0 and z_score > 5.0:
        print("\n✓ CLEAR CORRELATION PEAK!")
    else:
        print("\n⚠ Still weak")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(activity_time, activity, linewidth=0.6)
    axes[0].set_title('Activity Signal (combined magnitude + jerk)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)
    
    density_time = np.linspace(0, max_dur, len(density))
    axes[1].plot(density_time, density, linewidth=0.8)
    axes[1].set_title('Note Density from .sm (2s window)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_lags, corr, linewidth=0.8)
    axes[2].axvline(peak_lag, color='r', linestyle='--', linewidth=2)
    axes[2].set_title(f'Cross-correlation: ratio={peak_ratio:.2f}, z={z_score:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_xlim([-60, 60])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{capture_path.stem}_density.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    
    print("="*70)


if __name__ == '__main__':
    main()
