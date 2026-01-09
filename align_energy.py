#!/usr/bin/env python3
"""
Improved alignment using signal energy/envelope detection.

Key insight: DDR movements create bursts of activity in accelerometer.
Use envelope/energy of signal variation rather than raw values.

Usage:
    python align_energy.py <capture_zip> <sm_file> <difficulty>
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt, hilbert


def load_sensor_data(zip_path):
    """Load accelerometer data."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    time_sec = df['seconds_elapsed'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    
    return time_sec, x, y, z


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
    
    raise ValueError(f"No chart found for difficulty: {difficulty}")


def parse_note_times(notes_data, bpm, offset):
    """Extract note timestamps."""
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


def compute_movement_energy(x, y, z, time_sec, target_rate=100):
    """Compute energy/envelope of movement.
    
    Key: Remove gravity baseline and look at variations = actual movements.
    """
    # Resample
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_rate)
    new_time = np.linspace(0, duration, num_samples)
    
    x_resamp = np.interp(new_time, time_sec - time_sec[0], x)
    y_resamp = np.interp(new_time, time_sec - time_sec[0], y)
    z_resamp = np.interp(new_time, time_sec - time_sec[0], z)
    
    # High-pass filter to remove DC/gravity (keep only dynamic movements)
    b, a = butter(3, 0.5, btype='high', fs=target_rate)
    x_filt = filtfilt(b, a, x_resamp)
    y_filt = filtfilt(b, a, y_resamp)
    z_filt = filtfilt(b, a, z_resamp)
    
    # Compute magnitude of movement vector
    movement_mag = np.sqrt(x_filt**2 + y_filt**2 + z_filt**2)
    
    # Compute envelope (absolute value smoothed)
    envelope = np.abs(movement_mag)
    
    # Smooth envelope with moving average
    window = int(0.1 * target_rate)  # 100ms smoothing
    if window > 0:
        envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
    
    return new_time, envelope


def create_expected_signal(note_times, duration, sample_rate):
    """Create expected signal - pulse at each note."""
    num_samples = int(duration * sample_rate)
    signal = np.zeros(num_samples)
    
    # Gaussian pulse for each note
    sigma = 0.08 * sample_rate  # 80ms width
    half_width = int(3 * sigma)
    t_kernel = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (t_kernel / sigma) ** 2)
    
    for note_time in note_times:
        idx = int(note_time * sample_rate)
        start = max(0, idx - half_width)
        end = min(num_samples, idx + half_width + 1)
        k_start = max(0, half_width - idx)
        k_end = k_start + (end - start)
        
        if start < end:
            signal[start:end] += kernel[k_start:k_end]
    
    return signal


def compute_correlation(signal1, signal2, sample_rate):
    """Compute cross-correlation."""
    # Normalize
    s1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    s2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
    
    # Correlate
    corr = correlate(s1, s2, mode='full')
    lags = np.arange(-len(s2) + 1, len(s1))
    time_lags = lags / sample_rate
    
    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = time_lags[peak_idx]
    peak_val = corr[peak_idx]
    
    # Metrics
    window = int(2 * sample_rate)
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx - window):min(len(corr), peak_idx + window)] = False
    
    if np.any(mask):
        second_peak = np.max(corr[mask])
        peak_ratio = peak_val / second_peak if second_peak > 0 else np.inf
    else:
        peak_ratio = np.inf
    
    z_score = (peak_val - np.mean(corr)) / (np.std(corr) + 1e-10)
    
    return time_lags, corr, peak_lag, peak_ratio, z_score


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    
    print("="*70)
    print("Energy-based Alignment")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading sensor data...")
    time_sec, x, y, z = load_sensor_data(capture_path)
    print(f"  Duration: {time_sec[-1] - time_sec[0]:.2f}s")
    
    # Parse .sm
    print("\n[2/4] Parsing .sm file...")
    note_times, bpm, offset = parse_sm_file(sm_path, difficulty)
    print(f"  Notes: {len(note_times)}, BPM: {bpm}")
    
    # Compute movement energy
    print("\n[3/4] Computing movement energy...")
    target_rate = 100
    energy_time, energy = compute_movement_energy(x, y, z, time_sec, target_rate)
    print(f"  Energy signal: {len(energy)} samples")
    
    # Create expected
    max_dur = max(energy_time[-1], note_times[-1] if len(note_times) > 0 else 0)
    expected = create_expected_signal(note_times, max_dur, target_rate)
    
    # Trim to same length
    min_len = min(len(energy), len(expected))
    energy = energy[:min_len]
    expected = expected[:min_len]
    
    # Correlate
    print("\n[4/4] Computing correlation...")
    time_lags, corr, peak_lag, peak_ratio, z_score = compute_correlation(
        energy, expected, target_rate)
    
    print(f"\nResults:")
    print(f"  Peak lag: {peak_lag:.3f}s")
    print(f"  Peak ratio: {peak_ratio:.2f} (target: >2.0)")
    print(f"  Z-score: {z_score:.2f} (target: >5.0)")
    
    if peak_ratio > 2.0 and z_score > 5.0:
        print("\n✓ CLEAR PEAK FOUND!")
    else:
        print("\n⚠ Peak is still weak")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(energy_time, energy, linewidth=0.8)
    axes[0].set_title('Movement Energy (envelope of high-pass filtered acceleration)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)
    
    exp_time = np.linspace(0, max_dur, len(expected))
    axes[1].plot(exp_time, expected, linewidth=0.8)
    axes[1].set_title('Expected Signal from .sm arrows')
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
    output_file = output_dir / f'{capture_path.stem}_energy.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    print("="*70)


if __name__ == '__main__':
    main()
