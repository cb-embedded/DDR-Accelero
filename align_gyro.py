#!/usr/bin/env python3
"""
Try gyroscope data instead of accelerometer.
Gyroscope might show clearer patterns for DDR movements (rotation of phone/body).
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


def load_gyroscope_data(zip_path):
    """Load gyroscope data."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('Gyroscope.csv') as f:
            df = pd.read_csv(f)
    
    time_sec = df['seconds_elapsed'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return time_sec, magnitude


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


def process_gyro_signal(magnitude, time_sec, target_rate=100):
    """Process gyroscope magnitude."""
    # Resample
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC and apply bandpass to focus on movement frequencies
    b, a = butter(3, [0.5, 15], btype='band', fs=target_rate)
    filtered = filtfilt(b, a, resampled)
    
    # Take absolute value and smooth (envelope)
    envelope = np.abs(filtered)
    envelope = gaussian_filter1d(envelope, sigma=target_rate*0.05)
    
    return new_time, envelope


def create_expected_signal(note_times, duration, sample_rate):
    """Create expected signal."""
    num_samples = int(duration * sample_rate)
    signal = np.zeros(num_samples)
    
    # Gaussian pulse for each note
    sigma = 0.06 * sample_rate  # 60ms
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


def compute_correlation(s1, s2, sample_rate):
    """Compute correlation."""
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
    
    corr = correlate(s1_norm, s2_norm, mode='full')
    lags = np.arange(-len(s2) + 1, len(s1))
    time_lags = lags / sample_rate
    
    peak_idx = np.argmax(corr)
    peak_lag = time_lags[peak_idx]
    peak_val = corr[peak_idx]
    
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
    print("Gyroscope-based Alignment")
    print("="*70)
    
    # Load
    print("\n[1/4] Loading gyroscope data...")
    time_sec, gyro_mag = load_gyroscope_data(capture_path)
    print(f"  Gyroscope: {len(gyro_mag)} samples, {time_sec[-1]-time_sec[0]:.1f}s")
    
    # Parse
    print("\n[2/4] Parsing .sm...")
    note_times, bpm, offset = parse_sm_file(sm_path, difficulty)
    print(f"  Notes: {len(note_times)}, BPM: {bpm}")
    
    # Process
    print("\n[3/4] Processing signal...")
    target_rate = 100
    gyro_time, gyro_processed = process_gyro_signal(gyro_mag, time_sec, target_rate)
    
    max_dur = max(gyro_time[-1], note_times[-1] if len(note_times) > 0 else 0)
    expected = create_expected_signal(note_times, max_dur, target_rate)
    
    min_len = min(len(gyro_processed), len(expected))
    gyro_processed = gyro_processed[:min_len]
    expected = expected[:min_len]
    
    # Correlate
    print("\n[4/4] Computing correlation...")
    time_lags, corr, peak_lag, peak_ratio, z_score = compute_correlation(
        gyro_processed, expected, target_rate)
    
    print(f"\nResults:")
    print(f"  Peak lag: {peak_lag:.3f}s")
    print(f"  Peak ratio: {peak_ratio:.2f} (target: >2.0)")
    print(f"  Z-score: {z_score:.2f} (target: >5.0)")
    
    if peak_ratio > 2.0 and z_score > 5.0:
        print("\n✓ CLEAR PEAK!")
    else:
        print("\n⚠ Weak")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(gyro_time, gyro_processed, linewidth=0.6)
    axes[0].set_title('Gyroscope envelope (filtered, absolute value)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)
    
    exp_time = np.linspace(0, max_dur, len(expected))
    axes[1].plot(exp_time, expected, linewidth=0.8)
    axes[1].set_title('Expected from .sm')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_lags, corr, linewidth=0.8)
    axes[2].axvline(peak_lag, color='r', linestyle='--', linewidth=2)
    axes[2].set_title(f'Correlation: ratio={peak_ratio:.2f}, z={z_score:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_xlim([-60, 60])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{capture_path.stem}_gyro.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    
    print("="*70)


if __name__ == '__main__':
    main()
