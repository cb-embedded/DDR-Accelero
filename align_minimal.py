#!/usr/bin/env python3
"""
Minimal alignment: Generate intermediate signal from .sm arrows and correlate with sensor data.

Focus: Get clear correlation peak by properly modeling the expected signal.

Usage:
    python align_minimal.py <capture_zip> <sm_file> <difficulty>

Example:
    python align_minimal.py "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" Medium
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, butter, filtfilt


def load_sensor_data(zip_path):
    """Load accelerometer data from zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if 'Gravity.csv' not in zip_ref.namelist():
            raise ValueError("No Gravity.csv found in zip file")
        
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    time_sec = df['seconds_elapsed'].values
    # Try all axes - let's see which works best
    axes = {
        'x': df['x'].values,
        'y': df['y'].values,
        'z': df['z'].values,
        'magnitude': np.sqrt(df['x'].values**2 + df['y'].values**2 + df['z'].values**2)
    }
    
    return time_sec, axes


def parse_sm_file(sm_path, difficulty):
    """Parse .sm file and extract note timestamps."""
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract metadata
    offset_match = re.search(r'#OFFSET:([^;]+);', content)
    bpm_match = re.search(r'#BPMS:([^;]+);', content)
    
    offset = float(offset_match.group(1)) if offset_match else 0.0
    
    # Parse BPM (take first value if multiple)
    bpm = 120.0
    if bpm_match:
        bpm_str = bpm_match.group(1).strip()
        if '=' in bpm_str:
            bpm = float(bpm_str.split('=')[1].split(',')[0])
        else:
            bpm = float(bpm_str.split(',')[0])
    
    # Find chart with matching difficulty
    chart_pattern = r'#NOTES:\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^;]+);'
    
    for match in re.finditer(chart_pattern, content, re.MULTILINE | re.DOTALL):
        chart_type = match.group(1).strip()
        chart_difficulty = match.group(3).strip()
        notes_data = match.group(6).strip()
        
        if 'dance-single' not in chart_type:
            continue
        
        if difficulty.lower() not in chart_difficulty.lower():
            continue
        
        # Parse notes
        note_times = parse_note_times(notes_data, bpm, offset)
        return note_times, bpm, offset
    
    raise ValueError(f"No chart found for difficulty: {difficulty}")


def parse_note_times(notes_data, bpm, offset):
    """Extract note timestamps from chart data."""
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


def create_intermediate_signal(note_times, duration, sample_rate):
    """Create intermediate signal from note times.
    
    Model: Each note causes a brief impulse response in acceleration.
    Use a simple gaussian pulse to represent each note.
    """
    num_samples = int(duration * sample_rate)
    signal = np.zeros(num_samples)
    
    # Create a gaussian kernel for each note
    # Width should be short (~100-200ms) to represent brief foot impact
    sigma = 0.05 * sample_rate  # 50ms standard deviation
    kernel_half_width = int(3 * sigma)  # 3-sigma on each side
    t_kernel = np.arange(-kernel_half_width, kernel_half_width + 1)
    kernel = np.exp(-0.5 * (t_kernel / sigma) ** 2)
    kernel = kernel / np.sum(kernel)  # Normalize to sum to 1
    
    # Place kernel at each note time
    for note_time in note_times:
        idx = int(note_time * sample_rate)
        
        # Add kernel centered at this note
        start_idx = idx - kernel_half_width
        end_idx = idx + kernel_half_width + 1
        
        # Handle boundaries
        k_start = max(0, -start_idx)
        k_end = len(kernel) - max(0, end_idx - num_samples)
        s_start = max(0, start_idx)
        s_end = min(num_samples, end_idx)
        
        if s_start < s_end and k_start < k_end:
            signal[s_start:s_end] += kernel[k_start:k_end]
    
    return signal


def preprocess_sensor_signal(time_sec, signal_raw, target_rate=100):
    """Preprocess sensor signal: resample and minimal filtering."""
    # Resample to consistent rate
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], signal_raw)
    
    # Remove DC offset (mean)
    resampled = resampled - np.mean(resampled)
    
    # Optional: light bandpass filter to remove very low and very high frequencies
    # Keep movement frequencies (0.5 Hz to 20 Hz)
    b, a = butter(2, [0.5, 20], btype='band', fs=target_rate)
    filtered = filtfilt(b, a, resampled)
    
    return new_time, filtered


def compute_correlation_full(sensor_signal, expected_signal, sample_rate):
    """Compute cross-correlation over full signal length."""
    # Normalize both signals
    sensor_norm = (sensor_signal - np.mean(sensor_signal)) / (np.std(sensor_signal) + 1e-10)
    expected_norm = (expected_signal - np.mean(expected_signal)) / (np.std(expected_signal) + 1e-10)
    
    # Full cross-correlation
    corr = correlate(sensor_norm, expected_norm, mode='full')
    
    # Calculate lags in time
    lags = np.arange(-len(expected_norm) + 1, len(sensor_norm))
    time_lags = lags / sample_rate
    
    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = time_lags[peak_idx]
    peak_value = corr[peak_idx]
    
    # Calculate dominance metrics
    # Peak ratio: highest vs second highest (excluding window around peak)
    window = int(2 * sample_rate)  # 2 second exclusion
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx - window):min(len(corr), peak_idx + window)] = False
    
    if np.any(mask):
        second_peak = np.max(corr[mask])
        peak_ratio = peak_value / second_peak if second_peak > 0 else np.inf
    else:
        peak_ratio = np.inf
    
    # Z-score
    z_score = (peak_value - np.mean(corr)) / (np.std(corr) + 1e-10)
    
    return time_lags, corr, peak_lag, peak_ratio, z_score


def plot_results(time_sec, sensor_raw, sensor_time, sensor_processed, 
                 expected_time, expected, time_lags, corr, 
                 peak_lag, peak_ratio, z_score, axis_name, base_name):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Raw sensor signal
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(time_sec, sensor_raw, linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'{axis_name} raw (m/s²)')
    ax1.set_title(f'Raw Sensor Signal: {axis_name}-axis')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Processed sensor signal
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(sensor_time, sensor_processed, linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'{axis_name} processed')
    ax2.set_title(f'Processed Sensor Signal (filtered, normalized)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Expected intermediate signal
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(expected_time, expected, linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Expected signal')
    ax3.set_title(f'Expected Intermediate Signal from .sm arrows')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cross-correlation (full)
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(time_lags, corr, linewidth=0.8)
    ax4.axvline(peak_lag, color='r', linestyle='--', linewidth=2, 
                label=f'Peak: {peak_lag:.2f}s')
    ax4.set_xlabel('Time lag (s)')
    ax4.set_ylabel('Cross-correlation')
    ax4.set_title(f'Cross-correlation (full): Peak ratio={peak_ratio:.2f}, Z-score={z_score:.2f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Zoom into reasonable range for clarity
    reasonable_range = 60  # Show +/- 60 seconds
    ax4.set_xlim([-reasonable_range, reasonable_range])
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{base_name}_minimal_{axis_name}.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    
    print("="*70)
    print("Minimal Alignment: Intermediate Signal from Arrows")
    print("="*70)
    print(f"\nCapture: {capture_path.name}")
    print(f"Song: {sm_path.name}")
    print(f"Difficulty: {difficulty}\n")
    
    # Load sensor data
    print("[1/5] Loading sensor data...")
    time_sec, axes_data = load_sensor_data(capture_path)
    duration = time_sec[-1] - time_sec[0]
    sample_rate = len(time_sec) / duration
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sample_rate:.1f} Hz")
    
    # Parse .sm file
    print("\n[2/5] Parsing .sm file...")
    note_times, bpm, offset = parse_sm_file(sm_path, difficulty)
    print(f"  BPM: {bpm}")
    print(f"  Offset: {offset}s")
    print(f"  Total notes: {len(note_times)}")
    print(f"  First note: {note_times[0]:.3f}s")
    print(f"  Last note: {note_times[-1]:.3f}s")
    
    # Create intermediate signal
    print("\n[3/5] Creating intermediate signal from arrows...")
    target_rate = 100.0
    max_duration = max(duration, note_times[-1])
    expected = create_intermediate_signal(note_times, max_duration, target_rate)
    expected_time = np.linspace(0, max_duration, len(expected))
    print(f"  Signal length: {len(expected)} samples at {target_rate} Hz")
    
    # Test all axes
    print("\n[4/5] Testing correlation for each axis...")
    print(f"\n{'Axis':<12} {'Peak Lag':<12} {'Peak Ratio':<12} {'Z-score':<12} {'Status'}")
    print("-" * 70)
    
    results = {}
    for axis_name in ['x', 'y', 'z', 'magnitude']:
        # Preprocess sensor signal
        sensor_time, sensor_processed = preprocess_sensor_signal(
            time_sec, axes_data[axis_name], target_rate)
        
        # Make sure both signals have same length for fair comparison
        min_len = min(len(sensor_processed), len(expected))
        sensor_processed = sensor_processed[:min_len]
        expected_trimmed = expected[:min_len]
        
        # Compute correlation
        time_lags, corr, peak_lag, peak_ratio, z_score = compute_correlation_full(
            sensor_processed, expected_trimmed, target_rate)
        
        status = "✓ GOOD" if peak_ratio > 2.0 and z_score > 5.0 else "⚠ WEAK"
        
        results[axis_name] = {
            'sensor_processed': sensor_processed,
            'sensor_time': sensor_time,
            'peak_lag': peak_lag,
            'peak_ratio': peak_ratio,
            'z_score': z_score,
            'corr': corr,
            'time_lags': time_lags
        }
        
        print(f"{axis_name:<12} {peak_lag:<12.2f} {peak_ratio:<12.2f} {z_score:<12.2f} {status}")
    
    # Find best axis
    best_axis = max(results.keys(), key=lambda k: results[k]['peak_ratio'])
    best = results[best_axis]
    
    print("\n" + "="*70)
    print("BEST RESULT")
    print("="*70)
    print(f"Axis: {best_axis}")
    print(f"Peak lag: {best['peak_lag']:.3f}s")
    print(f"Peak ratio: {best['peak_ratio']:.2f} (target: >2.0)")
    print(f"Z-score: {best['z_score']:.2f} (target: >5.0)")
    
    if best['peak_ratio'] > 2.0 and best['z_score'] > 5.0:
        print("\n✓ CLEAR CORRELATION PEAK FOUND!")
        print(f"  First note ({note_times[0]:.2f}s in .sm)")
        print(f"  occurs at {best['peak_lag'] + note_times[0]:.2f}s in recording")
    else:
        print("\n⚠ Peak is still weak, may need further investigation")
    
    # Generate visualizations
    print("\n[5/5] Generating visualization...")
    base_name = capture_path.stem
    
    for axis_name in ['x', 'y', 'z', 'magnitude']:
        result = results[axis_name]
        plot_results(
            time_sec, axes_data[axis_name],
            result['sensor_time'], result['sensor_processed'],
            expected_time, expected,
            result['time_lags'], result['corr'],
            result['peak_lag'], result['peak_ratio'], result['z_score'],
            axis_name, base_name
        )
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
