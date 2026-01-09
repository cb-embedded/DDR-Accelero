#!/usr/bin/env python3
"""
Exact replication of the experiment 03 approach but with proper metrics.

Use magnitude envelope as sensor signal.
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
    """Load sensor data and compute magnitude."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    time = df['seconds_elapsed'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return time, magnitude


def parse_sm(sm_path, difficulty):
    """Parse .sm file."""
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
    """Exact preprocessing from experiment 03."""
    # Resample
    duration = time_sec[-1] - time_sec[0]
    new_time = np.linspace(0, duration, int(duration * rate))
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC
    resampled = resampled - np.mean(resampled)
    
    # Bandpass filter (0.5-10 Hz)
    sos = signal.butter(4, [0.5, 10], btype='band', fs=rate, output='sos')
    filtered = signal.sosfiltfilt(sos, resampled)
    
    # Envelope
    envelope = np.abs(filtered)
    window_size = int(0.2 * rate)  # 200ms
    hann = signal.windows.hann(window_size)
    envelope = signal.convolve(envelope, hann/np.sum(hann), mode='same')
    
    return new_time, envelope


def create_reference(note_times, duration, rate=100):
    """Create reference signal like experiment 03."""
    num_samples = int(duration * rate)
    ref = np.zeros(num_samples)
    
    # Place spike at each note
    for t in note_times:
        idx = int(t * rate)
        if 0 <= idx < num_samples:
            ref[idx] = 1.0
    
    # Smooth with Gaussian
    window_size = int(0.1 * rate)  # 100ms
    kernel = signal.windows.gaussian(window_size, window_size/6)
    ref = signal.convolve(ref, kernel, mode='same')
    
    return ref


def correlate_full(sensor, reference, rate):
    """Compute correlation with metrics."""
    # Normalize
    s1 = (sensor - np.mean(sensor)) / (np.std(sensor) + 1e-10)
    s2 = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)
    
    # Correlate
    corr = correlate(s1, s2, mode='full')
    lags = np.arange(-len(s2)+1, len(s1)) / rate
    
    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = lags[peak_idx]
    peak_val = corr[peak_idx]
    
    # Metrics
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
        print("Usage: python align_replication.py <capture.zip> <song.sm> <difficulty>")
        sys.exit(1)
    
    cap = Path(sys.argv[1])
    sm = Path(sys.argv[2])
    diff = sys.argv[3]
    
    print("="*70)
    print("REPLICATION OF EXPERIMENT 03 APPROACH")
    print("="*70)
    
    # Load
    print("\n[1/4] Loading...")
    time, mag = load_data(cap)
    notes, bpm, ofs = parse_sm(sm, diff)
    print(f"  Sensor: {len(time)} samples, {time[-1]-time[0]:.1f}s")
    print(f"  Notes: {len(notes)}, BPM={bpm}")
    
    # Process
    print("\n[2/4] Processing sensor (magnitude envelope)...")
    rate = 100
    sensor_time, sensor_envelope = preprocess_sensor(time, mag, rate)
    print(f"  Envelope: {len(sensor_envelope)} samples")
    
    # Reference
    print("\n[3/4] Creating reference from notes...")
    duration = max(sensor_time[-1], notes[-1] if len(notes) > 0 else 0)
    ref = create_reference(notes, duration, rate)
    print(f"  Reference: {len(ref)} samples")
    
    # Match lengths
    min_len = min(len(sensor_envelope), len(ref))
    sensor_envelope = sensor_envelope[:min_len]
    ref = ref[:min_len]
    
    # Correlate
    print("\n[4/4] Correlating...")
    lags, corr, peak_lag, ratio, z = correlate_full(sensor_envelope, ref, rate)
    
    print(f"\nRESULTS:")
    print(f"  Peak lag: {peak_lag:.3f}s")
    print(f"  Peak ratio: {ratio:.2f} {'✓ GOOD' if ratio > 2.0 else '⚠ WEAK'}")
    print(f"  Z-score: {z:.2f} {'✓ GOOD' if z > 5.0 else '⚠ WEAK'}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    axes[0].plot(sensor_time, sensor_envelope, linewidth=0.7)
    axes[0].set_title('Sensor: Magnitude Envelope (filtered 0.5-10Hz, absolute, smoothed)')
    axes[0].set_xlabel('Time (s)')
    axes[0].grid(True, alpha=0.3)
    
    ref_time = np.linspace(0, duration, len(ref))
    axes[1].plot(ref_time, ref, linewidth=0.8)
    axes[1].set_title('Reference from .sm (Gaussian-smoothed impulses)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lags, corr, linewidth=0.8)
    axes[2].axvline(peak_lag, color='r', linestyle='--', linewidth=2)
    axes[2].set_title(f'Cross-correlation: Peak ratio={ratio:.2f}, Z-score={z:.2f}')
    axes[2].set_xlabel('Lag (s)')
    axes[2].set_xlim([-60, 60])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'{cap.stem}_REPLICATION.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_file}")
    
    if ratio > 2.0 and z > 5.0:
        print("\n✓✓✓ CLEAR PEAK FOUND!")
    else:
        print("\n⚠⚠⚠ Weak correlation persists with this approach too")
    
    print("="*70)


if __name__ == '__main__':
    main()
