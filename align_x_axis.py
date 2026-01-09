#!/usr/bin/env python3
"""
Minimal alignment script: raw x-axis vs .sm expected signal using cross-correlation.

Phase 1: Evidence-driven, step-by-step approach.
Goal: Achieve unambiguous alignment with a dominant correlation peak.

Usage:
    python align_x_axis.py <capture_zip> <sm_file> <difficulty>

Example:
    python align_x_axis.py raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip "sm_files/Lucky Orb.sm" Medium
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate


def load_raw_x_axis(zip_path):
    """Load raw x-axis accelerometer data from capture zip."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if 'Gravity.csv' not in zip_ref.namelist():
            raise ValueError("No Gravity.csv found in zip file")
        
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    # Extract time (seconds_elapsed) and x-axis (column 4)
    time_sec = df['seconds_elapsed'].values
    x_raw = df['x'].values
    
    return time_sec, x_raw


def parse_sm_notes(sm_path, difficulty):
    """Parse .sm file and extract note timestamps for specified difficulty."""
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


def create_expected_signal(note_times, duration, sample_rate):
    """Create expected signal from note times as impulse train."""
    num_samples = int(duration * sample_rate)
    expected = np.zeros(num_samples)
    
    for t in note_times:
        idx = int(t * sample_rate)
        if 0 <= idx < num_samples:
            expected[idx] = 1.0
    
    return expected


def resample_signal(time_sec, signal, target_rate):
    """Resample signal to target sample rate."""
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], signal)
    return new_time, resampled


def compute_correlation(x_signal, expected_signal, sample_rate):
    """Compute cross-correlation and find dominant peak."""
    # Minimal normalization: remove mean
    x_norm = x_signal - np.mean(x_signal)
    exp_norm = expected_signal - np.mean(expected_signal)
    
    # Normalize by standard deviation for correlation coefficient
    x_norm = x_norm / (np.std(x_norm) + 1e-10)
    exp_norm = exp_norm / (np.std(exp_norm) + 1e-10)
    
    # Cross-correlation
    corr = correlate(x_norm, exp_norm, mode='full')
    lags = np.arange(-len(exp_norm) + 1, len(x_norm))
    time_lags = lags / sample_rate
    
    # Find peak
    peak_idx = np.argmax(corr)
    peak_lag = time_lags[peak_idx]
    peak_value = corr[peak_idx]
    
    # Compute dominance metric: ratio of highest to second highest peak
    # Exclude window around main peak
    window_size = int(2.0 * sample_rate)  # 2 second exclusion
    mask = np.ones(len(corr), dtype=bool)
    mask[max(0, peak_idx - window_size):min(len(corr), peak_idx + window_size)] = False
    
    if np.any(mask):
        second_peak_value = np.max(corr[mask])
        peak_ratio = peak_value / second_peak_value if second_peak_value > 0 else np.inf
    else:
        peak_ratio = np.inf
    
    # Z-score of peak
    z_score = (peak_value - np.mean(corr)) / (np.std(corr) + 1e-10)
    
    return time_lags, corr, peak_lag, peak_ratio, z_score


def save_plot(fig, filename, output_dir='artifacts'):
    """Save figure to artifacts directory."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_raw_x(time_sec, x_raw, title):
    """Plot raw x-axis signal."""
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(time_sec, x_raw, linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X acceleration (m/s²)')
    ax.set_title(f'Raw X-axis Signal: {title}')
    ax.grid(True, alpha=0.3)
    return fig


def plot_expected_signal(time_sec, expected, note_times, title):
    """Plot expected signal from .sm notes."""
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(time_sec, expected, linewidth=1)
    ax.scatter(note_times, np.ones_like(note_times) * np.max(expected) * 0.5, 
               c='red', s=10, alpha=0.5, label='Note times')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Expected signal (a.u.)')
    ax.set_title(f'Expected Signal from .sm: {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_correlation(time_lags, corr, peak_lag, peak_ratio, z_score, title):
    """Plot cross-correlation curve with peak marked."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_lags, corr, linewidth=1)
    ax.axvline(peak_lag, color='r', linestyle='--', linewidth=2, 
               label=f'Peak at {peak_lag:.3f}s')
    ax.set_xlabel('Time lag (s)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(f'Cross-correlation: {title}\nPeak ratio: {peak_ratio:.2f}, Z-score: {z_score:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-30, 30])  # Focus on reasonable range
    return fig


def plot_alignment_overlay(x_time, x_signal, exp_time, exp_signal, offset, title, window_start=20, window_dur=40):
    """Plot aligned signals overlay."""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Normalize for comparison
    x_norm = (x_signal - np.mean(x_signal)) / (np.std(x_signal) + 1e-10)
    exp_norm = (exp_signal - np.mean(exp_signal)) / (np.std(exp_signal) + 1e-10)
    
    # Shift expected signal by detected offset
    exp_time_shifted = exp_time + offset
    
    # Plot in window
    ax.plot(x_time, x_norm, linewidth=0.8, alpha=0.7, label='Raw X (normalized)')
    ax.plot(exp_time_shifted, exp_norm, linewidth=1.5, alpha=0.8, label='Expected (shifted)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized signal')
    ax.set_title(f'Alignment Overlay: {title}\nOffset: {offset:.3f}s')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([window_start, window_start + window_dur])
    return fig


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    
    print("="*70)
    print("DDR-Accelero: Minimal X-axis Alignment (Phase 1)")
    print("="*70)
    print(f"\nCapture: {capture_path.name}")
    print(f"Song: {sm_path.name}")
    print(f"Difficulty: {difficulty}\n")
    
    # Step 1: Load raw x-axis
    print("[1/6] Loading raw x-axis data...")
    time_sec, x_raw = load_raw_x_axis(capture_path)
    duration = time_sec[-1] - time_sec[0]
    sample_rate = len(time_sec) / duration
    print(f"  Duration: {duration:.2f}s")
    print(f"  Samples: {len(x_raw)}")
    print(f"  Sample rate: {sample_rate:.1f} Hz")
    
    # Step 2: Parse .sm file
    print("\n[2/6] Parsing .sm file...")
    note_times, bpm, offset = parse_sm_notes(sm_path, difficulty)
    print(f"  BPM: {bpm}")
    print(f"  Offset: {offset}s")
    print(f"  Total notes: {len(note_times)}")
    if len(note_times) > 0:
        print(f"  First note: {note_times[0]:.3f}s")
        print(f"  Last note: {note_times[-1]:.3f}s")
    
    # Step 3: Resample x-axis to consistent rate
    print("\n[3/6] Resampling to 100 Hz...")
    target_rate = 100.0
    x_time, x_resampled = resample_signal(time_sec, x_raw, target_rate)
    
    # Step 4: Create expected signal
    print("\n[4/6] Creating expected signal...")
    max_duration = max(x_time[-1], note_times[-1] if len(note_times) > 0 else 0)
    expected = create_expected_signal(note_times, max_duration, target_rate)
    exp_time = np.linspace(0, max_duration, len(expected))
    
    # Step 5: Cross-correlation
    print("\n[5/6] Computing cross-correlation...")
    time_lags, corr, peak_lag, peak_ratio, z_score = compute_correlation(
        x_resampled, expected, target_rate)
    
    print(f"  Detected offset: {peak_lag:.3f}s")
    print(f"  Peak ratio: {peak_ratio:.2f}")
    print(f"  Z-score: {z_score:.2f}")
    
    # Evaluate dominance
    if peak_ratio > 2.0 and z_score > 5.0:
        print("  ✓ DOMINANT PEAK DETECTED (unambiguous alignment)")
    else:
        print("  ⚠ WEAK PEAK (alignment may be ambiguous)")
    
    # Step 6: Generate PNG artifacts
    print("\n[6/6] Generating PNG artifacts...")
    base_name = capture_path.stem
    
    fig1 = plot_raw_x(time_sec, x_raw, base_name)
    save_plot(fig1, f'{base_name}_raw_x.png')
    
    fig2 = plot_expected_signal(exp_time, expected, note_times, base_name)
    save_plot(fig2, f'{base_name}_expected.png')
    
    fig3 = plot_correlation(time_lags, corr, peak_lag, peak_ratio, z_score, base_name)
    save_plot(fig3, f'{base_name}_correlation.png')
    
    fig4 = plot_alignment_overlay(x_time, x_resampled, exp_time, expected, 
                                   peak_lag, base_name)
    save_plot(fig4, f'{base_name}_alignment.png')
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Capture duration: {duration:.2f}s")
    print(f"Chart duration: {note_times[-1]:.2f}s (last note)")
    print(f"Sample rate: {sample_rate:.1f} Hz (original) → 100 Hz (resampled)")
    print(f"Total notes: {len(note_times)}")
    print(f"\nAlignment result:")
    print(f"  Offset: {peak_lag:.3f}s")
    print(f"  Peak dominance ratio: {peak_ratio:.2f}")
    print(f"  Peak z-score: {z_score:.2f}")
    print(f"\nInterpretation:")
    print(f"  The first note ({note_times[0]:.3f}s in .sm file)")
    print(f"  occurs at {peak_lag + note_times[0]:.3f}s in the sensor recording")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
