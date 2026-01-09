#!/usr/bin/env python3
"""
Visual verification of alignment: Inspect if detected offset makes sense.

This script creates detailed diagnostic plots to manually verify if the
detected alignment offset is approximately correct by visual inspection.

Usage:
    python verify_alignment_visual.py <capture_zip> <sm_file> <difficulty> <axis>

Example:
    python verify_alignment_visual.py raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip "sm_files/Lucky Orb.sm" Medium y
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import functions from align_x_axis.py
import importlib.util
spec = importlib.util.spec_from_file_location("align_module", 
                                               Path(__file__).parent / "align_x_axis.py")
align_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_module)

parse_sm_notes = align_module.parse_sm_notes
create_expected_signal = align_module.create_expected_signal
resample_signal = align_module.resample_signal
compute_correlation = align_module.compute_correlation


def load_axis(zip_path, axis_name):
    """Load specific axis from capture zip."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if 'Gravity.csv' not in zip_ref.namelist():
            raise ValueError("No Gravity.csv found in zip file")
        
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    time_sec = df['seconds_elapsed'].values
    
    if axis_name == 'magnitude':
        x, y, z = df['x'].values, df['y'].values, df['z'].values
        signal = np.sqrt(x**2 + y**2 + z**2)
    else:
        signal = df[axis_name].values
    
    return time_sec, signal


def find_high_density_regions(note_times, window_sec=10, top_n=3):
    """Find time windows with highest note density."""
    if len(note_times) == 0:
        return []
    
    max_time = note_times[-1]
    densities = []
    
    # Slide window through song
    for t in np.arange(0, max_time - window_sec, 1.0):
        notes_in_window = np.sum((note_times >= t) & (note_times < t + window_sec))
        densities.append((t, notes_in_window))
    
    # Sort by density and get top regions
    densities.sort(key=lambda x: x[1], reverse=True)
    return densities[:top_n]


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    axis_name = sys.argv[4]
    
    print("="*70)
    print("Visual Alignment Verification")
    print("="*70)
    print(f"\nCapture: {capture_path.name}")
    print(f"Song: {sm_path.name}")
    print(f"Difficulty: {difficulty}")
    print(f"Axis: {axis_name}\n")
    
    # Load data
    print("[1/4] Loading sensor data...")
    time_sec, signal_raw = load_axis(capture_path, axis_name)
    duration = time_sec[-1] - time_sec[0]
    print(f"  Duration: {duration:.2f}s")
    
    # Parse .sm
    print("\n[2/4] Parsing .sm file...")
    note_times, bpm, offset = parse_sm_notes(sm_path, difficulty)
    print(f"  BPM: {bpm}")
    print(f"  Total notes: {len(note_times)}")
    print(f"  First note: {note_times[0]:.3f}s")
    print(f"  Last note: {note_times[-1]:.3f}s")
    
    # Find high-density regions
    high_density = find_high_density_regions(note_times, window_sec=10, top_n=3)
    print(f"\n  High-density regions (10s windows):")
    for t, count in high_density:
        print(f"    {t:.1f}s - {t+10:.1f}s: {count} notes")
    
    # Compute alignment
    print("\n[3/4] Computing alignment...")
    target_rate = 100.0
    _, signal_resampled = resample_signal(time_sec, signal_raw, target_rate)
    signal_time = np.linspace(0, duration, len(signal_resampled))
    
    max_duration = max(duration, note_times[-1] if len(note_times) > 0 else 0)
    expected = create_expected_signal(note_times, max_duration, target_rate, 'bipolar')
    
    _, _, peak_lag, peak_ratio, z_score = compute_correlation(
        signal_resampled, expected, target_rate)
    
    print(f"  Detected offset: {peak_lag:.3f}s")
    print(f"  Peak ratio: {peak_ratio:.2f}")
    print(f"  First note predicted at: {peak_lag + note_times[0]:.3f}s in recording")
    
    # Create diagnostic figure
    print("\n[4/4] Creating diagnostic visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full raw signal with key times marked
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_sec, signal_raw, linewidth=0.5, alpha=0.7, label=f'{axis_name} axis')
    
    # Mark first note prediction
    first_note_time = peak_lag + note_times[0]
    ax1.axvline(first_note_time, color='red', linestyle='--', linewidth=2, 
                label=f'First note (predicted: {first_note_time:.1f}s)')
    
    # Mark high-density regions
    for i, (t, count) in enumerate(high_density[:3]):
        region_time = peak_lag + t
        ax1.axvspan(region_time, region_time + 10, alpha=0.2, color='orange',
                    label=f'High density region {i+1}' if i == 0 else '')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'{axis_name} (m/s²)')
    ax1.set_title(f'Full Recording: {axis_name}-axis with Predicted Note Regions')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom around first note
    ax2 = fig.add_subplot(gs[1, 0])
    window_center = first_note_time
    window_size = 10
    mask = (time_sec >= window_center - window_size/2) & (time_sec <= window_center + window_size/2)
    
    ax2.plot(time_sec[mask], signal_raw[mask], linewidth=1, label='Raw signal')
    ax2.axvline(first_note_time, color='red', linestyle='--', linewidth=2, label='First note')
    
    # Mark all notes in this window
    notes_in_window = note_times[(note_times + peak_lag >= window_center - window_size/2) & 
                                  (note_times + peak_lag <= window_center + window_size/2)]
    for nt in notes_in_window:
        ax2.axvline(nt + peak_lag, color='orange', alpha=0.3, linewidth=0.8)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(f'{axis_name} (m/s²)')
    ax2.set_title(f'Zoom: First Note Region (±{window_size/2:.0f}s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: High-density region 1
    if len(high_density) > 0:
        ax3 = fig.add_subplot(gs[1, 1])
        hd_center = peak_lag + high_density[0][0] + 5
        window_size = 10
        mask = (time_sec >= hd_center - window_size/2) & (time_sec <= hd_center + window_size/2)
        
        ax3.plot(time_sec[mask], signal_raw[mask], linewidth=1, label='Raw signal')
        
        # Mark notes in this window
        notes_in_window = note_times[(note_times + peak_lag >= hd_center - window_size/2) & 
                                      (note_times + peak_lag <= hd_center + window_size/2)]
        for nt in notes_in_window:
            ax3.axvline(nt + peak_lag, color='orange', alpha=0.5, linewidth=0.8)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel(f'{axis_name} (m/s²)')
        ax3.set_title(f'High-Density Region 1: {len(notes_in_window)} notes in {window_size}s')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Periodicity check (autocorrelation or zoomed section)
    ax4 = fig.add_subplot(gs[2, 0])
    beat_period = 60.0 / bpm
    
    # Take a 5-second window from high-density region
    if len(high_density) > 0:
        hd_start = peak_lag + high_density[0][0]
        short_window = 5.0
        mask = (time_sec >= hd_start) & (time_sec <= hd_start + short_window)
        
        ax4.plot(time_sec[mask] - hd_start, signal_raw[mask], linewidth=1.5, label='Raw signal')
        
        # Mark expected beat times
        for beat_num in range(int(short_window / beat_period) + 1):
            beat_time = beat_num * beat_period
            ax4.axvline(beat_time, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Mark actual notes
        notes_in_window = note_times[(note_times + peak_lag >= hd_start) & 
                                      (note_times + peak_lag <= hd_start + short_window)]
        for nt in notes_in_window:
            ax4.axvline(nt + peak_lag - hd_start, color='orange', alpha=0.7, linewidth=1.5)
        
        ax4.set_xlabel('Time relative to window start (s)')
        ax4.set_ylabel(f'{axis_name} (m/s²)')
        ax4.set_title(f'Periodicity Check: {short_window}s window (gray lines = beat period {beat_period:.3f}s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Note count comparison
    ax5 = fig.add_subplot(gs[2, 1])
    
    # For multiple 10s windows, compare expected vs visible peaks
    windows = [(peak_lag + t, t, count) for t, count in high_density[:3]]
    window_labels = []
    note_counts = []
    
    for rec_time, sm_time, expected_count in windows:
        window_labels.append(f'{sm_time:.0f}s\n({expected_count} notes)')
        note_counts.append(expected_count)
    
    x_pos = np.arange(len(window_labels))
    ax5.bar(x_pos, note_counts, color='orange', alpha=0.7)
    ax5.set_xlabel('Region (time in .sm file)')
    ax5.set_ylabel('Note count (10s window)')
    ax5.set_title('Expected Note Counts in High-Density Regions')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(window_labels)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary text
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    summary_text = f"""
VERIFICATION CHECKLIST:

1. First Note Check ({first_note_time:.1f}s in recording):
   □ Is there visible activity/peaks around {first_note_time:.1f}s? (see upper-left zoom)
   □ Does the activity start make sense visually?

2. High-Density Region Check (orange shaded regions in top plot):
   □ Do these regions show more activity than quiet regions?
   □ Can you visually see ~{high_density[0][1] if high_density else 0} distinct peaks in 10s? (see upper-right zoom)

3. Periodicity Check (lower-left plot):
   □ Do orange bars (notes) align with sensor peaks?
   □ Is there roughly one peak per beat period ({beat_period:.3f}s)?

4. Overall Assessment:
   Detected offset: {peak_lag:.3f}s
   Peak ratio: {peak_ratio:.2f} (target: >2.0)
   Z-score: {z_score:.2f} (target: >5.0)
   
   Based on visual inspection:
   □ Alignment appears correct → Problem is weak correlation due to noise/variability
   □ Alignment appears wrong → Problem is timing error in .sm file or gameplay
   □ Inconclusive → Need more analysis or different approach
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{capture_path.stem}_visual_verification_{axis_name}.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved: {output_file}")
    plt.close(fig)
    
    print("\n" + "="*70)
    print("VISUAL VERIFICATION COMPLETE")
    print("="*70)
    print("\nPlease inspect the saved PNG and answer the checklist questions.")
    print("This will inform the next steps for alignment improvement.")
    print("="*70)


if __name__ == '__main__':
    main()
