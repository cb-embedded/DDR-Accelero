#!/usr/bin/env python3
"""
Test all axes to find which gives best correlation with .sm expected signal.

Usage:
    python test_all_axes.py <capture_zip> <sm_file> <difficulty> [kernel_type]

Example:
    python test_all_axes.py raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip "sm_files/Lucky Orb.sm" Medium bipolar
"""

import sys
import zipfile
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate


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


def load_all_axes(zip_path):
    """Load all accelerometer axes from capture zip."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if 'Gravity.csv' not in zip_ref.namelist():
            raise ValueError("No Gravity.csv found in zip file")
        
        with zip_ref.open('Gravity.csv') as f:
            df = pd.read_csv(f)
    
    time_sec = df['seconds_elapsed'].values
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return time_sec, {'x': x, 'y': y, 'z': z, 'magnitude': magnitude}


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(__doc__)
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    difficulty = sys.argv[3]
    kernel_type = sys.argv[4] if len(sys.argv) == 5 else 'bipolar'
    
    print("="*70)
    print("DDR-Accelero: Test All Axes for Best Correlation")
    print("="*70)
    print(f"\nCapture: {capture_path.name}")
    print(f"Song: {sm_path.name}")
    print(f"Difficulty: {difficulty}")
    print(f"Kernel type: {kernel_type}\n")
    
    # Load all axes
    print("[1/3] Loading all accelerometer axes...")
    time_sec, axes_data = load_all_axes(capture_path)
    duration = time_sec[-1] - time_sec[0]
    print(f"  Duration: {duration:.2f}s")
    
    # Parse .sm file
    print("\n[2/3] Parsing .sm file...")
    note_times, bpm, offset = parse_sm_notes(sm_path, difficulty)
    print(f"  Total notes: {len(note_times)}")
    
    # Resample and create expected signal
    target_rate = 100.0
    max_duration = max(time_sec[-1] - time_sec[0], note_times[-1] if len(note_times) > 0 else 0)
    expected = create_expected_signal(note_times, max_duration, target_rate, kernel_type)
    
    # Test each axis
    print("\n[3/3] Testing correlation for each axis...")
    print(f"\n{'Axis':<12} {'Peak Ratio':<12} {'Z-score':<12} {'Offset (s)':<12} {'Status'}")
    print("-" * 70)
    
    results = {}
    for axis_name in ['x', 'y', 'z', 'magnitude']:
        # Resample axis data
        _, signal_resampled = resample_signal(time_sec, axes_data[axis_name], target_rate)
        
        # Compute correlation
        time_lags, corr, peak_lag, peak_ratio, z_score = compute_correlation(
            signal_resampled, expected, target_rate)
        
        # Evaluate dominance
        status = "✓ DOMINANT" if (peak_ratio > 2.0 and z_score > 5.0) else "⚠ WEAK"
        
        results[axis_name] = {
            'peak_ratio': peak_ratio,
            'z_score': z_score,
            'offset': peak_lag,
            'corr': corr,
            'time_lags': time_lags,
            'signal': signal_resampled
        }
        
        print(f"{axis_name:<12} {peak_ratio:<12.2f} {z_score:<12.2f} {peak_lag:<12.3f} {status}")
    
    # Find best axis
    best_axis = max(results.keys(), key=lambda k: results[k]['peak_ratio'])
    best_result = results[best_axis]
    
    print("\n" + "="*70)
    print("BEST AXIS FOUND")
    print("="*70)
    print(f"Axis: {best_axis}")
    print(f"Peak ratio: {best_result['peak_ratio']:.2f}")
    print(f"Z-score: {best_result['z_score']:.2f}")
    print(f"Offset: {best_result['offset']:.3f}s")
    
    if best_result['peak_ratio'] > 2.0 and best_result['z_score'] > 5.0:
        print("✓ DOMINANT PEAK - Alignment is unambiguous!")
    else:
        print("⚠ Still weak peak - may need different approach")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, axis_name in enumerate(['x', 'y', 'z', 'magnitude']):
        ax = axes[idx]
        result = results[axis_name]
        
        ax.plot(result['time_lags'], result['corr'], linewidth=0.8)
        ax.axvline(result['offset'], color='r', linestyle='--', 
                   label=f"Peak: {result['offset']:.3f}s")
        ax.set_xlabel('Time lag (s)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'{axis_name.upper()}: ratio={result["peak_ratio"]:.2f}, z={result["z_score"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-30, 30])
        
        # Highlight if best
        if axis_name == best_axis:
            ax.patch.set_facecolor('#e8f5e9')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{capture_path.stem}_all_axes_comparison_{kernel_type}.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {output_file}")
    plt.close(fig)
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
