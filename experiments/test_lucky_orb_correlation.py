#!/usr/bin/env python3
"""
Test Lucky Orb alignment with extended correlation window to verify signal quality.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import the alignment module
import importlib.util
spec = importlib.util.spec_from_file_location("align_signals", Path(__file__).parent / "03_align_signals.py")
align_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_module)

load_sensor_data = align_module.load_sensor_data
compute_acceleration_magnitude = align_module.compute_acceleration_magnitude
preprocess_sensor_signal = align_module.preprocess_sensor_signal
create_reference_signal = align_module.create_reference_signal
parse_sm_file = align_module.parse_sm_file

from scipy.signal import correlate


def analyze_correlation_quality(sensor_envelope, ref_signal, best_offset, lags, corr):
    """Analyze the signal-to-noise ratio of correlation peak."""
    # Find peak index
    peak_idx = np.argmax(corr)
    peak_value = corr[peak_idx]
    
    # Calculate noise floor (exclude region around peak)
    peak_window = 50  # samples around peak to exclude
    noise_mask = np.ones(len(corr), dtype=bool)
    noise_mask[max(0, peak_idx - peak_window):min(len(corr), peak_idx + peak_window)] = False
    noise_floor = np.mean(corr[noise_mask])
    noise_std = np.std(corr[noise_mask])
    
    # Signal-to-noise ratio
    snr = (peak_value - noise_floor) / noise_std if noise_std > 0 else 0
    
    # Peak prominence (how much it stands out)
    prominence = peak_value - noise_floor
    
    return {
        'peak_value': peak_value,
        'noise_floor': noise_floor,
        'noise_std': noise_std,
        'snr': snr,
        'prominence': prominence,
        'peak_idx': peak_idx
    }


def main():
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    
    # Find Lucky Orb files
    sensor_files = list(raw_data_dir.glob('*Lucky*Orb*.zip'))
    sm_file = None
    for f in sm_dir.glob('*Lucky*Orb*.sm'):
        sm_file = f
        break
    
    if not sensor_files or not sm_file:
        print("Error: Could not find Lucky Orb files")
        sys.exit(1)
    
    print("="*70)
    print("LUCKY ORB ALIGNMENT ANALYSIS WITH EXTENDED CORRELATION WINDOW")
    print("="*70)
    
    for sensor_file in sensor_files[:3]:  # Test up to 3 Lucky Orb captures
        print(f"\n{'='*70}")
        print(f"Processing: {sensor_file.name}")
        print(f"{'='*70}")
        
        # Load and process data
        print("Loading sensor data...")
        sensor_df = load_sensor_data(sensor_file)
        time_ms, magnitude = compute_acceleration_magnitude(sensor_df)
        print(f"  Sensor duration: {(time_ms[-1] - time_ms[0]) / 1000:.2f}s")
        
        print("Parsing StepMania file...")
        charts = parse_sm_file(sm_file)
        chart = None
        for c in charts:
            if 'Medium' in c.difficulty or 'medium' in c.difficulty.lower():
                chart = c
                break
        if not chart and charts:
            chart = charts[0]
        
        print(f"  Chart: {chart.difficulty}, {len(chart.notes)} notes, {chart.notes[-1].time:.2f}s")
        
        # Preprocess signals
        print("Processing signals...")
        sensor_time, sensor_envelope = preprocess_sensor_signal(time_ms, magnitude)
        ref_duration = max(sensor_time[-1], chart.notes[-1].time)
        ref_time, ref_signal = create_reference_signal(chart.notes, ref_duration)
        
        # Normalize signals for correlation
        sensor_norm = (sensor_envelope - np.mean(sensor_envelope)) / (np.std(sensor_envelope) + 1e-8)
        ref_norm = (ref_signal - np.mean(ref_signal)) / (np.std(ref_signal) + 1e-8)
        
        # Compute cross-correlation
        print("Computing correlation...")
        corr = correlate(sensor_norm, ref_norm, mode='full')
        lags = np.arange(-len(ref_norm) + 1, len(sensor_norm))
        time_lags = lags / 100.0  # sample_rate = 100
        
        # Find best offset
        peak_idx = np.argmax(corr)
        best_offset = time_lags[peak_idx]
        
        # Analyze quality
        quality = analyze_correlation_quality(sensor_envelope, ref_signal, best_offset, time_lags, corr)
        
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"{'='*70}")
        print(f"Best offset: {best_offset:.3f}s")
        print(f"Peak correlation: {quality['peak_value']:.2f}")
        print(f"Noise floor: {quality['noise_floor']:.2f}")
        print(f"Noise std: {quality['noise_std']:.2f}")
        print(f"Signal-to-Noise Ratio: {quality['snr']:.2f}")
        print(f"Peak prominence: {quality['prominence']:.2f}")
        
        if quality['snr'] > 10:
            print("✓ EXCELLENT: Peak is very clear above noise floor (SNR > 10)")
        elif quality['snr'] > 5:
            print("✓ GOOD: Peak is clearly distinguishable (SNR > 5)")
        elif quality['snr'] > 3:
            print("⚠ MARGINAL: Peak is visible but not strong (SNR > 3)")
        else:
            print("✗ POOR: Peak is not clearly distinguishable (SNR < 3)")
        
        # Create detailed visualization with extended range (-40 to +40 seconds)
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        
        # Plot 1: Sensor envelope
        ax = axes[0]
        ax.plot(sensor_time, sensor_envelope, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Acceleration envelope', fontsize=10)
        ax.set_title(f'Processed Sensor Signal - {sensor_file.stem}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reference signal
        ax = axes[1]
        ax.plot(ref_time, ref_signal, linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Reference signal', fontsize=10)
        ax.set_title(f'Reference Signal - Lucky Orb ({chart.difficulty})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Full correlation (extended range -40 to +40)
        ax = axes[2]
        ax.plot(time_lags, corr, linewidth=0.8, alpha=0.7)
        ax.axvline(best_offset, color='r', linestyle='--', linewidth=2, 
                   label=f'Best offset: {best_offset:.3f}s')
        ax.axhline(quality['noise_floor'], color='orange', linestyle=':', linewidth=1.5,
                   label=f'Noise floor: {quality["noise_floor"]:.1f}')
        ax.axhline(quality['noise_floor'] + 3*quality['noise_std'], color='yellow', linestyle=':', 
                   linewidth=1, label=f'3σ noise: {quality["noise_floor"] + 3*quality["noise_std"]:.1f}')
        ax.set_xlabel('Time offset (s)', fontsize=10)
        ax.set_ylabel('Cross-correlation', fontsize=10)
        ax.set_title(f'Cross-correlation (Extended Range) - SNR: {quality["snr"]:.2f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-40, 40])  # Extended range as requested
        
        # Plot 4: Zoomed correlation around peak
        ax = axes[3]
        zoom_range = 10  # ±10 seconds around peak
        zoom_mask = (time_lags >= best_offset - zoom_range) & (time_lags <= best_offset + zoom_range)
        ax.plot(time_lags[zoom_mask], corr[zoom_mask], linewidth=1.2, color='blue')
        ax.axvline(best_offset, color='r', linestyle='--', linewidth=2, 
                   label=f'Peak: {quality["peak_value"]:.1f}')
        ax.axhline(quality['noise_floor'], color='orange', linestyle=':', linewidth=1.5,
                   label=f'Noise: {quality["noise_floor"]:.1f}')
        ax.fill_between([best_offset - zoom_range, best_offset + zoom_range], 
                        quality['noise_floor'] - quality['noise_std'],
                        quality['noise_floor'] + quality['noise_std'],
                        alpha=0.2, color='orange', label='±1σ noise')
        ax.set_xlabel('Time offset (s)', fontsize=10)
        ax.set_ylabel('Cross-correlation', fontsize=10)
        ax.set_title('Zoomed View Around Peak', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(__file__).parent
        output_file = output_dir / f'{sensor_file.stem}_detailed_correlation.png'
        fig.savefig(output_file, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nVisualization saved: {output_file.name}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
