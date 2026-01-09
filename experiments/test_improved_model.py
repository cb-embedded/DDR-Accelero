#!/usr/bin/env python3
"""
Test improved alignment models to boost SNR.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.signal import correlate

# Import alignment module
import importlib.util
spec = importlib.util.spec_from_file_location("align_signals", Path(__file__).parent / "03_align_signals.py")
align_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_module)

load_sensor_data = align_module.load_sensor_data
compute_acceleration_magnitude = align_module.compute_acceleration_magnitude
parse_sm_file = align_module.parse_sm_file

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 10.0
BANDPASS_ORDER = 4


def preprocess_sensor_v2(time_ms, magnitude, target_sample_rate=100):
    """Improved preprocessing: use squared signal energy instead of envelope."""
    time_sec = time_ms / 1000.0
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_sample_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC and filter
    resampled = resampled - np.mean(resampled)
    sos = sp_signal.butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                           btype='band', fs=target_sample_rate, output='sos')
    filtered = sp_signal.sosfiltfilt(sos, resampled)
    
    # Use squared signal (energy) instead of absolute value
    # This better represents the intensity of movement
    energy = filtered ** 2
    
    # Smooth the energy signal
    window_size = int(0.15 * target_sample_rate)  # 150ms window
    energy_smooth = sp_signal.convolve(energy, 
                                       sp_signal.windows.hann(window_size)/sum(sp_signal.windows.hann(window_size)), 
                                       mode='same')
    
    return new_time, energy_smooth


def create_reference_v2(notes, duration, sample_rate=100):
    """Improved reference signal: squared exponential to match energy."""
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    ref_signal = np.zeros(num_samples)
    
    # Place spikes at notes
    for note in notes:
        idx = int(note.time * sample_rate)
        if 0 <= idx < num_samples:
            ref_signal[idx] = 1.0
    
    # Create impulse response - exponential decay
    decay_time = 0.15  # 150ms
    impulse_duration = 0.5  # 500ms
    impulse_samples = int(impulse_duration * sample_rate)
    t_impulse = np.arange(impulse_samples) / sample_rate
    impulse_response = np.exp(-t_impulse / decay_time)
    impulse_response = impulse_response / np.sum(impulse_response)
    
    # Convolve
    ref_signal = sp_signal.convolve(ref_signal, impulse_response, mode='same')
    
    # Apply bandpass filter
    sos = sp_signal.butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                           btype='band', fs=sample_rate, output='sos')
    ref_signal = sp_signal.sosfiltfilt(sos, ref_signal)
    
    # Square the reference signal to match sensor energy representation
    ref_signal = ref_signal ** 2
    
    return time, ref_signal


def analyze_snr(sensor, ref, best_offset, lags, corr):
    """Calculate SNR."""
    peak_idx = np.argmax(corr)
    peak_value = corr[peak_idx]
    
    # Exclude peak region for noise calculation
    peak_window = 50
    noise_mask = np.ones(len(corr), dtype=bool)
    noise_mask[max(0, peak_idx - peak_window):min(len(corr), peak_idx + peak_window)] = False
    noise_floor = np.mean(corr[noise_mask])
    noise_std = np.std(corr[noise_mask])
    
    snr = (peak_value - noise_floor) / noise_std if noise_std > 0 else 0
    return {
        'peak': peak_value,
        'noise_floor': noise_floor,
        'noise_std': noise_std,
        'snr': snr
    }


def main():
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    
    # Find Lucky Orb
    sensor_file = list(raw_data_dir.glob('*Lucky*Orb*5_Medium-2026-01-06*.zip'))[0]
    sm_file = list(sm_dir.glob('*Lucky*Orb*.sm'))[0]
    
    print("="*70)
    print("TESTING IMPROVED ALIGNMENT MODEL")
    print("="*70)
    print(f"Sensor: {sensor_file.name}")
    print(f"Chart: {sm_file.name}\n")
    
    # Load data
    sensor_df = load_sensor_data(sensor_file)
    time_ms, magnitude = compute_acceleration_magnitude(sensor_df)
    charts = parse_sm_file(sm_file)
    chart = [c for c in charts if 'Medium' in c.difficulty][0]
    
    print(f"Chart: {len(chart.notes)} notes, {chart.notes[-1].time:.2f}s\n")
    
    # Test OLD approach (current)
    print("Testing OLD model (envelope)...")
    sensor_time, sensor_old = align_module.preprocess_sensor_signal(time_ms, magnitude)
    ref_duration = max(sensor_time[-1], chart.notes[-1].time)
    ref_time, ref_old = align_module.create_reference_signal(chart.notes, ref_duration)
    
    sensor_norm = (sensor_old - np.mean(sensor_old)) / (np.std(sensor_old) + 1e-8)
    ref_norm = (ref_old - np.mean(ref_old)) / (np.std(ref_old) + 1e-8)
    corr_old = correlate(sensor_norm, ref_norm, mode='full')
    lags = np.arange(-len(ref_norm) + 1, len(sensor_norm))
    time_lags = lags / 100.0
    
    peak_idx_old = np.argmax(corr_old)
    offset_old = time_lags[peak_idx_old]
    snr_old = analyze_snr(sensor_old, ref_old, offset_old, time_lags, corr_old)
    
    print(f"  Offset: {offset_old:.3f}s")
    print(f"  SNR: {snr_old['snr']:.2f}")
    
    # Test NEW approach (energy-based)
    print("\nTesting NEW model (energy-based)...")
    sensor_new_time, sensor_new = preprocess_sensor_v2(time_ms, magnitude)
    ref_new_time, ref_new = create_reference_v2(chart.notes, ref_duration)
    
    sensor_norm2 = (sensor_new - np.mean(sensor_new)) / (np.std(sensor_new) + 1e-8)
    ref_norm2 = (ref_new - np.mean(ref_new)) / (np.std(ref_new) + 1e-8)
    corr_new = correlate(sensor_norm2, ref_norm2, mode='full')
    
    peak_idx_new = np.argmax(corr_new)
    offset_new = time_lags[peak_idx_new]
    snr_new = analyze_snr(sensor_new, ref_new, offset_new, time_lags, corr_new)
    
    print(f"  Offset: {offset_new:.3f}s")
    print(f"  SNR: {snr_new['snr']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT: SNR {snr_old['snr']:.2f} → {snr_new['snr']:.2f} ({(snr_new['snr']/snr_old['snr']-1)*100:+.1f}%)")
    print(f"{'='*70}\n")
    
    # Visualize comparison
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Sensor signals
    axes[0, 0].plot(sensor_time, sensor_old, linewidth=0.6, alpha=0.7)
    axes[0, 0].set_title('OLD: Sensor Envelope (|filtered|)', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(sensor_new_time, sensor_new, linewidth=0.6, alpha=0.7, color='green')
    axes[0, 1].set_title('NEW: Sensor Energy (filtered²)', fontweight='bold')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reference signals
    axes[1, 0].plot(ref_time, ref_old, linewidth=0.6, alpha=0.7)
    axes[1, 0].set_title('OLD: Reference (filtered exponential)', fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(ref_new_time, ref_new, linewidth=0.6, alpha=0.7, color='green')
    axes[1, 1].set_title('NEW: Reference (exponential²)', fontweight='bold')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlations
    axes[2, 0].plot(time_lags, corr_old, linewidth=0.8, alpha=0.7)
    axes[2, 0].axvline(offset_old, color='r', linestyle='--', linewidth=2)
    axes[2, 0].axhline(snr_old['noise_floor'], color='orange', linestyle=':', linewidth=1.5)
    axes[2, 0].set_title(f'OLD: Correlation (SNR={snr_old["snr"]:.2f})', fontweight='bold')
    axes[2, 0].set_xlabel('Time offset (s)')
    axes[2, 0].set_ylabel('Correlation')
    axes[2, 0].set_xlim([-40, 40])
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time_lags, corr_new, linewidth=0.8, alpha=0.7, color='green')
    axes[2, 1].axvline(offset_new, color='r', linestyle='--', linewidth=2)
    axes[2, 1].axhline(snr_new['noise_floor'], color='orange', linestyle=':', linewidth=1.5)
    axes[2, 1].set_title(f'NEW: Correlation (SNR={snr_new["snr"]:.2f})', fontweight='bold')
    axes[2, 1].set_xlabel('Time offset (s)')
    axes[2, 1].set_ylabel('Correlation')
    axes[2, 1].set_xlim([-40, 40])
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=120)
    print("Saved: model_comparison.png\n")


if __name__ == '__main__':
    main()
