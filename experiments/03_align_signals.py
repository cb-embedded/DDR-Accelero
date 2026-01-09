#!/usr/bin/env python3
"""
Experiment 3: Align sensor data with StepMania charts using correlation.

This script attempts to find the time offset between sensor data capture and the
StepMania chart by:
1. Creating a reference signal from the chart using causal impulse response
2. Computing signal features from sensor data (acceleration magnitude)
3. Using cross-correlation to find the best alignment
"""
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.signal import correlate
import sys
import os
import importlib.util

# Signal processing constants
BANDPASS_LOW_HZ = 0.5  # Low cutoff for bandpass filter (Hz)
BANDPASS_HIGH_HZ = 8.0  # High cutoff for bandpass filter (Hz) - matched to typical body motion
BANDPASS_ORDER = 4  # Butterworth filter order

# Biomechanical model constants for impulse response (from research on human biomechanics)
BODY_DECAY_TIME_SEC = 0.10  # Decay time constant τ for damped body response (80-150ms typical)
IMPULSE_DURATION_SEC = 0.8  # Total duration of impulse response (capture full decay)

# Load the parse_stepmania module dynamically
spec = importlib.util.spec_from_file_location("parse_sm", Path(__file__).parent / "02_parse_stepmania.py")
parse_sm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_sm_module)
parse_sm_file = parse_sm_module.parse_sm_file


def load_sensor_data(zip_path):
    """Load accelerometer data from zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Try to load Gravity or Accelerometer data
        sensor_files = ['Gravity.csv', 'Accelerometer.csv']
        
        for sensor_file in sensor_files:
            if sensor_file in zip_ref.namelist():
                with zip_ref.open(sensor_file) as f:
                    df = pd.read_csv(f)
                    return df
    
    raise ValueError("No accelerometer data found in zip file")


def compute_acceleration_magnitude(df):
    """Compute the magnitude of 3D acceleration vector."""
    # CSV format: time, seconds_elapsed, z, y, x
    # Use seconds_elapsed column (index 1) as it's already in seconds
    time = df.iloc[:, 1].values * 1000.0  # Convert to milliseconds
    x = df.iloc[:, 4].values  # x is at index 4
    y = df.iloc[:, 3].values  # y is at index 3
    z = df.iloc[:, 2].values  # z is at index 2
    
    # Compute magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return time, magnitude


def create_reference_signal(notes, duration, sample_rate=100):
    """Create reference signal using proper biomechanical model per explicit formula.
    
    Implements the mathematically correct transformation:
    1. Step onsets are the discrete note events e_i at times t_i
    2. Apply causal impulse response: h(t) = exp(-t/τ) · 1_{t≥0} with τ ≈ 80-150ms
    3. Generate pseudo-acceleration: p(t) = Σ_i h(t - t_i)
    4. Apply bandpass filter matching accelerometer: BandPass_{[0.5,8]}(p(t))
    5. Normalize: (p(t) - μ) / σ
    
    This respects causality, damping, and human biomechanics for sharp correlation peaks.
    
    Args:
        notes: List of SMNote objects (discrete step onset events)
        duration: Duration of signal in seconds
        sample_rate: Samples per second
    
    Returns:
        time array and normalized pseudo-acceleration signal
    """
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    
    # Step 1: Notes ARE the step onsets (discrete events e_i at times t_i)
    # Create impulse train at note times
    step_onsets = np.zeros(num_samples)
    for note in notes:
        idx = int(note.time * sample_rate)
        if 0 <= idx < num_samples:
            step_onsets[idx] = 1.0
    
    # Step 2: Create causal impulse response h(t) = exp(-t/τ) for t ≥ 0
    impulse_samples = int(IMPULSE_DURATION_SEC * sample_rate)
    t_impulse = np.arange(impulse_samples) / sample_rate
    impulse_response = np.exp(-t_impulse / BODY_DECAY_TIME_SEC)
    
    # Normalize impulse response to unit energy
    impulse_response = impulse_response / np.sum(impulse_response)
    
    # Step 3: Generate pseudo-acceleration by convolution p(t) = Σ_i h(t - t_i)
    pseudo_accel = signal.convolve(step_onsets, impulse_response, mode='same')
    
    # Step 4: Apply bandpass filter matching accelerometer bandwidth [0.5, 8] Hz
    # This removes DC drift and very high frequency noise
    sos = signal.butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ], 
                        btype='band', fs=sample_rate, output='sos')
    pseudo_accel_filtered = signal.sosfiltfilt(sos, pseudo_accel)
    
    # Step 5: Local normalization (z-score): (p(t) - μ) / σ
    # This ensures the reference signal has zero mean and unit variance
    mean_val = np.mean(pseudo_accel_filtered)
    std_val = np.std(pseudo_accel_filtered)
    if std_val > 1e-10:
        ref_signal = (pseudo_accel_filtered - mean_val) / std_val
    else:
        ref_signal = pseudo_accel_filtered - mean_val
    
    return time, ref_signal


def preprocess_sensor_signal(time_ms, magnitude, target_sample_rate=100):
    """Preprocess sensor signal to match reference signal processing.
    
    Applies the same processing as the reference signal:
    1. Resample to target rate
    2. Remove DC component  
    3. Apply bandpass filter [0.5, 8] Hz
    4. Normalize (z-score)
    
    Returns filtered and normalized acceleration signal, not envelope.
    """
    # Convert time to seconds
    time_sec = time_ms / 1000.0
    
    # Resample to target sample rate
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_sample_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC component
    resampled = resampled - np.mean(resampled)
    
    # Apply bandpass filter matching reference signal [0.5, 8] Hz
    sos = signal.butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                        btype='band', fs=target_sample_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, resampled)
    
    # Normalize (z-score) to match reference signal normalization
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    if std_val > 1e-10:
        normalized = (filtered - mean_val) / std_val
    else:
        normalized = filtered - mean_val
    
    return new_time, normalized


def find_alignment(sensor_signal, ref_signal, sample_rate=100):
    """Find the best time offset using cross-correlation.
    
    Both signals are already normalized (z-score), so we can correlate directly.
    """
    # Signals are already normalized, compute cross-correlation directly
    corr = correlate(sensor_signal, ref_signal, mode='full')
    lags = np.arange(-len(ref_signal) + 1, len(sensor_signal))
    
    # Convert lags to time
    time_lags = lags / sample_rate
    
    # Find the peak
    peak_idx = np.argmax(corr)
    best_offset = time_lags[peak_idx]
    
    return best_offset, time_lags, corr


def main():
    # Get directories
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    
    # Find Lucky Orb files as an example
    sensor_file = None
    for f in raw_data_dir.glob('*Lucky*Orb*.zip'):
        sensor_file = f
        break
    
    sm_file = None
    for f in sm_dir.glob('*Lucky*Orb*.sm'):
        sm_file = f
        break
    
    if not sensor_file or not sm_file:
        print("Error: Could not find matching Lucky Orb files")
        print("Trying with any available files...")
        zip_files = list(raw_data_dir.glob('*.zip'))
        sm_files_list = list(sm_dir.glob('*.sm'))
        
        if not zip_files or not sm_files_list:
            print("Error: No sensor or StepMania files found")
            sys.exit(1)
        
        sensor_file = zip_files[0]
        sm_file = sm_files_list[0]
    
    print(f"Sensor data: {sensor_file.name}")
    print(f"StepMania file: {sm_file.name}\n")
    
    # Load sensor data
    print("Loading sensor data...")
    sensor_df = load_sensor_data(sensor_file)
    time_ms, magnitude = compute_acceleration_magnitude(sensor_df)
    print(f"  Sensor duration: {(time_ms[-1] - time_ms[0]) / 1000:.2f}s")
    
    # Parse StepMania file
    print("Parsing StepMania file...")
    charts = parse_sm_file(sm_file)
    
    # Use Medium difficulty chart
    chart = None
    for c in charts:
        if 'Medium' in c.difficulty or 'medium' in c.difficulty.lower():
            chart = c
            break
    
    if not chart and charts:
        chart = charts[0]
    
    if not chart:
        print("Error: No chart found in StepMania file")
        sys.exit(1)
    
    print(f"  Chart difficulty: {chart.difficulty}")
    print(f"  Chart duration: {chart.notes[-1].time:.2f}s")
    print(f"  Total notes: {len(chart.notes)}\n")
    
    # Preprocess sensor signal
    print("Processing sensor signal...")
    sensor_time, sensor_envelope = preprocess_sensor_signal(time_ms, magnitude)
    
    # Create reference signal
    print("Creating reference signal from notes...")
    ref_duration = max(sensor_time[-1], chart.notes[-1].time)
    ref_time, ref_signal = create_reference_signal(chart.notes, ref_duration)
    
    # Align signals
    print("Finding alignment...")
    best_offset, lags, corr = find_alignment(sensor_envelope, ref_signal)
    
    print(f"\nBest time offset: {best_offset:.3f} seconds")
    print(f"This means the first note occurs at {best_offset:.3f}s in the sensor recording")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Sensor envelope
    ax = axes[0]
    ax.plot(sensor_time, sensor_envelope)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration envelope')
    ax.set_title('Processed Sensor Signal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reference signal
    ax = axes[1]
    ax.plot(ref_time, ref_signal)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reference signal')
    ax.set_title('Reference Signal from StepMania Chart')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cross-correlation
    ax = axes[2]
    ax.plot(lags, corr)
    ax.axvline(best_offset, color='r', linestyle='--', label=f'Best offset: {best_offset:.3f}s')
    ax.set_xlabel('Time offset (s)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Cross-correlation for Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-20, 20])  # Focus on reasonable offsets
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{sensor_file.stem}_alignment.png'
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    print("\n" + "="*50)
    print("Experiment 3 completed!")
    print("="*50)
    print("\nKey observations:")
    print("1. Cross-correlation can help find the time alignment")
    print("2. The offset tells us when the chart starts in the sensor recording")
    print("3. This approach may need refinement based on signal characteristics")
    print("4. Next step: use this alignment to create labeled training data")


if __name__ == '__main__':
    main()
