#!/usr/bin/env python3
"""
Experiment 3: Align sensor data with StepMania charts using correlation.

This script attempts to find the time offset between sensor data capture and the
StepMania chart by:
1. Creating a reference signal from the chart (spike at each note)
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
    # Assume columns are: Time, X, Y, Z (or similar)
    time = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    z = df.iloc[:, 3].values
    
    # Compute magnitude
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return time, magnitude


def create_reference_signal(notes, duration, sample_rate=100):
    """Create a reference signal from note timings.
    
    Args:
        notes: List of SMNote objects
        duration: Duration of signal in seconds
        sample_rate: Samples per second
    
    Returns:
        time array and signal array
    """
    num_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, num_samples)
    ref_signal = np.zeros(num_samples)
    
    # Place a spike at each note
    for note in notes:
        idx = int(note.time * sample_rate)
        if 0 <= idx < num_samples:
            ref_signal[idx] = 1.0
    
    # Smooth the reference signal (simulates the temporal spread of sensor response)
    window_size = int(0.1 * sample_rate)  # 100ms window
    ref_signal = signal.convolve(ref_signal, signal.windows.gaussian(window_size, window_size/6), mode='same')
    
    return time, ref_signal


def preprocess_sensor_signal(time_ms, magnitude, target_sample_rate=100):
    """Preprocess sensor signal: resample and filter."""
    # Convert time to seconds
    time_sec = time_ms / 1000.0
    
    # Resample to target sample rate
    duration = time_sec[-1] - time_sec[0]
    num_samples = int(duration * target_sample_rate)
    new_time = np.linspace(0, duration, num_samples)
    resampled = np.interp(new_time, time_sec - time_sec[0], magnitude)
    
    # Remove DC component and apply bandpass filter
    resampled = resampled - np.mean(resampled)
    
    # Bandpass filter (0.5 Hz to 10 Hz) to focus on human movement frequencies
    sos = signal.butter(4, [0.5, 10], btype='band', fs=target_sample_rate, output='sos')
    filtered = signal.sosfiltfilt(sos, resampled)
    
    # Compute envelope (absolute value + smoothing)
    envelope = np.abs(filtered)
    window_size = int(0.2 * target_sample_rate)  # 200ms window
    envelope = signal.convolve(envelope, signal.windows.hann(window_size)/sum(signal.windows.hann(window_size)), mode='same')
    
    return new_time, envelope


def find_alignment(sensor_signal, ref_signal, sample_rate=100):
    """Find the best time offset using cross-correlation."""
    # Normalize signals
    sensor_norm = (sensor_signal - np.mean(sensor_signal)) / (np.std(sensor_signal) + 1e-8)
    ref_norm = (ref_signal - np.mean(ref_signal)) / (np.std(ref_signal) + 1e-8)
    
    # Compute cross-correlation
    corr = correlate(sensor_norm, ref_norm, mode='full')
    lags = np.arange(-len(ref_norm) + 1, len(sensor_norm))
    
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
