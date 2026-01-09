#!/usr/bin/env python3
"""
Experiment 4: Generate a labeled dataset for arrow prediction.

This script creates a dataset by:
1. Aligning sensor data with StepMania charts
2. Extracting features from time windows around each arrow press
3. Creating labels (which arrows were pressed)
4. Saving the dataset for training
"""
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
import sys
import os
import importlib.util

# Load the parse_stepmania module dynamically
spec = importlib.util.spec_from_file_location("parse_sm", Path(__file__).parent / "02_parse_stepmania.py")
parse_sm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_sm_module)
parse_sm_file = parse_sm_module.parse_sm_file


def load_sensor_data(zip_path):
    """Load all sensor data from zip file."""
    data = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for sensor_name in ['Gravity', 'Gyroscope', 'Accelerometer']:
            csv_name = f'{sensor_name}.csv'
            if csv_name in zip_ref.namelist():
                with zip_ref.open(csv_name) as f:
                    data[sensor_name.lower()] = pd.read_csv(f)
    return data


def extract_window_features(df, center_time_ms, window_ms=500):
    """Extract features from a time window around a specific time.
    
    Args:
        df: DataFrame with columns [Time, X, Y, Z]
        center_time_ms: Center time in milliseconds
        window_ms: Window size in milliseconds (±window_ms/2)
    
    Returns:
        Dictionary of features
    """
    # Get data in window
    half_window = window_ms / 2
    mask = (df.iloc[:, 0] >= center_time_ms - half_window) & \
           (df.iloc[:, 0] <= center_time_ms + half_window)
    
    window_data = df[mask]
    
    if len(window_data) < 5:  # Not enough data
        return None
    
    features = {}
    
    # Extract X, Y, Z values
    for axis_idx, axis_name in enumerate(['x', 'y', 'z'], start=1):
        values = window_data.iloc[:, axis_idx].values
        
        if len(values) == 0:
            continue
        
        # Statistical features
        features[f'{axis_name}_mean'] = np.mean(values)
        features[f'{axis_name}_std'] = np.std(values)
        features[f'{axis_name}_min'] = np.min(values)
        features[f'{axis_name}_max'] = np.max(values)
        features[f'{axis_name}_range'] = np.max(values) - np.min(values)
        features[f'{axis_name}_skew'] = skew(values)
        features[f'{axis_name}_kurtosis'] = kurtosis(values)
        
        # Peak detection
        peaks, _ = signal.find_peaks(np.abs(values), height=np.std(values))
        features[f'{axis_name}_num_peaks'] = len(peaks)
    
    # Magnitude features
    x = window_data.iloc[:, 1].values
    y = window_data.iloc[:, 2].values
    z = window_data.iloc[:, 3].values
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    features['mag_mean'] = np.mean(magnitude)
    features['mag_std'] = np.std(magnitude)
    features['mag_max'] = np.max(magnitude)
    features['mag_energy'] = np.sum(magnitude**2)
    
    return features


def create_dataset(sensor_data, chart, time_offset_ms=0, window_ms=500):
    """Create a dataset from aligned sensor data and chart.
    
    Args:
        sensor_data: Dictionary of sensor DataFrames
        chart: SMChart object
        time_offset_ms: Time offset in milliseconds (when chart starts in sensor data)
        window_ms: Window size for feature extraction
    
    Returns:
        DataFrame with features and labels
    """
    dataset = []
    
    # Get reference time from first sensor
    sensor_name = list(sensor_data.keys())[0]
    start_time_ms = sensor_data[sensor_name].iloc[0, 0]
    
    # Process each note
    for note in chart.notes:
        # Convert note time to sensor time
        note_time_ms = start_time_ms + time_offset_ms + (note.time * 1000)
        
        # Extract features from each sensor
        sample = {}
        
        for sensor_name, df in sensor_data.items():
            features = extract_window_features(df, note_time_ms, window_ms)
            
            if features is None:
                continue
            
            # Prefix features with sensor name
            for key, value in features.items():
                sample[f'{sensor_name}_{key}'] = value
        
        if not sample:  # No features extracted
            continue
        
        # Add labels (which arrows are pressed)
        sample['arrow_left'] = 1 if note.arrows[0] != '0' else 0
        sample['arrow_down'] = 1 if note.arrows[1] != '0' else 0
        sample['arrow_up'] = 1 if note.arrows[2] != '0' else 0
        sample['arrow_right'] = 1 if note.arrows[3] != '0' else 0
        
        # Add metadata
        sample['time_sec'] = note.time
        sample['note_pattern'] = note.arrows
        
        dataset.append(sample)
    
    return pd.DataFrame(dataset)


def create_negative_samples(sensor_data, chart, time_offset_ms=0, window_ms=500, num_samples=100):
    """Create negative samples (no arrow press) for the dataset."""
    dataset = []
    
    sensor_name = list(sensor_data.keys())[0]
    start_time_ms = sensor_data[sensor_name].iloc[0, 0]
    end_time_ms = sensor_data[sensor_name].iloc[-1, 0]
    
    # Get note times
    note_times = [(start_time_ms + time_offset_ms + (note.time * 1000)) for note in chart.notes]
    
    # Sample random times that are far from any note
    min_distance_ms = window_ms * 2  # Stay at least 2 windows away from notes
    
    for _ in range(num_samples):
        # Random time in recording
        random_time = np.random.uniform(start_time_ms + min_distance_ms, 
                                       end_time_ms - min_distance_ms)
        
        # Check if far enough from all notes
        distances = [abs(random_time - nt) for nt in note_times]
        if min(distances) < min_distance_ms:
            continue
        
        # Extract features
        sample = {}
        for sensor_name, df in sensor_data.items():
            features = extract_window_features(df, random_time, window_ms)
            
            if features is None:
                continue
            
            for key, value in features.items():
                sample[f'{sensor_name}_{key}'] = value
        
        if not sample:
            continue
        
        # All arrows are 0 (no press)
        sample['arrow_left'] = 0
        sample['arrow_down'] = 0
        sample['arrow_up'] = 0
        sample['arrow_right'] = 0
        sample['time_sec'] = (random_time - start_time_ms) / 1000.0
        sample['note_pattern'] = '0000'
        
        dataset.append(sample)
    
    return pd.DataFrame(dataset)


def main():
    # Get directories
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Find Lucky Orb files
    sensor_file = None
    for f in raw_data_dir.glob('*Lucky*Orb*.zip'):
        sensor_file = f
        break
    
    sm_file = None
    for f in sm_dir.glob('*Lucky*Orb*.sm'):
        sm_file = f
        break
    
    if not sensor_file or not sm_file:
        print("Using any available files...")
        zip_files = list(raw_data_dir.glob('*.zip'))
        sm_files_list = list(sm_dir.glob('*.sm'))
        
        if not zip_files or not sm_files_list:
            print("Error: No sensor or StepMania files found")
            sys.exit(1)
        
        sensor_file = zip_files[0]
        sm_file = sm_files_list[0]
    
    print(f"Processing:")
    print(f"  Sensor: {sensor_file.name}")
    print(f"  Chart: {sm_file.name}\n")
    
    # Load data
    print("Loading sensor data...")
    sensor_data = load_sensor_data(sensor_file)
    print(f"  Loaded sensors: {', '.join(sensor_data.keys())}")
    
    print("Parsing StepMania file...")
    charts = parse_sm_file(sm_file)
    
    # Use Medium difficulty
    chart = None
    for c in charts:
        if 'Medium' in c.difficulty or 'medium' in c.difficulty.lower():
            chart = c
            break
    
    if not chart and charts:
        chart = charts[0]
    
    print(f"  Using difficulty: {chart.difficulty}")
    print(f"  Total notes: {len(chart.notes)}\n")
    
    # For this experiment, assume time offset is 0 (in practice, use experiment 3)
    time_offset_ms = 0
    window_ms = 500  # ±250ms window
    
    print("Extracting features from note windows...")
    positive_df = create_dataset(sensor_data, chart, time_offset_ms, window_ms)
    print(f"  Positive samples (with notes): {len(positive_df)}")
    
    print("Creating negative samples...")
    negative_df = create_negative_samples(sensor_data, chart, time_offset_ms, window_ms, num_samples=len(positive_df) // 2)
    print(f"  Negative samples (no notes): {len(negative_df)}")
    
    # Combine datasets
    full_dataset = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal dataset shape: {full_dataset.shape}")
    print(f"  Features: {full_dataset.shape[1] - 6}")  # Subtract label columns and metadata
    print(f"  Samples: {full_dataset.shape[0]}")
    
    # Show class distribution
    print("\nClass distribution:")
    for arrow in ['left', 'down', 'up', 'right']:
        count = full_dataset[f'arrow_{arrow}'].sum()
        print(f"  {arrow}: {count} ({count/len(full_dataset)*100:.1f}%)")
    
    # Save dataset
    output_file = output_dir / f'{sensor_file.stem}_dataset.csv'
    full_dataset.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Show sample
    print("\nSample features (first row):")
    feature_cols = [col for col in full_dataset.columns if not col.startswith('arrow_') and col not in ['time_sec', 'note_pattern']]
    print(full_dataset[feature_cols[:5]].head(1))
    
    print("\n" + "="*50)
    print("Experiment 4 completed!")
    print("="*50)
    print("\nKey observations:")
    print("1. We extracted statistical features from sensor windows")
    print("2. Dataset contains both positive (arrow press) and negative (no press) samples")
    print("3. Features include mean, std, min, max, peaks, energy, etc.")
    print("4. Next step: train a machine learning model on this dataset")


if __name__ == '__main__':
    main()
