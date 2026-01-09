#!/usr/bin/env python3
"""
Synchronize sensor data and apply arrow labels.

This script:
1. Loads processed sensor data
2. Aligns timestamps across sensors
3. Applies labels from label files
4. Creates synchronized dataset with labels
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import interpolate


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sensor_data(session_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load sensor data for a session.
    
    Args:
        session_dir: Directory containing sensor CSV files
        
    Returns:
        Dictionary mapping sensor type to DataFrame
    """
    sensor_data = {}
    sensor_types = ['accelerometer', 'gyroscope', 'magnetometer']
    
    for sensor_type in sensor_types:
        csv_path = os.path.join(session_dir, f"{sensor_type}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sensor_data[sensor_type] = df
            print(f"Loaded {sensor_type}: {len(df)} samples")
    
    return sensor_data


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load arrow labels from CSV file.
    
    Expected format:
    timestamp,arrow
    1234567890.123,LEFT
    1234567890.456,DOWN
    
    Args:
        labels_path: Path to labels CSV file
        
    Returns:
        DataFrame with columns: timestamp, arrow
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    
    # Validate columns
    if 'timestamp' not in df.columns or 'arrow' not in df.columns:
        raise ValueError("Labels file must contain 'timestamp' and 'arrow' columns")
    
    # Convert timestamp to float
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # If timestamps are in milliseconds, convert to seconds
    if df['timestamp'].mean() > 1e12:
        df['timestamp'] = df['timestamp'] / 1000.0
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} arrow labels")
    print(f"  Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s")
    print(f"  Arrow distribution: {df['arrow'].value_counts().to_dict()}")
    
    return df


def resample_sensor(df: pd.DataFrame, target_rate_hz: float) -> pd.DataFrame:
    """
    Resample sensor data to target rate using interpolation.
    
    Args:
        df: DataFrame with columns: timestamp, x, y, z
        target_rate_hz: Target sampling rate in Hz
        
    Returns:
        Resampled DataFrame
    """
    # Create uniform time grid
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration = end_time - start_time
    
    num_samples = int(duration * target_rate_hz)
    new_timestamps = np.linspace(start_time, end_time, num_samples)
    
    # Interpolate each axis
    resampled_data = {'timestamp': new_timestamps}
    
    for axis in ['x', 'y', 'z']:
        interp_func = interpolate.interp1d(
            df['timestamp'].values,
            df[axis].values,
            kind='linear',
            fill_value='extrapolate'
        )
        resampled_data[axis] = interp_func(new_timestamps)
    
    return pd.DataFrame(resampled_data)


def synchronize_sensors(sensor_data: Dict[str, pd.DataFrame], 
                       target_rate_hz: float) -> pd.DataFrame:
    """
    Synchronize all sensors to common timestamp grid.
    
    Args:
        sensor_data: Dictionary of sensor type to DataFrame
        target_rate_hz: Target sampling rate
        
    Returns:
        DataFrame with synchronized sensor data
    """
    print(f"\nSynchronizing sensors to {target_rate_hz} Hz...")
    
    # Find common time range
    start_times = [df['timestamp'].min() for df in sensor_data.values()]
    end_times = [df['timestamp'].max() for df in sensor_data.values()]
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    print(f"Common time range: {common_start:.2f} - {common_end:.2f}s "
          f"({common_end - common_start:.2f}s duration)")
    
    # Resample each sensor
    resampled_sensors = {}
    for sensor_type, df in sensor_data.items():
        # Filter to common time range
        df_filtered = df[(df['timestamp'] >= common_start) & 
                         (df['timestamp'] <= common_end)].copy()
        
        # Resample
        df_resampled = resample_sensor(df_filtered, target_rate_hz)
        resampled_sensors[sensor_type] = df_resampled
        print(f"  Resampled {sensor_type}: {len(df_resampled)} samples")
    
    # Merge all sensors
    result = resampled_sensors['accelerometer'].copy()
    result = result.rename(columns={
        'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'
    })
    
    if 'gyroscope' in resampled_sensors:
        gyro = resampled_sensors['gyroscope']
        result['gyro_x'] = gyro['x']
        result['gyro_y'] = gyro['y']
        result['gyro_z'] = gyro['z']
    
    if 'magnetometer' in resampled_sensors:
        mag = resampled_sensors['magnetometer']
        result['mag_x'] = mag['x']
        result['mag_y'] = mag['y']
        result['mag_z'] = mag['z']
    
    print(f"\nSynchronized dataset: {len(result)} samples, {len(result.columns)} columns")
    return result


def apply_labels(sensor_df: pd.DataFrame, labels_df: pd.DataFrame, 
                tolerance_ms: float, arrow_classes: List[str]) -> pd.DataFrame:
    """
    Apply arrow labels to sensor data based on timestamps.
    
    Args:
        sensor_df: Synchronized sensor data
        labels_df: Arrow labels with timestamps
        tolerance_ms: Time tolerance for matching labels (milliseconds)
        arrow_classes: List of valid arrow class names
        
    Returns:
        DataFrame with added 'label' column
    """
    print(f"\nApplying labels with {tolerance_ms}ms tolerance...")
    
    tolerance_s = tolerance_ms / 1000.0
    
    # Initialize label column (0 = no arrow)
    sensor_df['label'] = 0
    sensor_df['arrow'] = 'NONE'
    
    # Create arrow to integer mapping
    arrow_to_int = {arrow: i+1 for i, arrow in enumerate(arrow_classes)}
    arrow_to_int['NONE'] = 0
    
    # For each label, find matching sensor samples
    label_counts = {arrow: 0 for arrow in arrow_classes}
    label_counts['NONE'] = len(sensor_df)
    
    for _, label_row in labels_df.iterrows():
        label_time = label_row['timestamp']
        arrow = label_row['arrow']
        
        if arrow not in arrow_classes:
            print(f"  Warning: Unknown arrow type '{arrow}', skipping")
            continue
        
        # Find samples within tolerance
        time_diff = np.abs(sensor_df['timestamp'] - label_time)
        matching_indices = time_diff <= tolerance_s
        
        if matching_indices.any():
            sensor_df.loc[matching_indices, 'label'] = arrow_to_int[arrow]
            sensor_df.loc[matching_indices, 'arrow'] = arrow
            label_counts[arrow] += matching_indices.sum()
            label_counts['NONE'] -= matching_indices.sum()
    
    print(f"\nLabel distribution:")
    for arrow, count in label_counts.items():
        percentage = (count / len(sensor_df)) * 100
        print(f"  {arrow}: {count} samples ({percentage:.2f}%)")
    
    return sensor_df


def save_synchronized_data(session_name: str, data: pd.DataFrame, 
                          output_dir: str) -> str:
    """
    Save synchronized and labeled data.
    
    Args:
        session_name: Name of the session
        data: Synchronized DataFrame with labels
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{session_name}_sync.csv")
    data.to_csv(output_path, index=False)
    
    print(f"\nSaved synchronized data to {output_path}")
    return output_path


def process_session(session_name: str, config: Dict, labels_path: str = None) -> None:
    """
    Process a single session: synchronize sensors and apply labels.
    
    Args:
        session_name: Name of the session
        config: Configuration dictionary
        labels_path: Optional path to labels file
    """
    print(f"\n{'='*60}")
    print(f"Processing session: {session_name}")
    print(f"{'='*60}")
    
    # Load sensor data
    session_dir = os.path.join(config['data']['processed_dir'], session_name)
    if not os.path.exists(session_dir):
        print(f"ERROR: Session directory not found: {session_dir}")
        return
    
    sensor_data = load_sensor_data(session_dir)
    
    if not sensor_data:
        print("ERROR: No sensor data found")
        return
    
    if 'accelerometer' not in sensor_data:
        print("ERROR: Accelerometer data is required")
        return
    
    # Synchronize sensors
    target_rate = config['sensors']['sampling']['target_rate_hz']
    sync_data = synchronize_sensors(sensor_data, target_rate)
    
    # Apply labels if provided
    if labels_path and os.path.exists(labels_path):
        try:
            labels_df = load_labels(labels_path)
            tolerance_ms = config['arrows']['label_tolerance_ms']
            arrow_classes = config['arrows']['classes']
            
            sync_data = apply_labels(sync_data, labels_df, tolerance_ms, arrow_classes)
        except Exception as e:
            print(f"ERROR loading labels: {e}")
            print("Continuing without labels...")
            sync_data['label'] = 0
            sync_data['arrow'] = 'NONE'
    else:
        print("\nNo labels provided, creating unlabeled dataset")
        sync_data['label'] = 0
        sync_data['arrow'] = 'NONE'
    
    # Save synchronized data
    features_dir = config['data']['features_dir']
    save_synchronized_data(session_name, sync_data, features_dir)
    
    print(f"\n✓ Successfully processed {session_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Synchronize sensor data and apply arrow labels'
    )
    parser.add_argument(
        'session_name',
        help='Name of the session to process'
    )
    parser.add_argument(
        '--labels',
        help='Path to labels CSV file (optional)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    os.makedirs(config['data']['features_dir'], exist_ok=True)
    
    # Process session
    process_session(args.session_name, config, args.labels)
    
    print(f"\n{'='*60}")
    print("Synchronization complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
