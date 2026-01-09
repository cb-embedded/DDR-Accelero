#!/usr/bin/env python3
"""
Extract and validate sensor data from Sensor Logger ZIP files.

This script:
1. Extracts CSV files from ZIP archives
2. Validates data integrity
3. Parses sensor data into structured format
4. Saves extracted data to the processed directory
"""

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_zip(zip_path: str, output_dir: str) -> str:
    """
    Extract ZIP file to output directory.
    
    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract files to
        
    Returns:
        Path to extracted directory
    """
    session_name = Path(zip_path).stem
    extract_path = os.path.join(output_dir, session_name)
    
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"Extracted {zip_path} to {extract_path}")
    return extract_path


def find_sensor_files(directory: str) -> Dict[str, str]:
    """
    Find sensor CSV files in extracted directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        Dictionary mapping sensor type to file path
    """
    sensor_files = {}
    sensor_names = ['accelerometer', 'gyroscope', 'magnetometer']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_lower = file.lower()
            for sensor in sensor_names:
                if sensor in file_lower and file.endswith('.csv'):
                    sensor_files[sensor] = os.path.join(root, file)
    
    return sensor_files


def parse_sensor_csv(file_path: str, sensor_type: str) -> pd.DataFrame:
    """
    Parse sensor CSV file.
    
    Args:
        file_path: Path to CSV file
        sensor_type: Type of sensor (accelerometer, gyroscope, magnetometer)
        
    Returns:
        DataFrame with columns: timestamp, x, y, z
    """
    # Try different CSV formats that Sensor Logger might use
    try:
        # Format 1: timestamp, x, y, z
        df = pd.read_csv(file_path)
        
        # Standardize column names
        columns = df.columns.tolist()
        if len(columns) >= 4:
            df = df.iloc[:, :4]  # Take first 4 columns
            df.columns = ['timestamp', 'x', 'y', 'z']
        elif len(columns) == 4:
            df.columns = ['timestamp', 'x', 'y', 'z']
        else:
            raise ValueError(f"Unexpected number of columns: {len(columns)}")
        
        # Convert timestamp to float (handle both seconds and milliseconds)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # If timestamps are in milliseconds, convert to seconds
        if df['timestamp'].mean() > 1e12:
            df['timestamp'] = df['timestamp'] / 1000.0
        
        # Ensure numeric types for sensor values
        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Parsed {sensor_type}: {len(df)} samples, "
              f"time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}s")
        
        return df
        
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        raise


def validate_sensor_data(df: pd.DataFrame, sensor_type: str) -> Tuple[bool, List[str]]:
    """
    Validate sensor data quality.
    
    Args:
        df: DataFrame with sensor data
        sensor_type: Type of sensor
        
    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []
    
    # Check for minimum number of samples
    if len(df) < 100:
        warnings.append(f"Too few samples: {len(df)}")
    
    # Check for reasonable timestamp range
    time_range = df['timestamp'].max() - df['timestamp'].min()
    if time_range < 1.0:
        warnings.append(f"Time range too short: {time_range:.2f}s")
    
    # Check sampling rate
    time_diffs = df['timestamp'].diff().dropna()
    median_interval = time_diffs.median()
    if median_interval > 0:
        sampling_rate = 1.0 / median_interval
        print(f"  Estimated sampling rate: {sampling_rate:.1f} Hz")
        
        if sampling_rate < 10:
            warnings.append(f"Low sampling rate: {sampling_rate:.1f} Hz")
    
    # Check for gaps in data
    max_gap = time_diffs.max()
    if max_gap > 1.0:
        warnings.append(f"Large time gap detected: {max_gap:.2f}s")
    
    # Check for reasonable sensor value ranges
    for axis in ['x', 'y', 'z']:
        values = df[axis]
        if values.std() < 0.01:
            warnings.append(f"{axis}-axis has very low variance")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def save_processed_data(session_name: str, sensor_data: Dict[str, pd.DataFrame], 
                       output_dir: str) -> str:
    """
    Save processed sensor data.
    
    Args:
        session_name: Name of the session
        sensor_data: Dictionary of sensor type to DataFrame
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    session_dir = os.path.join(output_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)
    
    # Save each sensor to separate CSV
    for sensor_type, df in sensor_data.items():
        output_path = os.path.join(session_dir, f"{sensor_type}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {sensor_type} data to {output_path}")
    
    # Save metadata
    metadata = {
        'session_name': session_name,
        'sensors': list(sensor_data.keys()),
        'sample_counts': {sensor: len(df) for sensor, df in sensor_data.items()},
        'time_ranges': {
            sensor: {
                'start': float(df['timestamp'].min()),
                'end': float(df['timestamp'].max()),
                'duration': float(df['timestamp'].max() - df['timestamp'].min())
            }
            for sensor, df in sensor_data.items()
        }
    }
    
    metadata_path = os.path.join(session_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_path}")
    return session_dir


def process_zip_file(zip_path: str, config: Dict) -> None:
    """
    Process a single ZIP file from Sensor Logger.
    
    Args:
        zip_path: Path to ZIP file
        config: Configuration dictionary
    """
    print(f"\n{'='*60}")
    print(f"Processing: {zip_path}")
    print(f"{'='*60}")
    
    # Extract ZIP
    raw_dir = config['data']['raw_dir']
    extract_path = extract_zip(zip_path, raw_dir)
    
    # Find sensor files
    sensor_files = find_sensor_files(extract_path)
    print(f"\nFound sensor files: {list(sensor_files.keys())}")
    
    if not sensor_files:
        print("WARNING: No sensor CSV files found in ZIP!")
        return
    
    # Parse and validate each sensor
    sensor_data = {}
    all_valid = True
    
    for sensor_type, file_path in sensor_files.items():
        print(f"\nProcessing {sensor_type}...")
        try:
            df = parse_sensor_csv(file_path, sensor_type)
            is_valid, warnings = validate_sensor_data(df, sensor_type)
            
            if warnings:
                print(f"  Warnings for {sensor_type}:")
                for warning in warnings:
                    print(f"    - {warning}")
                all_valid = False
            
            sensor_data[sensor_type] = df
            
        except Exception as e:
            print(f"  ERROR: Failed to process {sensor_type}: {e}")
            all_valid = False
    
    # Save processed data
    if sensor_data:
        session_name = Path(zip_path).stem
        processed_dir = config['data']['processed_dir']
        save_processed_data(session_name, sensor_data, processed_dir)
        
        if all_valid:
            print(f"\n✓ Successfully processed {session_name}")
        else:
            print(f"\n⚠ Processed {session_name} with warnings")
    else:
        print(f"\n✗ Failed to process any sensors from {zip_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract and validate sensor data from Sensor Logger ZIP files'
    )
    parser.add_argument(
        'zip_files',
        nargs='+',
        help='Path(s) to ZIP file(s) from Sensor Logger'
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
    for dir_key in ['raw_dir', 'processed_dir']:
        os.makedirs(config['data'][dir_key], exist_ok=True)
    
    # Process each ZIP file
    for zip_path in args.zip_files:
        if not os.path.exists(zip_path):
            print(f"ERROR: File not found: {zip_path}")
            continue
        
        process_zip_file(zip_path, config)
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
