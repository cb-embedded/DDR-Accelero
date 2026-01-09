#!/usr/bin/env python3
"""
Experiment 1: Extract and visualize raw sensor data from Sensor Logger zip files.

This script extracts accelerometer/gyroscope data from Android Sensor Logger captures
and visualizes the signals to understand the data characteristics.
"""
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def extract_and_load_sensor_data(zip_path):
    """Extract sensor data from a zip file and load into pandas DataFrames."""
    data = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List available files
        files = zip_ref.namelist()
        print(f"Files in {zip_path.name}:")
        for f in files:
            print(f"  - {f}")
        
        # Load gravity/accelerometer data if available
        if 'Gravity.csv' in files:
            with zip_ref.open('Gravity.csv') as f:
                data['gravity'] = pd.read_csv(f)
        
        # Load gyroscope data if available
        if 'Gyroscope.csv' in files:
            with zip_ref.open('Gyroscope.csv') as f:
                data['gyroscope'] = pd.read_csv(f)
        
        # Load metadata
        if 'Metadata.csv' in files:
            with zip_ref.open('Metadata.csv') as f:
                data['metadata'] = pd.read_csv(f)
    
    return data


def plot_sensor_data(data, title):
    """Plot sensor data (gravity and gyroscope) over time."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot gravity/accelerometer data
    if 'gravity' in data:
        df = data['gravity']
        # Assume columns are: Time, X, Y, Z or similar
        time_col = df.columns[0]
        
        ax = axes[0]
        ax.plot(df[time_col], df.iloc[:, 1], label='X', alpha=0.7)
        ax.plot(df[time_col], df.iloc[:, 2], label='Y', alpha=0.7)
        ax.plot(df[time_col], df.iloc[:, 3], label='Z', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Gravity (m/s²)')
        ax.set_title(f'{title} - Gravity/Accelerometer')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot gyroscope data
    if 'gyroscope' in data:
        df = data['gyroscope']
        time_col = df.columns[0]
        
        ax = axes[1]
        ax.plot(df[time_col], df.iloc[:, 1], label='X', alpha=0.7)
        ax.plot(df[time_col], df.iloc[:, 2], label='Y', alpha=0.7)
        ax.plot(df[time_col], df.iloc[:, 3], label='Z', alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Angular velocity (rad/s)')
        ax.set_title(f'{title} - Gyroscope')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    # Get the raw_data directory
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    
    if not raw_data_dir.exists():
        print(f"Error: raw_data directory not found at {raw_data_dir}")
        sys.exit(1)
    
    # List all zip files
    zip_files = list(raw_data_dir.glob('*.zip'))
    
    if not zip_files:
        print("No zip files found in raw_data directory")
        sys.exit(1)
    
    print(f"Found {len(zip_files)} capture files\n")
    
    # Analyze the first file as an example
    example_file = zip_files[0]
    print(f"Analyzing: {example_file.name}\n")
    
    # Extract and load data
    data = extract_and_load_sensor_data(example_file)
    
    # Print data summary
    print("\nData summary:")
    for sensor, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{sensor}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Duration: {(df.iloc[-1, 0] - df.iloc[0, 0]) / 1000:.2f} seconds")
            print(f"  Sample rate: ~{len(df) / ((df.iloc[-1, 0] - df.iloc[0, 0]) / 1000):.1f} Hz")
    
    # Create visualization
    fig = plot_sensor_data(data, example_file.stem)
    
    # Save plot
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{example_file.stem}_visualization.png'
    fig.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    print("\n" + "="*50)
    print("Experiment 1 completed successfully!")
    print("="*50)
    print("\nKey observations to note:")
    print("1. Check the signal characteristics (noise, amplitude, patterns)")
    print("2. Look for periodic patterns that might correspond to steps/arrows")
    print("3. Consider which axes (X, Y, Z) are most relevant for dance pad detection")


if __name__ == '__main__':
    main()
