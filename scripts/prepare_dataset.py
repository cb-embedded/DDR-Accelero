#!/usr/bin/env python3
"""
Prepare final dataset with feature extraction and train/test split.

This script:
1. Loads synchronized sensor data
2. Extracts features from time windows
3. Splits data into train/validation/test sets
4. Saves final dataset in NumPy format
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import signal, fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_synchronized_data(features_dir: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all synchronized session data.
    
    Args:
        features_dir: Directory containing synchronized CSV files
        
    Returns:
        List of (session_name, DataFrame) tuples
    """
    sessions = []
    
    for filename in os.listdir(features_dir):
        if filename.endswith('_sync.csv'):
            session_name = filename.replace('_sync.csv', '')
            filepath = os.path.join(features_dir, filename)
            df = pd.read_csv(filepath)
            sessions.append((session_name, df))
            print(f"Loaded session '{session_name}': {len(df)} samples")
    
    return sessions


def compute_magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute magnitude of 3D vector."""
    return np.sqrt(x**2 + y**2 + z**2)


def compute_jerk(values: np.ndarray, dt: float) -> np.ndarray:
    """Compute jerk (rate of change of acceleration)."""
    return np.gradient(values, dt)


def extract_time_domain_features(window: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """
    Extract time-domain features from a window.
    
    Args:
        window: DataFrame window with sensor data
        prefix: Prefix for feature names (e.g., 'acc', 'gyro', 'mag')
        
    Returns:
        Dictionary of feature names to values
    """
    features = {}
    
    for axis in ['x', 'y', 'z']:
        col = f"{prefix}_{axis}"
        if col in window.columns:
            values = window[col].values
            
            features[f"{col}_mean"] = np.mean(values)
            features[f"{col}_std"] = np.std(values)
            features[f"{col}_min"] = np.min(values)
            features[f"{col}_max"] = np.max(values)
    
    # Compute magnitude features
    if all(f"{prefix}_{axis}" in window.columns for axis in ['x', 'y', 'z']):
        mag = compute_magnitude(
            window[f"{prefix}_x"].values,
            window[f"{prefix}_y"].values,
            window[f"{prefix}_z"].values
        )
        features[f"{prefix}_mag_mean"] = np.mean(mag)
        features[f"{prefix}_mag_std"] = np.std(mag)
        features[f"{prefix}_mag_max"] = np.max(mag)
    
    return features


def extract_frequency_features(window: pd.DataFrame, prefix: str, 
                               n_coeffs: int, sampling_rate: float) -> Dict[str, float]:
    """
    Extract frequency-domain features using FFT.
    
    Args:
        window: DataFrame window with sensor data
        prefix: Prefix for feature names
        n_coeffs: Number of FFT coefficients to extract
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of feature names to values
    """
    features = {}
    
    for axis in ['x', 'y', 'z']:
        col = f"{prefix}_{axis}"
        if col in window.columns:
            values = window[col].values
            
            # Compute FFT
            fft_vals = np.abs(fft.rfft(values))
            
            # Take first n_coeffs (excluding DC component)
            for i in range(min(n_coeffs, len(fft_vals) - 1)):
                features[f"{col}_fft_{i}"] = fft_vals[i + 1]
    
    return features


def extract_features_from_window(window: pd.DataFrame, config: Dict, 
                                 sampling_rate: float) -> Dict[str, float]:
    """
    Extract all features from a single window.
    
    Args:
        window: DataFrame window with sensor data
        config: Configuration dictionary
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of feature names to values
    """
    features = {}
    
    # Time-domain features
    for prefix in ['acc', 'gyro', 'mag']:
        if any(col.startswith(prefix) for col in window.columns):
            time_features = extract_time_domain_features(window, prefix)
            features.update(time_features)
    
    # Frequency-domain features
    if config['features']['frequency_domain']:
        n_coeffs = config['features']['frequency_domain'][0]['fft_coefficients']
        for prefix in ['acc', 'gyro', 'mag']:
            if any(col.startswith(prefix) for col in window.columns):
                freq_features = extract_frequency_features(
                    window, prefix, n_coeffs, sampling_rate
                )
                features.update(freq_features)
    
    # Jerk features
    if config['features']['compute_jerk']:
        dt = 1.0 / sampling_rate
        for axis in ['x', 'y', 'z']:
            col = f"acc_{axis}"
            if col in window.columns:
                jerk = compute_jerk(window[col].values, dt)
                features[f"jerk_{axis}_mean"] = np.mean(jerk)
                features[f"jerk_{axis}_std"] = np.std(jerk)
    
    return features


def create_windows(df: pd.DataFrame, window_size_ms: float, 
                  overlap: float, sampling_rate: float) -> List[Tuple[pd.DataFrame, int]]:
    """
    Create sliding windows from data.
    
    Args:
        df: DataFrame with synchronized sensor data
        window_size_ms: Window size in milliseconds
        overlap: Overlap fraction (0-1)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        List of (window_df, label) tuples
    """
    window_size_s = window_size_ms / 1000.0
    window_samples = int(window_size_s * sampling_rate)
    step_samples = int(window_samples * (1 - overlap))
    
    windows = []
    
    for start_idx in range(0, len(df) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window = df.iloc[start_idx:end_idx]
        
        # Use majority label for the window
        label = window['label'].mode()[0] if 'label' in window.columns else 0
        
        windows.append((window, label))
    
    return windows


def extract_features_from_session(session_name: str, df: pd.DataFrame, 
                                  config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a session.
    
    Args:
        session_name: Name of the session
        df: Synchronized DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (features array, labels array)
    """
    print(f"\nExtracting features from {session_name}...")
    
    window_size_ms = config['sensors']['sampling']['window_size_ms']
    overlap = config['sensors']['sampling']['window_overlap']
    sampling_rate = config['sensors']['sampling']['target_rate_hz']
    
    # Create windows
    windows = create_windows(df, window_size_ms, overlap, sampling_rate)
    print(f"  Created {len(windows)} windows")
    
    # Extract features from each window
    feature_list = []
    label_list = []
    
    for window_df, label in windows:
        features = extract_features_from_window(window_df, config, sampling_rate)
        feature_list.append(features)
        label_list.append(label)
    
    # Convert to arrays
    if feature_list:
        # Get consistent feature names
        feature_names = sorted(feature_list[0].keys())
        
        # Create feature matrix
        X = np.array([[f[name] for name in feature_names] for f in feature_list])
        y = np.array(label_list)
        
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Label distribution: {np.bincount(y)}")
        
        return X, y, feature_names
    else:
        return np.array([]), np.array([]), []


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                      method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Normalize features.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Tuple of (X_train_norm, X_val_norm, X_test_norm, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def split_dataset(sessions_data: List[Tuple[str, np.ndarray, np.ndarray]], 
                 config: Dict) -> Dict:
    """
    Split dataset into train/val/test sets.
    
    Args:
        sessions_data: List of (session_name, X, y) tuples
        config: Configuration dictionary
        
    Returns:
        Dictionary with split datasets
    """
    print(f"\nSplitting dataset...")
    
    split_config = config['split']
    train_ratio = split_config['train']
    val_ratio = split_config['val']
    test_ratio = split_config['test']
    random_seed = split_config['random_seed']
    
    if split_config['split_by_session']:
        # Split by session to avoid data leakage
        n_sessions = len(sessions_data)
        session_indices = np.arange(n_sessions)
        
        # Split into train+val and test
        train_val_idx, test_idx = train_test_split(
            session_indices,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=random_seed
        )
        
        # Combine data from sessions
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []
        
        for idx in train_idx:
            _, X, y = sessions_data[idx]
            X_train_list.append(X)
            y_train_list.append(y)
        
        for idx in val_idx:
            _, X, y = sessions_data[idx]
            X_val_list.append(X)
            y_val_list.append(y)
        
        for idx in test_idx:
            _, X, y = sessions_data[idx]
            X_test_list.append(X)
            y_test_list.append(y)
        
        X_train = np.vstack(X_train_list) if X_train_list else np.array([])
        y_train = np.concatenate(y_train_list) if y_train_list else np.array([])
        X_val = np.vstack(X_val_list) if X_val_list else np.array([])
        y_val = np.concatenate(y_val_list) if y_val_list else np.array([])
        X_test = np.vstack(X_test_list) if X_test_list else np.array([])
        y_test = np.concatenate(y_test_list) if y_test_list else np.array([])
        
    else:
        # Split randomly across all samples
        X = np.vstack([X for _, X, _ in sessions_data])
        y = np.concatenate([y for _, _, y in sessions_data])
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_seed, stratify=y
        )
        
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted,
            random_state=random_seed, stratify=y_train_val
        )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def save_dataset(dataset: Dict, feature_names: List[str], config: Dict) -> None:
    """
    Save final dataset to disk.
    
    Args:
        dataset: Dictionary with train/val/test splits
        feature_names: List of feature names
        config: Configuration dictionary
    """
    output_dir = config['data']['splits_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize if configured
    if config['cleaning']['normalize']:
        method = config['cleaning']['normalization_method']
        print(f"\nNormalizing features using {method} method...")
        
        X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
            dataset['X_train'],
            dataset['X_val'],
            dataset['X_test'],
            method
        )
        
        dataset['X_train'] = X_train_norm
        dataset['X_val'] = X_val_norm
        dataset['X_test'] = X_test_norm
    
    # Save as NPZ file
    output_path = os.path.join(output_dir, 'dataset.npz')
    np.savez(
        output_path,
        X_train=dataset['X_train'],
        y_train=dataset['y_train'],
        X_val=dataset['X_val'],
        y_val=dataset['y_val'],
        X_test=dataset['X_test'],
        y_test=dataset['y_test']
    )
    print(f"\nSaved dataset to {output_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved feature names to {feature_names_path}")
    
    # Save metadata
    metadata = {
        'n_features': len(feature_names),
        'n_classes': len(config['arrows']['classes']) + 1,  # +1 for NONE
        'class_names': ['NONE'] + config['arrows']['classes'],
        'train_samples': len(dataset['y_train']),
        'val_samples': len(dataset['y_val']),
        'test_samples': len(dataset['y_test']),
        'train_label_dist': np.bincount(dataset['y_train']).tolist(),
        'val_label_dist': np.bincount(dataset['y_val']).tolist(),
        'test_label_dist': np.bincount(dataset['y_test']).tolist(),
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare final dataset with feature extraction and splitting'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load all synchronized sessions
    features_dir = config['data']['features_dir']
    sessions = load_synchronized_data(features_dir)
    
    if not sessions:
        print("ERROR: No synchronized sessions found")
        return
    
    # Extract features from each session
    sessions_data = []
    feature_names = None
    
    for session_name, df in sessions:
        X, y, names = extract_features_from_session(session_name, df, config)
        if len(X) > 0:
            sessions_data.append((session_name, X, y))
            if feature_names is None:
                feature_names = names
    
    if not sessions_data:
        print("ERROR: No features extracted")
        return
    
    # Split dataset
    dataset = split_dataset(sessions_data, config)
    
    # Save dataset
    save_dataset(dataset, feature_names, config)
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
