#!/usr/bin/env python3
"""
Train a machine learning model to predict DDR arrow presses from sensor data.

This script:
1. Loads multiple captures and creates a dataset using create_dataset.py functions
2. Splits data into train/test sets
3. Trains a classifier to predict arrow labels
4. Evaluates accuracy and compares to random baseline
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.preprocessing import StandardScaler
import pickle

# Import functions from existing scripts
from create_dataset import load_sensor_data, parse_sm_file, create_dataset
from align_clean import align_capture


def load_datasets_from_captures(capture_configs, window_size=1.0):
    """
    Load multiple captures and create unified dataset.
    
    Args:
        capture_configs: List of tuples (capture_path, sm_path, diff_level)
        window_size: Window size for samples (default 1.0s)
    
    Returns:
        X: Combined sensor data [N x window_samples x 9]
        Y: Combined arrow labels [N x 4]
    """
    all_X = []
    all_Y = []
    
    for i, (capture_path, sm_path, diff_level) in enumerate(capture_configs):
        print(f"\n[{i+1}/{len(capture_configs)}] Processing: {Path(capture_path).name}")
        
        # Load sensor data
        print("  Loading sensor data...")
        t_sensor, sensors = load_sensor_data(capture_path)
        
        # Parse SM file
        print("  Parsing SM file...")
        t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level, diff_type='medium')
        
        # Align
        print("  Aligning...")
        result = align_capture(capture_path, sm_path, diff_level, diff_type='medium', verbose=False)
        offset = result['offset']
        
        # Create dataset
        print("  Creating dataset...")
        X, Y, _ = create_dataset(t_sensor, sensors, t_arrows, arrows, offset, window_size=window_size)
        
        print(f"  Samples: {len(X)}")
        all_X.append(X)
        all_Y.append(Y)
    
    # Combine all datasets
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    
    return X, Y


def prepare_features(X):
    """
    Extract features from raw sensor windows.
    
    For simplicity, we'll flatten the windows and use basic statistics.
    More advanced approaches could use CNN features.
    
    Args:
        X: Sensor windows [N x time_steps x 9_channels]
    
    Returns:
        X_features: Feature matrix [N x feature_dim]
    """
    N = X.shape[0]
    
    # Extract statistical features per channel
    features = []
    
    for i in range(N):
        sample_features = []
        
        # For each of 9 channels
        for ch in range(9):
            channel_data = X[i, :, ch]
            
            # Basic statistics
            sample_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        features.append(sample_features)
    
    return np.array(features)


def train_multilabel_model(X_train, Y_train):
    """
    Train a multi-label classifier for arrow prediction.
    
    Uses Binary Relevance with Random Forest for each arrow.
    
    Args:
        X_train: Training features
        Y_train: Training labels [N x 4]
    
    Returns:
        models: List of 4 trained models (one per arrow)
    """
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    models = []
    
    print("\nTraining models for each arrow...")
    for i, name in enumerate(arrow_names):
        print(f"  Training {name} arrow classifier...")
        
        # Train binary classifier for this arrow
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, Y_train[:, i])
        models.append(clf)
    
    return models


def evaluate_model(models, X_test, Y_test):
    """
    Evaluate multi-label model performance.
    
    Args:
        models: List of trained models
        X_test: Test features
        Y_test: Test labels
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    
    # Predict each arrow
    Y_pred = np.zeros_like(Y_test)
    
    for i, model in enumerate(models):
        Y_pred[:, i] = model.predict(X_test)
    
    # Compute metrics
    
    # Exact match accuracy (all 4 arrows must match)
    exact_match = np.mean(np.all(Y_pred == Y_test, axis=1))
    
    # Per-arrow accuracy
    per_arrow_acc = []
    for i in range(4):
        acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
        per_arrow_acc.append(acc)
    
    # Hamming loss (fraction of wrong labels)
    ham_loss = hamming_loss(Y_test, Y_pred)
    
    # Average accuracy across arrows
    avg_accuracy = np.mean(per_arrow_acc)
    
    metrics = {
        'exact_match': exact_match,
        'per_arrow_accuracy': per_arrow_acc,
        'average_accuracy': avg_accuracy,
        'hamming_loss': ham_loss,
        'arrow_names': arrow_names,
        'Y_pred': Y_pred,
        'Y_test': Y_test
    }
    
    return metrics


def print_results(metrics, Y_train):
    """Print evaluation results and compare to baseline."""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Calculate random baseline
    # For multi-label, we need to consider the label distribution
    label_dist = np.mean(Y_train, axis=0)
    
    # Random baseline: predict based on training distribution
    random_per_arrow = []
    for i, prob in enumerate(label_dist):
        # Accuracy when predicting randomly with correct probability
        random_acc = prob * prob + (1-prob) * (1-prob)
        random_per_arrow.append(random_acc)
    
    random_baseline = np.mean(random_per_arrow)
    
    print(f"\nBaseline (Random with training distribution): {random_baseline:.4f}")
    print(f"Model Average Accuracy: {metrics['average_accuracy']:.4f}")
    print(f"Improvement over baseline: {(metrics['average_accuracy'] - random_baseline):.4f}")
    
    print(f"\nExact Match Accuracy: {metrics['exact_match']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    
    print("\nPer-Arrow Accuracy:")
    for i, name in enumerate(metrics['arrow_names']):
        acc = metrics['per_arrow_accuracy'][i]
        baseline = random_per_arrow[i]
        improvement = acc - baseline
        print(f"  {name:5s}: {acc:.4f} (baseline: {baseline:.4f}, +{improvement:.4f})")
    
    # Check if better than random
    if metrics['average_accuracy'] > random_baseline:
        print("\n✓ Model performs BETTER than random baseline!")
    else:
        print("\n✗ Model does NOT outperform random baseline")
    
    # Distribution in test set
    print("\nLabel Distribution in Test Set:")
    for i, name in enumerate(metrics['arrow_names']):
        count = np.sum(metrics['Y_test'][:, i])
        total = len(metrics['Y_test'])
        print(f"  {name:5s}: {count}/{total} ({count/total*100:.1f}%)")
    
    print("="*70)


def save_model(models, scaler, filepath='trained_model.pkl'):
    """Save trained models and scaler."""
    model_data = {
        'models': models,
        'scaler': scaler,
        'arrow_names': ['Left', 'Down', 'Up', 'Right']
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {filepath}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python train_model.py <capture1_zip> <sm1_file> <diff1_level> [<capture2_zip> <sm2_file> <diff2_level> ...]")
        print("\nExample:")
        print("  python train_model.py \\")
        print("    'raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip' 'sm_files/Lucky Orb.sm' 5 \\")
        print("    'raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip' 'sm_files/Decorator.sm' 6")
        sys.exit(1)
    
    # Parse arguments (groups of 3)
    args = sys.argv[1:]
    if len(args) % 3 != 0:
        print("Error: Arguments must be in groups of 3 (capture_zip, sm_file, diff_level)")
        sys.exit(1)
    
    capture_configs = []
    for i in range(0, len(args), 3):
        capture_configs.append((args[i], args[i+1], int(args[i+2])))
    
    print("="*70)
    print("DDR MACHINE LEARNING PIPELINE")
    print("="*70)
    print(f"\nCaptures to process: {len(capture_configs)}")
    
    # Load datasets
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASETS")
    print("="*70)
    X, Y = load_datasets_from_captures(capture_configs, window_size=1.0)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"X shape: {X.shape} (samples x time_steps x channels)")
    print(f"Y shape: {Y.shape} (samples x arrows)")
    
    # Prepare features
    print("\n" + "="*70)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*70)
    print("Extracting statistical features from raw sensor windows...")
    X_features = prepare_features(X)
    print(f"Feature matrix shape: {X_features.shape}")
    
    # Split into train/test
    print("\n" + "="*70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("="*70)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_features, Y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    print("\n" + "="*70)
    print("STEP 4: TRAINING MODEL")
    print("="*70)
    models = train_multilabel_model(X_train, Y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("="*70)
    metrics = evaluate_model(models, X_test, Y_test)
    print_results(metrics, Y_train)
    
    # Save model
    save_model(models, scaler, filepath='artifacts/trained_model.pkl')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
