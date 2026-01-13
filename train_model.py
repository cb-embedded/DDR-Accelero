#!/usr/bin/env python3
"""
Train a CNN model to predict DDR arrow presses from sensor data using Keras.

This script:
1. Loads multiple captures and creates a dataset
2. Splits data into train/test sets
3. Trains a 1D CNN classifier to predict arrow labels from raw sensor time series
4. Evaluates accuracy using exact combination matching
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
import matplotlib.pyplot as plt

# Keras/TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Import functions from existing scripts
from core.dataset import load_sensor_data, parse_sm_file, create_dataset
from core.align import align_capture


def load_datasets_from_captures(capture_configs, window_size=1.0):
    """
    Load multiple captures and create unified dataset.
    
    Args:
        capture_configs: List of tuples (capture_path, sm_path, diff_level[, diff_type])
        window_size: Window size for samples (default 1.0s)
    
    Returns:
        X: Combined sensor data [N x window_samples x 9]
        Y: Combined arrow labels [N x 4]
    """
    all_X = []
    all_Y = []
    
    DEFAULT_DIFFICULTY_TYPE = 'medium'
    
    for i, config in enumerate(capture_configs):
        if len(config) == 3:
            capture_path, sm_path, diff_level = config
            diff_type = DEFAULT_DIFFICULTY_TYPE
        else:
            capture_path, sm_path, diff_level, diff_type = config
        
        print(f"\n[{i+1}/{len(capture_configs)}] Processing: {Path(capture_path).name}")
        
        print("  Loading sensor data...")
        t_sensor, sensors = load_sensor_data(capture_path)
        
        print("  Parsing SM file...")
        t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level, diff_type=diff_type)
        
        print("  Aligning...")
        result = align_capture(capture_path, sm_path, diff_level, diff_type=diff_type, verbose=False)
        offset = result['offset']
        
        print("  Creating balanced dataset...")
        X, Y, _ = create_dataset(t_sensor, sensors, t_arrows, arrows, offset, 
                                 window_size=window_size, balance_classes=True)
        
        print(f"  Samples: {len(X)}")
        if len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)
        else:
            print(f"  Warning: Skipping capture with 0 samples")
    
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    
    return X, Y


def create_arrow_cnn(input_shape=(198, 9)):
    """
    Create 1D CNN for predicting arrow combinations from sensor time series.
    
    Architecture:
    - Input: [time_steps, 9 channels]
    - 3 Conv1D layers with batch norm and max pooling
    - Fully connected layers
    - Output: 4 arrows (sigmoid for multi-label classification)
    
    Args:
        input_shape: Shape of input data (time_steps, channels)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Conv layer 1
        layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', 
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # Conv layer 2
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # Conv layer 3
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer - 4 arrows (sigmoid for independent predictions)
        layers.Dense(4, activation='sigmoid')
    ])
    
    return model


def exact_match_accuracy(y_true, y_pred):
    """
    Calculate exact match accuracy (all 4 arrows must match).
    
    Args:
        y_true: True labels [N x 4]
        y_pred: Predicted labels [N x 4]
    
    Returns:
        Exact match accuracy (0-1)
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    matches = np.all(y_true == y_pred_binary, axis=1)
    return np.mean(matches)


def train_model(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, lr=0.001):
    """
    Train CNN model for arrow prediction.
    
    Args:
        X_train: Training data [N x time_steps x 9]
        Y_train: Training arrow labels [N x 4]
        X_val: Validation data [N x time_steps x 9]
        Y_val: Validation arrow labels [N x 4]
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        model: Trained Keras model
        history: Training history dictionary
    """
    print(f"\nUsing TensorFlow/Keras {tf.__version__}")
    
    # Create model
    model = create_arrow_cnn(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Compile with binary crossentropy for multi-label classification
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train
    print(f"\nTraining for up to {epochs} epochs...")
    history_obj = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Calculate exact match accuracy for validation set
    Y_val_pred = model.predict(X_val, verbose=0)
    val_exact = exact_match_accuracy(Y_val, Y_val_pred)
    
    # Build history dictionary
    history = {
        'train_loss': history_obj.history['loss'],
        'val_loss': history_obj.history['val_loss'],
        'train_accuracy': history_obj.history['accuracy'],
        'val_accuracy': history_obj.history['val_accuracy'],
        'val_exact_match': val_exact
    }
    
    print(f"\nFinal validation exact match accuracy: {val_exact:.4f}")
    
    return model, history


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        X_test: Test data [N x time_steps x 9]
        Y_test: Test arrow labels [N x 4]
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating on test set...")
    
    # Get predictions
    Y_pred = model.predict(X_test, verbose=0)
    Y_pred_binary = (Y_pred > 0.5).astype(int)
    
    # Calculate metrics
    exact_match = exact_match_accuracy(Y_test, Y_pred)
    hamming = 1 - hamming_loss(Y_test, Y_pred_binary)
    
    # Per-arrow accuracy
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    per_arrow_acc = []
    for i in range(4):
        acc = accuracy_score(Y_test[:, i], Y_pred_binary[:, i])
        per_arrow_acc.append(acc)
    
    metrics = {
        'Y_pred': Y_pred_binary,
        'exact_match': exact_match,
        'hamming': hamming,
        'per_arrow_accuracy': per_arrow_acc,
        'arrow_names': arrow_names
    }
    
    return metrics


def print_results(metrics, Y_train):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"\nExact Match Accuracy: {metrics['exact_match']:.4f}")
    print(f"Hamming Score: {metrics['hamming']:.4f}")
    
    print("\nPer-Arrow Accuracy:")
    for name, acc in zip(metrics['arrow_names'], metrics['per_arrow_accuracy']):
        print(f"  {name:>6}: {acc:.4f}")
    
    # Class distribution
    num_nothing = np.sum(np.all(Y_train == 0, axis=1))
    num_arrows = len(Y_train) - num_nothing
    print(f"\nTraining set distribution:")
    print(f"  Arrows: {num_arrows} ({num_arrows/len(Y_train)*100:.1f}%)")
    print(f"  Nothing: {num_nothing} ({num_nothing/len(Y_train)*100:.1f}%)")


def save_model(model, filepath='artifacts/trained_model.h5'):
    """Save trained Keras model."""
    Path(filepath).parent.mkdir(exist_ok=True)
    model.save(filepath)
    print(f"\n✓ Model saved to: {filepath}")


def visualize_predictions(X_raw, Y_test, Y_pred, indices, output_dir='docs'):
    """
    Generate visualizations comparing predictions vs ground truth.
    
    Args:
        X_raw: Raw sensor data [N x time_steps x 9]
        Y_test: Ground truth labels [N x 4]
        Y_pred: Predicted labels [N x 4]
        indices: Indices of samples to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    
    for idx_num, sample_idx in enumerate(indices):
        fig, axes = plt.subplots(10, 1, figsize=(12, 14))
        
        # Plot each sensor channel
        for ch in range(9):
            axes[ch].plot(X_raw[sample_idx, :, ch], linewidth=0.8)
            axes[ch].set_ylabel(f'Ch{ch}')
            axes[ch].grid(True, alpha=0.3)
            axes[ch].set_xlim([0, X_raw.shape[1]])
        
        # Ground truth
        gt_text = ', '.join([arrow_names[i] for i in range(4) if Y_test[sample_idx, i] == 1])
        if not gt_text:
            gt_text = 'Nothing'
        axes[9].text(0.5, 0.7, f'Ground Truth: {gt_text}', 
                     ha='center', fontsize=12, weight='bold')
        
        # Prediction
        pred_text = ', '.join([arrow_names[i] for i in range(4) if Y_pred[sample_idx, i] == 1])
        if not pred_text:
            pred_text = 'Nothing'
        
        match = np.all(Y_test[sample_idx] == Y_pred[sample_idx])
        color = 'green' if match else 'red'
        axes[9].text(0.5, 0.3, f'Prediction: {pred_text}', 
                     ha='center', fontsize=12, weight='bold', color=color)
        axes[9].axis('off')
        
        plt.tight_layout()
        
        out_file = output_dir / f'prediction_sample_{idx_num+1:02d}.png'
        fig.savefig(out_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {out_file}")


def main():
    DEFAULT_DIFFICULTY_TYPE = 'medium'
    VALID_DIFFICULTY_TYPES = ['easy', 'medium', 'hard']
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python train_model.py <capture1_zip> <sm1_file> <diff1_level> [diff1_type] [...]")
        print("\nExample:")
        print("  python train_model.py \\")
        print("    'raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip' 'sm_files/Lucky Orb.sm' 5 \\")
        print("    'raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip' 'sm_files/DECORATOR.sm' 6")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    capture_configs = []
    i = 0
    while i < len(args):
        if i + 2 >= len(args):
            print(f"Error: Incomplete argument group at position {i}")
            sys.exit(1)
        
        capture = args[i]
        sm_file = args[i+1]
        diff_level = int(args[i+2])
        
        if i + 3 < len(args) and args[i+3].lower() in VALID_DIFFICULTY_TYPES:
            diff_type = args[i+3].lower()
            i += 4
        else:
            diff_type = DEFAULT_DIFFICULTY_TYPE
            i += 3
        
        capture_configs.append((capture, sm_file, diff_level, diff_type))
    
    print("="*70)
    print("DDR MACHINE LEARNING - KERAS CNN")
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
    
    num_nothing = np.sum(np.all(Y == 0, axis=1))
    num_arrows = len(Y) - num_nothing
    print(f"Samples with arrows: {num_arrows} ({num_arrows/len(Y)*100:.1f}%)")
    print(f"Samples with nothing: {num_nothing} ({num_nothing/len(Y)*100:.1f}%)")
    
    # Split data
    print("\n" + "="*70)
    print("STEP 2: TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.25, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\n" + "="*70)
    print("STEP 3: TRAINING KERAS CNN MODEL")
    print("="*70)
    model, history = train_model(X_train, Y_train, X_val, Y_val,
                                  epochs=100, batch_size=32, lr=0.001)
    
    # Plot training history
    print("\nPlotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_accuracy'], label='Train Acc')
    axes[1].plot(history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/training_history.png', dpi=150, bbox_inches='tight')
    print("  Saved: docs/training_history.png")
    plt.close()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 4: EVALUATION ON TEST SET")
    print("="*70)
    metrics = evaluate_model(model, X_test, Y_test)
    print_results(metrics, Y_train)
    
    # Save model
    save_model(model, filepath='artifacts/trained_model.h5')
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 5: GENERATING SAMPLE VISUALIZATIONS")
    print("="*70)
    
    num_viz = min(10, len(X_test))
    print(f"\nGenerating {num_viz} sample visualizations...")
    
    np.random.seed(42)
    viz_indices = np.random.choice(len(X_test), size=num_viz, replace=False)
    
    visualize_predictions(X_test, Y_test, metrics['Y_pred'],
                         viz_indices, output_dir='docs')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✓ Model saved to: artifacts/trained_model.h5")
    print(f"✓ Training history saved to: docs/training_history.png")
    print(f"✓ Sample predictions saved to: docs/prediction_sample_*.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
