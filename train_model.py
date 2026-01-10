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
import matplotlib.pyplot as plt

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
    
    # Calculate random baseline for EXACT MATCH (entire combination)
    # Count the frequency of each unique combination
    Y_train_tuples = [tuple(row) for row in Y_train]
    Y_test_tuples = [tuple(row) for row in metrics['Y_test']]
    
    from collections import Counter
    train_combo_counts = Counter(Y_train_tuples)
    
    # Calculate expected accuracy if randomly guessing based on training distribution
    exact_match_random_baseline = 0.0
    for combo in set(Y_test_tuples):
        # Probability of this combo in training set
        prob_train = train_combo_counts.get(combo, 0) / len(Y_train)
        # Count how many times this combo appears in test
        count_test = Y_test_tuples.count(combo)
        # Add to expected correct predictions
        exact_match_random_baseline += prob_train * count_test
    
    exact_match_random_baseline /= len(metrics['Y_test'])
    
    # CLEARER BASELINE EXPLANATION
    print("\n" + "-"*70)
    print("EVALUATION METRIC: EXACT COMBINATION MATCH")
    print("-"*70)
    print("\nIMPORTANT: We evaluate by predicting the EXACT arrow combination!")
    print("\nExamples:")
    print("  • TRUE: Left only [1,0,0,0]  →  PREDICTED: Left only [1,0,0,0]  ✓ CORRECT")
    print("  • TRUE: Left only [1,0,0,0]  →  PREDICTED: Left+Up [1,0,1,0]   ✗ WRONG")
    print("  • TRUE: Left+Up [1,0,1,0]    →  PREDICTED: Left+Up [1,0,1,0]   ✓ CORRECT")
    print("\nA prediction is correct ONLY if all 4 arrows match exactly.")
    print("This is much harder than per-arrow accuracy!")
    print("-"*70)
    
    # Main metric: Exact Match
    print(f"\n{'='*70}")
    print(f"PRIMARY METRIC: EXACT COMBINATION MATCH ACCURACY")
    print(f"{'='*70}")
    print(f"\nRandom Baseline (guessing by training frequency): {exact_match_random_baseline:.4f} ({exact_match_random_baseline*100:.1f}%)")
    print(f"Model Accuracy:                                   {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.1f}%)")
    print(f"Absolute Improvement:                             +{(metrics['exact_match'] - exact_match_random_baseline):.4f} (+{(metrics['exact_match'] - exact_match_random_baseline)*100:.1f}%)")
    
    if exact_match_random_baseline > 0:
        rel_improvement = (metrics['exact_match'] - exact_match_random_baseline) / exact_match_random_baseline * 100
        print(f"Relative Improvement:                             +{rel_improvement:.1f}%")
    
    # Check if better than random
    if metrics['exact_match'] > exact_match_random_baseline:
        print("\n✓✓✓ Model performs BETTER than random baseline!")
    else:
        print("\n✗✗✗ Model does NOT outperform random baseline")
    
    print(f"{'='*70}")
    
    # Distribution in test set
    print("\nLabel Distribution in Test Set:")
    for i, name in enumerate(metrics['arrow_names']):
        count = np.sum(metrics['Y_test'][:, i])
        total = len(metrics['Y_test'])
        print(f"  {name:5s}: {count}/{total} ({count/total*100:.1f}%)")
    
    # Show most common combinations
    print("\nMost Common Arrow Combinations in Test Set:")
    test_combo_counts = Counter(Y_test_tuples)
    arrow_names = metrics['arrow_names']
    for combo, count in test_combo_counts.most_common(10):
        combo_str = ' + '.join([arrow_names[i] for i, v in enumerate(combo) if v == 1])
        if not combo_str:
            combo_str = 'None'
        pct = count / len(metrics['Y_test']) * 100
        print(f"  {combo_str:20s}: {count:3d} ({pct:4.1f}%)")
    
    # ANALYSIS OF DOUBLE/MULTIPLE PRESSES
    print("\n" + "-"*70)
    print("ANALYSIS BY NUMBER OF SIMULTANEOUS ARROWS")
    print("-"*70)
    
    # Count samples by number of arrows
    Y_test = metrics['Y_test']
    Y_pred = metrics['Y_pred']
    
    num_arrows_true = np.sum(Y_test, axis=1)
    num_arrows_pred = np.sum(Y_pred, axis=1)
    
    print("\nDistribution of samples by number of simultaneous arrows:")
    for n in range(5):
        count = np.sum(num_arrows_true == n)
        if count > 0:
            pct = count / len(Y_test) * 100
            print(f"  {n} arrows: {count} samples ({pct:.1f}%)")
    
    # Accuracy for single vs double/multiple
    print("\nExact Match Accuracy by Number of Arrows:")
    for n in range(5):
        mask = num_arrows_true == n
        if np.sum(mask) > 0:
            acc = np.mean(np.all(Y_pred[mask] == Y_test[mask], axis=1))
            count = np.sum(mask)
            print(f"  {n} arrow(s): {acc:.1%} ({count} samples)")
    
    print("-"*70)
    
    # SECONDARY METRICS (for reference only)
    print("\n" + "-"*70)
    print("SECONDARY METRICS (for reference, not primary evaluation)")
    print("-"*70)
    
    # Calculate per-arrow random baseline
    label_dist = np.mean(Y_train, axis=0)
    random_per_arrow = []
    for i, prob in enumerate(label_dist):
        random_acc = prob * prob + (1-prob) * (1-prob)
        random_per_arrow.append(random_acc)
    
    per_arrow_random_baseline = np.mean(random_per_arrow)
    
    print(f"\nPer-Arrow Average Accuracy: {metrics['average_accuracy']:.4f}")
    print(f"  (Random baseline: {per_arrow_random_baseline:.4f})")
    print(f"\nHamming Loss (fraction of wrong arrows): {metrics['hamming_loss']:.4f}")
    
    print("\nPer-Arrow Breakdown:")
    for i, name in enumerate(metrics['arrow_names']):
        acc = metrics['per_arrow_accuracy'][i]
        baseline = random_per_arrow[i]
        print(f"  {name:5s}: {acc:.4f} (baseline: {baseline:.4f})")
    
    print("\nNote: Per-arrow metrics can be misleading because predicting")
    print("'not pressed' is often correct, inflating accuracy.")
    print("-"*70)
    
    # Add interpretation of results based on EXACT MATCH
    print("\n" + "="*70)
    print("RESULTS INTERPRETATION")
    print("="*70)
    
    # Calculate relative improvement for exact match
    if exact_match_random_baseline > 0:
        rel_improvement = (metrics['exact_match'] - exact_match_random_baseline) / exact_match_random_baseline * 100
    else:
        rel_improvement = float('inf') if metrics['exact_match'] > 0 else 0
    
    print(f"\n1. OVERALL PERFORMANCE (Exact Combination Match):")
    print(f"   • Exact Match Accuracy: {metrics['exact_match']:.1%}")
    print(f"   • Random Baseline: {exact_match_random_baseline:.1%}")
    if exact_match_random_baseline > 0:
        print(f"   • Relative improvement: +{rel_improvement:.1f}% over random")
    
    if rel_improvement > 100:
        print(f"   • Assessment: EXCELLENT - Strong predictive capability")
    elif rel_improvement > 50:
        print(f"   • Assessment: GOOD - Clear predictive signal")
    elif rel_improvement > 20:
        print(f"   • Assessment: MODERATE - Promising but needs improvement")
    else:
        print(f"   • Assessment: WEAK - Limited predictive capability")
    
    print(f"\n2. TASK DIFFICULTY:")
    print(f"   • Exact combination matching (all 4 arrows must be correct)")
    print(f"   • Real-world noisy sensor data from mobile device")
    print(f"   • No temporal alignment guarantees beyond biomechanical model")
    print(f"   • Complex human movement patterns with individual variations")
    
    print(f"\n3. CONTEXT & SIGNIFICANCE:")
    if metrics['exact_match'] > 0.10:
        print(f"   • Model demonstrates ability to predict full combinations")
        print(f"   • Learns meaningful patterns from accelerometer/gyro/mag data")
        print(f"   • Achievable with simple statistical features (no deep learning)")
    else:
        print(f"   • Exact matching is very challenging with current approach")
        print(f"   • Per-arrow predictions work better than full combinations")
        print(f"   • More sophisticated models (CNN/LSTM) may improve performance")
    
    print(f"\n4. COMBINATION-SPECIFIC INSIGHTS:")
    # Find which combinations are predicted best
    num_arrows_true = np.sum(Y_test, axis=1)
    print(f"   • Single arrow samples: {np.sum(num_arrows_true == 1)} ({np.sum(num_arrows_true == 1)/len(Y_test)*100:.1f}%)")
    if np.sum(num_arrows_true == 1) > 0:
        single_acc = np.mean(np.all(Y_pred[num_arrows_true == 1] == Y_test[num_arrows_true == 1], axis=1))
        print(f"     - Accuracy: {single_acc:.1%}")
    
    if np.sum(num_arrows_true >= 2) > 0:
        multi_acc = np.mean(np.all(Y_pred[num_arrows_true >= 2] == Y_test[num_arrows_true >= 2], axis=1))
        print(f"   • Multi-arrow samples: {np.sum(num_arrows_true >= 2)} ({np.sum(num_arrows_true >= 2)/len(Y_test)*100:.1f}%)")
        print(f"     - Accuracy: {multi_acc:.1%}")
    
    print(f"\n5. PRACTICAL IMPLICATIONS:")
    if metrics['exact_match'] > 0.15:
        print(f"   • Model shows promise for combination prediction")
        print(f"   • Could assist players with arrow pattern recognition")
        print(f"   • Ready for proof-of-concept testing")
    else:
        print(f"   • Current exact match accuracy is low")
        print(f"   • May be more useful for per-arrow guidance rather than full combinations")
        print(f"   • Consider hybrid approach or better features")
    
    print(f"\n6. NEXT STEPS FOR IMPROVEMENT:")
    print(f"   • Try CNN/LSTM to capture temporal patterns directly")
    print(f"   • Collect more diverse training data (different songs/players)")
    print(f"   • Engineer features specific to movement biomechanics")
    print(f"   • Apply data augmentation to increase training samples")
    print(f"   • Investigate ensemble methods or confidence thresholding")
    
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


def visualize_predictions(X_raw, X_features, Y_test, Y_pred, indices, output_dir='artifacts'):
    """
    Visualize sample predictions with input sensor data.
    
    Args:
        X_raw: Raw sensor data windows [N x time_steps x 9]
        X_features: Feature matrix [N x 54]
        Y_test: True labels [N x 4]
        Y_pred: Predicted labels [N x 4]
        indices: List of sample indices to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    channel_names = ['Accel X', 'Accel Y', 'Accel Z',
                     'Gyro X', 'Gyro Y', 'Gyro Z',
                     'Mag X', 'Mag Y', 'Mag Z']
    
    dt = 0.01  # 100 Hz sampling
    
    for sample_idx, idx in enumerate(indices):
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1, 0.8])
        
        # Get data
        sensor_window = X_raw[idx]  # [time_steps x 9]
        y_true = Y_test[idx]
        y_pred = Y_pred[idx]
        
        # Create time axis (centered at 0)
        n_samples = sensor_window.shape[0]
        t = np.arange(n_samples) * dt - (n_samples * dt / 2)
        
        # Plot 9 sensor channels
        for i in range(9):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t, sensor_window[:, i], linewidth=0.8, color='#1f77b4')
            ax.set_title(channel_names[i], fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add prediction comparison panel
        ax_pred = fig.add_subplot(gs[4, :])
        ax_pred.axis('off')
        
        # Format labels
        true_arrows = [arrow_names[i] for i, v in enumerate(y_true) if v == 1]
        pred_arrows = [arrow_names[i] for i, v in enumerate(y_pred) if v == 1]
        
        true_str = ' + '.join(true_arrows) if true_arrows else 'None'
        pred_str = ' + '.join(pred_arrows) if pred_arrows else 'None'
        
        # Check if prediction is correct
        is_correct = np.all(y_true == y_pred)
        result_color = 'green' if is_correct else 'red'
        result_text = '✓ CORRECT' if is_correct else '✗ WRONG'
        
        # Create comparison table
        comparison_text = f"""
PREDICTION RESULT: {result_text}

╔═══════════════════════════════════════════════════════════════╗
║  Arrow    │  True Label  │  Predicted   │  Match?            ║
╠═══════════════════════════════════════════════════════════════╣
"""
        
        for i, name in enumerate(arrow_names):
            true_val = '✓ YES' if y_true[i] == 1 else '✗ NO'
            pred_val = '✓ YES' if y_pred[i] == 1 else '✗ NO'
            match = '✓' if y_true[i] == y_pred[i] else '✗'
            comparison_text += f"║  {name:7s} │  {true_val:11s} │  {pred_val:11s} │  {match:17s} ║\n"
        
        comparison_text += "╚═══════════════════════════════════════════════════════════════╝\n"
        comparison_text += f"\nCombined Labels:\n"
        comparison_text += f"  • TRUE:      {true_str}\n"
        comparison_text += f"  • PREDICTED: {pred_str}"
        
        ax_pred.text(0.5, 0.5, comparison_text,
                    transform=ax_pred.transAxes,
                    fontsize=11,
                    family='monospace',
                    verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat' if is_correct else 'lightcoral', alpha=0.3))
        
        # Title
        fig.suptitle(f'Sample Prediction #{sample_idx + 1} (Test Index: {idx}) - {result_text}',
                    fontsize=14, fontweight='bold', color=result_color)
        
        # Save
        output_path = output_dir / f'prediction_sample_{sample_idx + 1:02d}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


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
    
    # Split both features and raw data to keep them aligned
    X_train_feat, X_test_feat, Y_train, Y_test, X_train_raw, X_test_raw = train_test_split(
        X_features, Y, X, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Training samples: {len(X_train_feat)}")
    print(f"Test samples: {len(X_test_feat)}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_feat)
    X_test_norm = scaler.transform(X_test_feat)
    
    # Train model
    print("\n" + "="*70)
    print("STEP 4: TRAINING MODEL")
    print("="*70)
    models = train_multilabel_model(X_train_norm, Y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("="*70)
    metrics = evaluate_model(models, X_test_norm, Y_test)
    print_results(metrics, Y_train)
    
    # Save model
    save_model(models, scaler, filepath='artifacts/trained_model.pkl')
    
    # Generate sample prediction visualizations
    print("\n" + "="*70)
    print("STEP 6: GENERATING SAMPLE PREDICTION VISUALIZATIONS")
    print("="*70)
    
    # Select 10 random samples from test set
    num_viz = min(10, len(X_test_raw))
    print(f"\nGenerating {num_viz} random sample visualizations...")
    
    np.random.seed(42)
    viz_indices = np.random.choice(len(X_test_raw), size=num_viz, replace=False)
    
    visualize_predictions(X_test_raw, X_test_feat, Y_test, metrics['Y_pred'], 
                         viz_indices, output_dir='artifacts')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✓ Model saved to: artifacts/trained_model.pkl")
    print(f"✓ Sample predictions saved to: artifacts/prediction_sample_*.png")
    print("="*70)


if __name__ == '__main__':
    main()
