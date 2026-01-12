#!/usr/bin/env python3
"""
Train a CNN model to predict DDR arrow presses from sensor data.

This script:
1. Loads multiple captures and creates a dataset using create_dataset.py functions
2. Splits data into train/test sets
3. Trains a 1D CNN classifier to predict arrow labels from raw sensor time series
4. Evaluates accuracy using exact combination matching
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
import pickle
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import functions from existing scripts
from create_dataset import load_sensor_data, parse_sm_file, create_dataset
from align_clean import align_capture



def load_datasets_from_captures(capture_configs, window_size=1.0):
    """
    Load multiple captures and create unified dataset.
    
    Args:
        capture_configs: List of tuples (capture_path, sm_path, diff_level[, diff_type])
                        For backward compatibility, diff_type is optional and defaults to 'medium'.
                        - 3-tuple format: (capture_path, sm_path, diff_level)
                        - 4-tuple format: (capture_path, sm_path, diff_level, diff_type)
        window_size: Window size for samples (default 1.0s)
    
    Returns:
        X: Combined sensor data [N x window_samples x 9]
        Y: Combined arrow labels [N x 4]
    """
    all_X = []
    all_Y = []
    
    DEFAULT_DIFFICULTY_TYPE = 'medium'
    
    for i, config in enumerate(capture_configs):
        # Support both old format (3 args) and new format (4 args)
        if len(config) == 3:
            capture_path, sm_path, diff_level = config
            diff_type = DEFAULT_DIFFICULTY_TYPE
        else:
            capture_path, sm_path, diff_level, diff_type = config
        
        print(f"\n[{i+1}/{len(capture_configs)}] Processing: {Path(capture_path).name}")
        
        # Load sensor data
        print("  Loading sensor data...")
        t_sensor, sensors = load_sensor_data(capture_path)
        
        # Parse SM file
        print("  Parsing SM file...")
        t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level, diff_type=diff_type)
        
        # Align
        print("  Aligning...")
        result = align_capture(capture_path, sm_path, diff_level, diff_type=diff_type, verbose=False)
        offset = result['offset']
        
        # Create dataset
        print("  Creating dataset...")
        X, Y, _ = create_dataset(t_sensor, sensors, t_arrows, arrows, offset, window_size=window_size)
        
        print(f"  Samples: {len(X)}")
        # Skip captures with no samples
        if len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)
        else:
            print(f"  Warning: Skipping capture with 0 samples")
    
    # Combine all datasets
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    
    return X, Y


class ArrowCNN(nn.Module):
    """
    1D CNN for predicting arrow combinations from sensor time series.
    
    Architecture:
    - Input: [batch_size, 9 channels, time_steps]
    - 3 Conv1D layers with batch norm and max pooling
    - Fully connected layers
    - Output: Arrow classification [batch_size, 4] (probabilities for each arrow)
    """
    
    def __init__(self, input_channels=9, seq_length=198):
        super(ArrowCNN, self).__init__()
        
        # Convolutional layers (feature extraction)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate size after convolutions
        # seq_length -> seq_length/2 -> seq_length/4 -> seq_length/8
        conv_output_size = (seq_length // 8) * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        # Arrow classification output
        self.fc_arrows = nn.Linear(128, 4)  # 4 arrows
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Conv layers (feature extraction)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Arrow output
        arrows = self.sigmoid(self.fc_arrows(x))  # Sigmoid for multi-label classification
        
        return arrows


def train_cnn_model(X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32, lr=0.001):
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
        model: Trained CNN model
        history: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Reshape X: [N, time_steps, channels] -> [N, channels, time_steps]
    X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
    Y_train_t = torch.FloatTensor(Y_train)
    
    X_val_t = torch.FloatTensor(X_val).permute(0, 2, 1)
    Y_val_t = torch.FloatTensor(Y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ArrowCNN(input_channels=9, seq_length=X_train.shape[1])
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label classification
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_exact_match': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining CNN model (Arrow Classification)...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            arrows_out = model(batch_X)
            
            # Calculate loss
            loss = criterion(arrows_out, batch_Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                
                arrows_out = model(batch_X)
                
                loss = criterion(arrows_out, batch_Y)
                
                val_loss += loss.item() * batch_X.size(0)
                
                # Convert probabilities to binary predictions
                preds = (arrows_out > 0.5).float().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch_Y.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        exact_match = np.mean(np.all(all_preds == all_labels, axis=1))
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_exact_match'].append(exact_match)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Exact Match: {exact_match:.1%}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    
    return model, history


def evaluate_cnn_model(model, X_test, Y_test, batch_size=32):
    """
    Evaluate CNN model on test set.
    
    Args:
        model: Trained CNN model
        X_test: Test data [N x time_steps x 9]
        Y_test: Test arrow labels [N x 4]
        batch_size: Batch size for evaluation
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Reshape X: [N, time_steps, channels] -> [N, channels, time_steps]
    X_test_t = torch.FloatTensor(X_test).permute(0, 2, 1)
    Y_test_t = torch.FloatTensor(Y_test)
    
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_arrow_preds = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            arrows_out = model(batch_X)
            arrow_preds = (arrows_out > 0.5).float().cpu().numpy()
            all_arrow_preds.append(arrow_preds)
    
    Y_pred = np.vstack(all_arrow_preds)
    
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    
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
    print("  • TRUE: Nothing [0,0,0,0]    →  PREDICTED: Nothing [0,0,0,0]   ✓ CORRECT")
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
    
    # Count "nothing" samples
    num_nothing = np.sum(np.all(metrics['Y_test'] == 0, axis=1))
    print(f"  Nothing: {num_nothing}/{len(metrics['Y_test'])} ({num_nothing/len(metrics['Y_test'])*100:.1f}%)")
    
    # Show most common combinations
    print("\nMost Common Arrow Combinations in Test Set:")
    test_combo_counts = Counter(Y_test_tuples)
    arrow_names = metrics['arrow_names']
    for combo, count in test_combo_counts.most_common(10):
        combo_str = ' + '.join([arrow_names[i] for i, v in enumerate(combo) if v == 1])
        if not combo_str:
            combo_str = 'Nothing'
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
    print(f"   • Includes 'nothing pressed' samples (> 50ms from arrows)")
    print(f"   • Real-world noisy sensor data from mobile device")
    print(f"   • Complex human movement patterns with individual variations")
    
    print(f"\n3. PRACTICAL IMPLICATIONS:")
    if metrics['exact_match'] > 0.15:
        print(f"   • Model successfully predicts arrow combinations")
        print(f"   • Ready for real-world gameplay assistance")
        print(f"   • Can guide players on arrow patterns")
    elif metrics['exact_match'] > 0.10:
        print(f"   • Model shows promise in arrow prediction")
        print(f"   • Useful for gameplay guidance with some limitations")
        print(f"   • Further training may improve performance")
    else:
        print(f"   • Model needs improvement for practical use")
        print(f"   • Consider more training data or architecture changes")
    
    print(f"\n4. KEY CHANGES IN THIS VERSION:")
    print(f"   • ✓ Removed offset prediction (simpler classification task)")
    print(f"   • ✓ 50ms threshold: arrows within 50ms → labeled, else 'nothing'")
    print(f"   • ✓ Cleaner task: predict arrows OR nothing")
    print(f"   • ✓ More realistic: model learns when NOT to predict")
    
    print("="*70)


def save_model(model, filepath='trained_model.pth'):
    """Save trained CNN model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'ArrowCNN',
        'arrow_names': ['Left', 'Down', 'Up', 'Right']
    }, filepath)
    
    print(f"\nModel saved to: {filepath}")


def visualize_predictions(X_raw, Y_test, Y_pred, indices, output_dir='artifacts'):
    """
    Visualize sample predictions with input sensor data.
    
    Args:
        X_raw: Raw sensor data windows [N x time_steps x 9]
        Y_test: True arrow labels [N x 4]
        Y_pred: Predicted arrow labels [N x 4]
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
            
            # Mark window center
            ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Window center')
            
            if i == 0:  # Only show legend on first plot
                ax.legend(fontsize=7, loc='upper right')
        
        # Add prediction comparison panel
        ax_pred = fig.add_subplot(gs[4, :])
        ax_pred.axis('off')
        
        # Format labels
        true_arrows = [arrow_names[i] for i, v in enumerate(y_true) if v == 1]
        pred_arrows = [arrow_names[i] for i, v in enumerate(y_pred) if v == 1]
        
        true_str = ' + '.join(true_arrows) if true_arrows else 'Nothing'
        pred_str = ' + '.join(pred_arrows) if pred_arrows else 'Nothing'
        
        # Check if prediction is correct
        is_correct = np.all(y_true == y_pred)
        
        arrow_result = '✓ CORRECT' if is_correct else '✗ WRONG'
        
        # Create comparison table
        comparison_text = f"""
PREDICTION RESULTS:
  • Arrows: {arrow_result}

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
        comparison_text += f"\nCombined Arrow Labels:\n"
        comparison_text += f"  • TRUE:      {true_str}\n"
        comparison_text += f"  • PREDICTED: {pred_str}\n"
        
        # Color based on result
        bg_color = 'lightgreen' if is_correct else 'lightcoral'
        
        ax_pred.text(0.5, 0.5, comparison_text,
                    transform=ax_pred.transAxes,
                    fontsize=10,
                    family='monospace',
                    verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3))
        
        # Title
        overall_status = "CORRECT" if is_correct else "WRONG"
        fig.suptitle(f'Sample Prediction #{sample_idx + 1} (Test Index: {idx}) - {overall_status}',
                    fontsize=14, fontweight='bold')
        
        # Save
        output_path = output_dir / f'prediction_sample_{sample_idx + 1:02d}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def main():
    DEFAULT_DIFFICULTY_TYPE = 'medium'
    VALID_DIFFICULTY_TYPES = ['easy', 'medium', 'hard']
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python train_model.py <capture1_zip> <sm1_file> <diff1_level> [diff1_type] [<capture2_zip> <sm2_file> <diff2_level> [diff2_type] ...]")
        print("\nExample:")
        print("  python train_model.py \\")
        print("    'raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip' 'sm_files/Lucky Orb.sm' 5 medium \\")
        print("    'raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip' 'sm_files/Decorator.sm' 6 medium")
        sys.exit(1)
    
    # Parse arguments (groups of 3 or 4: capture_zip, sm_file, diff_level, [diff_type])
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
        
        # Check if next argument is difficulty type or next capture
        if i + 3 < len(args) and args[i+3].lower() in VALID_DIFFICULTY_TYPES:
            diff_type = args[i+3].lower()
            i += 4
        else:
            diff_type = DEFAULT_DIFFICULTY_TYPE
            i += 3
        
        capture_configs.append((capture, sm_file, diff_level, diff_type))
    
    print("="*70)
    print("DDR MACHINE LEARNING PIPELINE - CNN")
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
    
    # Count "nothing" vs arrow samples
    num_nothing = np.sum(np.all(Y == 0, axis=1))
    num_arrows = len(Y) - num_nothing
    print(f"Samples with arrows: {num_arrows} ({num_arrows/len(Y)*100:.1f}%)")
    print(f"Samples with nothing: {num_nothing} ({num_nothing/len(Y)*100:.1f}%)")
    
    # Split into train/val/test (60/20/20)
    print("\n" + "="*70)
    print("STEP 2: TRAIN/VAL/TEST SPLIT")
    print("="*70)
    
    # First split: 80% train+val, 20% test
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.25, random_state=42, stratify=None
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train CNN model
    print("\n" + "="*70)
    print("STEP 3: TRAINING CNN MODEL (Arrow Classification)")
    print("="*70)
    model, history = train_cnn_model(X_train, Y_train, 
                                     X_val, Y_val,
                                     epochs=100, batch_size=32, lr=0.001)
    
    # Plot training history
    print("\nPlotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plots
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['val_exact_match'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Exact Match Accuracy')
    axes[1].set_title('Validation Exact Match Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/training_history.png', dpi=150, bbox_inches='tight')
    print("  Saved: artifacts/training_history.png")
    plt.close()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 4: EVALUATION ON TEST SET")
    print("="*70)
    metrics = evaluate_cnn_model(model, X_test, Y_test)
    print_results(metrics, Y_train)
    
    # Save model
    save_model(model, filepath='artifacts/trained_model.pth')
    
    # Generate sample prediction visualizations
    print("\n" + "="*70)
    print("STEP 5: GENERATING SAMPLE PREDICTION VISUALIZATIONS")
    print("="*70)
    
    # Select 10 random samples from test set
    num_viz = min(10, len(X_test))
    print(f"\nGenerating {num_viz} random sample visualizations...")
    
    np.random.seed(42)
    viz_indices = np.random.choice(len(X_test), size=num_viz, replace=False)
    
    visualize_predictions(X_test, Y_test, metrics['Y_pred'], 
                         viz_indices, output_dir='artifacts')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n✓ Model saved to: artifacts/trained_model.pth")
    print(f"✓ Training history saved to: artifacts/training_history.png")
    print(f"✓ Sample predictions saved to: artifacts/prediction_sample_*.png")
    print("\n" + "="*70)
    print("KEY CHANGES: Removed offset prediction for simpler classification!")
    print("="*70)
    print("\nNew approach:")
    print("  • 50ms threshold: arrows within 50ms → labeled, else 'nothing'")
    print("  • Model learns when to predict and when not to predict")
    print("  • Cleaner classification task without regression complexity")
    print("="*70)


if __name__ == '__main__':
    main()
