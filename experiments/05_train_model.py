#!/usr/bin/env python3
"""
Experiment 5: Train a simple model to predict arrow presses from sensor data.

This script trains a basic machine learning model (Random Forest) to predict
which DDR arrows are pressed based on accelerometer/gyroscope features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys


def load_dataset(dataset_path):
    """Load the generated dataset."""
    df = pd.read_csv(dataset_path)
    
    # Separate features and labels
    label_cols = ['arrow_left', 'arrow_down', 'arrow_up', 'arrow_right']
    metadata_cols = ['time_sec', 'note_pattern']
    
    feature_cols = [col for col in df.columns if col not in label_cols + metadata_cols]
    
    X = df[feature_cols].values
    y = df[label_cols].values
    
    return X, y, feature_cols, label_cols, df


def train_models(X_train, y_train):
    """Train a separate binary classifier for each arrow."""
    models = {}
    arrow_names = ['left', 'down', 'up', 'right']
    
    for i, arrow in enumerate(arrow_names):
        print(f"\nTraining model for '{arrow}' arrow...")
        
        # Train a Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train[:, i])
        models[arrow] = model
        
        print(f"  Model trained with {len(model.estimators_)} trees")
    
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate the trained models."""
    arrow_names = ['left', 'down', 'up', 'right']
    
    print("\n" + "="*60)
    print("Model Evaluation Results")
    print("="*60)
    
    for i, arrow in enumerate(arrow_names):
        print(f"\n{arrow.upper()} Arrow:")
        print("-" * 40)
        
        model = models[arrow]
        y_true = y_test[:, i]
        y_pred = model.predict(X_test)
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              No    Yes")
        print(f"Actual No   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['No press', 'Press'], zero_division=0))
    
    # Overall metrics (any arrow pressed)
    print("\n" + "="*60)
    print("Overall Performance (Any Arrow)")
    print("="*60)
    
    # Any arrow pressed in ground truth
    any_true = (y_test.sum(axis=1) > 0).astype(int)
    
    # Any arrow predicted
    all_preds = np.array([models[arrow].predict(X_test) for arrow in arrow_names]).T
    any_pred = (all_preds.sum(axis=1) > 0).astype(int)
    
    acc = accuracy_score(any_true, any_pred)
    print(f"Accuracy: {acc:.3f}")
    
    cm = confusion_matrix(any_true, any_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")


def analyze_feature_importance(models, feature_names):
    """Analyze which features are most important."""
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)
    
    arrow_names = ['left', 'down', 'up', 'right']
    
    for arrow in arrow_names:
        model = models[arrow]
        importance = model.feature_importances_
        
        # Get top 10 features
        indices = np.argsort(importance)[::-1][:10]
        
        print(f"\n{arrow.upper()} - Top 10 Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")


def main():
    # Find the generated dataset
    output_dir = Path(__file__).parent / 'output'
    
    if not output_dir.exists():
        print("Error: No output directory found. Run experiment 4 first.")
        sys.exit(1)
    
    # Find dataset file
    dataset_files = list(output_dir.glob('*_dataset.csv'))
    
    if not dataset_files:
        print("Error: No dataset file found. Run experiment 4 first to generate the dataset.")
        sys.exit(1)
    
    dataset_file = dataset_files[0]
    print(f"Loading dataset: {dataset_file.name}")
    
    # Load data
    X, y, feature_names, label_names, df = load_dataset(dataset_file)
    
    print(f"\nDataset information:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Labels: {len(label_names)}")
    
    # Show class distribution
    print(f"\nClass distribution:")
    for i, label in enumerate(label_names):
        count = y[:, i].sum()
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y[:, 0]  # Stratify on first arrow
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train models
    print(f"\n" + "="*60)
    print("Training Models")
    print("="*60)
    models = train_models(X_train, y_train)
    
    # Evaluate
    evaluate_models(models, X_test, y_test)
    
    # Feature importance
    analyze_feature_importance(models, feature_names)
    
    print("\n" + "="*60)
    print("Experiment 5 completed!")
    print("="*60)
    print("\nKey observations:")
    print("1. Random Forest can learn patterns from sensor data")
    print("2. Performance depends heavily on:")
    print("   - Quality of time alignment (experiment 3)")
    print("   - Feature engineering (experiment 4)")
    print("   - Amount and diversity of training data")
    print("3. Feature importance shows which sensors/axes are most useful")
    print("\nNext steps for improvement:")
    print("- Collect more diverse training data (multiple songs, users)")
    print("- Try deep learning models (LSTM, CNN)")
    print("- Improve time alignment algorithm")
    print("- Add more sophisticated features (frequency domain, temporal patterns)")
    print("- Consider multi-task learning (predict arrow combinations)")


if __name__ == '__main__':
    main()
