#!/usr/bin/env python3
"""
Example: Using visualize_arrows with ML model predictions.

This script demonstrates how to integrate the arrow visualization module
with machine learning predictions for DDR arrow detection.
"""

import sys
sys.path.insert(0, '.')
from visualize_arrows import extract_sm_window, visualize_arrows
import numpy as np


def simulate_ml_predictions(ground_truth_events, timing_error_std=0.1, miss_rate=0.1, random_seed=42):
    """
    Simulate ML model predictions with realistic errors.
    
    Args:
        ground_truth_events: List of ground truth events
        timing_error_std: Standard deviation of timing errors (in seconds)
        miss_rate: Probability of missing an arrow (0.0 to 1.0)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List of predicted events (same format as input)
    """
    np.random.seed(random_seed)
    predictions = []
    
    for event in ground_truth_events:
        # Skip this event with probability miss_rate
        if np.random.random() < miss_rate:
            continue
        
        # Add timing error (normally distributed)
        timing_error = np.random.normal(0, timing_error_std)
        pred_time = event['time'] + timing_error
        
        # Possibly miss individual arrows in a multi-arrow event
        pred_arrows = []
        for arrow in event['arrows']:
            if arrow == 1:
                # 90% chance to correctly predict this arrow
                pred_arrows.append(1 if np.random.random() > 0.1 else 0)
            else:
                # 5% chance of false positive
                pred_arrows.append(1 if np.random.random() < 0.05 else 0)
        
        # Only add event if at least one arrow is predicted
        if any(pred_arrows):
            predictions.append({
                'time': max(0, pred_time),  # Don't go negative
                'arrows': pred_arrows
            })
    
    return predictions


def example_ml_integration():
    """
    Example: Visualize ML predictions vs ground truth from .sm file.
    """
    print("="*70)
    print("EXAMPLE: ML PREDICTION VISUALIZATION")
    print("="*70)
    
    # Step 1: Extract ground truth from .sm file
    print("\n[1/3] Extracting ground truth from Lucky Orb...")
    sm_file = 'sm_files/Lucky Orb.sm'
    diff_level = 5
    start_time = 70.0  # Start at 70 seconds
    duration = 10.0
    
    ground_truth = extract_sm_window(
        sm_file, 
        diff_level, 
        'medium', 
        start_time=start_time,
        duration=duration
    )
    print(f"  Ground truth: {len(ground_truth)} arrow events")
    
    # Step 2: Simulate ML predictions
    print("\n[2/3] Simulating ML model predictions...")
    print("  (Adding realistic timing errors and occasional misses)")
    predictions = simulate_ml_predictions(
        ground_truth,
        timing_error_std=0.15,  # 150ms timing error std
        miss_rate=0.15  # 15% miss rate
    )
    print(f"  Predictions: {len(predictions)} arrow events")
    print(f"  Detected: {len(predictions)}/{len(ground_truth)} "
          f"({100*len(predictions)/len(ground_truth):.1f}%)")
    
    # Step 3: Visualize comparison
    print("\n[3/3] Creating visualization...")
    output_path = 'artifacts/ml_prediction_example.png'
    visualize_arrows(
        ground_truth,
        predictions,
        duration=duration,
        output_path=output_path,
        title1='Ground Truth (SM File)',
        title2='ML Model Predictions'
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Output: {output_path}")
    print("\nThis visualization shows:")
    print("  • Left: Ground truth arrow patterns from .sm file")
    print("  • Right: Simulated ML predictions with realistic errors")
    print("  • Useful for evaluating model performance visually")
    print("="*70)


def example_format_conversion():
    """
    Example: Convert numpy arrays (typical ML output) to visualization format.
    """
    print("\n" + "="*70)
    print("EXAMPLE: FORMAT CONVERSION (NumPy → Visualization)")
    print("="*70)
    
    # Typical ML model output format
    print("\nTypical ML output (numpy arrays):")
    pred_times = np.array([0.5, 1.2, 2.3, 3.1, 4.5])
    pred_arrows = np.array([
        [1, 0, 0, 0],  # Left
        [0, 1, 0, 0],  # Down
        [0, 0, 1, 0],  # Up
        [0, 0, 0, 1],  # Right
        [1, 0, 1, 0],  # Left+Up
    ])
    
    print(f"  pred_times shape: {pred_times.shape}")
    print(f"  pred_arrows shape: {pred_arrows.shape}")
    print(f"  Example: time={pred_times[0]}, arrows={pred_arrows[0]}")
    
    # Convert to visualization format
    print("\nConvert to visualization format:")
    events = []
    for t, arrows in zip(pred_times, pred_arrows):
        events.append({
            'time': float(t),
            'arrows': arrows.tolist()
        })
    
    print(f"  Converted to {len(events)} events")
    print(f"  Example: {events[0]}")
    print("\n✓ Ready to use with visualize_arrows()!")
    print("="*70)


if __name__ == '__main__':
    # Run ML integration example
    example_ml_integration()
    
    # Show format conversion example
    example_format_conversion()
