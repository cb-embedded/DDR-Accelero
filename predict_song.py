#!/usr/bin/env python3
"""
Predict arrows for a song and compare with ground truth using Keras model.

This script:
1. Loads the trained Keras model
2. Makes predictions on a song
3. Compares predictions with ground truth using visualization
"""

import sys
import numpy as np
from pathlib import Path
from tensorflow import keras

# Import from existing modules
from core.dataset import load_sensor_data, parse_sm_file
from core.align import align_capture
from utils.visualize import extract_sm_window, visualize_arrows


def load_trained_model(model_path='artifacts/trained_model.h5'):
    """Load the trained Keras model."""
    model = keras.models.load_model(model_path)
    print(f"✓ Loaded trained model from: {model_path}")
    return model


def make_predictions_on_song(model, capture_path, sm_path, diff_level, 
                              start_time, duration, window_size=1.98):
    """
    Make predictions on a specific time window of a song.
    
    Args:
        model: Trained Keras model
        capture_path: Path to sensor capture zip
        sm_path: Path to .sm file
        diff_level: Difficulty level
        start_time: Start time in seconds for prediction window
        duration: Duration of prediction window in seconds
        window_size: Size of sliding window for predictions (default 1.98s = 198 samples)
    
    Returns:
        predictions: List of predicted arrow events (time, arrows)
        ground_truth: List of ground truth arrow events (time, arrows)
    """
    print("\n[1/5] Loading sensor data...")
    t_sensor, sensors = load_sensor_data(capture_path)
    
    print("[2/5] Parsing SM file...")
    t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level, diff_type='medium')
    
    print("[3/5] Aligning...")
    result = align_capture(capture_path, sm_path, diff_level, diff_type='medium', verbose=False)
    offset = result['offset']
    print(f"  Offset: {offset:.3f}s")
    
    # Extract ground truth in prediction window
    print(f"[4/5] Extracting ground truth ({start_time:.1f}s - {start_time+duration:.1f}s)...")
    t_adjusted = t_arrows + offset
    
    mask = (t_adjusted >= start_time) & (t_adjusted < start_time + duration)
    ground_truth = [(t_adjusted[i], arrows[i]) for i in range(len(t_arrows)) if mask[i]]
    print(f"  Ground truth arrows: {len(ground_truth)}")
    
    # Make predictions
    print("[5/5] Making predictions...")
    
    # Convert sensors dictionary to 2D numpy array
    sensor_array = np.column_stack([
        sensors['acc_x'], sensors['acc_y'], sensors['acc_z'],
        sensors['gyro_x'], sensors['gyro_y'], sensors['gyro_z'],
        sensors['mag_x'], sensors['mag_y'], sensors['mag_z']
    ])
    
    dt = 0.01
    num_samples = int(window_size / dt)
    
    predictions = []
    t_current = start_time
    
    while t_current + window_size <= start_time + duration:
        # Extract sensor window
        mask = (t_sensor >= t_current) & (t_sensor < t_current + window_size)
        if np.sum(mask) < num_samples:
            t_current += 0.1
            continue
        
        sensor_window = sensor_array[mask][:num_samples]
        
        if len(sensor_window) < num_samples:
            t_current += 0.1
            continue
        
        # Make prediction
        X = sensor_window.reshape(1, num_samples, 9)
        pred = model.predict(X, verbose=0)[0]
        pred_binary = (pred > 0.3).astype(int)  # Lower threshold for more predictions
        
        # If any arrow predicted, add to predictions
        if np.any(pred_binary):
            t_pred = t_current + window_size / 2
            predictions.append((t_pred, pred_binary))
        
        t_current += 0.1
    
    print(f"  Raw predictions: {len(predictions)}")
    
    # Filter predictions by offset to avoid duplicates
    filtered_predictions = []
    if len(predictions) > 0:
        filtered_predictions.append(predictions[0])
        
        for t, arrows in predictions[1:]:
            if t - filtered_predictions[-1][0] > 0.3:
                filtered_predictions.append((t, arrows))
    
    print(f"  Filtered predictions: {len(filtered_predictions)}")
    
    return filtered_predictions, ground_truth


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nUsage: python predict_song.py <capture_zip> <sm_file> <diff_level> [start_time] [duration]")
        print("\nExample:")
        print("  python predict_song.py \\")
        print("    'raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip' \\")
        print("    'sm_files/Lucky Orb.sm' 5 70.0 10.0")
        sys.exit(1)
    
    capture_path = sys.argv[1]
    sm_path = sys.argv[2]
    diff_level = int(sys.argv[3])
    start_time = float(sys.argv[4]) if len(sys.argv) > 4 else 70.0
    duration = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    
    print("="*70)
    print("DDR PREDICTION - KERAS CNN")
    print("="*70)
    print(f"\nCapture: {Path(capture_path).name}")
    print(f"SM file: {Path(sm_path).name}")
    print(f"Difficulty: {diff_level}")
    print(f"Prediction window: {start_time:.1f}s - {start_time+duration:.1f}s")
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    model = load_trained_model('artifacts/trained_model.h5')
    
    # Make predictions
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    predictions, ground_truth = make_predictions_on_song(
        model, capture_path, sm_path, diff_level, start_time, duration
    )
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    
    output_dir = Path('docs')
    output_dir.mkdir(exist_ok=True)
    
    song_name = Path(sm_path).stem.replace(' ', '_')
    output_file = output_dir / f'{song_name}_prediction_comparison.png'
    
    # Convert tuples to dictionaries for visualization
    gt_dict = [{'time': t, 'arrows': arr} for t, arr in ground_truth]
    pred_dict = [{'time': t, 'arrows': arr} for t, arr in predictions]
    
    visualize_arrows(gt_dict, pred_dict, output_path=str(output_file))
    
    print(f"\n✓ Visualization saved to: {output_file}")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"\nGround truth arrows: {len(ground_truth)}")
    print(f"Predicted arrows: {len(predictions)}")
    print("="*70)


if __name__ == '__main__':
    main()
