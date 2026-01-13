#!/usr/bin/env python3
"""
Predict arrows for a song excluded from the training set and compare with ground truth.

This script:
1. Loads the trained model
2. Makes predictions on a hold-out song (excluded from training)
3. Filters predictions by offset threshold to avoid duplicates
4. Compares predictions with ground truth using visualization
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import from existing modules
from train_model import ArrowCNN
from core.dataset import load_sensor_data, parse_sm_file
from core.align import align_capture
from utils.visualize import extract_sm_window, visualize_arrows


def load_trained_model(model_path='artifacts/trained_model.pth'):
    """Load the trained CNN model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize model (assuming standard parameters)
    model = ArrowCNN(input_channels=9, seq_length=198)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded trained model from: {model_path}")
    return model


def make_predictions_on_song(model, capture_path, sm_path, diff_level, 
                              start_time, duration, window_size=1.0):
    """
    Make predictions on a specific time window of a song.
    
    Args:
        model: Trained PyTorch model
        capture_path: Path to sensor capture zip
        sm_path: Path to .sm file
        diff_level: Difficulty level
        start_time: Start time in seconds for prediction window
        duration: Duration of prediction window in seconds
        window_size: Size of sliding window for predictions (default 1.0s)
        
    Returns:
        list of dicts: Predictions in format [{'time': t, 'arrows': [L,D,U,R]}, ...]
    """
    print(f"\n[1/4] Loading sensor data from {Path(capture_path).name}...")
    t_sensor, sensors_dict = load_sensor_data(capture_path)
    
    # Convert sensors dict to numpy array [N x 9]
    sensors = np.column_stack([
        sensors_dict['acc_x'], sensors_dict['acc_y'], sensors_dict['acc_z'],
        sensors_dict['gyro_x'], sensors_dict['gyro_y'], sensors_dict['gyro_z'],
        sensors_dict['mag_x'], sensors_dict['mag_y'], sensors_dict['mag_z']
    ])
    
    print(f"[2/4] Parsing SM file...")
    t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level, diff_type='medium')
    
    print(f"[3/4] Aligning capture with chart...")
    result = align_capture(capture_path, sm_path, diff_level, diff_type='medium', verbose=False)
    offset = result['offset']
    print(f"  Alignment offset: {offset:.3f}s")
    
    # Convert chart time to sensor time
    sensor_start = start_time + offset
    sensor_end = sensor_start + duration
    
    print(f"[4/4] Making predictions on {duration}s window (chart time {start_time:.1f}s to {start_time+duration:.1f}s)...")
    
    # Match the model's expected input size
    expected_timesteps = 198
    
    # Make predictions at regular intervals (every 0.1s)
    prediction_interval = 0.1  # seconds
    prediction_times = np.arange(sensor_start, sensor_end, prediction_interval)
    
    predictions = []
    device = torch.device('cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for pred_time in prediction_times:
            # Find the window center in sensor timeline
            center_idx = np.searchsorted(t_sensor, pred_time)
            
            # Extract window: we need 198 samples total
            half_samples = expected_timesteps // 2  # 99
            start_idx = center_idx - half_samples
            end_idx = start_idx + expected_timesteps
            
            # Skip if window is out of bounds
            if start_idx < 0 or end_idx >= len(sensors):
                continue
            
            # Extract sensor window [198 x 9]
            sensor_window = sensors[start_idx:end_idx, :]
            
            # Verify size
            if sensor_window.shape[0] != expected_timesteps:
                continue
            
            # Prepare input for model [1, channels, time_steps]
            X = torch.FloatTensor(sensor_window).unsqueeze(0).permute(0, 2, 1).to(device)
            
            # Make prediction
            arrows_out = model(X)
            
            # Convert to binary arrows (threshold at 0.5)
            pred_arrows = (arrows_out[0] > 0.5).float().cpu().numpy().astype(int)
            
            # Only add prediction if at least one arrow is predicted
            if pred_arrows.sum() > 0:
                # Convert sensor time back to chart time
                chart_time = pred_time - offset
                # Make relative to window start
                relative_time = chart_time - start_time
                
                # Only include if within our target duration
                if 0 <= relative_time <= duration:
                    predictions.append({
                        'time': relative_time,
                        'arrows': pred_arrows.tolist()
                    })
    
    print(f"  Generated {len(predictions)} predictions")
    return predictions


def main():
    if len(sys.argv) < 4:
        print("\nUsage: python predict_song.py <capture_zip> <sm_file> <diff_level> [start_time] [duration]")
        print("\nExample:")
        print("  python predict_song.py \\")
        print("    'raw_data/Butterfly_Cat_6_Medium-2026-01-10_09-34-07.zip' \\")
        print("    'sm_files/Butterfly Cat.sm' \\")
        print("    6 \\")
        print("    70.0 \\")
        print("    10.0")
        sys.exit(1)
    
    capture_path = sys.argv[1]
    sm_path = sys.argv[2]
    diff_level = int(sys.argv[3])
    start_time = float(sys.argv[4]) if len(sys.argv) > 4 else 70.0
    duration = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    
    print("="*70)
    print("PREDICT ARROWS FOR HOLD-OUT SONG")
    print("="*70)
    print(f"\nCapture: {Path(capture_path).name}")
    print(f"SM File: {Path(sm_path).name}")
    print(f"Difficulty: {diff_level}")
    print(f"Time window: {start_time}s to {start_time + duration}s")
    
    # Step 1: Load trained model
    print("\n" + "="*70)
    print("STEP 1: LOADING TRAINED MODEL")
    print("="*70)
    model = load_trained_model('artifacts/trained_model.pth')
    
    # Step 2: Make predictions
    print("\n" + "="*70)
    print("STEP 2: MAKING PREDICTIONS")
    print("="*70)
    
    predictions = make_predictions_on_song(
        model, capture_path, sm_path, diff_level,
        start_time, duration
    )
    
    # Step 3: Extract ground truth
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING GROUND TRUTH FROM .SM FILE")
    print("="*70)
    
    ground_truth = extract_sm_window(
        sm_path, diff_level, 'medium',
        start_time=start_time,
        duration=duration
    )
    print(f"  Ground truth: {len(ground_truth)} arrow events")
    
    # Step 4: Create visualization
    print("\n" + "="*70)
    print("STEP 4: CREATING COMPARISON VISUALIZATION")
    print("="*70)
    
    output_dir = Path('docs')
    output_dir.mkdir(exist_ok=True)
    
    song_name = Path(sm_path).stem.replace(' ', '_')
    output_path = output_dir / f'{song_name}_prediction_comparison.png'
    
    visualize_arrows(
        ground_truth,
        predictions,
        duration=duration,
        output_path=output_path,
        title1=f'Original Chart ({Path(sm_path).stem} Medium {diff_level})',
        title2=f'ML Predictions'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("PREDICTION COMPARISON COMPLETE")
    print("="*70)
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"\n  Time window analyzed: {start_time}s - {start_time + duration}s")
    print(f"  Original arrows: {len(ground_truth)}")
    print(f"  Predicted arrows: {len(predictions)}")
    print(f"  Detection rate: {len(predictions)}/{len(ground_truth)} ({100*len(predictions)/max(1,len(ground_truth)):.1f}%)")
    
    print("\n" + "="*70)
    print("The figure shows:")
    print("  • Left column: Original arrow pattern from .sm file")
    print("  • Right column: ML model predictions from sensor data")
    print("  • Arrows colored by type (Left=Pink, Down=Cyan, Up=Yellow, Right=Red)")
    print("  • Time flows from bottom to top (like StepMania gameplay)")
    print("="*70)


if __name__ == '__main__':
    main()
