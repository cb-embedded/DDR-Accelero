#!/usr/bin/env python3
"""
Generate additional visualization examples showing different prediction scenarios.

This demonstrates the visualization tool's ability to show various types of
prediction errors and patterns that might occur in real ML predictions.
"""

import sys
sys.path.insert(0, '.')
from visualize_arrows import extract_sm_window, visualize_arrows
from example_ml_integration import simulate_ml_predictions


def generate_example_1_high_accuracy():
    """Example 1: High accuracy predictions (90% detection, low timing error)"""
    print("\n[Example 1] High Accuracy Predictions...")
    
    # Extract from Lucky Orb
    ground_truth = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', 
                                      start_time=50.0, duration=15.0)
    
    # Simulate high accuracy predictions
    predictions = simulate_ml_predictions(
        ground_truth,
        timing_error_std=0.05,  # Very low timing error (50ms)
        miss_rate=0.10,  # Only 10% miss rate
        random_seed=10
    )
    
    visualize_arrows(
        ground_truth,
        predictions,
        duration=15.0,
        output_path='artifacts/example_1_high_accuracy.png',
        title1='Ground Truth',
        title2='High Accuracy Predictions (90% detect, 50ms error)'
    )
    
    print(f"  Detected: {len(predictions)}/{len(ground_truth)} "
          f"({100*len(predictions)/len(ground_truth):.1f}%)")


def generate_example_2_timing_errors():
    """Example 2: Good detection but significant timing errors"""
    print("\n[Example 2] Timing Errors (good detection, poor timing)...")
    
    # Extract from Seyana
    ground_truth = extract_sm_window('sm_files/Seyana.sm', 6, 'medium',
                                      start_time=40.0, duration=15.0)
    
    # Simulate predictions with timing issues
    predictions = simulate_ml_predictions(
        ground_truth,
        timing_error_std=0.25,  # Significant timing error (250ms)
        miss_rate=0.05,  # Very low miss rate
        random_seed=20
    )
    
    visualize_arrows(
        ground_truth,
        predictions,
        duration=15.0,
        output_path='artifacts/example_2_timing_errors.png',
        title1='Ground Truth',
        title2='Timing Error Predictions (95% detect, 250ms error)'
    )
    
    print(f"  Detected: {len(predictions)}/{len(ground_truth)} "
          f"({100*len(predictions)/len(ground_truth):.1f}%)")


def generate_example_3_poor_detection():
    """Example 3: Poor detection rate but good timing when detected"""
    print("\n[Example 3] Poor Detection Rate...")
    
    # Extract from Charles
    ground_truth = extract_sm_window('sm_files/Charles.sm', 5, 'medium',
                                      start_time=35.0, duration=15.0)
    
    # Simulate predictions with many misses
    predictions = simulate_ml_predictions(
        ground_truth,
        timing_error_std=0.08,  # Low timing error (80ms)
        miss_rate=0.35,  # High miss rate
        random_seed=30
    )
    
    visualize_arrows(
        ground_truth,
        predictions,
        duration=15.0,
        output_path='artifacts/example_3_poor_detection.png',
        title1='Ground Truth',
        title2='Poor Detection Predictions (65% detect, 80ms error)'
    )
    
    print(f"  Detected: {len(predictions)}/{len(ground_truth)} "
          f"({100*len(predictions)/len(ground_truth):.1f}%)")


def generate_example_4_mixed_quality():
    """Example 4: Variable quality predictions across time window"""
    print("\n[Example 4] Variable Quality (mixed performance)...")
    
    # Extract from Black Rock Shooter
    ground_truth = extract_sm_window('sm_files/Black Rock Shooter.sm', 6, 'medium',
                                      start_time=30.0, duration=15.0)
    
    # Simulate medium quality predictions
    predictions = simulate_ml_predictions(
        ground_truth,
        timing_error_std=0.15,  # Medium timing error (150ms)
        miss_rate=0.20,  # Medium miss rate
        random_seed=40
    )
    
    visualize_arrows(
        ground_truth,
        predictions,
        duration=15.0,
        output_path='artifacts/example_4_mixed_quality.png',
        title1='Ground Truth',
        title2='Mixed Quality Predictions (80% detect, 150ms error)'
    )
    
    print(f"  Detected: {len(predictions)}/{len(ground_truth)} "
          f"({100*len(predictions)/len(ground_truth):.1f}%)")


def generate_example_5_different_songs():
    """Example 5: Comparing two different songs at same difficulty"""
    print("\n[Example 5] Different Songs Comparison...")
    
    # Extract from two different songs
    song1 = extract_sm_window('sm_files/Chururira Chururira Daddadda.sm', 6, 'medium',
                               start_time=45.0, duration=15.0)
    song2 = extract_sm_window('sm_files/Remote Control.sm', 6, 'medium',
                               start_time=40.0, duration=15.0)
    
    visualize_arrows(
        song1,
        song2,
        duration=15.0,
        output_path='artifacts/example_5_song_comparison.png',
        title1='Chururira Chururira Daddadda (Medium 6)',
        title2='Remote Control (Medium 6)'
    )
    
    print(f"  Song 1: {len(song1)} events")
    print(f"  Song 2: {len(song2)} events")


def main():
    print("="*70)
    print("GENERATING ADDITIONAL VISUALIZATION EXAMPLES")
    print("="*70)
    print("\nNote: These use simulated predictions to demonstrate")
    print("      the visualization tool's capabilities with various")
    print("      prediction quality scenarios.")
    
    try:
        generate_example_1_high_accuracy()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    try:
        generate_example_2_timing_errors()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    try:
        generate_example_3_poor_detection()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    try:
        generate_example_4_mixed_quality()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    try:
        generate_example_5_different_songs()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    print("\n" + "="*70)
    print("EXAMPLES GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. example_1_high_accuracy.png - 90% detection, low timing error")
    print("  2. example_2_timing_errors.png - Good detection, poor timing")
    print("  3. example_3_poor_detection.png - Poor detection, good timing")
    print("  4. example_4_mixed_quality.png - Medium quality overall")
    print("  5. example_5_song_comparison.png - Two different songs")
    print("="*70)


if __name__ == '__main__':
    main()
