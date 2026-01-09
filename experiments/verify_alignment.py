#!/usr/bin/env python3
"""
Verify alignment script by running on multiple captures.
This script tests the alignment algorithm on various recordings to ensure it works correctly.
"""
import sys
from pathlib import Path
import re

# Import the main alignment script
import importlib.util
spec = importlib.util.spec_from_file_location("align_signals", Path(__file__).parent / "03_align_signals.py")
align_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_module)

# Access functions from the module
load_sensor_data = align_module.load_sensor_data
compute_acceleration_magnitude = align_module.compute_acceleration_magnitude
preprocess_sensor_signal = align_module.preprocess_sensor_signal
create_reference_signal = align_module.create_reference_signal
find_alignment = align_module.find_alignment
parse_sm_file = align_module.parse_sm_file

import numpy as np
import matplotlib.pyplot as plt


def extract_song_name(filename):
    """Extract song name from filename."""
    # Remove difficulty and date suffix
    name = re.sub(r'_\d+_?(Medium|Easy|Hard|Expert)?.*-\d{4}-\d{2}-\d{2}.*', '', filename)
    name = name.replace('_', ' ')
    return name.strip()


def find_matching_sm_file(sensor_file, sm_dir):
    """Find matching .sm file for a sensor capture."""
    song_name = extract_song_name(sensor_file.stem)
    
    # Try exact match first
    for sm_file in sm_dir.glob('*.sm'):
        sm_name_clean = sm_file.stem.replace('_', ' ').replace('-', ' ')
        song_name_clean = song_name.replace('_', ' ').replace('-', ' ')
        if song_name_clean.lower() in sm_name_clean.lower():
            return sm_file
    
    # Try word-by-word match
    words = [w.lower() for w in song_name.split() if len(w) > 3]
    best_match = None
    best_score = 0
    
    for sm_file in sm_dir.glob('*.sm'):
        sm_name = sm_file.stem.lower().replace('_', ' ').replace('-', ' ')
        score = sum(1 for word in words if word in sm_name)
        if score > best_score and score >= len(words) * 0.5:  # At least 50% match
            best_score = score
            best_match = sm_file
    
    return best_match


def process_alignment(sensor_file, sm_file, output_dir):
    """Process alignment for a single sensor/sm file pair."""
    print(f"\n{'='*60}")
    print(f"Processing: {sensor_file.stem}")
    print(f"{'='*60}")
    print(f"Sensor file: {sensor_file.name}")
    print(f"StepMania file: {sm_file.name}\n")
    
    try:
        # Load sensor data
        print("Loading sensor data...")
        sensor_df = load_sensor_data(sensor_file)
        time_ms, magnitude = compute_acceleration_magnitude(sensor_df)
        print(f"  Sensor duration: {(time_ms[-1] - time_ms[0]) / 1000:.2f}s")
        
        # Parse StepMania file
        print("Parsing StepMania file...")
        charts = parse_sm_file(sm_file)
        
        # Select appropriate chart
        chart = None
        for c in charts:
            if 'Medium' in c.difficulty or 'medium' in c.difficulty.lower():
                chart = c
                break
        
        if not chart and charts:
            chart = charts[0]
        
        if not chart:
            print("  ERROR: No chart found")
            return None
        
        print(f"  Chart difficulty: {chart.difficulty}")
        print(f"  Chart duration: {chart.notes[-1].time:.2f}s")
        print(f"  Total notes: {len(chart.notes)}\n")
        
        # Preprocess sensor signal
        print("Processing sensor signal...")
        sensor_time, sensor_envelope = preprocess_sensor_signal(time_ms, magnitude)
        
        # Create reference signal
        print("Creating reference signal...")
        ref_duration = max(sensor_time[-1], chart.notes[-1].time)
        ref_time, ref_signal = create_reference_signal(chart.notes, ref_duration)
        
        # Align signals
        print("Finding alignment...")
        best_offset, lags, corr = find_alignment(sensor_envelope, ref_signal)
        
        print(f"\nBest time offset: {best_offset:.3f} seconds")
        print(f"Correlation peak: {np.max(corr):.3f}")
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Sensor envelope
        ax = axes[0]
        ax.plot(sensor_time, sensor_envelope, linewidth=0.8)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Acceleration envelope', fontsize=10)
        ax.set_title(f'Processed Sensor Signal - {sensor_file.stem}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reference signal
        ax = axes[1]
        ax.plot(ref_time, ref_signal, linewidth=0.8)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Reference signal', fontsize=10)
        ax.set_title(f'Reference Signal - {sm_file.stem} ({chart.difficulty})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Cross-correlation
        ax = axes[2]
        ax.plot(lags, corr, linewidth=0.8)
        peak_idx = np.argmax(corr)
        ax.axvline(best_offset, color='r', linestyle='--', linewidth=2, 
                   label=f'Best offset: {best_offset:.3f}s (peak: {corr[peak_idx]:.3f})')
        ax.set_xlabel('Time offset (s)', fontsize=10)
        ax.set_ylabel('Cross-correlation', fontsize=10)
        ax.set_title('Cross-correlation for Alignment', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-20, 20])
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / f'{sensor_file.stem}_correlation.png'
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization saved: {output_file.name}")
        
        return {
            'sensor_file': sensor_file.name,
            'sm_file': sm_file.name,
            'offset': best_offset,
            'correlation_peak': np.max(corr),
            'sensor_duration': (time_ms[-1] - time_ms[0]) / 1000,
            'chart_duration': chart.notes[-1].time,
            'num_notes': len(chart.notes),
            'success': True
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'sensor_file': sensor_file.name,
            'sm_file': sm_file.name if sm_file else 'N/A',
            'error': str(e),
            'success': False
        }


def main():
    raw_data_dir = Path(__file__).parent.parent / 'raw_data'
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Get all sensor files
    sensor_files = sorted(list(raw_data_dir.glob('*.zip')))
    
    print(f"Found {len(sensor_files)} sensor capture files")
    print(f"Testing alignment on multiple captures...\n")
    
    results = []
    
    # Process up to 5 different songs for verification
    processed_songs = set()
    max_samples = 5
    
    for sensor_file in sensor_files:
        if len(results) >= max_samples:
            break
        
        # Extract song name to avoid duplicates
        song_name = extract_song_name(sensor_file.stem)
        if song_name in processed_songs:
            continue
        
        # Find matching .sm file
        sm_file = find_matching_sm_file(sensor_file, sm_dir)
        if not sm_file:
            print(f"Skipping {sensor_file.name} - no matching .sm file found")
            continue
        
        processed_songs.add(song_name)
        result = process_alignment(sensor_file, sm_file, output_dir)
        if result:
            results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ALIGNMENT VERIFICATION SUMMARY")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}\n")
    
    if successful:
        print("Successful alignments:")
        for r in successful:
            print(f"  • {r['sensor_file'][:50]}")
            print(f"    Offset: {r['offset']:.3f}s | Peak: {r['correlation_peak']:.3f} | Notes: {r['num_notes']}")
    
    if failed:
        print("\nFailed alignments:")
        for r in failed:
            print(f"  • {r['sensor_file']}: {r.get('error', 'Unknown error')}")
    
    print(f"\nCorrelation plots saved to: {output_dir}")
    print("\nVerification complete! Check the correlation plots to verify alignment quality.")


if __name__ == '__main__':
    main()
