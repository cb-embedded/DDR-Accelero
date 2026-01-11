#!/usr/bin/env python3
"""
Batch process all captures to generate alignment artifacts.
Manual mapping between captures and .sm files.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from align_clean import align_capture


# Manual mapping: capture filename prefix -> (sm_filename, difficulty_type, difficulty_level)
MAPPINGS = {
    '39_Music_6_Medium': ('39 Music!.sm', 'medium', 6),
    'Butterfly_Cat_6_Medium': ('Butterfly Cat.sm', 'medium', 6),
    'Catch_the_wave_7_Medium': ('Catch The Wave.sm', 'medium', 7),
    'Charles_5_Medium': ('Charles.sm', 'medium', 5),
    'Chigau_Easy_5': ('Chigau.sm', 'easy', 5),
    'Confession_Medium_7': ('Confession.sm', 'medium', 7),
    'Decorator_Medium_6': ('DECORATOR.sm', 'medium', 6),
    'Even_a_Kunoichi_needs_love_6_Medium': ('Even a Kunoichi Needs Love.sm', 'medium', 6),
    'Failure_Girl_6_Medium_Failed_': ('Failure Girl.sm', 'medium', 6),
    'Fantasy_Film_5_Medium': ('Fantasy Film.sm', 'medium', 5),
    'Friend_Medium_6': ('F(R)IEND.sm', 'medium', 6),
    'Getting_Faster_and_Faster_5_Medium': ('Getting Faster and Faster.sm', 'medium', 5),
    'Isolation_Thanatos_Easy_6': ('Isolation=Thanatos.sm', 'easy', 6),
    'Kimagure_Mercy_Medium_6_Failed_': ('Kimagure Mercy.sm', 'medium', 6),
    'Kimagure_Mercy_Moyen_6': ('Kimagure Mercy.sm', 'medium', 6),
    'Little_But_Adult_Hit_7_Medium': ('Little Bit Adult Hit.sm', 'medium', 7),
    'Love_song_6_Medium': ('Love Song.sm', 'medium', 6),
    'Lucky_Orb_5_Medium': ('Lucky Orb.sm', 'medium', 5),
    'Lucky_Orb_5_Medium_3_': ('Lucky Orb.sm', 'medium', 5),
    'Lucky_Orb_5_Medium_4_': ('Lucky Orb.sm', 'medium', 5),
    'Lucky_Orb_Medium_5': ('Lucky Orb.sm', 'medium', 5),
    'Lucky_Orb_Medium_5_2_': ('Lucky Orb.sm', 'medium', 5),
    'Melt_Hard_6': ('Melt.sm', 'hard', 6),
    'Neko_Neko_Super_Fever_Night_6_Medium': ('Neko Neko Super Fever Night.sm', 'medium', 6),
    'Night_Sky_Patrol_of_Tomorrow_7_Medium': ('Night Sky Patrol of Tomorrow.sm', 'medium', 7),
    'Nostalgic_Winds_of_Autumn_Medium_5': ('Nostalgic Winds of Autumn.sm', 'medium', 5),
}


def plot_alignment_artifact(result, capture_path, sm_path, out_path):
    """
    Create 2-pane artifact showing:
    1. Correlation peak
    2. Aligned signals comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Pane 1: Correlation peak
    axes[0].plot(result['lags'], result['corr'], linewidth=0.8, color='blue')
    axes[0].axvline(result['offset'], color='r', linestyle='--', linewidth=2, label=f'Peak: {result["offset"]:.2f}s')
    axes[0].set_title(f'Correlation Peak (ratio={result["ratio"]:.2f}, z={result["z_score"]:.2f})')
    axes[0].set_xlabel('Lag (s)')
    axes[0].set_ylabel('Correlation')
    axes[0].set_xlim([0, min(200, result['lags'][-1])])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pane 2: Aligned signals comparison
    # Shift chart signal by offset to align with accelerometer
    t_chart_shifted = result['t_chart'] + result['offset']
    
    # Plot both signals on the same time axis
    axes[1].plot(result['t_acc'], result['ax'], linewidth=0.6, alpha=0.7, label='Accelerometer (captured)')
    axes[1].plot(t_chart_shifted, result['p'], linewidth=0.8, alpha=0.7, color='orange', label='Chart signal (recreated)')
    axes[1].set_title('Aligned Signals Comparison')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Normalized Signal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Limit x-axis to show a reasonable window
    t_min = max(result['t_acc'][0], t_chart_shifted[0])
    t_max = min(result['t_acc'][-1], t_chart_shifted[-1])
    window_size = min(60, (t_max - t_min) / 2)  # Show up to 60 seconds
    axes[1].set_xlim([t_min, t_min + window_size])
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    raw_data_dir = Path('raw_data')
    sm_files_dir = Path('sm_files')
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    
    # Get all capture files
    capture_files = sorted(raw_data_dir.glob('*.zip'))
    
    print(f"Found {len(capture_files)} capture files")
    print(f"Processing captures with mappings...\n")
    
    processed = 0
    skipped = 0
    
    for capture_path in capture_files:
        # Find matching mapping
        matched = False
        for prefix, (sm_file, diff_type, diff_level) in MAPPINGS.items():
            if capture_path.stem.startswith(prefix):
                sm_path = sm_files_dir / sm_file
                
                if not sm_path.exists():
                    print(f"⚠ SKIP: {capture_path.name} - .sm file not found: {sm_file}")
                    skipped += 1
                    matched = True
                    break
                
                print(f"Processing: {capture_path.name}")
                print(f"  -> {sm_file} ({diff_type} level {diff_level})")
                
                try:
                    result = align_capture(capture_path, sm_path, diff_level, diff_type, verbose=False)
                    
                    out_path = artifacts_dir / f'{capture_path.stem}_ALIGN.png'
                    plot_alignment_artifact(result, capture_path, sm_path, out_path)
                    
                    print(f"  ✓ Offset: {result['offset']:.2f}s, Ratio: {result['ratio']:.2f}, Z: {result['z_score']:.2f}")
                    print(f"  ✓ Saved: {out_path.name}\n")
                    processed += 1
                except Exception as e:
                    print(f"  ✗ ERROR: {e}\n")
                    skipped += 1
                
                matched = True
                break
        
        if not matched:
            print(f"⚠ SKIP: {capture_path.name} - no mapping found")
            skipped += 1
    
    print("="*70)
    print(f"SUMMARY: Processed {processed}, Skipped {skipped}")
    print("="*70)


if __name__ == '__main__':
    main()
