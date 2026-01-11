#!/usr/bin/env python3
"""
Train model with many more songs for better generalization.
"""

import subprocess
import sys

# Select diverse songs for training (exclude one for testing)
# We'll use songs with clear difficulty levels and good alignment

training_songs = [
    # Lucky Orb variations
    ("raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip", "sm_files/Lucky Orb.sm", 5),
    ("raw_data/Lucky_Orb_Medium_5-2026-01-10_09-07-24.zip", "sm_files/Lucky Orb.sm", 5),
    
    # DECORATOR
    ("raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip", "sm_files/DECORATOR.sm", 6),
    
    # Nostalgic Winds of Autumn
    ("raw_data/Nostalgic_Winds_of_Autumn_Medium_5-2026-01-07_06-24-52.zip", "sm_files/Nostalgic Winds of Autumn.sm", 5),
    
    # Charles (will be used for validation but not for final prediction test)
    ("raw_data/Charles_5_Medium-2026-01-10_09-22-48.zip", "sm_files/Charles.sm", 5),
    
    # Additional diverse songs
    ("raw_data/Confession_Medium_7-2026-01-10_09-40-48.zip", "sm_files/Confession.sm", 7),
    ("raw_data/Getting_Faster_and_Faster_5_Medium-2026-01-09_06-30-45.zip", "sm_files/Getting Faster and Faster.sm", 5),
    ("raw_data/Friend_Medium_6-2026-01-11_09-57-53.zip", "sm_files/F(R)IEND.sm", 6),
    ("raw_data/Fantasy_Film_5_Medium-2026-01-11_09-54-45.zip", "sm_files/Fantasy Film.sm", 5),
    ("raw_data/Night_Sky_Patrol_of_Tomorrow_7_Medium-2026-01-10_09-11-02.zip", "sm_files/Night Sky Patrol of Tomorrow.sm", 7),
    
    # More variety
    ("raw_data/Neko_Neko_Super_Fever_Night_6_Medium-2026-01-10_09-15-23.zip", "sm_files/Neko Neko Super Fever Night.sm", 6),
    ("raw_data/Love_song_6_Medium-2026-01-11_09-41-52.zip", "sm_files/Love Song.sm", 6),
]

# Hold-out song for final testing (completely excluded from training)
# Using Butterfly Cat as it's different from training set
hold_out_song = ("raw_data/Butterfly_Cat_6_Medium-2026-01-10_09-34-07.zip", "sm_files/Butterfly Cat.sm", 6)

print("="*70)
print("TRAINING WITH MORE SONGS")
print("="*70)
print(f"\nTraining songs: {len(training_songs)}")
print(f"Hold-out song: Butterfly Cat (Medium 6)")
print("\nTraining set:")
for i, (cap, sm, diff) in enumerate(training_songs, 1):
    song_name = sm.split('/')[-1].replace('.sm', '')
    print(f"  {i:2d}. {song_name:40s} (Medium/Easy {diff})")

# Build command
cmd = ["python", "train_model.py"]
for cap, sm, diff in training_songs:
    cmd.extend([cap, sm, str(diff)])

print(f"\nRunning training with {len(training_songs)} songs...")
print("="*70)

# Run training
subprocess.run(cmd)

