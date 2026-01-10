#!/bin/bash
# Train with expanded dataset for better performance

python train_model.py \
  "raw_data/Lucky_Orb_5_Medium-2026-01-06_18-45-00.zip" "sm_files/Lucky Orb.sm" 5 \
  "raw_data/Decorator_Medium_6-2026-01-07_06-27-54.zip" "sm_files/DECORATOR.sm" 6 \
  "raw_data/Charles_5_Medium-2026-01-10_09-22-48.zip" "sm_files/Charles.sm" 5 \
  "raw_data/Catch_the_wave_7_Medium-2026-01-10_09-26-07.zip" "sm_files/Catch The Wave.sm" 7 \
  "raw_data/Neko_Neko_Super_Fever_Night_6_Medium-2026-01-10_09-15-23.zip" "sm_files/Neko Neko Super Fever Night.sm" 6 \
  "raw_data/Butterfly_Cat_6_Medium-2026-01-10_09-34-07.zip" "sm_files/Butterfly Cat.sm" 6 \
  "raw_data/39_Music_6_Medium-2026-01-10_09-37-22.zip" "sm_files/39 Music!.sm" 6 \
  "raw_data/Confession_Medium_7-2026-01-10_09-40-48.zip" "sm_files/Confession.sm" 7
