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
  "raw_data/Confession_Medium_7-2026-01-10_09-40-48.zip" "sm_files/Confession.sm" 7 \
  "raw_data/Even_a_Kunoichi_needs_love_6_Medium-2026-01-11_10-00-28.zip" "sm_files/Even a Kunoichi Needs Love.sm" 6 \
  "raw_data/Fantasy_Film_5_Medium-2026-01-11_09-54-45.zip" "sm_files/Fantasy Film.sm" 5 \
  "raw_data/Friend_Medium_6-2026-01-11_09-57-53.zip" "sm_files/F(R)IEND.sm" 6 \
  "raw_data/Kimagure_Mercy_Moyen_6-2026-01-11_09-48-54.zip" "sm_files/Kimagure Mercy.sm" 6 \
  "raw_data/Little_But_Adult_Hit_7_Medium-2026-01-11_09-45-34.zip" "sm_files/Little Bit Adult Hit.sm" 7 \
  "raw_data/Love_song_6_Medium-2026-01-11_09-41-52.zip" "sm_files/Love Song.sm" 6
