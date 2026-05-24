#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iliad   
#SBATCH --account=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_MODE=offline

# Local LeRobot mirror: .../lerobot_data/johnson906/droid_light{0,1,2}_*
# (names are droid_light*, not droid_flower_light*)
# data_id="droid_light0_40"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# # No precomputed stats for light0_40 — using light1_40 as proxy; run compute_norm_stats for this repo for best results.
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --overwrite \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_light1_40" \
#     --num_train_steps=4001 \
#     --save_interval=2000 \
#     --fsdp_devices=1

# data_id="droid_light1_25"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # No folder for light1_25 in assets/ — using light1_40 stats as proxy; prefer compute_norm_stats on this repo.
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --overwrite \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_light1_40" \
#     --num_train_steps=4001 \
#     --save_interval=2000 \
#     --fsdp_devices=1

data_id="droid_pick_cube_10"

# Proxy norm stats: closest available is light2_30; run compute_norm_stats for light2_25 if distributions differ.
uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --data.assets.asset_id="johnson906/droid_pick_cube_15" \
    --num_train_steps=4001 \
    --save_interval=2000 \
    --fsdp_devices=1