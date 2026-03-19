#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iliad
#SBATCH --account=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

export CUDA_VISIBLE_DEVICES=1

data_id="droid_eggflip_20"

# Optional: override assets so norm stats are loaded from assets_dir / asset_id
# assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --data.assets.asset_id="johnson906/droid_eggflip_50" \
    --num_train_steps=8001 \
    --save_interval=2000 \
    --fsdp_devices=1

data_id="droid_eggflip_25"

# Optional: override assets so norm stats are loaded from assets_dir / asset_id
# assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --data.assets.asset_id="johnson906/droid_eggflip_50" \
    --num_train_steps=8001 \
    --save_interval=2000 \
    --fsdp_devices=1