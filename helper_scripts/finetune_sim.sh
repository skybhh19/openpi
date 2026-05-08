#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iliad
#SBATCH --account=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_id="sim_pickenv_50"

# Optional: override assets so norm stats are loaded from assets_dir / asset_id
# assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
uv run scripts/train.py expo_pi05_sim_lora_finetune_sft \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_sim_lora_finetune_sft" \
    --data.assets.asset_id="johnson906/sim_twocubes_60" \
    --num_train_steps=5001 \
    --save_interval=1000 \
    --fsdp_devices=1


