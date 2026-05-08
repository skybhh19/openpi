#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iliad-lo
#SBATCH --account=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:4
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_id="droid_flower_1_120"

uv run scripts/train.py expo_pi05_droid_full_finetune_sft_cartesian_state \
    --exp-name=${data_id}_full_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --data.assets.asset_id="johnson906/droid_flower_1_120" \
    --num_train_steps=4001 \
    --save_interval=1000 \
    --fsdp_devices=1

