#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"


data_id="droid_ppv1_2_cartesian_random"

uv run scripts/train.py expo_pi05_droid_lora_finetune_sft \
    --exp-name=${data_id}_lora_sft_base \
    --resume \
    --data.repo_id="johnson906/$data_id" \
    --num_train_steps=10001 \
    --save_interval=1000 \
    --fsdp_devices=1
