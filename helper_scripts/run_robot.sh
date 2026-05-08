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


data_id="droid_scoop_rotate_20"
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft_h1_test \
    --resume \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_sim_lora_finetune_sft" \
    --data.assets.asset_id="johnson906/sim_twocubes_60" \
    --data.repo_id="johnson906/$data_id" \
    --num_train_steps=100001 \
    --save_interval=10000000 \
    --fsdp_devices=1

