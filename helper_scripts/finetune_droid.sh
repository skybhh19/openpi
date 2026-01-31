#!/bin/bash

#SBATCH --job-name=pi
#SBATCH --partition=iliad
#SBATCH --nodelist=iliad-hgx-1
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --account=iliad 
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

# uv run scripts/train.py pi05_droid_lora_finetune \
#     --exp-name=droid_ppv2_5 \
#     --overwrite \
#     --data.repo_id="johnson906/droid_ppv2_5" \
#     --num_train_steps=5001 \
#     --save_interval=1000 \
#     --fsdp_devices=1
    
# uv run scripts/train.py pi05_droid_lora_finetune \
#     --exp-name=droid_ppv2_3 \
#     --overwrite \
#     --data.repo_id="johnson906/droid_ppv2_3" \
#     --num_train_steps=5001 \
#     --save_interval=1000 \
#     --fsdp_devices=1

uv run scripts/train.py pi05_droid_lora_finetune \
    --exp-name=droid_ppv2_2 \
    --overwrite \
    --data.repo_id="johnson906/droid_ppv2_1" \
    --num_train_steps=2001 \
    --save_interval=1000 \
    --fsdp_devices=1

# uv run scripts/train.py pi05_droid_lora_finetune \
#     --exp-name=droid_ppv2_1 \
#     --overwrite \
#     --data.repo_id="johnson906/droid_ppv2_1" \
#     --num_train_steps=5001 \
#     --save_interval=1000 \
#     --fsdp_devices=1
