#!/bin/bash

#SBATCH --job-name=pi
#SBATCH --partition=iris
#SBATCH --nodelist=iris-hgx-1
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --account=iris
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

uv run scripts/train.py pi05_droid_finetune \
    --exp-name=pp \
    --overwrite \
    --data.repo_id="johnson906/droid_pp" \
    --num_train_steps=3001 \
    --save_interval=1000 \
    --fsdp_devices=4 
