#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes, optionally iris-hi
#SBATCH --time=48:00:00 # Max job length is 2 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=30 # Request 30 CPUs for this task
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pi05_droid_pen_cup_randompct50_low_mem_finetune_0605 # Name the job (for easier monitoring)
#SBATCH --output=slurm-%j.out
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-1
#SBATCH --mail-user tiangao@stanford.edu
#SBATCH --mail-type END,FAIL,REQUEUE,TIME_LIMIT_80


source .venv/bin/activate


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --group rlds scripts/train.py pi05_droid_pen_cup_randompct50_low_mem_finetune_0605 \
  --exp-name=pi05_droid_pen_cup_randompct50_low_mem_finetune_0605 \
  --overwrite

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_robomimic_square_random_post_left_close_low_low_mem_finetune_validation --exp-name=pi0_fast_robomimic_square_random_post_left_close_low_low_mem_finetune_validation --overwrite
