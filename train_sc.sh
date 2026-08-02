#!/bin/bash
#SBATCH --partition=iliad # Run on IRIS nodes, optionally iris-hi
#SBATCH --time=48:00:00 # Max job length is 2 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --gres=gpu
#SBATCH --job-name=pi05_droid_pen_in_blue_cup_07272026_gmm_k5_std1em02_cross_fit_score5pct75_low_mem_finetune # Name the job (for easier monitoring)
#SBATCH --constraint=[hopper] # Use only Hopper or Ampere nodes
#SBATCH --output=slurm-%j.out
#SBATCH --account=iliad
#SBATCH --mail-user tiangao@stanford.edu
#SBATCH --mail-type END,FAIL,REQUEUE,TIME_LIMIT_80


source .venv/bin/activate

export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_07272026_gmm_k5_std1em02_cross_fit_score5pct75_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_07272026_gmm_k5_std1em02_cross_fit_score5pct75_low_mem_finetune \
  --resume
