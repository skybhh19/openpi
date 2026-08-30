#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes, optionally iris-hi
#SBATCH --time=48:00:00 # Max job length is 2 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune # Name the job (for easier monitoring)
#SBATCH --constraint=[hopper] # Use only Hopper or Ampere nodes
#SBATCH --output=slurm-%j.out
#SBATCH --account=iris
#SBATCH --mail-user tiangao@stanford.edu
#SBATCH --mail-type END,FAIL,REQUEUE,TIME_LIMIT_80


source .venv/bin/activate

export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune \
  --exp-name=pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune \
  --overwrite
