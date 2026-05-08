#!/bin/bash

#SBATCH --job-name=pi-sft
#SBATCH --partition=iliad-lo   
#SBATCH --account=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --gres=gpu:h200:2
#SBATCH --output=runs/%A.out
#SBATCH --error=runs/%A.err

source /iris/u/khhung/projects/openpi/.venv/bin/activate

echo "Starting training"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_MODE=offline

data_id="droid_flower1_350"

# Optional: override assets so norm stats are loaded from assets_dir / asset_id
# assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --exp-name=${data_id}_lora_sft \
    --overwrite \
    --data.repo_id="johnson906/$data_id" \
    --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
    --data.assets.asset_id="johnson906/droid_flower1_50" \
    --num_train_steps=10001 \
    --save_interval=2000 \
    --fsdp_devices=1


# data_id="droid_flower1_150"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --resume \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_flower1_50" \
#     --num_train_steps=4001 \
#     --save_interval=2000 \
#     --fsdp_devices=1


# data_id="droid_light1_30"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --resume \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_flower1_50" \
#     --num_train_steps=4001 \
#     --save_interval=2000 \
#     --fsdp_devices=2

# data_id="droid_light2_20"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --overwrite \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_light2_30" \
#     --num_train_steps=8001 \
#     --save_interval=2000 \
#     --fsdp_devices=1

# data_id="droid_flower_1_80"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# uv run scripts/train.py expo_pi05_droid_lora_finetune_sft_cartesian_state \
#     --exp-name=${data_id}_lora_sft \
#     --overwrite \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_droid_lora_finetune_sft_cartesian_state" \
#     --data.assets.asset_id="johnson906/droid_flower_1_120" \
#     --num_train_steps=8001 \
#     --save_interval=2000 \
#     --fsdp_devices=1


# data_id="sim_pickenv_50"

# # Optional: override assets so norm stats are loaded from assets_dir / asset_id
# # assets_dir is the base path; norm stats are loaded from assets_dir/asset_id
# uv run scripts/train.py expo_pi05_sim_lora_finetune_sft \
#     --exp-name=${data_id}_lora_sft \
#     --overwrite \
#     --data.repo_id="johnson906/$data_id" \
#     --data.assets.assets_dir="/iris/u/khhung/projects/openpi/assets/expo_pi05_sim_lora_finetune_sft" \
#     --data.assets.asset_id="johnson906/sim_twocubes_60" \
#     --num_train_steps=500001 \
#     --save_interval=100000 \
#     --fsdp_devices=1

