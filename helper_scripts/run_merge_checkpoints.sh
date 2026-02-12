#!/bin/bash

source /iris/u/khhung/projects/openpi/.venv/bin/activate

# Default: save to ./checkpoints/pi05_base_droid_merged
# Pass extra args to override, e.g.:
#   ./helper_scripts/run_merge_checkpoints.sh --output_dir ./checkpoints/my_merged
#   ./helper_scripts/run_merge_checkpoints.sh --no_save

uv run helper_scripts/merge_checkpoints.py \
        --output_dir ./checkpoints/pi05_base_droid_expert \
        --backbone_checkpoint "gs://openpi-assets/checkpoints/pi05_droid/params" \
        --action_expert_checkpoint "gs://openpi-assets/checkpoints/pi05_base/params"