source /iris/u/khhung/projects/openpi/.venv/bin/activate

# Alternative: run zero-shot inference with base model

# uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=expo_pi05_droid_lora_finetune_sft \
#     --policy.dir=/iris/u/khhung/projects/openpi/checkpoints/expo_pi05_droid_lora_finetune_sft/droid_ppv2_50_cartesian_same_pos_lora_sft_base/1000


uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=expo_pi05_droid_lora_finetune_sft \
    --policy.dir=/iris/u/khhung/projects/openpi/checkpoints/expo_pi05_droid_lora_finetune_sft/droid_ppv2_50_cartesian_same_pos_lora_sft_long/5000
