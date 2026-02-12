source /iris/u/khhung/projects/openpi/.venv/bin/activate

# use the norm stats from droid on state, but use new action norm stats
uv run scripts/compute_norm_stats.py \
    --config-name expo_pi05_droid_lora_finetune_sft