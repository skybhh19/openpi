source /iris/u/khhung/projects/openpi/.venv/bin/activate

# Run trained model from finetune_droid.sh
# The checkpoint is saved at: checkpoints/pi05_droid_finetune/pp/{step}/
# Replace {step} with the actual checkpoint step (e.g., 1000, 2000, etc.)
# Or use the latest checkpoint by pointing to the step directory
# uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_droid_finetune_pp \
#     --policy.dir=./checkpoints/pi05_droid_finetune/pp/1000

# # Alternative: run zero-shot inference with base model
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_lora_finetune_pp \
    --policy.dir=/iris/u/khhung/projects/openpi/checkpoints/pi05_droid_lora_finetune/droid_ppv2_1/3000

# Alternative: run zero-shot inference with base model
# uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_droid \
#     --policy.dir=gs://openpi-assets/checkpoints/pi05_droid