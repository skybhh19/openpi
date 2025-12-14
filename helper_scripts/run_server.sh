source /iris/u/khhung/projects/openpi/.venv/bin/activate

# Run trained model from finetune_droid.sh
# The checkpoint is saved at: checkpoints/pi05_droid_finetune/pp/{step}/
# Replace {step} with the actual checkpoint step (e.g., 1000, 2000, etc.)
# Or use the latest checkpoint by pointing to the step directory
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_finetune \
    --policy.dir=./checkpoints/pi05_droid_finetune/pp_5000/4999

# # Alternative: run zero-shot inference with base model
# uv run scripts/serve_policy.py policy:checkpoint \
#     --policy.config=pi05_droid \
#     --policy.dir=gs://openpi-assets/checkpoints/pi05_droid