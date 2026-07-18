# DROID Wrench-On-Hook

This folder contains the LeRobot conversion script, human-label annotations, and
episode-subset filters for the DROID wrench-on-hook task.

## Convert To LeRobot

Use the task-specific converter:

```bash
source .venv/bin/activate
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/

uv run examples/droid/wrench_on_hook/convert_wrench_to_hook_data_to_lerobot.py \
  --data-dir /iris/u/tiangao/projects/droid/data/success/2026-06-13 \
  --repo-id skybhh19/droid_wrench_on_hook \
  --overwrite
```

Each episode uses the prompt `Hang the wrench on the hook`.
Only valid transitions are written to LeRobot: `movement_enabled` must be true
and `skip_action` must be false.

The converter writes the DROID observations expected by `LeRobotDROIDDataConfig`:
`exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`,
`joint_position`, `gripper_position`, and 8-D `actions`.

## Datasets

| Raw DROID data | LeRobot repo id | Full-data training config |
| --- | --- | --- |
| `/iris/u/tiangao/projects/droid/data/success/2026-06-13` | `skybhh19/droid_wrench_on_hook` | `pi05_droid_wrench_on_hook_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/2026-06-15` | `skybhh19/droid_wrench_on_hook_06152026` | `pi05_droid_wrench_on_hook_06152026_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/wrench-on-hook-0620` | `skybhh19/droid_wrench_on_hook_06202026` | `pi05_droid_wrench_on_hook_06202026_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/wrench-on-hook-0622` | `skybhh19/droid_wrench_on_hook_06222026` | `pi05_droid_wrench_on_hook_06222026_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/wrench-on-hook-0622` | `skybhh19/droid_wrench_on_hook_06282026` | `pi05_droid_wrench_on_hook_06282026_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/wrench-on-hook-0629` | `skybhh19/droid_wrench_on_hook_06292026` | `pi05_droid_wrench_on_hook_06292026_low_mem_finetune` |
| `/iris/u/tiangao/projects/droid/data/success/wrench-to-hook-filtered-combined-0617` | `skybhh19/droid_wrench_to_hook_filtered_combined_0617` | `pi05_droid_wrench_to_hook_filtered_combined_0617_low_mem_finetune` |

## Train

All wrench-on-hook fine-tuning configs start from the pretrained pi05-DROID
checkpoint and reuse the pi05-DROID normalization stats:
`gs://openpi-assets/checkpoints/pi05_droid`.

Use the same local LeRobot root when training:

```bash
source .venv/bin/activate
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/
```

Train any full-data config:

```bash
CONFIG=pi05_droid_wrench_on_hook_06292026_low_mem_finetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py ${CONFIG} \
  --exp-name=${CONFIG} \
  --resume
```

Train a filtered subset by using the corresponding filtered config name:

```bash
CONFIG=pi05_droid_wrench_on_hook_06292026_observabilitypct80_low_mem_finetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py ${CONFIG} \
  --exp-name=${CONFIG} \
  --resume
```

## Episode Filters

The subset JSON files live in `examples/droid/wrench_on_hook/lerobot_filtering_keys/`.
Each `*_episode_indices.json` stores LeRobot episode indices and is wired into
training through `DataConfig(lerobot_episode_indices_path=...)`.

Available training suffixes:

- Base June 13 dataset: `randompct25`, `randompct50`, `randompct75`,
  `observabilitypct25`, `observabilitypct50`, `observabilitypct75`
- June 15 dataset: `randompct25`, `randompct50`, `randompct75`,
  `observabilitypct25`, `observabilitypct50`, `observabilitypct75`
- June 20 dataset: `randompct25`, `randompct50`, `randompct75`,
  `observabilitypct25`, `observabilitypct50`, `observabilitypct75`
- June 22 dataset: `randompct60`, `randompct70`, `randompct80`,
  `observabilitypct60`, `observabilitypct70`, `observabilitypct80`
- June 28 dataset: `randompct60`, `randompct70`, `randompct80`,
  `observabilitypct60`, `observabilitypct70`, `observabilitypct80`
- June 29 dataset: `randompct25`, `randompct40`, `randompct50`,
  `randompct60`, `randompct75`, `randompct80`, `observabilitypct25`,
  `observabilitypct40`, `observabilitypct50`, `observabilitypct60`,
  `observabilitypct75`, `observabilitypct80`
- Filtered combined June 17 dataset: `randompct25`, `randompct50`,
  `randompct75`, `observabilitypct25`, `observabilitypct50`,
  `observabilitypct75`

For example:

```bash
CONFIG=pi05_droid_wrench_on_hook_06222026_randompct70_low_mem_finetune
CONFIG=pi05_droid_wrench_on_hook_06222026_randompct70_30k_low_mem_finetune
CONFIG=pi05_droid_wrench_to_hook_filtered_combined_0617_observabilitypct75_low_mem_finetune
```

See `lerobot_filtering_keys/README.md` for episode counts, annotation sources,
and validation details for each dataset.

Regenerate filters after editing annotations:

```bash
source .venv/bin/activate

python examples/droid/wrench_on_hook/build_human_label_episode_filters.py
python examples/droid/wrench_on_hook/build_06152026_episode_filters.py
python examples/droid/wrench_on_hook/build_06202026_episode_filters.py
python examples/droid/wrench_on_hook/build_filtered_combined_0617_episode_filters.py
```

The label-based filter builders validate episode indices against the converted
LeRobot dataset and ignore documented extra annotation rows when those rows are
not present in the converted dataset.
