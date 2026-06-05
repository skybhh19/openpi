# DROID Pen-Cup RLDS Subset

This pipeline builds a small DROID RLDS subset from `examples/droid/droid_pen_cup_yes_index.jsonl` while preserving the original DROID TFDS/RLDS schema. The subset stays episode-level: frames are not dropped, action chunks are not precomputed, and idle filtering remains a dataloader-time keep-ranges filter.

The subset is intended for fine-tuning `pi05_droid` with:

```text
action space: joint_velocity[7] + gripper_position[1]
prompt:       Put the pen in the cup
```

## 1. Build The Subset RLDS Dataset

```bash
cd /iliad/u/tiangao/projects/openpi
source .venv/bin/activate

uv run --group rlds python examples/droid/droid_subset_rlds/build_droid_subset_rlds.py \
  --index-jsonl examples/droid/droid_pen_cup_yes_index.jsonl \
  --source-builder-dir /iliad/group/datasets/droid/1.0.1 \
  --output-builder-dir /iliad/group/datasets/droid_pen_cup_fixed_prompt/1.0.0 \
  --fixed-prompt "Put the pen in the cup" \
  --examples-per-shard 64 \
  --overwrite
```

This copies selected serialized DROID episodes and patches all three language fields to the fixed prompt. It also writes:

```text
/iliad/group/datasets/droid_pen_cup_fixed_prompt/1.0.0/subset_manifest.json
```

## 2. Build The Subset Keep-Ranges JSON

```bash
cd /iliad/u/tiangao/projects/openpi
source .venv/bin/activate

uv run --group rlds python examples/droid/droid_subset_rlds/build_subset_keep_ranges.py \
  --subset-builder-dir /iliad/group/datasets/droid_pen_cup_fixed_prompt/1.0.0 \
  --full-keep-ranges examples/droid/droid_refinement/keep_ranges_1_0_1.json \
  --output-keep-ranges examples/droid/droid_pen_cup_keep_ranges_1_0_1.json \
  --output-episode-map examples/droid/droid_pen_cup_episode_map.json \
  --fixed-prompt "Put the pen in the cup"
```

The keep-ranges file is keyed by:

```text
recording_folderpath--file_path
```

which is the same key format used by `DroidRldsDataset`.

## 3. Validate The Dataset

```bash
cd /iliad/u/tiangao/projects/openpi
source .venv/bin/activate

uv run --group rlds python examples/droid/droid_subset_rlds/validate_droid_subset_rlds.py \
  --subset-builder-dir /iliad/group/datasets/droid_pen_cup_fixed_prompt/1.0.0 \
  --keep-ranges examples/droid/droid_pen_cup_keep_ranges_1_0_1.json \
  --expected-episodes 1108 \
  --fixed-prompt "Put the pen in the cup"
```

## 4. Dataloader Configuration

The subset can use the existing DROID RLDS dataloader. Point the dataset entry at the local TFDS builder directory:

```python
data=RLDSDroidDataConfig(
    repo_id="droid_pen_cup_fixed_prompt",
    rlds_data_dir="/iliad2/group/datasets",
    action_space=droid_rlds_dataset.DroidActionSpace.JOINT_VELOCITY,
    shuffle_buffer_size=50_000,
    datasets=(
        droid_rlds_dataset.RLDSDataset(
            name="droid_pen_cup_fixed_prompt",
            version="1.0.0",
            weight=1.0,
            builder_dir="/iliad2/group/datasets/droid_pen_cup_fixed_prompt/1.0.0",
            filter_dict_path="examples/droid/droid_pen_cup_keep_ranges_1_0_1.json",
        ),
    ),
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
        asset_id="droid",
    ),
)
```

Use `pi05_droid` weights:

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://openpi-assets/checkpoints/pi05_droid/params"
)
```

For this subset, `shuffle_buffer_size=50_000` is a reasonable starting point. For quick debugging, `10_000` is fine. The full DROID default remains `250_000`.

## Smoke Test With Two Episodes

Use this before launching the full copy job:

```bash
cd /iliad2/u/tiangao/projects/openpi
source .venv/bin/activate

rm -rf /tmp/openpi_droid_pen_cup_smoke/1.0.0

uv run --group rlds python examples/droid/droid_subset_rlds/build_droid_subset_rlds.py \
  --index-jsonl examples/droid/droid_pen_cup_yes_index.jsonl \
  --source-builder-dir /iliad2/group/datasets/droid/1.0.1 \
  --output-builder-dir /tmp/openpi_droid_pen_cup_smoke/1.0.0 \
  --fixed-prompt "Put the pen in the cup" \
  --examples-per-shard 2 \
  --max-episodes 2 \
  --overwrite

uv run --group rlds python examples/droid/droid_subset_rlds/build_subset_keep_ranges.py \
  --subset-builder-dir /tmp/openpi_droid_pen_cup_smoke/1.0.0 \
  --full-keep-ranges examples/droid/droid_refinement/keep_ranges_1_0_1.json \
  --output-keep-ranges /tmp/openpi_droid_pen_cup_smoke_keep_ranges.json \
  --output-episode-map /tmp/openpi_droid_pen_cup_smoke_episode_map.json \
  --fixed-prompt "Put the pen in the cup"

uv run --group rlds python examples/droid/droid_subset_rlds/validate_droid_subset_rlds.py \
  --subset-builder-dir /tmp/openpi_droid_pen_cup_smoke/1.0.0 \
  --keep-ranges /tmp/openpi_droid_pen_cup_smoke_keep_ranges.json \
  --expected-episodes 2 \
  --fixed-prompt "Put the pen in the cup"
```

## Why This Shape

We do not drop idle frames while building the subset. The existing RLDS dataloader first builds future action chunks from the full original trajectory and only then filters current timesteps with the keep-ranges table. Preserving full episodes keeps that behavior intact.
