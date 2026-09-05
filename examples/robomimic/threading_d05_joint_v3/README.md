# Threading_D05 v3 absolute-joint pipeline

The OpenPI repository now lives under `/iris/u/tiangao/projects/openpi`. The
source data remains on the legacy `/iliad` data volume at:

```text
/iliad/u/tiangao/projects/robomimic/datasets/threading_d05_joint_v3/image_v15_256.hdf5
```

The source contains 300 demonstrations and 71,222 frames. Conversion defaults
to `filter_key=all`, so the randomized `train` / `valid` masks are ignored.
The explicit label masks are disjoint and exhaustive:

- `full`: 150 episodes / 36,250 frames
- `partial`: 150 episodes / 34,972 frames

Actions are `[absolute Panda joint target (7), gripper command (1)]`. LeRobot
stores the absolute arm target and maps the gripper to closure in `[0, 1]`.
`LeRobotRobomimicDataConfig` applies deltas only to the seven arm dimensions;
the gripper remains absolute.

## Convert and validate locally

Run commands from the OpenPI repository root. Local staging avoids writing
individual conversion artifacts directly to NFS.

```bash
export HF_LEROBOT_HOME=/tmp/openpi_threading_d05_joint_v3_lerobot
uv run examples/robomimic/threading_d05_joint_v3/convert_robomimic_data_to_lerobot.py
```

After publishing the verified dataset to
`/iris/u/tiangao/lerobot_datasets/local/robomimic_threading_d05_joint_v3_256`,
rebuild the checked label filters:

```bash
uv run examples/robomimic/threading_d05_joint_v3/build_filtering_keys.py
```

The filtering script validates source path/size/environment, repo ID, task,
episode order, every episode length, totals, FPS, and complete label coverage
before writing `full_only` and `partial_only` episode indices.

## Normalization and training

```bash
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
uv run scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d05_joint_v3_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d05_joint_v3_low_mem_finetune`
- `pi05_robomimic_threading_d05_joint_v3_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d05_joint_v3_partial_only_low_mem_finetune`

The two filtered configs intentionally reuse this v3 dataset's all-episode
normalization statistics.

## Tests

```bash
uv run pytest -q \
  examples/robomimic/threading_d05_joint_v3/threading_d05_joint_v3_conversion_test.py \
  examples/robomimic/threading_d05_joint_v3/threading_d05_joint_v3_filtering_test.py \
  src/openpi/training/robomimic_config_test.py
```
