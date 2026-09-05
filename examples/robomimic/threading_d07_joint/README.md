# Threading_D07 absolute-joint pipeline

This pipeline uses every demonstration in
`/iliad/u/tiangao/projects/robomimic/datasets/threading_d07_joint/image_v15_256.hdf5`.
The source contains 400 episodes and 96,290 frames. Its `train` and `valid`
masks are deliberately ignored.

The environment implementation is in the relocated robosuite checkout at
`/iris/u/tiangao/projects/robosuite`. The HDF5 records robosuite 1.5.2
`Threading_D07` with a 20 Hz absolute `JOINT_POSITION` Panda controller and
256x256 `agentview` and `robot0_eye_in_hand` RGB observations.

The raw action is `[absolute Panda joint target (7), gripper command (1)]`.
Conversion stores Pi-style `[absolute joint target (7), closure in [0,1] (1)]`.
Training uses `THREADING_JOINT`, which converts only the seven arm targets to
deltas and leaves the gripper absolute. Evaluation reverses the arm transform
before sending absolute targets to robosuite.

## Convert all episodes

Use local NVMe for the expensive conversion:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/tmp/openpi_threading_d07_joint_lerobot
uv run examples/robomimic/threading_d07_joint/convert_robomimic_data_to_lerobot.py
```

The converter defaults to `--filter-key all`. It validates the exact audited
episode/frame totals and writes `meta/robomimic_source_manifest.json`, tying
every numeric source demo name and length to its LeRobot episode index.

After verifying and publishing the dataset to the shared LeRobot root, rebuild
the exclusive subset keys:

```bash
uv run examples/robomimic/threading_d07_joint/build_filtering_keys.py
```

The builder checks source path, size, modification time, task, environment,
repo ID, FPS, episode order, episode lengths, and total counts before writing
keys. Source `mask/full` becomes `full_only`; source `mask/partial` becomes
`partial_only`. The masks are disjoint and together cover all 400 demos.

## Normalization and training

The canonical shared LeRobot root is now on Iris:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
uv run scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d07_joint_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d07_joint_low_mem_finetune`
- `pi05_robomimic_threading_d07_joint_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d07_joint_partial_only_low_mem_finetune`

The filtered configs intentionally reuse the all-episode D07 normalization
statistics. They never reuse D0 or D05 statistics.

## Evaluate

The evaluator reads the HDF5 only for environment/controller metadata and uses
fresh seeded `Threading_D07` resets. The launcher defaults to the relocated
`/iris/u/tiangao/projects/robosuite` checkout:

```bash
cd /iris/u/tiangao/projects/openpi
examples/robomimic/threading_d07_joint/run_eval.sh \
  pi05_robomimic_threading_d07_joint_partial_only_low_mem_finetune
```

## Tests

```bash
uv run pytest -q \
  examples/robomimic/threading_d07_joint/threading_d07_conversion_test.py \
  examples/robomimic/threading_d07_joint/threading_d07_filtering_test.py \
  examples/robomimic/threading_d07_joint/threading_d07_config_test.py
```
