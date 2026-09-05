# Threading_D08 absolute-joint pipeline

This pipeline uses all 300 demonstrations (97,077 frames) in
`/iris/u/tiangao/projects/robomimic/robomimic/datasets/threading_d08_joint/image_v15_256.hdf5`.
Its `train` and `valid` masks are deliberately ignored.

The environment implementation is in `/iris/u/tiangao/projects/robosuite`.
The HDF5 records robosuite 1.5.2 `Threading_D08` with a 20 Hz absolute
`JOINT_POSITION` Panda controller and 256x256 `agentview_full` and
`robot0_eye_in_hand` RGB observations.

The raw action is `[absolute Panda joint target (7), gripper command (1)]`.
Conversion stores `[absolute joint target (7), closure in [0,1] (1)]`.
Training uses `THREADING_JOINT`, which converts only the seven arm targets to
deltas and leaves the gripper absolute. Evaluation reverses that arm transform
before sending absolute targets to robosuite.

## Convert all episodes

Use local NVMe for conversion:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/tmp/openpi_threading_d08_joint_lerobot
uv run examples/robomimic/threading_d08_joint/convert_robomimic_data_to_lerobot.py
```

The converter defaults to `--filter-key all`, validates the audited totals,
and writes `meta/robomimic_source_manifest.json` with every source demo name,
length, and resulting LeRobot episode index.

After publishing the verified dataset under the canonical Iris LeRobot root,
rebuild the exclusive subset keys:

```bash
uv run examples/robomimic/threading_d08_joint/build_filtering_keys.py
```

Source `mask/full` becomes `full_only`; source `mask/partial` becomes
`partial_only`. They are disjoint and cover all 300 demos.

## Normalization and training

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
uv run scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d08_joint_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d08_joint_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_partial_only_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_30k_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_full_only_30k_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_partial_only_30k_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_full_only_40k_low_mem_finetune`
- `pi05_robomimic_threading_d08_joint_full_only_50k_low_mem_finetune`

The extended-step and filtered configs intentionally reuse the existing
all-episode D08 normalization statistics; they never reuse another threading
variant's statistics.

## Evaluate

The evaluator reads the HDF5 for environment/controller metadata and performs
fresh seeded `Threading_D08` resets. Put the Iris robosuite checkout first on
`PYTHONPATH` when launching it:

```bash
cd /iris/u/tiangao/projects/openpi
PYTHONPATH=/iris/u/tiangao/projects/robosuite:$PYTHONPATH \
  uv run examples/robomimic/threading_d08_joint/main.py --help
```

## Tests

```bash
uv run pytest -q examples/robomimic/threading_d08_joint
```
