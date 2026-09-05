# Threading_D06_Harder 600-demo joint-position pipeline

This pipeline uses all 600 demonstrations (253,863 frames) in
`/iris/u/tiangao/projects/robomimic/robomimic/datasets/threading_d06_harder_joint_600/image_v15_256.hdf5`.
Its `train` and `valid` masks are deliberately ignored.

The HDF5 records robosuite 1.5.2 `Threading_D06_Harder` at 20 Hz with a
Panda absolute `JOINT_POSITION` controller and 256x256 `agentview_full` and
`robot0_eye_in_hand` RGB observations. The environment implementation is in
`/iris/u/tiangao/projects/robosuite`; this variant uses a 20 mm outer ring
with a 12 mm square aperture.

The raw action is `[absolute Panda joint target (7), gripper command (1)]`.
Conversion stores `[absolute joint target (7), closure in [0,1] (1)]`.
Fine-tuning uses `THREADING_JOINT`, which turns only the seven arm targets into
deltas and leaves gripper closure absolute. Evaluation reconstructs absolute
arm targets before sending the 8-D command to robosuite.

## Convert and filter

Always use the Iris checkout's virtual environment and LeRobot home:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
./.venv/bin/python \
  examples/robomimic/threading_d06_harder_joint_600/convert_robomimic_data_to_lerobot.py
./.venv/bin/python \
  examples/robomimic/threading_d06_harder_joint_600/build_filtering_keys.py
```

The converter defaults to `--filter-key all`, validates the audited source
totals, and writes `meta/robomimic_source_manifest.json` with every source
demo's target episode index and frame count. For faster conversion, an exact
local HDF5 copy may be provided with `--data-path`; use
`--source-provenance-path` for the canonical Iris source so its fingerprint is
retained in the manifest.

The output is
`/iris/u/tiangao/lerobot_datasets/local/robomimic_threading_d06_harder_joint_600_256`.
Source `mask/full` becomes `full_only` (300 demos / 133,708 frames) and source
`mask/partial` becomes `partial_only` (300 demos / 120,155 frames). They are
disjoint and cover every converted demo.

## Normalization and training

```bash
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d06_harder_joint_600_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d06_harder_joint_600_low_mem_finetune`
- `pi05_robomimic_threading_d06_harder_joint_600_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d06_harder_joint_600_partial_only_low_mem_finetune`

The filtered configs reuse the all-episode normalization stats for this exact
dataset. All three train for 20,000 steps and save/keep checkpoints every
5,000 steps.

## Evaluate and test

The evaluator reads controller metadata from the source but uses fresh seeded
`Threading_D06_Harder` environment resets:

```bash
PYTHONPATH=/iris/u/tiangao/projects/robosuite:/iris/u/tiangao/projects/openpi \
  ./.venv/bin/python \
  examples/robomimic/threading_d06_harder_joint_600/main.py --help

./.venv/bin/pytest -q examples/robomimic/threading_d06_harder_joint_600
```
