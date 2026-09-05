# Threading_D06_Hard absolute-joint pipeline

This pipeline uses all 400 demonstrations (128,232 frames) in
`/iris/u/tiangao/projects/robomimic/robomimic/datasets/threading_d06_hard_joint/image_v15_256.hdf5`.
Its `train` and `valid` masks are deliberately ignored.

The environment implementation is in `/iris/u/tiangao/projects/robosuite`.
The HDF5 records robosuite 1.5.2 `Threading_D06_Hard` with a 20 Hz
absolute `JOINT_POSITION` Panda controller and 256x256 `agentview_full` and
`robot0_eye_in_hand` RGB observations. The hard variant uses a 22 mm outer
ring with a 14 mm square aperture.

The raw action is `[absolute Panda joint target (7), gripper command (1)]`.
Conversion stores `[absolute joint target (7), closure in [0,1] (1)]`.
Training uses `THREADING_JOINT`, which converts only the seven arm targets to
deltas and leaves the gripper absolute. Evaluation reverses that arm transform
before sending absolute targets to robosuite. The optional
`THREADING_JOINT_ABSOLUTE` path instead trains directly on the stored absolute
arm targets. Both paths use closure in `[0,1]` for the model's gripper output
and send robosuite gripper sign in `[-1,1]` during evaluation. Both training
representations use the same converted LeRobot dataset; no reconversion is
needed.

## Convert all episodes

Use local storage for conversion and the Iris checkout's virtual environment:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/tmp/openpi_threading_d06_hard_joint_lerobot
./.venv/bin/python examples/robomimic/threading_d06_hard_joint/convert_robomimic_data_to_lerobot.py
```

The converter defaults to `--filter-key all`, validates the audited totals,
and writes `meta/robomimic_source_manifest.json` with every source demo name,
length, and resulting LeRobot episode index.

After publishing the verified dataset under the canonical Iris LeRobot root,
rebuild the exclusive subset keys:

```bash
./.venv/bin/python examples/robomimic/threading_d06_hard_joint/build_filtering_keys.py
```

Source `mask/full` becomes `full_only`; source `mask/partial` becomes
`partial_only`. Each contains 200 demos, they are disjoint, and they cover all
400 source demos. The same script also uses deterministic seed 0 to sample 50
full and 50 partial episodes, producing a balanced 100-demo filter and the two
50-demo component filters.

## Normalization and training

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d06_hard_joint_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d06_hard_joint_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_partial_only_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_sampled_100_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_sampled_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_sampled_partial_only_low_mem_finetune`

The filtered configs intentionally reuse the all-episode D06-hard
normalization statistics; they never use another threading variant's stats.

For direct absolute-joint fine-tuning, compute a separate normalization asset:

```bash
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d06_hard_joint_absolute_low_mem_finetune
```

Absolute-action training configs:

- `pi05_robomimic_threading_d06_hard_joint_absolute_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_absolute_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d06_hard_joint_absolute_partial_only_low_mem_finetune`

The two filtered absolute configs reuse the all-episode *absolute-action*
normalization statistics. They must not use the delta-action asset above.

## Evaluate

The evaluator reads the HDF5 only for environment/controller metadata and
performs fresh seeded `Threading_D06_Hard` resets. The policy server always
returns absolute arm targets: it reconstructs them for delta-trained configs
and passes them through for absolute-trained configs. Put the Iris robosuite
checkout first on `PYTHONPATH`:

```bash
cd /iris/u/tiangao/projects/openpi
PYTHONPATH=/iris/u/tiangao/projects/robosuite:/iris/u/tiangao/projects/openpi \
  ./.venv/bin/python examples/robomimic/threading_d06_hard_joint/main.py --help
```

## Tests

```bash
./.venv/bin/pytest -q examples/robomimic/threading_d06_hard_joint
```
