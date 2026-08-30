# Threading_D05 absolute-joint pipeline

This directory converts all 200 demonstrations in
`/iliad/u/tiangao/projects/robomimic/datasets/threading_d05_joint_v2/image_v15_256.hdf5`.
The default deliberately ignores the randomized `train` / `valid` masks.

The source action is `[absolute Panda joint target (7), gripper command (1)]`.
The converter stores Pi-style `[absolute joint target (7), closure in [0,1] (1)]`.
At training time, `LeRobotRobomimicDataConfig` uses the `THREADING_JOINT` task
configuration to convert only the seven arm targets to deltas and keep the gripper absolute.

```bash
export HF_LEROBOT_HOME=/tmp/openpi_threading_d05_joint_lerobot
uv run examples/robomimic/threading_d05_joint/convert_robomimic_data_to_lerobot.py
```

After copying the verified dataset to the shared LeRobot root, reproduce the
HDF5-mask-to-episode-index mapping with:

```bash
uv run examples/robomimic/threading_d05_joint/build_filtering_keys.py
```

The source masks are named `full` and `partial`. The generated files and
training config names call them `full_only` and `partial_only` to make their
exclusive membership explicit. Filtered configs intentionally reuse the
all-200 D05 normalization statistics.

```bash
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets
uv run scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d05_joint_low_mem_finetune
```

Training configs:

- `pi05_robomimic_threading_d05_joint_low_mem_finetune`
- `pi05_robomimic_threading_d05_joint_full_only_low_mem_finetune`
- `pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune`

## Evaluate

After the final `19999` checkpoint has been written, run the partial-only policy
server and simulator evaluation together with:

```bash
examples/robomimic/threading_d05_joint/run_eval.sh
```

Pass another config name as the first argument to evaluate a different D05 run:

```bash
examples/robomimic/threading_d05_joint/run_eval.sh \
  pi05_robomimic_threading_d05_joint_full_only_low_mem_finetune
```

The launcher accepts evaluator flags after the config name. Environment
variables such as `CHECKPOINT_STEP`, `PORT`, `OUTPUT_PATH`, `VIDEO_DIR`,
`ROBOSUITE_ROOT`, and `POLICY_MEM_FRACTION` override its defaults.

Evaluation uses the source HDF5 only for environment/controller metadata. Each
closed-loop trial starts from a fresh seeded `env.reset()` sample; no
demonstration XML or simulator state is restored.

## Tests

```bash
uv run pytest -q \
  examples/robomimic/threading_d05_joint/threading_d05_joint_conversion_test.py \
  examples/robomimic/threading_d05_joint/threading_d05_filtering_test.py \
  src/openpi/training/robomimic_config_test.py
```
