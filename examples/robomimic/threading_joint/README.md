# Threading_D0 absolute-joint pipeline

This directory converts and evaluates the 256×256 robomimic dataset:

```text
/iliad/u/tiangao/projects/robomimic/datasets/threading_d0_joint_v3/image_v15_256.hdf5
```

The default converter uses all 63,927 transitions across all 200 demonstrations. Closed-loop evaluation does not
restore any demonstration state: every trial samples a fresh randomized initial state with `env.reset()`.

## Action-space decision

The raw action is eight-dimensional:

```text
[absolute Panda joint target in radians (7), robosuite gripper command (1)]
```

This is established by both the controller implementation and the data:

- HDF5 `env_args` selects `JOINT_POSITION` with `input_type="absolute"`; robosuite assigns the seven arm values
  directly to `goal_qpos` without scaling.
- Across all 63,927 transitions, each target is at most 0.045 rad from `obs/robot0_joint_pos`. The mean absolute
  target error falls from 0.0140 rad at the current observation to 0.0111 rad at `next_obs`, showing that the robot
  moves toward the supplied absolute target. Arm action / observed-joint correlations are 0.988–0.999.
- The gripper action contains exactly `{-1, +1}`. Panda's gripper code defines `-1` as open and `+1` as closed.
  Recorded open frames have about 0.08 m finger width, while grasping frames have about 0.03 m width.
- `rewards` and `dones` are identical binary success labels. Every demonstration ends in a persistent positive
  suffix (3,765 positive frames overall), which is expected because collection used `ignore_done=true`.

The LeRobot dataset uses Pi's standard Franka convention:

```text
state  = [current_joint_position_rad (7), measured_closure (1)]
action = [absolute_joint_target_rad (7), target_closure (1)]

measured_closure = clip(1 - (finger_qpos[0] - finger_qpos[1]) / 0.08, 0, 1)
target_closure   = (robosuite_gripper_command + 1) / 2
```

Only the seven arm dimensions pass through `DeltaActions` during training. This makes the model target
`q_target - q_current`, matching the Pi joint-angle convention. The inverse `AbsoluteActions` transform adds the
current joint state back at inference. The gripper remains absolute throughout; the final policy output maps Pi
closure back with `robosuite_command = 2 * closure - 1`.

Thus, `main.py` sends absolute joint targets directly to robosuite. It must not add the current joint state again.

## Images

Both stored cameras are upright 256×256 RGB. The converter copies them without resizing or flipping. OpenPI's model
transform performs the single resize-with-padding to 224×224 expected by π0.5. Live raw MuJoCo observations are
vertically flipped once to match the stored robomimic images and are also left at 256×256 until that model transform.

## Convert

Validate the controller, arrays, masks, image resolution, absolute-action alignment, and gripper values without
writing a LeRobot dataset:

```bash
uv run examples/robomimic/threading_joint/convert_robomimic_data_to_lerobot.py --dry-run
```

Convert all training data:

```bash
uv run examples/robomimic/threading_joint/convert_robomimic_data_to_lerobot.py
```

The result is `local/robomimic_threading_d0_joint_v3_256` under `HF_LEROBOT_HOME`. Existing output is preserved
unless `--overwrite` is explicitly supplied. `--filter-key` remains available only for intentional subset studies.

### Full and partial label subsets

The source HDF5 has two disjoint 100-demo masks: `mask/partial` contains `demo_1` through `demo_100`, and `mask/full`
contains `demo_101` through `demo_200`. In the all-200 LeRobot conversion, those map to episode indices `0..99` and
`100..199`, respectively. The checked-in filters are under `lerobot_filtering_keys/`, and these LoRA configs use them:

- `pi05_robomimic_threading_d0_joint_label_partial_low_mem_finetune`
- `pi05_robomimic_threading_d0_joint_label_full_low_mem_finetune`

The index files target the all-200 converted dataset. Do not use them with a separately converted single-mask dataset,
whose episode indices would be renumbered. Both filtered configs intentionally reuse the all-200 statistics from
`pi05_robomimic_threading_d0_joint_low_mem_finetune`, keeping normalization fixed across the label comparison.

## Fine-tune π0.5

Four configs cover full / dual-LoRA training and base / fresh normalization statistics:

| Config | Parameters | Normalization |
|---|---|---|
| `pi05_robomimic_threading_d0_joint_franka_stats_low_mem_finetune` | dual LoRA | π0.5 base `franka` |
| `pi05_robomimic_threading_d0_joint_franka_stats_finetune` | full | π0.5 base `franka` |
| `pi05_robomimic_threading_d0_joint_low_mem_finetune` | dual LoRA | fresh dataset stats |
| `pi05_robomimic_threading_d0_joint_finetune` | full | fresh dataset stats |

The Franka-stat LoRA config is the recommended first run: robot, 7-radian-joint + `[0,1]` gripper convention, and
20 Hz control rate match the π0.5 Franka pretraining definition. It intentionally reuses base statistics, so do not
run `compute_norm_stats.py` for a `franka_stats` config.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_robomimic_threading_d0_joint_franka_stats_low_mem_finetune \
  --exp-name=threading_d0_joint_v3 --overwrite
```

It is still useful to compare fresh statistics, as recommended in `docs/norm_stats.md`. Compute them before using a
config without `franka_stats`:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d0_joint_low_mem_finetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_robomimic_threading_d0_joint_low_mem_finetune \
  --exp-name=threading_d0_joint_v3 --overwrite
```

All configs use π0.5 base weights, 16-step action chunks, an 8-D unpadded state/action representation, and the
model's standard 32-D padded action head.

## Evaluate in robosuite

Start the websocket policy server using the same config as training:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_robomimic_threading_d0_joint_franka_stats_low_mem_finetune \
  --policy.dir=checkpoints/pi05_robomimic_threading_d0_joint_franka_stats_low_mem_finetune/threading_d0_joint_v3/19999
```

In a Python environment with this robosuite checkout importable, run:

```bash
PYTHONPATH=/iliad/u/tiangao/projects/robosuite:$PYTHONPATH \
python examples/robomimic/threading_joint/main.py \
  --seed 0 \
  --video-dir data/robomimic_threading_joint_videos
```

The HDF5 is read only for the recorded environment/controller metadata. The evaluator seeds robosuite once, calls
`env.reset()` for every episode, and therefore samples fresh task initial states from the environment distribution.
It clips absolute targets to `env.action_spec` and replans every eight simulator steps by default. The seed, reset
source, individual trials, and aggregate metrics are written to `data/robomimic_threading_d0_joint_256_eval.json`.

To extend an existing 20-episode report with the next 30 seeded resets, use `--start-episode=20
--num-episodes=30`. Resume mode validates the existing report, fast-forwards the environment RNG through the saved
reset prefix, and appends episodes 20–49 before recomputing the aggregate metrics.

## Tests

```bash
uv run pytest -q \
  examples/robomimic/threading_joint/threading_joint_conversion_test.py \
  examples/robomimic/threading_joint/threading_joint_main_test.py \
  src/openpi/policies/robomimic_policy_test.py \
  src/openpi/training/robomimic_config_test.py
```
