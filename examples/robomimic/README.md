# Robomimic threading fine-tuning and evaluation

This directory contains the end-to-end pipeline for fine-tuning π0.5 on the
Robomimic threading tasks and evaluating the resulting policy in robosuite.
Run all OpenPI commands from the Iris checkout:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
```

Use this repository's virtual environment at
`/iris/u/tiangao/projects/openpi/.venv`. The examples below invoke its Python
executable directly so a stale environment from `/iliad` or `/iliad2` cannot
be selected accidentally.

## Required data flow and image resolution

The image-rendering step and the LeRobot conversion are separate:

```text
raw/state Robomimic HDF5
  -> re-render camera observations at 256x256 in the Robomimic repository
  -> image_v15_256.hdf5
  -> convert the 256x256 HDF5 to LeRobot in this OpenPI repository
  -> build label filters and normalization statistics
  -> fine-tune π0.5
  -> serve the checkpoint and evaluate it in robosuite
```

**The 256×256 re-render in the Robomimic repository is required. Do not use
the old 84×84 `image_v15.hdf5` files and do not upscale those images inside the
LeRobot converter.** The converters validate that both camera arrays are
256×256 RGB. They preserve those images in LeRobot; OpenPI performs the one
resize-with-padding from 256×256 to π0.5's 224×224 model input during model
preprocessing.

A typical re-render command, run in the Robomimic repository with its
robomimic/robosuite environment, is:

```bash
cd /iris/u/tiangao/projects/robomimic
python robomimic/scripts/dataset_states_to_obs.py \
  --dataset /path/to/the/state_dataset.hdf5 \
  --output_name image_v15_256.hdf5 \
  --done_mode 2 \
  --camera_names agentview_full robot0_eye_in_hand \
  --camera_height 256 \
  --camera_width 256
```

Use the camera names recorded for the task: D0, D05, and D07 use `agentview`,
while the current D06-hard, D06-harder, and D08 datasets use
`agentview_full`. Preserve the actions, states, masks, `env_args`, and the
action dictionary where present in the rendered output. The task-specific
converter is the final authority and fails if the HDF5 schema, controller,
task name, images, or audited episode totals do not match.

## Data and action conventions

The joint-position HDF5 files record an absolute Panda `JOINT_POSITION`
controller. A converted frame contains:

```text
state  = [current Panda joint position in radians (7), measured closure (1)]
action = [absolute Panda joint target in radians (7), target closure (1)]
```

Measured and target closure use `[0, 1]`, where 0 is open and 1 is closed.
The model state and action are logically 8-D and are padded to the π0.5
model's 32-D action width. Policies predict 16-step action chunks.

The default joint configuration is `THREADING_JOINT`. It converts only the
seven arm dimensions to
`absolute target - current joint position` for training. The gripper remains
an absolute closure target. At inference, the output transform adds the
current arm state back, and the evaluator sends an 8-D absolute joint command
to robosuite.

`THREADING_JOINT_ABSOLUTE` instead trains directly on the seven stored
absolute arm targets. It is currently available for D06-hard and the
D06-hard 1,000-demo dataset. Delta and absolute training use the same LeRobot
dataset, but they require separate normalization statistics. The D0 OSC
configs are a separate 9-D-state/7-D-action pipeline and preserve their native
controller actions; the joint-position rules above do not apply to them.

## Available joint-position pipelines

Each task directory documents its source dataset, exact totals, LeRobot repo
ID, config names, label counts, and any task-specific camera alias:

| Task | Pipeline |
| --- | --- |
| Threading D0 joint | [threading_joint](threading_joint/README.md) |
| Threading D05 v2 | [threading_d05_joint](threading_d05_joint/README.md) |
| Threading D05 v3 | [threading_d05_joint_v3](threading_d05_joint_v3/README.md) |
| Threading D06 hard, 400 demos | [threading_d06_hard_joint](threading_d06_hard_joint/README.md) |
| Threading D06 hard, 1,000 demos | [threading_d06_hard_joint_1000](threading_d06_hard_joint_1000/README.md) |
| Threading D06 hard wrist-up, 200 demos | [threading_d06_hard_wristup_joint_200](threading_d06_hard_wristup_joint_200/README.md) |
| Threading D06 hard wrist-up, 300 demos | [threading_d06_hard_wristup_joint_300](threading_d06_hard_wristup_joint_300/README.md) |
| Threading D06 hard wrist-up, 400 demos | [threading_d06_hard_wristup_joint_400](threading_d06_hard_wristup_joint_400/README.md) |
| Threading D06 hard wrist-up, 600 demos | [threading_d06_hard_wristup_joint_600](threading_d06_hard_wristup_joint_600/README.md) |
| Threading D06 harder, 600 demos | [threading_d06_harder_joint_600](threading_d06_harder_joint_600/README.md) |
| Threading D07 | [threading_d07_joint](threading_d07_joint/README.md) |
| Threading D08 | [threading_d08_joint](threading_d08_joint/README.md) |

Some source HDF5 files remain on a legacy data volume. That does not change
the required OpenPI checkout, virtual environment, robosuite checkout, or
LeRobot destination on Iris. Pass `--data-path` explicitly when a source HDF5
has moved.

## 1. Validate and convert all episodes

Select the task directory, then validate the source without writing output:

```bash
cd /iris/u/tiangao/projects/openpi
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets

./.venv/bin/python \
  examples/robomimic/threading_d06_hard_joint/convert_robomimic_data_to_lerobot.py \
  --dry-run
```

Convert after validation succeeds:

```bash
./.venv/bin/python \
  examples/robomimic/threading_d06_hard_joint/convert_robomimic_data_to_lerobot.py
```

The default is `--filter-key all` with no episode limit. This deliberately
uses every episode in `/data` and ignores the source `train` and `valid`
masks. Do not pass a train/validation mask for the canonical conversion.
Full/partial comparisons are filters over this one complete LeRobot dataset,
not separately converted datasets.

The converter writes the dataset below
`$HF_LEROBOT_HOME/local/<task-repo-id>` and records the source demo name,
length, and resulting episode index in
`meta/robomimic_source_manifest.json`. Existing output is preserved unless
`--overwrite` is explicitly supplied.

Conversion can be staged under `/tmp` if needed. Set `HF_LEROBOT_HOME` to a
dedicated temporary directory, validate the completed dataset, then copy the
entire `local/<task-repo-id>` directory to
`/iris/u/tiangao/lerobot_datasets/local/`. Build filters and compute stats only
after the complete dataset is at that canonical path.

## 2. Build full and partial filters

For a task with explicit HDF5 `mask/full` and `mask/partial` labels, run:

```bash
./.venv/bin/python \
  examples/robomimic/threading_d06_hard_joint/build_filtering_keys.py
```

The builder verifies the source manifest, episode ordering, lengths, totals,
and label coverage before writing LeRobot episode-index JSON files under the
task's `lerobot_filtering_keys/` directory. `full_only` and `partial_only`
must be disjoint and together cover the complete dataset.

Do not infer labels from reward, success, episode order, or train/valid masks.
If `/mask/full` or `/mask/partial` is absent, stop and establish the intended
labels before creating filtered configs. If the source HDF5 was updated after
conversion, reconvert it before rebuilding filters so the manifest remains a
valid provenance check.

## 3. Compute normalization statistics

Compute statistics with the all-episode config for the exact task dataset:

```bash
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d06_hard_joint_low_mem_finetune
```

The `full_only`, `partial_only`, sampled-subset, and longer-schedule configs
reuse that dataset's all-episode statistics. Do not recompute subset-specific
statistics and do not reuse statistics from another threading variant.

For an absolute-action config, compute the separate all-episode absolute
asset once:

```bash
./.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_robomimic_threading_d06_hard_joint_absolute_low_mem_finetune
```

All absolute subsets reuse the matching all-episode absolute statistics.
Never use delta-action statistics for an absolute-action config or vice
versa. D0 configs whose names contain `franka_stats` intentionally load π0.5
base Franka statistics and do not need a local stats computation.

## 4. Fine-tune π0.5

Use the config name as the experiment name when using the supplied evaluation
launcher:

```bash
CONFIG=pi05_robomimic_threading_d06_hard_joint_full_only_low_mem_finetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  ./.venv/bin/python scripts/train.py "$CONFIG" \
  --exp-name="$CONFIG"
```

Add `--overwrite` only when the existing checkpoint directory is intentionally
being replaced. Use `--resume` to continue an interrupted compatible run.

For Slurm, edit both the job name and config in `train_sc.sh`, then submit it
from the Iris checkout:

```bash
cd /iris/u/tiangao/projects/openpi
sbatch train_sc.sh
```

This matters because `train_sc.sh` activates `.venv` and resolves source files
relative to its working directory. A job submitted from `/iliad2` can load an
old checkout and fail with `Config ... not found` even though the config exists
on Iris. `HF_LEROBOT_HOME` must remain
`/iris/u/tiangao/lerobot_datasets` in the job.

All current threading configs save and retain checkpoints every 5,000 steps.
The final checkpoint is named with the zero-based last loop step:

| Training length | Final checkpoint |
| --- | --- |
| 20,000 steps | `19999` |
| 30,000 steps | `29999` |
| 40,000 steps | `39999` |
| 50,000 steps | `49999` |

## 5. Evaluate in robosuite

Activate the Iris OpenPI environment and call the task wrapper. It starts the
policy server, waits for it to become ready, runs closed-loop evaluation, and
then stops the server:

```bash
cd /iris/u/tiangao/projects/openpi
source .venv/bin/activate

CONFIG=pi05_robomimic_threading_d06_hard_joint_full_only_low_mem_finetune
CHECKPOINT_STEP=19999 \
  examples/robomimic/threading_d06_hard_joint/run_eval.sh "$CONFIG" \
  --num-episodes=50 \
  --seed=0
```

The launcher defaults to `/iris/u/tiangao/projects/robosuite`, port 8000, and
GPU 0. Useful overrides include `CHECKPOINT_DIR`, `CHECKPOINT_STEP`,
`CUDA_VISIBLE_DEVICES`, `PORT`, `OUTPUT_PATH`, `VIDEO_DIR`, `ROBOSUITE_ROOT`,
and `POLICY_MEM_FRACTION`. If training used a custom experiment name, set
`CHECKPOINT_DIR` because the launcher's default is:

```text
checkpoints/<config>/<config>/<step>
```

The policy server must use the same config as training so it applies the
matching normalization and delta/absolute output transform. Both delta-trained
and absolute-trained joint policies ultimately send absolute 7-D arm targets
plus a robosuite gripper sign to the environment.

The evaluator reads the source HDF5 only for task and controller metadata. It
does not replay demonstration states or XML. Each trial uses a fresh seeded
`env.reset()`, observes live 256×256 cameras, replans from the policy's action
chunk, writes an aggregate JSON report, and optionally writes videos.

## Tests and preflight checks

Run the task-specific tests plus the shared Robomimic transform/config tests:

```bash
./.venv/bin/pytest -q \
  examples/robomimic/threading_d06_hard_joint \
  src/openpi/policies/robomimic_policy_test.py \
  src/openpi/training/robomimic_config_test.py \
  src/openpi/training/robomimic_threading_config_test.py
```

Before launching a long job, check the paths that will actually be used:

```bash
pwd
readlink -f .venv/bin/python
./.venv/bin/python -c 'import openpi; print(openpi.__file__)'
printf '%s\n' "$HF_LEROBOT_HOME"
```

Expected values begin with `/iris/u/tiangao/projects/openpi` for OpenPI and
`/iris/u/tiangao/lerobot_datasets` for LeRobot data.
