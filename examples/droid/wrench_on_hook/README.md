# Wrench On Hook

This task uses DROID data from a Franka Emika Panda and fine-tunes `pi05_base` with joint-position control.

Default prompt:

```text
Hang the wrench on the hook
```

For any DROID dataset with the same raw format, set these variables first and reuse the commands below. The values shown here are the 0801 example.

```bash
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets
export RAW_DATA_DIR=/iris/u/tiangao/wrench_on_hook_0801
export DATASET_TAG=08012026
export REPO_ID=skybhh19/droid_wrench_on_hook_${DATASET_TAG}_jointpos
export LEROBOT_DATASET_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
export FILTER_PREFIX=wrench_on_hook_${DATASET_TAG}_jointpos
export LABEL_CSV=/iris/u/tiangao/wrench_on_hook_0801_scores.csv
```

For a new dataset, change `RAW_DATA_DIR`, `DATASET_TAG`, `REPO_ID`, and `LABEL_CSV`. Keep the `_jointpos` suffix when the converted actions are joint-position targets.

## Data Conversion

Use the joint-position converter. It writes actions as `[action/joint_position(7), action/gripper_position(1)]`.

```bash
uv run examples/droid/wrench_on_hook/convert_wrench_on_hook_joint_position_data_to_lerobot.py \
  --data-dir "$RAW_DATA_DIR" \
  --repo-id "$REPO_ID"
```

For a Slurm conversion job:

```bash
sbatch --parsable \
  --partition=iliad \
  --time=12:00:00 \
  --nodes=1 \
  --cpus-per-task=28 \
  --mem=128G \
  --job-name="convert_wrench_${DATASET_TAG}_jointpos" \
  --output="slurm-%j-convert-wrench-${DATASET_TAG}-jointpos.out" \
  --account=iliad \
  --wrap="bash -lc 'cd /iliad/u/tiangao/projects/openpi && source .venv/bin/activate && export HF_LEROBOT_HOME=${HF_LEROBOT_HOME} && uv run examples/droid/wrench_on_hook/convert_wrench_on_hook_joint_position_data_to_lerobot.py --data-dir ${RAW_DATA_DIR} --repo-id ${REPO_ID}'"
```

Verify the converted dataset:

```bash
python -m json.tool "$LEROBOT_DATASET_DIR/meta/info.json"
```

The 0801 example has `100` episodes, `17933` frames, `15` fps, and robot type `panda`.

## Filtering Keys

Filtering keys are JSON files containing LeRobot episode indices. The filter builder supports different datasets as long as the CSV has one row per raw episode and an `episode` column matching the raw episode directory name. For human-ranked filters, pass the numeric label column with `--label-column`; for the 0801 example this is `score`.

Run a dry check first:

```bash
uv run examples/droid/wrench_on_hook/build_wrench_on_hook_episode_filters.py \
  --annotations-csv "$LABEL_CSV" \
  --raw-data-dir "$RAW_DATA_DIR" \
  --lerobot-dataset-dir "$LEROBOT_DATASET_DIR" \
  --file-prefix "$FILTER_PREFIX" \
  --dataset-name "$REPO_ID" \
  --label-column score \
  --filter-pcts 25,50,75 \
  --check-only
```

Write the filter files:

```bash
uv run examples/droid/wrench_on_hook/build_wrench_on_hook_episode_filters.py \
  --annotations-csv "$LABEL_CSV" \
  --raw-data-dir "$RAW_DATA_DIR" \
  --lerobot-dataset-dir "$LEROBOT_DATASET_DIR" \
  --file-prefix "$FILTER_PREFIX" \
  --dataset-name "$REPO_ID" \
  --label-column score \
  --filter-pcts 25,50,75
```

This writes files under:

```text
examples/droid/wrench_on_hook/lerobot_filtering_keys/
```

For the 0801 example, the configured files are:

```text
wrench_on_hook_08012026_jointpos_randompct25_episode_indices.json
wrench_on_hook_08012026_jointpos_randompct50_episode_indices.json
wrench_on_hook_08012026_jointpos_randompct75_episode_indices.json
wrench_on_hook_08012026_jointpos_observabilitypct25_episode_indices.json
wrench_on_hook_08012026_jointpos_observabilitypct50_episode_indices.json
wrench_on_hook_08012026_jointpos_observabilitypct75_episode_indices.json
```

## Policy Fine-Tuning

The recommended joint-position runs use `pi05_base` LoRA fine-tuning with Franka norm stats:

```text
assets_dir = gs://openpi-assets/checkpoints/pi05_base/assets
asset_id   = franka
```

Do not run `compute_norm_stats.py` for `_franka_stats` configs. They intentionally use the base Franka stats.

Training configs are still explicit OpenPI config entries. For a new dataset, add a matching config in `src/openpi/training/config.py` by copying the 0801 pattern and changing `name`, `repo_id`, and any filter filenames. Then set `CONFIG` to that config name.

For the 0801 full-data run:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CONFIG=pi05_base_droid_wrench_on_hook_08012026_jointpos_franka_stats_low_mem_finetune

uv run scripts/train.py "$CONFIG" \
  --exp-name="$CONFIG" \
  --resume
```

For the 0801 filtered runs:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

for FILTER in randompct25 randompct50 randompct75 observabilitypct25 observabilitypct50 observabilitypct75; do
  CONFIG="pi05_base_droid_wrench_on_hook_08012026_jointpos_franka_stats_${FILTER}_low_mem_finetune"
  uv run scripts/train.py "$CONFIG" \
    --exp-name="$CONFIG" \
    --resume
done
```

For Slurm, use the same `CONFIG` value in `train_sc.sh` and submit:

```bash
sbatch train_sc.sh
```

If Hopper is busy, L40S is a good fallback for these LoRA jobs:

```bash
#SBATCH --partition=iliad
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=ada
```

## Evaluation

The robot-side eval script is task-specific but dataset-agnostic. It only needs a policy server serving a checkpoint trained with the matching config.

Start the OpenPI policy server from this repository. Replace `CONFIG` and `STEP` with the run to evaluate.

```bash
export CONFIG=pi05_base_droid_wrench_on_hook_08012026_jointpos_franka_stats_low_mem_finetune
export STEP=19999

uv run scripts/serve_policy.py --port 8123 \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="checkpoints/$CONFIG/$CONFIG/$STEP"
```

On the robot/DROID side, run:

```bash
python /iris/u/tiangao/projects/droid/scripts/main_wrench_to_hook_joint_position.py \
  --remote-host <POLICY_SERVER_IP> \
  --remote-port 8123
```

The eval script defaults to the training prompt, `Hang the wrench on the hook`, and uses:

```text
RobotEnv(action_space="joint_position", gripper_action_space="position")
```

It treats policy actions as absolute joint targets plus gripper position, clips to Panda limits, and limits per-step joint target changes with `--max-joint-delta`.
