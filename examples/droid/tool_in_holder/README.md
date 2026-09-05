# Tool In Holder

This task uses DROID data from a Franka Emika Panda and fine-tunes `pi05_base` with joint-position control.

Default prompt:

```text
Insert the tool into the holder
```

The scripts are not tied to one dataset. For any DROID dataset with the same raw format, set these variables first and reuse the commands below. The values shown here are the 0801 example.

```bash
export HF_LEROBOT_HOME=/iris/u/tiangao/lerobot_datasets
export RAW_DATA_DIR=/iris/u/tiangao/tool_in_holder_0801
export DATASET_TAG=08012026
export REPO_ID=skybhh19/droid_tool_in_holder_${DATASET_TAG}_jointpos
export LEROBOT_DATASET_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
export FILTER_PREFIX=tool_in_holder_${DATASET_TAG}_jointpos
export LABEL_CSV=/path/to/tool_in_holder_${DATASET_TAG}_scores.csv
```

For a new dataset, change `RAW_DATA_DIR`, `DATASET_TAG`, `REPO_ID`, and, if filtering is needed, `LABEL_CSV`. Keep the `_jointpos` suffix when the converted actions are joint-position targets.

## Data Conversion

Use the joint-position converter. It writes actions as `[action/joint_position(7), action/gripper_position(1)]`.

```bash
uv run examples/droid/tool_in_holder/convert_tool_in_holder_joint_position_data_to_lerobot.py \
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
  --job-name="convert_tool_${DATASET_TAG}_jointpos" \
  --output="slurm-%j-convert-tool-${DATASET_TAG}-jointpos.out" \
  --account=iliad \
  --wrap="bash -lc 'cd /iliad/u/tiangao/projects/openpi && source .venv/bin/activate && export HF_LEROBOT_HOME=${HF_LEROBOT_HOME} && uv run examples/droid/tool_in_holder/convert_tool_in_holder_joint_position_data_to_lerobot.py --data-dir ${RAW_DATA_DIR} --repo-id ${REPO_ID}'"
```

Verify the converted dataset:

```bash
python -m json.tool "$LEROBOT_DATASET_DIR/meta/info.json"
```

The 0801 example has `30` episodes, `4948` frames, `15` fps, and robot type `panda`.

## Filtering Keys

The current configured training path for tool-in-holder is full-data only. If you add labels later, create a CSV with one row per LeRobot episode and an `episode` column matching the raw episode directory names. A numeric label column such as `score` can be used for ranked filtering.

The wrench filter builder is generic enough to generate episode-index JSONs for this task when all paths are passed explicitly:

```bash
mkdir -p examples/droid/tool_in_holder/lerobot_filtering_keys

uv run examples/droid/wrench_on_hook/build_wrench_on_hook_episode_filters.py \
  --annotations-csv "$LABEL_CSV" \
  --raw-data-dir "$RAW_DATA_DIR" \
  --lerobot-dataset-dir "$LEROBOT_DATASET_DIR" \
  --output-dir examples/droid/tool_in_holder/lerobot_filtering_keys \
  --file-prefix "$FILTER_PREFIX" \
  --dataset-name "$REPO_ID" \
  --label-column score \
  --filter-pcts 25,50,75 \
  --check-only
```

Remove `--check-only` to write the files. To train filtered variants, also add matching `episode_indices_path` configs in `src/openpi/training/config.py`, following the `wrench_on_hook_08012026_jointpos` pattern.

## Policy Fine-Tuning

The configured full-data run uses `pi05_base` LoRA fine-tuning with Franka norm stats:

```text
assets_dir = gs://openpi-assets/checkpoints/pi05_base/assets
asset_id   = franka
```

Do not run `compute_norm_stats.py` for `_franka_stats` configs. They intentionally use the base Franka stats.

Training configs are explicit OpenPI config entries. For a new dataset, add a matching config in `src/openpi/training/config.py` by copying the 0801 pattern and changing `name` and `repo_id`. Then set `CONFIG` to that config name.

For the 0801 full-data run:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CONFIG=pi05_base_droid_tool_in_holder_08012026_jointpos_franka_stats_low_mem_finetune

uv run scripts/train.py "$CONFIG" \
  --exp-name="$CONFIG" \
  --resume
```

For Slurm, use the same `CONFIG` value in `train_sc.sh` and submit:

```bash
sbatch train_sc.sh
```

If Hopper is busy, L40S is a good fallback for this LoRA job:

```bash
#SBATCH --partition=iliad
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=ada
```

## Evaluation

The robot-side eval script is task-specific but dataset-agnostic. It only needs a policy server serving a checkpoint trained with the matching config.

Start the OpenPI policy server from this repository. Replace `CONFIG` and `STEP` with the run to evaluate.

```bash
export CONFIG=pi05_base_droid_tool_in_holder_08012026_jointpos_franka_stats_low_mem_finetune
export STEP=19999

uv run scripts/serve_policy.py --port 8123 \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="checkpoints/$CONFIG/$CONFIG/$STEP"
```

On the robot/DROID side, run:

```bash
python /iris/u/tiangao/projects/droid/scripts/main_tool_in_holder_joint_position.py \
  --remote-host <POLICY_SERVER_IP> \
  --remote-port 8123
```

The eval script defaults to the training prompt, `Insert the tool into the holder`, and uses:

```text
RobotEnv(action_space="joint_position", gripper_action_space="position")
```

It treats policy actions as absolute joint targets plus gripper position, clips to Panda limits, and limits per-step joint target changes with `--max-joint-delta`.
