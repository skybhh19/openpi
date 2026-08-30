# DROID Pen-In-Blue-Cup

This folder contains the LeRobot conversion script and LeRobot episode-subset filters for the DROID pen-in-blue-cup task.

## Convert To LeRobot

The converter reads the raw DROID data from:

```bash
/iris/u/tiangao/projects/droid/data/success/2026-06-10
```

Run the full conversion:

```bash
source .venv/bin/activate
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/
uv run examples/droid/pen_in_blue_cup/convert_pen_in_blue_cup_data_to_lerobot.py --overwrite
```

The default output repo id is `skybhh19/droid_pen_in_blue_cup`, and each episode uses the prompt `Put the pen in the cup`.
Only valid transitions are written to LeRobot: `movement_enabled` must be true and `skip_action` must be false.

## Train

Use the same local LeRobot root when training:

```bash
source .venv/bin/activate
export HF_LEROBOT_HOME=/iliad/u/tiangao/lerobot_datasets/
```

Full fine-tune:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup \
  --resume
```

Low-memory LoRA fine-tune:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_low_mem_finetune \
  --resume
```

Low-memory LoRA fine-tunes on deterministic random LeRobot episode subsets:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_randompct25_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_randompct25_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_randompct50_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_randompct50_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_randompct75_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_randompct75_low_mem_finetune \
  --resume
```

Low-memory LoRA fine-tunes on human-label LeRobot episode subsets:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_observabilitypct25_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_observabilitypct25_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_observabilitypct50_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_observabilitypct50_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_observabilitypct75_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_observabilitypct75_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_optimalitypct25_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_optimalitypct25_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_optimalitypct50_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_optimalitypct50_low_mem_finetune \
  --resume

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_optimalitypct75_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_optimalitypct75_low_mem_finetune \
  --resume
```

Low-memory LoRA fine-tunes on score-ranked LeRobot episode subsets use the same
suffixes as the score filter files:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_pen_in_blue_cup_rank_h_wrist_ext_instant_abspct50_low_mem_finetune \
  --exp-name=pi05_droid_pen_in_blue_cup_rank_h_wrist_ext_instant_abspct50_low_mem_finetune \
  --resume
```

Available score suffixes are:

- `rank_h_wrist_ext_instant_abspct25`, `rank_h_wrist_ext_instant_abspct50`, `rank_h_wrist_ext_instant_abspct75`
- `rank_h_robot_wrist_ext_instant_abspct25`, `rank_h_robot_wrist_ext_instant_abspct50`, `rank_h_robot_wrist_ext_instant_abspct75`
- `rank_nll_robot_wrist_ext_minus_robot_instant_abspct25`, `rank_nll_robot_wrist_ext_minus_robot_instant_abspct50`, `rank_nll_robot_wrist_ext_minus_robot_instant_abspct75`
- `rank_nll_robot_wrist_ext_minus_action_prior_instant_abspct25`, `rank_nll_robot_wrist_ext_minus_action_prior_instant_abspct50`, `rank_nll_robot_wrist_ext_minus_action_prior_instant_abspct75`

Regenerate the random and human-label filters after editing annotations:

```bash
source .venv/bin/activate
python examples/droid/pen_in_blue_cup/build_pen_in_blue_cup_episode_filters.py
```

The generator verifies that the CSV episode names, raw DROID episode timestamp folders,
source-video episode names, and local LeRobot episode parquet files all match.
Human-label filters rank higher numeric labels first and randomly break ties with a
fixed seed.

The subset files live in `examples/droid/pen_in_blue_cup/lerobot_filtering_keys/`.
