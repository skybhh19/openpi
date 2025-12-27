source /iris/u/khhung/projects/openpi/.venv/bin/activate

TASK_NAME="${1:-pick_and_place_cube}"
REPO_NAME="${1:-johnson906/droid_pp_1}"
DATA_DIR="${2:-/iris/u/khhung/projects/expo_sampling/data/$TASK_NAME/success}"

uv run helper_scripts/convert_droid_data_to_lerobot.py \
    --data_dir="$DATA_DIR" \
    --repo_name="$REPO_NAME" \
    --language_annotation="pick up the cube and place in the box" \
    --push-to-hub