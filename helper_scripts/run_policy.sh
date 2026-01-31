# source /iris/u/khhung/projects/openpi/.venv/bin/activate


python3 examples/droid/main.py \
    --remote_host=iliad-hgx-1.stanford.edu \
    --remote_port=8000 \
    --external_camera="left" \
    --left_camera_id=27904255_left \
    --right_camera_id=27904255_left \
    --wrist_camera_id=12841040_left \
    --instruction="pick up the red block and blue block and place them in the box" \
    --max_timesteps=400