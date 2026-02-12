# source /iris/u/khhung/projects/openpi/.venv/bin/activate


python3 examples/droid/main_cartesian.py \
    --remote_host=iliad-hgx-1.stanford.edu \
    --remote_port=8000 \
    --external_camera="left" \
    --left_camera_id=27904255_left \
    --right_camera_id=27904255_left \
    --wrist_camera_id=12841040_left \
    --instruction="pick up the green block to the box and then pick up the yellow block and place it in the box" \
    --max_timesteps=1000