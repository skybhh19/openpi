"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import h5py
import numpy as np

import os
# os.environ['HF_LEROBOT_HOME'] = '/iris/u/tiangao/lerobot_datasets/'

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils import get_language_instruction
import tyro
print("finish import")

ENV_NAME = "NutAssemblySquare"
REPO_NAME = f"skybhh19/lerobot_robomimic_{ENV_NAME}_ph"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = f"/iris/u/tiangao/projects/aiorl/robomimic/datasets/{ENV_NAME}_ph/image_v141_n30.hdf5"  # For simplicity we will combine multiple Libero datasets into one training dataset


    
def main(push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    hdf5_file = h5py.File(RAW_DATASET_PATH, 'r')
    demo_keys = sorted([k for k in hdf5_file['data'].keys() if k.startswith('demo_')])
    first_demo = hdf5_file['data'][demo_keys[0]]
    action_dim = first_demo['actions'].shape[1]
    state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    state_dim = sum(first_demo['obs'][k].shape[1] for k in state_keys)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=20, # TODO: check the fps for robomimic, (20Hz for robomimic? 10Hz for libero?)
        features={
            "image": { # agentview_image
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {# robot0_eye_in_hand_image
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for demo_key in demo_keys:
        demo = hdf5_file['data'][demo_key]
        traj_len = len(demo['actions'])
        for t in range(traj_len):
            state = np.concatenate([demo['obs'][k][t] for k in state_keys])
            dataset.add_frame(
                {
                    "image": demo["obs"]["agentview_image"][t].astype(np.uint8),
                    "wrist_image": demo["obs"]["robot0_eye_in_hand_image"][t].astype(np.uint8),
                    "state": state.astype(np.float32),
                    "actions": demo["actions"][t].astype(np.float32),
                    "task": get_language_instruction(ENV_NAME),
                }
            )
        dataset.save_episode()
        print(f"Saved {demo_key}")

    hdf5_file.close()
    
    # Optionally push to Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "panda", "rlds"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)