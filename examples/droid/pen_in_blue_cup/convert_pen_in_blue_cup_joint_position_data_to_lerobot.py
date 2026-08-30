"""
Convert the DROID pen-in-blue-cup dataset to LeRobot format with joint-position actions.

Example:
uv run examples/droid/pen_in_blue_cup/convert_pen_in_blue_cup_joint_position_data_to_lerobot.py \
    --data-dir /iris/u/tiangao/projects/droid/data/success/2026-06-10 \
    --repo-id skybhh19/droid_pen_in_blue_cup_jointpos

The resulting dataset is saved under $LEROBOT_HOME / <repo-id>.
"""

# ruff: noqa: E402, I001

from pathlib import Path
import shutil
import sys

import numpy as np
from tqdm import tqdm
import tyro

DROID_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(DROID_EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(DROID_EXAMPLES_DIR))
TASK_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(TASK_EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(TASK_EXAMPLES_DIR))

from convert_droid_data_to_lerobot import is_valid_transition
from convert_droid_data_to_lerobot import load_trajectory
from convert_droid_data_to_lerobot import resize_image
from convert_pen_in_blue_cup_data_to_lerobot import create_lerobot_dataset
from convert_pen_in_blue_cup_data_to_lerobot import DEFAULT_DATA_DIR
from convert_pen_in_blue_cup_data_to_lerobot import find_episode_paths
from convert_pen_in_blue_cup_data_to_lerobot import get_camera_ids
from convert_pen_in_blue_cup_data_to_lerobot import TASK_PROMPT

DEFAULT_REPO_ID = "skybhh19/droid_pen_in_blue_cup_jointpos"


def convert_step(step: dict) -> dict:
    wrist_id, exterior_id = get_camera_ids(step)

    exterior_image = resize_image(step["observation"]["image"][exterior_id][..., ::-1], (320, 180))
    wrist_image = resize_image(step["observation"]["image"][wrist_id][..., ::-1], (320, 180))

    action_joint_position = np.asarray(step["action"]["joint_position"], dtype=np.float32)
    action_gripper_position = np.asarray(step["action"]["gripper_position"], dtype=np.float32).reshape(1)

    return {
        "exterior_image_1_left": exterior_image,
        # The June 10 pen-in-cup data has one exterior camera. Duplicate it so the dataset remains
        # compatible with the DROID LeRobot data configs.
        "exterior_image_2_left": exterior_image,
        "wrist_image_left": wrist_image,
        "joint_position": np.asarray(step["observation"]["robot_state"]["joint_positions"], dtype=np.float32),
        "gripper_position": np.asarray(
            step["observation"]["robot_state"]["gripper_position"], dtype=np.float32
        ).reshape(1),
        "actions": np.concatenate([action_joint_position, action_gripper_position]).astype(np.float32, copy=False),
        "task": TASK_PROMPT,
    }


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    overwrite: bool = False,
    dry_run: bool = False,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
):
    data_dir_path = Path(data_dir)
    episode_paths = find_episode_paths(data_dir_path)
    if not episode_paths:
        raise FileNotFoundError(f"No trajectory.h5 files found under {data_dir_path}")
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    if not dry_run:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        output_path = HF_LEROBOT_HOME / repo_id
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(output_path)

    dataset = None if dry_run else create_lerobot_dataset(repo_id)

    total_frames = 0
    total_valid_frames = 0
    converted_episodes = 0
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        recording_folderpath = episode_path.parent / "recordings" / "MP4"
        trajectory = load_trajectory(
            str(episode_path), recording_folderpath=str(recording_folderpath), remove_skipped_steps=True
        )
        if len(trajectory) == 0:
            continue

        converted_frames = 0
        for step in trajectory:
            total_frames += 1
            if not is_valid_transition(step):
                continue
            total_valid_frames += 1

            if dataset is not None:
                dataset.add_frame(convert_step(step))
            converted_frames += 1

        if converted_frames == 0:
            continue

        if dataset is not None:
            dataset.save_episode()
        converted_episodes += 1

    print(
        f"Converted {converted_episodes} episodes with {total_valid_frames} valid frames "
        f"out of {total_frames} loaded frames."
    )

    if push_to_hub:
        if dataset is None:
            raise ValueError("Cannot push to hub during --dry-run.")
        dataset.push_to_hub(
            tags=["droid", "panda", "pen-in-cup", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
