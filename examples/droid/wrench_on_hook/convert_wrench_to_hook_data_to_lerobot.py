"""
Convert the DROID wrench-on-hook dataset to LeRobot format.

Example:
uv run examples/droid/wrench_to_hook/convert_wrench_to_hook_data_to_lerobot.py \
    --data-dir /iris/u/tiangao/projects/droid/data/success/2026-06-13 \
    --repo-id skybhh19/droid_wrench_on_hook

The resulting dataset is saved under $LEROBOT_HOME / <repo-id>.
"""

from pathlib import Path
import shutil
import sys

import numpy as np
from tqdm import tqdm
import tyro

DROID_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(DROID_EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(DROID_EXAMPLES_DIR))

from convert_droid_data_to_lerobot import is_valid_transition
from convert_droid_data_to_lerobot import load_trajectory
from convert_droid_data_to_lerobot import resize_image

DEFAULT_DATA_DIR = "/iris/u/tiangao/projects/droid/data/success/2026-06-13"
DEFAULT_REPO_ID = "skybhh19/droid_wrench_on_hook"
TASK_PROMPT = "Hang the wrench on the hook"


def create_lerobot_dataset(repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def get_camera_ids(step: dict) -> tuple[str, str]:
    camera_type_dict = step["observation"]["camera_type"]
    image_dict = step["observation"].get("image", {})
    wrist_ids = [k for k, v in camera_type_dict.items() if v == 0 and k in image_dict]
    exterior_ids = [k for k, v in camera_type_dict.items() if v != 0 and k in image_dict]
    if len(wrist_ids) != 1 or len(exterior_ids) != 1:
        raise ValueError(
            "Expected exactly one wrist and one exterior camera with loaded images, "
            f"got wrist={wrist_ids}, exterior={exterior_ids}, available_images={sorted(image_dict)}"
        )
    return wrist_ids[0], exterior_ids[0]


def convert_step(step: dict) -> dict:
    wrist_id, exterior_id = get_camera_ids(step)

    exterior_image = resize_image(step["observation"]["image"][exterior_id][..., ::-1], (320, 180))
    wrist_image = resize_image(step["observation"]["image"][wrist_id][..., ::-1], (320, 180))

    return {
        "exterior_image_1_left": exterior_image,
        # The June 13 wrench-on-hook data has one exterior camera. Duplicate it so the dataset remains
        # compatible with the existing LeRobotDROIDDataConfig schema.
        "exterior_image_2_left": exterior_image,
        "wrist_image_left": wrist_image,
        "joint_position": np.asarray(step["observation"]["robot_state"]["joint_positions"], dtype=np.float32),
        "gripper_position": np.asarray(
            step["observation"]["robot_state"]["gripper_position"][None], dtype=np.float32
        ),
        "actions": np.concatenate(
            [step["action"]["joint_velocity"], step["action"]["gripper_position"][None]], dtype=np.float32
        ),
        "task": TASK_PROMPT,
    }


def find_episode_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("**/trajectory.h5"))


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
    skipped_episodes = 0
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        recording_folderpath = episode_path.parent / "recordings" / "MP4"
        trajectory = load_trajectory(
            str(episode_path), recording_folderpath=str(recording_folderpath), remove_skipped_steps=True
        )
        if len(trajectory) == 0:
            continue

        total_frames += len(trajectory)
        valid_steps = [step for step in trajectory if is_valid_transition(step)]
        total_valid_frames += len(valid_steps)
        if not valid_steps:
            continue

        try:
            converted_frames = [convert_step(step) for step in valid_steps]
        except ValueError as exc:
            skipped_episodes += 1
            print(f"Skipping {episode_path}: {exc}")
            continue

        if dataset is not None:
            for frame in converted_frames:
                dataset.add_frame(frame)
            dataset.save_episode()

        converted_episodes += 1

    print(
        f"Converted {converted_episodes} episodes with {total_valid_frames} valid frames "
        f"out of {total_frames} loaded frames."
    )
    if skipped_episodes:
        print(f"Skipped {skipped_episodes} episodes because required camera images were missing.")

    if push_to_hub:
        if dataset is None:
            raise ValueError("Cannot push to hub during --dry-run.")
        dataset.push_to_hub(
            tags=["droid", "panda", "wrench-on-hook"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
