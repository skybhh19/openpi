"""
Convert the DROID pen-in-blue-cup dataset to LeRobot format.

Example:
uv run examples/droid/pen_in_blue_cup/convert_pen_in_blue_cup_data_to_lerobot.py \
    --data-dir /iris/u/tiangao/projects/droid/data/success/2026-06-10 \
    --repo-id skybhh19/droid_pen_in_blue_cup

The resulting dataset is saved under $LEROBOT_HOME / <repo-id>.
"""

# ruff: noqa: E402

import json
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

DEFAULT_DATA_DIR = "/iris/u/tiangao/projects/droid/data/success/2026-06-10"
DEFAULT_REPO_ID = "skybhh19/droid_pen_in_blue_cup"
TASK_PROMPT = "Put the pen in the cup"


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


def get_camera_ids(
    step: dict, *, wrist_camera_id: str | None = None, exterior_camera_id: str | None = None
) -> tuple[str, str]:
    camera_type_dict = step["observation"]["camera_type"]
    image_dict = step["observation"].get("image", {})
    if wrist_camera_id is not None or exterior_camera_id is not None:
        if wrist_camera_id is None or exterior_camera_id is None:
            raise ValueError("wrist_camera_id and exterior_camera_id must be set together")
        missing = [camera_id for camera_id in (wrist_camera_id, exterior_camera_id) if camera_id not in image_dict]
        if missing:
            raise ValueError(f"Selected camera images are missing: {missing}; available_images={sorted(image_dict)}")
        if camera_type_dict.get(wrist_camera_id) != 0 or camera_type_dict.get(exterior_camera_id) == 0:
            raise ValueError(
                f"Selected cameras have unexpected types: wrist={camera_type_dict.get(wrist_camera_id)}, "
                f"exterior={camera_type_dict.get(exterior_camera_id)}"
            )
        return wrist_camera_id, exterior_camera_id

    wrist_ids = [k for k, v in camera_type_dict.items() if v == 0 and k in image_dict]
    exterior_ids = [k for k, v in camera_type_dict.items() if v != 0 and k in image_dict]
    if len(wrist_ids) != 1 or len(exterior_ids) != 1:
        raise ValueError(
            f"Expected exactly one wrist and one exterior camera, got wrist={wrist_ids}, exterior={exterior_ids}"
        )
    return wrist_ids[0], exterior_ids[0]


def convert_step(
    step: dict, *, wrist_camera_id: str | None = None, exterior_camera_id: str | None = None
) -> dict:
    wrist_id, exterior_id = get_camera_ids(
        step, wrist_camera_id=wrist_camera_id, exterior_camera_id=exterior_camera_id
    )

    exterior_image = resize_image(step["observation"]["image"][exterior_id][..., ::-1], (320, 180))
    wrist_image = resize_image(step["observation"]["image"][wrist_id][..., ::-1], (320, 180))

    return {
        "exterior_image_1_left": exterior_image,
        # The June 10 pen-in-cup data has one exterior camera. Duplicate it so the dataset remains
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
    wrist_camera_id: str | None = None,
    exterior_camera_id: str | None = None,
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
    source_episodes = []
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

            converted_step = convert_step(
                step, wrist_camera_id=wrist_camera_id, exterior_camera_id=exterior_camera_id
            )
            if dataset is not None:
                dataset.add_frame(converted_step)
            converted_frames += 1

        if converted_frames == 0:
            continue

        if dataset is not None:
            dataset.save_episode()
        source_episodes.append(
            {
                "episode_index": converted_episodes,
                "trajectory_path": str(episode_path.resolve()),
                "raw_episode_dir": str(episode_path.parent.resolve()),
                "frames": converted_frames,
            }
        )
        converted_episodes += 1

    if dataset is not None:
        manifest = {
            "format_version": 1,
            "repo_id": repo_id,
            "task_prompt": TASK_PROMPT,
            "fps": 15,
            "source_roots": [str(data_dir_path.resolve())],
            "wrist_camera_id": wrist_camera_id,
            "exterior_camera_id": exterior_camera_id,
            "total_episodes": converted_episodes,
            "total_frames": sum(episode["frames"] for episode in source_episodes),
            "episodes": source_episodes,
        }
        manifest_path = output_path / "meta" / "droid_source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote source manifest to {manifest_path}")

    print(
        f"Converted {converted_episodes} episodes with {total_valid_frames} valid frames "
        f"out of {total_frames} loaded frames."
    )

    if push_to_hub:
        if dataset is None:
            raise ValueError("Cannot push to hub during --dry-run.")
        dataset.push_to_hub(
            tags=["droid", "panda", "pen-in-cup"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
