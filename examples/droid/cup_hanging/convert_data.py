"""Convert cup hanging once into aligned joint-position and joint-velocity datasets."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_droid_data_to_lerobot import is_valid_transition
from convert_droid_data_to_lerobot import load_trajectory
from convert_droid_data_to_lerobot import resize_image
from pen_in_blue_cup.convert_pen_in_blue_cup_data_to_lerobot import create_lerobot_dataset
from pen_in_blue_cup.convert_pen_in_blue_cup_data_to_lerobot import get_camera_ids

PROMPT = "Hang the cup on the hook"
REPOS = {
    "joint_position": "skybhh19/droid_cup_hanging_09132026_jointpos",
    "joint_velocity": "skybhh19/droid_cup_hanging_09132026_jointvel",
}


def main(data_dir: str = "/iris/u/am208/droid/data/success/2026-09-13"):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    root = Path(data_dir).resolve()
    paths = sorted(root.rglob("trajectory.h5"))
    if not paths:
        raise ValueError(f"No trajectories under {root}")
    for repo in REPOS.values():
        if (HF_LEROBOT_HOME / repo).exists():
            raise FileExistsError(HF_LEROBOT_HOME / repo)
    # Fail before creating datasets when a selected camera file is absent.
    for path in paths:
        for camera in ("17471093", "31078156"):
            video = path.parent / "recordings/MP4" / f"{camera}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
    datasets = {action: create_lerobot_dataset(repo) for action, repo in REPOS.items()}
    episodes = []
    for path in paths:
        # Only expose the requested cameras to the shared reader. Ignore 23404442 entirely.
        with tempfile.TemporaryDirectory(prefix="cup-hanging-cameras-") as selected:
            for camera in ("17471093", "31078156"):
                (Path(selected) / f"{camera}.mp4").symlink_to(path.parent / "recordings/MP4" / f"{camera}.mp4")
            trajectory = load_trajectory(str(path), recording_folderpath=selected, remove_skipped_steps=False)
        with h5py.File(path) as raw:
            expected_frames = len(raw["action/joint_position"])
        # Every audited recording has one terminal HDF5 step without a video frame.
        # Retain all paired frames and explicitly record the unpaired terminal step.
        if expected_frames - len(trajectory) not in (0, 1):
            raise ValueError(f"Truncated camera stream in {path}: {len(trajectory)} of {expected_frames}")
        indices = []
        for index, step in enumerate(trajectory):
            if not is_valid_transition(step):
                continue
            wrist, exterior = get_camera_ids(step, wrist_camera_id="17471093", exterior_camera_id="31078156")
            ext_image = resize_image(step["observation"]["image"][exterior][..., ::-1], (320, 180))
            wrist_image = resize_image(step["observation"]["image"][wrist][..., ::-1], (320, 180))
            state = step["observation"]["robot_state"]
            frame = {
                "exterior_image_1_left": ext_image,
                "exterior_image_2_left": ext_image.copy(),
                "wrist_image_left": wrist_image,
                "joint_position": np.asarray(state["joint_positions"], dtype=np.float32),
                "gripper_position": np.asarray(state["gripper_position"], dtype=np.float32).reshape(1),
                "task": PROMPT,
            }
            for action, dataset in datasets.items():
                target = np.concatenate(
                    [
                        np.asarray(step["action"][action], dtype=np.float32).reshape(7),
                        np.asarray(step["action"]["gripper_position"], dtype=np.float32).reshape(1),
                    ]
                )
                if not np.isfinite(target).all():
                    raise ValueError(f"Nonfinite {action} at {path}:{index}")
                dataset.add_frame({**frame, "actions": target})
            indices.append(index)
        if not indices:
            raise ValueError(f"No valid transitions: {path}")
        for dataset in datasets.values():
            dataset.save_episode()
        episodes.append(
            {
                "episode_index": len(episodes),
                "trajectory_path": str(path),
                "raw_episode_dir": str(path.parent),
                "frames": len(indices),
                "raw_frame_indices": indices,
                "raw_hdf5_frames": expected_frames,
                "unpaired_terminal_frame_indices": list(range(len(trajectory), expected_frames)),
            }
        )
        print(f"Converted {len(episodes)}/{len(paths)}: {path.parent.name}, {len(indices)} frames", flush=True)
    for action, repo in REPOS.items():
        manifest = {
            "format_version": 1,
            "repo_id": repo,
            "task_prompt": PROMPT,
            "fps": 15,
            "action_space": action,
            "gripper_action": "absolute position",
            "source_roots": [str(root)],
            "wrist_camera_id": "17471093",
            "exterior_camera_id": "31078156",
            "ignored_camera_ids": ["23404442"],
            "total_source_trajectories": len(paths),
            "total_episodes": len(episodes),
            "total_frames": sum(e["frames"] for e in episodes),
            "episodes": episodes,
            "excluded_episodes": [],
        }
        (HF_LEROBOT_HOME / repo / "meta/droid_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Both datasets complete", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
