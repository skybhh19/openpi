"""
Convert the DROID pen-in-white-cup robomimic-style dataset to LeRobot format.

Example:
uv run python examples/droid/pen_in_white_cup/convert_pen_in_white_cup_data_to_lerobot.py \
    --data-dir /iris/u/tiangao/pen_in_cup_0705 \
    --repo-id skybhh19/droid_pen_in_white_cup

The resulting dataset is saved under $LEROBOT_HOME / <repo-id>.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import h5py
import numpy as np
from tqdm import tqdm

DEFAULT_DATA_DIR = "/iris/u/tiangao/pen_in_cup_0705"
DEFAULT_REPO_ID = "skybhh19/droid_pen_in_white_cup"
TASK_PROMPT = "Put the pen in the cup"

EXPECTED_FPS = 15
EXPECTED_VIDEO_SIZE = (448, 224)
VIDEO_PIXEL_MSE_THRESHOLD = 40.0

DEMO_FILE_RE = re.compile(r"^demos_(\d+)\.hdf5$")


@dataclass(frozen=True)
class DatasetSpec:
    shape_tail: tuple[int, ...]
    dtype_kind: str


REQUIRED_DATASETS = {
    "obs/agent_view": DatasetSpec((224, 224, 3), "u"),
    "obs/wrist": DatasetSpec((224, 224, 3), "u"),
    "obs/JOINT_POS": DatasetSpec((7,), "f"),
    "obs/GRIPPER": DatasetSpec((1,), "f"),
    "actions/joint_velocity": DatasetSpec((7,), "f"),
    "actions/gripper_position": DatasetSpec((1,), "f"),
}


def create_lerobot_dataset(repo_id: str):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=EXPECTED_FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
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


def parse_episode_id(path: Path) -> int:
    match = DEMO_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Expected a demos_<id>.hdf5 filename, got {path.name}")
    return int(match.group(1))


def episode_group_name(path: Path) -> str:
    return f"data/demo_{parse_episode_id(path)}"


def find_episode_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("demos_*.hdf5"), key=parse_episode_id)


def get_demo_group(hdf5_file: h5py.File, episode_path: Path) -> h5py.Group:
    group_name = episode_group_name(episode_path)
    if group_name not in hdf5_file:
        raise ValueError(f"{episode_path} is missing group {group_name}")
    return hdf5_file[group_name]


def validate_episode_file(episode_path: Path, *, check_finite: bool = True) -> int:
    with h5py.File(episode_path, "r") as hdf5_file:
        demo = get_demo_group(hdf5_file, episode_path)
        lengths = []
        for dataset_name, spec in REQUIRED_DATASETS.items():
            if dataset_name not in demo:
                raise ValueError(f"{episode_path} is missing dataset {episode_group_name(episode_path)}/{dataset_name}")

            dataset = demo[dataset_name]
            if dataset.ndim != 1 + len(spec.shape_tail) or dataset.shape[1:] != spec.shape_tail:
                raise ValueError(
                    f"{episode_path}:{dataset_name} has shape {dataset.shape}, "
                    f"expected (T, {', '.join(str(dim) for dim in spec.shape_tail)})"
                )
            if dataset.dtype.kind != spec.dtype_kind:
                raise ValueError(
                    f"{episode_path}:{dataset_name} has dtype {dataset.dtype}, expected dtype kind {spec.dtype_kind}"
                )
            if dataset_name.endswith(("agent_view", "wrist")) and dataset.dtype != np.dtype("uint8"):
                raise ValueError(f"{episode_path}:{dataset_name} has dtype {dataset.dtype}, expected uint8")
            if check_finite and dataset.dtype.kind == "f" and not np.isfinite(dataset[:]).all():
                raise ValueError(f"{episode_path}:{dataset_name} contains non-finite values")

            lengths.append(dataset.shape[0])

        unique_lengths = sorted(set(lengths))
        if len(unique_lengths) != 1:
            raise ValueError(f"{episode_path} has inconsistent timestep counts: {unique_lengths}")

        return unique_lengths[0]


def convert_frame(demo: h5py.Group, index: int) -> dict:
    exterior_image = np.asarray(demo["obs/agent_view"][index])
    wrist_image = np.asarray(demo["obs/wrist"][index])
    joint_position = np.asarray(demo["obs/JOINT_POS"][index], dtype=np.float32)
    gripper_position = np.asarray(demo["obs/GRIPPER"][index], dtype=np.float32).reshape(1)
    joint_velocity = np.asarray(demo["actions/joint_velocity"][index], dtype=np.float32)
    action_gripper = np.asarray(demo["actions/gripper_position"][index], dtype=np.float32).reshape(1)

    return {
        "exterior_image_1_left": exterior_image,
        # This dataset has one exterior camera. Duplicate it so the dataset remains compatible with
        # the existing LeRobotDROIDDataConfig schema.
        "exterior_image_2_left": exterior_image,
        "wrist_image_left": wrist_image,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
        "actions": np.concatenate([joint_velocity, action_gripper]).astype(np.float32, copy=False),
        "task": TASK_PROMPT,
    }


def validate_video_alignment(episode_path: Path, expected_length: int) -> None:
    import cv2

    episode_id = parse_episode_id(episode_path)
    video_path = episode_path.with_name(f"video_{episode_id}.mp4")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing paired video for {episode_path}: {video_path}")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    try:
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count != expected_length:
            raise ValueError(f"{video_path} has {frame_count} frames, expected {expected_length}")
        if round(fps) != EXPECTED_FPS:
            raise ValueError(f"{video_path} has fps={fps}, expected {EXPECTED_FPS}")
        if (width, height) != EXPECTED_VIDEO_SIZE:
            raise ValueError(f"{video_path} has size {(width, height)}, expected {EXPECTED_VIDEO_SIZE}")

        sample_indices = sorted({0, expected_length // 2, expected_length - 1})
        with h5py.File(episode_path, "r") as hdf5_file:
            demo = get_demo_group(hdf5_file, episode_path)
            for index in sample_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = video.read()
                if not success:
                    raise RuntimeError(f"Could not read frame {index} from {video_path}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                left = frame_rgb[:, : EXPECTED_VIDEO_SIZE[0] // 2]
                right = frame_rgb[:, EXPECTED_VIDEO_SIZE[0] // 2 :]
                agent_mse = float(
                    np.mean((left.astype(np.float32) - demo["obs/agent_view"][index].astype(np.float32)) ** 2)
                )
                wrist_mse = float(
                    np.mean((right.astype(np.float32) - demo["obs/wrist"][index].astype(np.float32)) ** 2)
                )
                if max(agent_mse, wrist_mse) > VIDEO_PIXEL_MSE_THRESHOLD:
                    raise ValueError(
                        f"{video_path} frame {index} does not match HDF5 images: "
                        f"agent_mse={agent_mse:.3f}, wrist_mse={wrist_mse:.3f}"
                    )
    finally:
        video.release()


def convert_episodes(
    episode_paths: Sequence[Path],
    *,
    repo_id: str,
    dry_run: bool,
    check_finite: bool,
    validate_videos: bool,
) -> tuple[int, int, object | None]:
    dataset = None if dry_run else create_lerobot_dataset(repo_id)
    converted_episodes = 0
    total_frames = 0

    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        episode_length = validate_episode_file(episode_path, check_finite=check_finite)
        if validate_videos:
            validate_video_alignment(episode_path, episode_length)

        total_frames += episode_length
        if dataset is not None:
            with h5py.File(episode_path, "r") as hdf5_file:
                demo = get_demo_group(hdf5_file, episode_path)
                for index in range(episode_length):
                    dataset.add_frame(convert_frame(demo, index))
            dataset.save_episode()

        converted_episodes += 1

    return converted_episodes, total_frames, dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing demos_*.hdf5 and video_*.mp4")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing local LeRobot dataset with repo-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing a LeRobot dataset")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N sorted episodes")
    parser.add_argument("--skip-finite-check", action="store_true", help="Skip finite-value validation for numeric arrays")
    parser.add_argument("--validate-videos", action="store_true", help="Check paired MP4 frame counts and sampled pixels")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the converted dataset to the Hugging Face Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    episode_paths = find_episode_paths(data_dir)
    if not episode_paths:
        raise FileNotFoundError(f"No demos_*.hdf5 files found under {data_dir}")
    if args.max_episodes is not None:
        episode_paths = episode_paths[: args.max_episodes]

    if not args.dry_run:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        output_path = HF_LEROBOT_HOME / args.repo_id
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(output_path)

    converted_episodes, total_frames, dataset = convert_episodes(
        episode_paths,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
        check_finite=not args.skip_finite_check,
        validate_videos=args.validate_videos,
    )
    action = "Validated" if args.dry_run else "Converted"
    print(f"{action} {converted_episodes} episodes with {total_frames} frames.")

    if args.push_to_hub:
        if args.dry_run:
            raise ValueError("Cannot push to hub during --dry-run.")
        if dataset is None:
            raise RuntimeError("Expected a converted dataset before pushing to hub.")
        dataset.push_to_hub(
            tags=["droid", "panda", "pen-in-white-cup"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main()
