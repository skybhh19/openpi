"""Convert all 600 Threading_D06_Harder joint demonstrations to LeRobot.

The HDF5 stores seven absolute Panda joint targets and one robosuite gripper
command. Conversion preserves the arm targets and maps the gripper to Pi
closure; the fine-tuning data transform makes only the seven arm dimensions
relative.
"""

import importlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import h5py
from tqdm import tqdm
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

shared = importlib.import_module(
    "examples.robomimic.threading_d06_hard_joint_1000.convert_robomimic_data_to_lerobot"
)

DEFAULT_DATA_PATH = Path(
    "/iris/u/tiangao/projects/robomimic/robomimic/datasets/"
    "threading_d06_harder_joint_600/image_v15_256.hdf5"
)
DEFAULT_REPO_ID = "local/robomimic_threading_d06_harder_joint_600_256"
DEFAULT_TASK_PROMPT = "Thread the needle through the ring"
DEFAULT_FILTER_KEY = "all"
EXPECTED_ENV_NAME = "Threading_D06_Harder"
EXPECTED_ENV_VERSION = "1.5.2"
EXPECTED_SOURCE_EPISODES = 600
EXPECTED_SOURCE_FRAMES = 253_863
SOURCE_AGENTVIEW_KEY = shared.SOURCE_AGENTVIEW_KEY

IMAGE_HEIGHT = shared.IMAGE_HEIGHT
IMAGE_WIDTH = shared.IMAGE_WIDTH
IMAGE_SHAPE = shared.IMAGE_SHAPE
JOINT_DIM = shared.JOINT_DIM
STATE_DIM = shared.STATE_DIM
ACTION_DIM = shared.ACTION_DIM

selected_demo_names = shared.selected_demo_names
validate_episode = shared.validate_episode
load_episode = shared.load_episode
convert_frame = shared.convert_frame
create_lerobot_dataset = shared.create_lerobot_dataset
resolve_source_provenance_path = shared.resolve_source_provenance_path


def validate_environment_metadata(env_args: dict[str, Any]) -> int:
    """Validate the harder task and its recorded absolute-joint schema."""
    if env_args.get("env_name") != EXPECTED_ENV_NAME:
        raise ValueError(f"Expected env_name={EXPECTED_ENV_NAME!r}, got {env_args.get('env_name')!r}")
    if env_args.get("env_version") != EXPECTED_ENV_VERSION:
        raise ValueError(
            f"Expected robosuite env_version={EXPECTED_ENV_VERSION!r}, got {env_args.get('env_version')!r}"
        )

    shared_metadata = json.loads(json.dumps(env_args))
    shared_metadata["env_name"] = "Threading_D06_Hard"
    return shared.validate_environment_metadata(shared_metadata)


def validate_complete_selection(frame_counts: dict[str, int]) -> None:
    """Fail closed if the audited all-episode source has changed."""
    if len(frame_counts) != EXPECTED_SOURCE_EPISODES:
        raise ValueError(f"Expected all {EXPECTED_SOURCE_EPISODES} source demos, got {len(frame_counts)}")
    total_frames = sum(frame_counts.values())
    if total_frames != EXPECTED_SOURCE_FRAMES:
        raise ValueError(f"Expected {EXPECTED_SOURCE_FRAMES} source frames, got {total_frames}")


def _write_source_manifest(
    output_path: Path,
    *,
    source_path: Path,
    repo_id: str,
    task_prompt: str,
    fps: int,
    frame_counts: dict[str, int],
) -> None:
    """Record the exact source-to-LeRobot episode mapping for subset filters."""
    source_stat = source_path.stat()
    manifest = {
        "format_version": 1,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_env_name": EXPECTED_ENV_NAME,
        "source_env_version": EXPECTED_ENV_VERSION,
        "repo_id": repo_id,
        "task_prompt": task_prompt,
        "fps": fps,
        "total_episodes": len(frame_counts),
        "total_frames": sum(frame_counts.values()),
        "episodes": [
            {"episode_index": index, "source_demo": demo_name, "length": length}
            for index, (demo_name, length) in enumerate(frame_counts.items())
        ],
    }
    manifest_path = output_path / "meta" / "robomimic_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(
    data_path: str = str(DEFAULT_DATA_PATH),
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filter_key: str = DEFAULT_FILTER_KEY,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_episodes: int | None = None,
    source_provenance_path: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    push_to_hub: bool = False,
) -> None:
    """Convert episodes; defaults deliberately select all and ignore train/valid."""
    source_path = Path(data_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    manifest_source_path = resolve_source_provenance_path(source_path, source_provenance_path)

    with h5py.File(source_path, "r") as source:
        if "env_args" not in source["data"].attrs:
            raise ValueError("Source HDF5 is missing /data.attrs['env_args']")
        fps = validate_environment_metadata(json.loads(source["data"].attrs["env_args"]))
        demos = selected_demo_names(source, filter_key)
        if max_episodes is not None:
            demos = demos[:max_episodes]
        if not demos:
            raise ValueError("No episodes selected")
        frame_counts = {name: validate_episode(name, source[f"data/{name}"]) for name in demos}

        is_complete_default = filter_key.lower() in {"", "all", "none"} and max_episodes is None
        if is_complete_default:
            validate_complete_selection(frame_counts)
        if dry_run:
            print(f"Validated {len(demos)} episodes / {sum(frame_counts.values())} frames at {fps} Hz")
            return

        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        output_path = HF_LEROBOT_HOME / repo_id
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(output_path)

        dataset = create_lerobot_dataset(repo_id, fps)
        for demo_name in tqdm(demos, desc="Converting Threading_D06_Harder 600 episodes"):
            episode = load_episode(source[f"data/{demo_name}"])
            for index in range(frame_counts[demo_name]):
                dataset.add_frame(convert_frame(episode, index, task_prompt))
            dataset.save_episode()

    _write_source_manifest(
        output_path,
        source_path=manifest_source_path,
        repo_id=repo_id,
        task_prompt=task_prompt,
        fps=fps,
        frame_counts=frame_counts,
    )
    print(f"Converted {len(demos)} episodes / {sum(frame_counts.values())} frames to {repo_id}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "robosuite", "threading-d06-harder", "600-demos", "panda", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
