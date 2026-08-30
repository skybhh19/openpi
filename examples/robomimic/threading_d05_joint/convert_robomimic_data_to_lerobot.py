"""Convert all 200 Threading_D05 absolute-joint demonstrations to LeRobot.

The source uses the same Panda action convention as the Threading_D0 joint
dataset: seven absolute joint-angle targets followed by a binary robosuite
gripper command. The shared frame conversion maps the gripper to Pi closure;
the training data config later converts only the seven arm targets to deltas.
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
joint_conversion = importlib.import_module("examples.robomimic.threading_joint.convert_robomimic_data_to_lerobot")

DEFAULT_DATA_PATH = Path("/iliad/u/tiangao/projects/robomimic/datasets/threading_d05_joint_v2/image_v15_256.hdf5")
DEFAULT_REPO_ID = "local/robomimic_threading_d05_joint_v2_256"
DEFAULT_TASK_PROMPT = "Thread the needle through the ring"
DEFAULT_FILTER_KEY = "all"
EXPECTED_ENV_NAME = "Threading_D05"

IMAGE_HEIGHT = joint_conversion.IMAGE_HEIGHT
IMAGE_WIDTH = joint_conversion.IMAGE_WIDTH
IMAGE_SHAPE = joint_conversion.IMAGE_SHAPE
JOINT_DIM = joint_conversion.JOINT_DIM
STATE_DIM = joint_conversion.STATE_DIM
ACTION_DIM = joint_conversion.ACTION_DIM

selected_demo_names = joint_conversion.selected_demo_names
panda_gripper_qpos_to_closure = joint_conversion.panda_gripper_qpos_to_closure
robosuite_gripper_to_closure = joint_conversion.robosuite_gripper_to_closure
build_state = joint_conversion.build_state
validate_episode = joint_conversion.validate_episode
convert_frame = joint_conversion.convert_frame
create_lerobot_dataset = joint_conversion.create_lerobot_dataset


def validate_environment_metadata(env_args: dict[str, Any]) -> int:
    """Validate Threading_D05 and the shared absolute-joint recording schema."""
    if env_args.get("env_name") != EXPECTED_ENV_NAME:
        raise ValueError(f"Expected env_name={EXPECTED_ENV_NAME!r}, got {env_args.get('env_name')!r}")

    # The D0 validator owns the detailed Panda/controller/camera checks. D05
    # differs only in the task class name, which was checked above.
    shared_metadata = dict(env_args)
    shared_metadata["env_name"] = "Threading_D0"
    return joint_conversion.validate_environment_metadata(shared_metadata)


def _write_source_manifest(
    output_path: Path,
    *,
    source_path: Path,
    repo_id: str,
    frame_counts: dict[str, int],
) -> None:
    """Record the source-to-LeRobot episode mapping used by subset filters."""
    manifest = {
        "format_version": 1,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "source_env_name": EXPECTED_ENV_NAME,
        "repo_id": repo_id,
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
    overwrite: bool = False,
    dry_run: bool = False,
    push_to_hub: bool = False,
) -> None:
    """Convert episodes; the default deliberately ignores train/valid masks."""
    source_path = Path(data_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

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
        for demo_name in tqdm(demos, desc="Converting Threading_D05 episodes"):
            episode = source[f"data/{demo_name}"]
            for index in range(frame_counts[demo_name]):
                dataset.add_frame(convert_frame(episode, index, task_prompt))
            dataset.save_episode()

    _write_source_manifest(
        output_path,
        source_path=source_path,
        repo_id=repo_id,
        frame_counts=frame_counts,
    )
    print(f"Converted {len(demos)} episodes / {sum(frame_counts.values())} frames to {repo_id}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "robosuite", "threading-d05", "panda", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
