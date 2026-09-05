"""Convert every Threading_D05 v3 absolute-joint demonstration to LeRobot.

The source action is seven absolute Panda joint targets followed by a binary
robosuite gripper command. Frames retain absolute arm targets and convert the
gripper to Pi closure. The training config applies deltas only to the seven arm
dimensions, leaving the gripper target absolute.
"""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
joint_conversion = importlib.import_module("examples.robomimic.threading_joint.convert_robomimic_data_to_lerobot")

# The code repository moved to /iris, but this source dataset remains on the
# legacy /iliad data volume. Do not derive it from PROJECT_ROOT.
DEFAULT_DATA_PATH = Path("/iliad/u/tiangao/projects/robomimic/datasets/threading_d05_joint_v3/image_v15_256.hdf5")
DEFAULT_REPO_ID = "local/robomimic_threading_d05_joint_v3_256"
DEFAULT_TASK_PROMPT = "Thread the needle through the ring"
DEFAULT_FILTER_KEY = "all"
EXPECTED_ENV_NAME = "Threading_D05"
MANIFEST_FORMAT_VERSION = 2

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
convert_frame = joint_conversion.convert_frame
create_lerobot_dataset = joint_conversion.create_lerobot_dataset


def validate_environment_metadata(env_args: dict[str, Any]) -> int:
    """Validate Threading_D05 and the shared Panda absolute-joint schema."""
    if env_args.get("env_name") != EXPECTED_ENV_NAME:
        raise ValueError(f"Expected env_name={EXPECTED_ENV_NAME!r}, got {env_args.get('env_name')!r}")

    shared_metadata = dict(env_args)
    shared_metadata["env_name"] = "Threading_D0"
    return joint_conversion.validate_environment_metadata(shared_metadata)


def validate_episode(demo_name: str, episode: h5py.Group) -> int:
    """Validate the complete v3 frame schema and absolute-target evidence."""
    required = (
        "actions",
        "obs/agentview_image",
        "obs/robot0_eye_in_hand_image",
        "obs/robot0_joint_pos",
        "obs/robot0_gripper_qpos",
        "next_obs/robot0_joint_pos",
        "rewards",
        "dones",
    )
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(f"{demo_name} is missing required datasets: {missing}")

    actions = np.asarray(episode["actions"])
    num_steps = len(actions)
    if actions.shape != (num_steps, ACTION_DIM) or not np.all(np.isfinite(actions)):
        raise ValueError(f"{demo_name}/actions must be finite with shape (T, {ACTION_DIM}); got {actions.shape}")
    for key in required[1:]:
        if len(episode[key]) != num_steps:
            raise ValueError(f"{demo_name}/{key} has {len(episode[key])} frames, expected {num_steps}")
    for key in ("obs/agentview_image", "obs/robot0_eye_in_hand_image"):
        dataset = episode[key]
        if dataset.shape[1:] != IMAGE_SHAPE or dataset.dtype != np.uint8:
            raise ValueError(
                f"{demo_name}/{key} has shape={dataset.shape}, dtype={dataset.dtype}; "
                f"expected (T, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3) uint8"
            )

    current_joint = np.asarray(episode["obs/robot0_joint_pos"])
    next_joint = np.asarray(episode["next_obs/robot0_joint_pos"])
    gripper_qpos = np.asarray(episode["obs/robot0_gripper_qpos"])
    if current_joint.shape != (num_steps, JOINT_DIM) or next_joint.shape != (num_steps, JOINT_DIM):
        raise ValueError(f"{demo_name} must contain aligned 7-D current and next joint observations")
    if gripper_qpos.shape != (num_steps, 2):
        raise ValueError(f"{demo_name}/obs/robot0_gripper_qpos must have shape (T, 2)")
    if not all(np.all(np.isfinite(values)) for values in (current_joint, next_joint, gripper_qpos)):
        raise ValueError(f"{demo_name} contains non-finite proprioception")

    rewards = np.asarray(episode["rewards"])
    dones = np.asarray(episode["dones"])
    if not np.all(np.isin(rewards, (0, 1))) or not np.all(np.isin(dones, (0, 1))):
        raise ValueError(f"{demo_name} rewards and dones must be binary")
    if not np.array_equal(rewards, dones):
        raise ValueError(f"{demo_name} reward and done success labels disagree")
    successes = np.flatnonzero(rewards)
    if len(successes) == 0 or not np.all(rewards[successes[0] :] == 1):
        raise ValueError(f"{demo_name} must contain a persistent success suffix")

    # V3 target error peaks at 0.0587 rad because the measured joint state can
    # lag a changing absolute target slightly beyond the controller's 0.05-rad
    # output limit. Targets remain radian-scale poses, and the next state moves
    # closer to them in every source episode.
    current_error = np.abs(actions[:, :JOINT_DIM] - current_joint)
    next_error = np.abs(actions[:, :JOINT_DIM] - next_joint)
    if float(np.max(current_error)) > 0.061:
        raise ValueError(f"{demo_name} arm actions are not aligned absolute joint targets")
    if float(np.mean(next_error)) > float(np.mean(current_error)) + 1e-8:
        raise ValueError(f"{demo_name} next joint state does not move toward the commanded absolute target")
    robosuite_gripper_to_closure(actions[:, -1])
    return num_steps


def episode_order_length_sha256(frame_counts: dict[str, int]) -> str:
    """Return a stable digest of ordered source demo names and lengths."""
    payload = "\n".join(f"{name}\t{length}" for name, length in frame_counts.items())
    return hashlib.sha256(payload.encode()).hexdigest()


def build_source_manifest(
    *,
    source_path: Path,
    repo_id: str,
    env_args: dict[str, Any],
    fps: int,
    frame_counts: dict[str, int],
) -> dict[str, Any]:
    """Build complete source-to-LeRobot provenance for every episode."""
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "source_env_name": env_args["env_name"],
        "source_env_version": env_args.get("env_version"),
        "repo_id": repo_id,
        "task_prompt": DEFAULT_TASK_PROMPT,
        "fps": fps,
        "total_episodes": len(frame_counts),
        "total_frames": sum(frame_counts.values()),
        "episode_order_length_sha256": episode_order_length_sha256(frame_counts),
        "episodes": [
            {"episode_index": index, "source_demo": demo_name, "length": length}
            for index, (demo_name, length) in enumerate(frame_counts.items())
        ],
    }


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
        env_args = json.loads(source["data"].attrs["env_args"])
        fps = validate_environment_metadata(env_args)
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
        for demo_name in tqdm(demos, desc="Converting Threading_D05 v3 episodes"):
            episode = source[f"data/{demo_name}"]
            for index in range(frame_counts[demo_name]):
                dataset.add_frame(convert_frame(episode, index, task_prompt))
            dataset.save_episode()

    manifest = build_source_manifest(
        source_path=source_path,
        repo_id=repo_id,
        env_args=env_args,
        fps=fps,
        frame_counts=frame_counts,
    )
    manifest["task_prompt"] = task_prompt
    manifest_path = output_path / "meta" / "robomimic_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Converted {len(demos)} episodes / {sum(frame_counts.values())} frames to {repo_id}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "robosuite", "threading-d05", "v3", "panda", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
