"""Convert every Threading_D06_Hard absolute-joint demonstration to LeRobot.

The source controller accepts seven absolute Panda joint targets followed by a
binary robosuite gripper command. Conversion keeps the arm targets absolute and
maps the gripper to Pi closure. The shared training transform later makes only
the arm dimensions relative; the gripper remains absolute.
"""

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

DEFAULT_DATA_PATH = Path(
    "/iris/u/tiangao/projects/robomimic/robomimic/datasets/threading_d06_hard_joint/image_v15_256.hdf5"
)
DEFAULT_REPO_ID = "local/robomimic_threading_d06_hard_joint_256"
DEFAULT_TASK_PROMPT = "Thread the needle through the ring"
DEFAULT_FILTER_KEY = "all"
EXPECTED_ENV_NAME = "Threading_D06_Hard"
EXPECTED_ENV_VERSION = "1.5.2"
EXPECTED_SOURCE_EPISODES = 400
EXPECTED_SOURCE_FRAMES = 128_232
SOURCE_AGENTVIEW_KEY = "agentview_full_image"
MAX_ABS_JOINT_TARGET_ERROR = 0.06

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
create_lerobot_dataset = joint_conversion.create_lerobot_dataset


def validate_environment_metadata(env_args: dict[str, Any]) -> int:
    """Validate the D06-hard task and its Panda absolute-joint schema."""
    if env_args.get("env_name") != EXPECTED_ENV_NAME:
        raise ValueError(f"Expected env_name={EXPECTED_ENV_NAME!r}, got {env_args.get('env_name')!r}")
    if env_args.get("env_version") != EXPECTED_ENV_VERSION:
        raise ValueError(
            f"Expected robosuite env_version={EXPECTED_ENV_VERSION!r}, got {env_args.get('env_version')!r}"
        )

    shared_metadata = json.loads(json.dumps(env_args))
    shared_metadata["env_name"] = "Threading_D0"
    camera_names = shared_metadata.get("env_kwargs", {}).get("camera_names")
    if camera_names != ["agentview_full", "robot0_eye_in_hand"]:
        raise ValueError(f"Expected agentview_full and wrist RGB cameras, got {camera_names!r}")
    shared_metadata["env_kwargs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    return joint_conversion.validate_environment_metadata(shared_metadata)


def validate_episode(demo_name: str, episode: h5py.Group) -> int:
    """Validate D06-hard schema and audited absolute-joint semantics."""
    required = (
        "actions",
        f"obs/{SOURCE_AGENTVIEW_KEY}",
        "obs/robot0_eye_in_hand_image",
        "obs/robot0_joint_pos",
        "obs/robot0_gripper_qpos",
        "next_obs/robot0_joint_pos",
        "action_dict/abs_joint_pos",
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
    for key in (f"obs/{SOURCE_AGENTVIEW_KEY}", "obs/robot0_eye_in_hand_image"):
        dataset = episode[key]
        if dataset.shape[1:] != IMAGE_SHAPE or dataset.dtype != np.uint8:
            raise ValueError(
                f"{demo_name}/{key} has shape={dataset.shape}, dtype={dataset.dtype}; "
                f"expected (T, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3) uint8"
            )

    current_joint = np.asarray(episode["obs/robot0_joint_pos"])
    next_joint = np.asarray(episode["next_obs/robot0_joint_pos"])
    gripper_qpos = np.asarray(episode["obs/robot0_gripper_qpos"])
    action_dict_target = np.asarray(episode["action_dict/abs_joint_pos"])
    if current_joint.shape != (num_steps, JOINT_DIM) or next_joint.shape != (num_steps, JOINT_DIM):
        raise ValueError(f"{demo_name} must contain aligned 7-D current and next joint observations")
    if action_dict_target.shape != (num_steps, JOINT_DIM):
        raise ValueError(f"{demo_name}/action_dict/abs_joint_pos must have shape (T, {JOINT_DIM})")
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

    # Across all 128,232 audited frames, target-current error is <= 0.058253
    # rad. Every episode's mean error decreases at the next state, and the arm
    # action matches action_dict/abs_joint_pos to 1.2e-7. These are absolute
    # radian targets, not normalized delta commands.
    current_error = np.abs(actions[:, :JOINT_DIM] - current_joint)
    next_error = np.abs(actions[:, :JOINT_DIM] - next_joint)
    if float(np.max(current_error)) > MAX_ABS_JOINT_TARGET_ERROR:
        raise ValueError(f"{demo_name} arm actions are not aligned absolute joint targets")
    if float(np.mean(next_error)) > float(np.mean(current_error)) + 1e-8:
        raise ValueError(f"{demo_name} next joint state does not move toward the commanded absolute target")
    if not np.allclose(actions[:, :JOINT_DIM], action_dict_target, atol=1e-6, rtol=0.0):
        raise ValueError(f"{demo_name} actions disagree with action_dict/abs_joint_pos")
    robosuite_gripper_to_closure(actions[:, -1])
    return num_steps


def convert_frame(episode: h5py.Group, index: int, task_prompt: str) -> dict[str, Any]:
    """Map D06-hard's external camera into the canonical training schema."""
    observation = episode["obs"]
    raw_action = np.asarray(episode["actions"][index], dtype=np.float32)
    action = np.concatenate(
        [raw_action[:JOINT_DIM], np.asarray([robosuite_gripper_to_closure(raw_action[-1])], dtype=np.float32)]
    ).astype(np.float32, copy=False)
    return {
        "agentview_image": np.asarray(observation[SOURCE_AGENTVIEW_KEY][index]),
        "eye_in_hand_image": np.asarray(observation["robot0_eye_in_hand_image"][index]),
        "state": build_state(observation, index),
        "actions": action,
        "task": task_prompt,
    }


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
        for demo_name in tqdm(demos, desc="Converting Threading_D06_Hard episodes"):
            episode = source[f"data/{demo_name}"]
            for index in range(frame_counts[demo_name]):
                dataset.add_frame(convert_frame(episode, index, task_prompt))
            dataset.save_episode()

    _write_source_manifest(
        output_path,
        source_path=source_path,
        repo_id=repo_id,
        task_prompt=task_prompt,
        fps=fps,
        frame_counts=frame_counts,
    )
    print(f"Converted {len(demos)} episodes / {sum(frame_counts.values())} frames to {repo_id}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "robosuite", "threading-d06-hard", "panda", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
