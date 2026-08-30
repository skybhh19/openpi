"""Convert the 256px robomimic Threading_D0 joint-position dataset to LeRobot.

The converter uses Pi's standard Franka convention:

* state: current joint angles in radians followed by measured gripper closure;
* action: absolute target joint angles followed by target gripper closure.

Robosuite's binary gripper command is converted from ``-1=open, +1=close`` to
``0=open, 1=closed``. The data config converts only the seven absolute arm
targets to deltas for model training and reverses that operation at inference.
"""

from pathlib import Path
import shutil
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm
import tyro

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[4] / "robomimic" / "datasets" / "threading_d0_joint_v3" / "image_v15_256.hdf5"
)
DEFAULT_REPO_ID = "local/robomimic_threading_d0_joint_v3_256"
DEFAULT_TASK_PROMPT = "Thread the needle through the ring"
DEFAULT_FILTER_KEY = "all"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
PANDA_MAX_GRIPPER_WIDTH = 0.08
JOINT_DIM = 7
STATE_DIM = 8
ACTION_DIM = 8


def _natural_demo_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected a robomimic demo name such as 'demo_17', got {name!r}") from exc


def selected_demo_names(hdf5_file: h5py.File, filter_key: str | None) -> list[str]:
    """Return numerically sorted demos from a robomimic mask, or every demo."""
    if "data" not in hdf5_file:
        raise ValueError("Source HDF5 is missing the required /data group")

    use_all = filter_key is None or filter_key.lower() in {"", "all", "none"}
    if use_all:
        names = list(hdf5_file["data"].keys())
    else:
        mask_path = f"mask/{filter_key}"
        if mask_path not in hdf5_file:
            available = sorted(hdf5_file.get("mask", {}).keys())
            raise KeyError(f"Mask {mask_path!r} does not exist; available masks: {available}")
        names = [value.decode() if isinstance(value, bytes) else str(value) for value in hdf5_file[mask_path][()]]

    missing = [name for name in names if name not in hdf5_file["data"]]
    if missing:
        raise ValueError(f"Mask references demos that are absent from /data: {missing[:5]}")
    return sorted(names, key=_natural_demo_key)


def validate_environment_metadata(env_args: dict[str, Any]) -> int:
    """Validate the recorded Panda absolute-joint controller and return its rate."""
    if env_args.get("env_name") != "Threading_D0":
        raise ValueError(f"Expected env_name='Threading_D0', got {env_args.get('env_name')!r}")
    env_kwargs = env_args.get("env_kwargs", {})
    if env_kwargs.get("robots") != ["Panda"]:
        raise ValueError(f"Expected a single Panda robot, got {env_kwargs.get('robots')!r}")

    body_parts = env_kwargs.get("controller_configs", {}).get("body_parts", {})
    joint_parts = [part for part in body_parts.values() if part.get("type") == "JOINT_POSITION"]
    if len(joint_parts) != 1:
        raise ValueError(f"Expected exactly one JOINT_POSITION controller, got {body_parts!r}")
    controller = joint_parts[0]
    if controller.get("input_type") != "absolute":
        raise ValueError(f"Expected absolute joint-position inputs, got {controller.get('input_type')!r}")
    if len(controller.get("input_min", ())) != JOINT_DIM or len(controller.get("input_max", ())) != JOINT_DIM:
        raise ValueError("Expected seven joint-angle input bounds on the controller")
    if controller.get("gripper", {}).get("type") != "GRIP":
        raise ValueError(f"Expected the Panda GRIP controller, got {controller.get('gripper')!r}")

    camera_names = env_kwargs.get("camera_names")
    if camera_names != ["agentview", "robot0_eye_in_hand"]:
        raise ValueError(f"Expected agentview and wrist RGB cameras, got {camera_names!r}")
    camera_shape = (env_kwargs.get("camera_heights"), env_kwargs.get("camera_widths"))
    if camera_shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Expected recorded camera resolution {IMAGE_HEIGHT}x{IMAGE_WIDTH}, got {camera_shape}")

    # Robosuite defaults to 20 Hz when no explicit override was serialized.
    fps = int(env_kwargs.get("control_freq") or 20)
    if fps <= 0:
        raise ValueError(f"Invalid control frequency: {fps}")
    return fps


def panda_gripper_qpos_to_closure(qpos: np.ndarray) -> np.ndarray:
    """Map Panda finger positions to Pi closure: 0 is open and 1 is closed."""
    qpos = np.asarray(qpos)
    if qpos.shape[-1] != 2:
        raise ValueError(f"Expected two Panda finger positions, got {qpos.shape}")
    width = qpos[..., 0] - qpos[..., 1]
    return np.clip(1.0 - width / PANDA_MAX_GRIPPER_WIDTH, 0.0, 1.0)


def robosuite_gripper_to_closure(command: np.ndarray) -> np.ndarray:
    """Map robosuite's -1/open, +1/close command to Pi's [0, 1] closure."""
    command = np.asarray(command)
    if np.any(~np.isclose(np.abs(command), 1.0, atol=1e-6)):
        raise ValueError("Expected binary robosuite gripper commands in {-1, +1}")
    return (command + 1.0) / 2.0


def build_state(observation: h5py.Group | dict[str, np.ndarray], index: int) -> np.ndarray:
    """Build [joint angles (7), measured gripper closure (1)]."""
    joint_position = np.asarray(observation["robot0_joint_pos"][index], dtype=np.float32)
    closure = np.asarray([panda_gripper_qpos_to_closure(observation["robot0_gripper_qpos"][index])], dtype=np.float32)
    state = np.concatenate([joint_position, closure])
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid state at index {index}: shape={state.shape}, finite={np.all(np.isfinite(state))}")
    return state


def validate_episode(demo_name: str, episode: h5py.Group) -> int:
    """Validate schema and numeric evidence for absolute target semantics."""
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

    # These absolute targets remain within 0.045 rad of the state from which they are issued. They are
    # radian-scale joint poses, not normalized [-1, 1] commands. The next state must move closer to the target.
    current_error = np.abs(actions[:, :JOINT_DIM] - current_joint)
    next_error = np.abs(actions[:, :JOINT_DIM] - next_joint)
    if float(np.max(current_error)) > 0.051:
        raise ValueError(f"{demo_name} arm actions are not aligned absolute joint targets")
    if float(np.mean(next_error)) > float(np.mean(current_error)) + 1e-8:
        raise ValueError(f"{demo_name} next joint state does not move toward the commanded absolute target")
    robosuite_gripper_to_closure(actions[:, -1])
    return num_steps


def convert_frame(episode: h5py.Group, index: int, task_prompt: str) -> dict[str, Any]:
    """Convert one frame to canonical Pi Franka state and action conventions."""
    observation = episode["obs"]
    raw_action = np.asarray(episode["actions"][index], dtype=np.float32)
    action = np.concatenate(
        [raw_action[:JOINT_DIM], np.asarray([robosuite_gripper_to_closure(raw_action[-1])], dtype=np.float32)]
    ).astype(np.float32, copy=False)
    return {
        # The image-observation conversion already stored MuJoCo images upright.
        "agentview_image": np.asarray(observation["agentview_image"][index]),
        "eye_in_hand_image": np.asarray(observation["robot0_eye_in_hand_image"][index]),
        "state": build_state(observation, index),
        "actions": action,
        "task": task_prompt,
    }


def create_lerobot_dataset(repo_id: str, fps: int):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "agentview_image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "eye_in_hand_image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


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
):
    """Convert selected episodes. Use ``--filter-key all`` to include every demo."""
    source_path = Path(data_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    with h5py.File(source_path, "r") as source:
        if "env_args" not in source["data"].attrs:
            raise ValueError("Source HDF5 is missing /data.attrs['env_args']")
        import json

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
        for demo_name in tqdm(demos, desc="Converting robomimic episodes"):
            episode = source[f"data/{demo_name}"]
            for index in range(frame_counts[demo_name]):
                dataset.add_frame(convert_frame(episode, index, task_prompt))
            dataset.save_episode()

    print(f"Converted {len(demos)} episodes / {sum(frame_counts.values())} frames to {repo_id}")
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "robosuite", "threading", "panda", "joint-position"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
