"""Audit D06-hard absolute-action normalization and demonstration replay."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openpi import transforms
from openpi.policies import robomimic_policy
from openpi.training import checkpoints as checkpoint_utils
from openpi.training import config as training_config

from examples.robomimic.threading_d06_hard_joint import convert_robomimic_data_to_lerobot as conversion


def _demo_names(dataset: h5py.File) -> list[str]:
    return sorted(dataset["data"].keys(), key=lambda name: int(name.removeprefix("demo_")))


def audit_normalization(checkpoint_dir: str, *, max_frames: int = 10_000) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    config_name = checkpoint.parents[1].name
    train_config = training_config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    task_config = train_config.data.task_config
    if task_config != robomimic_policy.THREADING_JOINT_ABSOLUTE:
        raise ValueError(f"{config_name} is not configured for absolute joint actions")
    norm_stats = checkpoint_utils.load_norm_stats(checkpoint / "assets", data_config.asset_id)

    actions = []
    states = []
    with h5py.File(conversion.DEFAULT_DATA_PATH, "r") as dataset:
        for demo_name in _demo_names(dataset):
            demo = dataset[f"data/{demo_name}"]
            take = min(len(demo["actions"]), max_frames - len(actions))
            if take <= 0:
                break
            raw_demo_actions = np.asarray(demo["actions"][:take], dtype=np.float32)
            converted_actions = raw_demo_actions.copy()
            converted_actions[:, -1] = conversion.robosuite_gripper_to_closure(raw_demo_actions[:, -1])
            actions.extend(converted_actions)
            joint = np.asarray(demo["obs/robot0_joint_pos"][:take], dtype=np.float32)
            closure = conversion.panda_gripper_qpos_to_closure(demo["obs/robot0_gripper_qpos"][:take])[:, None]
            states.extend(np.concatenate([joint, closure], axis=-1))

    raw_actions = np.asarray(actions)
    raw_states = np.asarray(states)
    training_sample = {"state": raw_states.copy(), "actions": raw_actions.copy()}
    normalized = transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)(training_sample)
    deployed = transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)(
        {"state": normalized["state"].copy(), "actions": normalized["actions"].copy()}
    )
    deployed = robomimic_policy.RobomimicOutputs(task_config=task_config)(deployed)

    expected_gripper = raw_actions[:, -1] * 2.0 - 1.0
    report = {
        "config": config_name,
        "checkpoint": str(checkpoint.resolve()),
        "frames": len(raw_actions),
        "uses_quantile_normalization": data_config.use_quantile_norm,
        "max_arm_roundtrip_error": float(np.max(np.abs(deployed["actions"][:, :7] - raw_actions[:, :7]))),
        "max_gripper_roundtrip_error": float(np.max(np.abs(deployed["actions"][:, 7] - expected_gripper))),
        "normalized_action_min": np.min(normalized["actions"][:, :8], axis=0).tolist(),
        "normalized_action_max": np.max(normalized["actions"][:, :8], axis=0).tolist(),
    }
    if report["max_arm_roundtrip_error"] > 1e-5 or report["max_gripper_roundtrip_error"] > 1e-5:
        raise RuntimeError(f"Normalization deployment round-trip failed: {report}")
    return report


def replay_dataset(output_path: str, *, mask: str = "all", max_demos: int | None = None) -> dict[str, Any]:
    import robosuite

    with h5py.File(conversion.DEFAULT_DATA_PATH, "r") as dataset:
        env_args = json.loads(dataset["data"].attrs["env_args"])
        conversion.validate_environment_metadata(env_args)
        names = _demo_names(dataset)
        if mask != "all":
            selected = {value.decode() if isinstance(value, bytes) else str(value) for value in dataset[f"mask/{mask}"][()]}
            names = [name for name in names if name in selected]
        if max_demos is not None:
            names = names[:max_demos]

        env_kwargs = dict(env_args["env_kwargs"])
        env_kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            hard_reset=False,
            horizon=2000,
            ignore_done=True,
        )
        env = robosuite.make(env_args["env_name"], **env_kwargs)
        episodes = []
        try:
            for index, demo_name in enumerate(names):
                demo = dataset[f"data/{demo_name}"]
                states = np.asarray(demo["states"])
                actions = np.asarray(demo["actions"])
                expected_joint = np.asarray(demo["next_obs/robot0_joint_pos"])
                env.reset()
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()
                env._threading_initial_tripod_pos = None
                env._threading_max_insert_progress = -np.inf
                success = False
                joint_errors = []
                for action, target_joint in zip(actions, expected_joint):
                    observation, reward, _, _ = env.step(action)
                    success = success or bool(reward > 0.0 or env._check_success())
                    joint_errors.append(float(np.max(np.abs(observation["robot0_joint_pos"] - target_joint))))
                episodes.append(
                    {
                        "episode": index,
                        "source_demo": demo_name,
                        "success": success,
                        "steps": len(actions),
                        "max_next_joint_error": max(joint_errors),
                        "mean_next_joint_error": float(np.mean(joint_errors)),
                    }
                )
                print(f"{demo_name}: success={success} max_joint_error={max(joint_errors):.6g}", flush=True)
        finally:
            env.close()

    report = {
        "dataset": str(Path(conversion.DEFAULT_DATA_PATH).resolve()),
        "mask": mask,
        "summary": {
            "episodes": len(episodes),
            "successes": sum(episode["success"] for episode in episodes),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "max_next_joint_error": max(episode["max_next_joint_error"] for episode in episodes),
            "mean_next_joint_error": float(np.mean([episode["mean_next_joint_error"] for episode in episodes])),
        },
        "episodes": episodes,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(
    *,
    checkpoint_dir: str | None = None,
    normalization_output_path: str | None = None,
    replay_output_path: str | None = None,
    replay_mask: str = "all",
    replay_max_demos: int | None = None,
) -> None:
    if checkpoint_dir is None and replay_output_path is None:
        raise ValueError("Specify checkpoint_dir and/or replay_output_path")
    if checkpoint_dir is not None:
        report = audit_normalization(checkpoint_dir)
        print(json.dumps(report, indent=2))
        if normalization_output_path is not None:
            destination = Path(normalization_output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    if replay_output_path is not None:
        report = replay_dataset(replay_output_path, mask=replay_mask, max_demos=replay_max_demos)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    tyro.cli(main)
