"""Evaluate an OpenPI absolute-joint policy from fresh Threading_D08 resets."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.robomimic.threading_d08_joint import convert_robomimic_data_to_lerobot as d08_conversion  # noqa: E402
from examples.robomimic.threading_joint import main as joint_evaluation  # noqa: E402

_base_make_environment_kwargs = joint_evaluation.make_environment_kwargs
_base_run_rollout = joint_evaluation.run_rollout


def make_environment_kwargs(
    env_args: dict[str, Any],
    *,
    max_steps: int,
    render_gpu_device_id: int,
    seed: int,
    metadata_validator=None,
) -> dict[str, Any]:
    """Preserve D08's recorded external-camera identity during evaluation."""
    kwargs = _base_make_environment_kwargs(
        env_args,
        max_steps=max_steps,
        render_gpu_device_id=render_gpu_device_id,
        seed=seed,
        metadata_validator=metadata_validator or d08_conversion.validate_environment_metadata,
    )
    kwargs["camera_names"] = ["agentview_full", "robot0_eye_in_hand"]
    return kwargs


def _alias_agentview(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expose D08's agentview_full image under the canonical evaluator key."""
    if "agentview_full_image" not in observation:
        raise KeyError("Threading_D08 observation is missing agentview_full_image")
    aliased = dict(observation)
    aliased["agentview_image"] = observation["agentview_full_image"]
    return aliased


class _AgentviewAliasEnv:
    """Delegate to robosuite while aliasing step observations for shared code."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        return _alias_agentview(observation), reward, done, info


def run_rollout(env, policy, observation: dict[str, np.ndarray], **kwargs):
    """Run the shared absolute-joint rollout with D08 camera aliases."""
    return _base_run_rollout(_AgentviewAliasEnv(env), policy, _alias_agentview(observation), **kwargs)


# D08 has the same state/action controller semantics as D0, but its task name
# and camera identity differ. Replace only these integration hooks.
joint_evaluation.validate_environment_metadata = d08_conversion.validate_environment_metadata
joint_evaluation.make_environment_kwargs = make_environment_kwargs
joint_evaluation.run_rollout = run_rollout


@dataclasses.dataclass
class Args(joint_evaluation.Args):
    dataset_path: str = str(d08_conversion.DEFAULT_DATA_PATH)
    output_path: str = "data/robomimic_threading_d08_joint_256_eval.json"


def main(args: Args) -> None:
    joint_evaluation.main(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
