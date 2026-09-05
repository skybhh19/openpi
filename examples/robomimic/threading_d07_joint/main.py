"""Evaluate an OpenPI absolute-joint policy from fresh Threading_D07 resets."""

from __future__ import annotations

import dataclasses

import tyro

from examples.robomimic.threading_d07_joint import convert_robomimic_data_to_lerobot as d07_conversion
from examples.robomimic.threading_joint import main as joint_evaluation

# D07 has the same observations and controller semantics as D0. Reuse the
# audited rollout while replacing the task-specific metadata validation hook.
joint_evaluation.validate_environment_metadata = d07_conversion.validate_environment_metadata


@dataclasses.dataclass
class Args(joint_evaluation.Args):
    dataset_path: str = str(d07_conversion.DEFAULT_DATA_PATH)
    output_path: str = "data/robomimic_threading_d07_joint_256_eval.json"


def main(args: Args) -> None:
    joint_evaluation.main(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
