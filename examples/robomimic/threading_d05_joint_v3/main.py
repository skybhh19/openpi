"""Evaluate an OpenPI absolute-joint policy from fresh Threading_D05 v3 resets."""

from __future__ import annotations

import dataclasses

import tyro

from examples.robomimic.threading_d05_joint_v3 import convert_robomimic_data_to_lerobot as d05_conversion
from examples.robomimic.threading_joint import main as joint_evaluation

joint_evaluation.validate_environment_metadata = d05_conversion.validate_environment_metadata


@dataclasses.dataclass
class Args(joint_evaluation.Args):
    dataset_path: str = str(d05_conversion.DEFAULT_DATA_PATH)
    output_path: str = "data/robomimic_threading_d05_joint_v3_256_eval.json"


def main(args: Args) -> None:
    joint_evaluation.main(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
