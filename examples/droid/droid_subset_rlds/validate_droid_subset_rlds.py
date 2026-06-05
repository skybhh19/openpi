"""Validate a DROID RLDS subset and optional keep-ranges file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _decode_scalar_string(value: Any) -> str:
    raw = value.numpy()
    if hasattr(raw, "shape") and raw.shape:
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _first_step(steps):
    for step in steps.take(1):
        return step
    raise ValueError("Encountered an episode with no steps")


def _episode_key(episode: dict[str, Any]) -> str:
    metadata = episode["episode_metadata"]
    return f"{_decode_scalar_string(metadata['recording_folderpath'])}--{_decode_scalar_string(metadata['file_path'])}"


def _validate_step_schema(step: dict[str, Any], fixed_prompt: str) -> None:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if _decode_scalar_string(step[key]) != fixed_prompt:
            raise ValueError(f"{key} is not fixed to {fixed_prompt!r}")

    observation = step["observation"]
    action_dict = step["action_dict"]
    expected_shapes = {
        "observation/joint_position": _shape(observation["joint_position"]),
        "observation/gripper_position": _shape(observation["gripper_position"]),
        "observation/exterior_image_1_left": _shape(observation["exterior_image_1_left"]),
        "observation/exterior_image_2_left": _shape(observation["exterior_image_2_left"]),
        "observation/wrist_image_left": _shape(observation["wrist_image_left"]),
        "action_dict/joint_velocity": _shape(action_dict["joint_velocity"]),
        "action_dict/joint_position": _shape(action_dict["joint_position"]),
        "action_dict/gripper_position": _shape(action_dict["gripper_position"]),
    }
    required = {
        "observation/joint_position": (7,),
        "observation/gripper_position": (1,),
        "observation/exterior_image_1_left": (180, 320, 3),
        "observation/exterior_image_2_left": (180, 320, 3),
        "observation/wrist_image_left": (180, 320, 3),
        "action_dict/joint_velocity": (7,),
        "action_dict/joint_position": (7,),
        "action_dict/gripper_position": (1,),
    }
    for key, expected in required.items():
        if expected_shapes[key] != expected:
            raise ValueError(f"{key} shape {expected_shapes[key]} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-builder-dir", type=Path, required=True)
    parser.add_argument("--keep-ranges", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--fixed-prompt", default="Put the pen in the cup")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow_datasets as tfds

    keep_ranges = None
    if args.keep_ranges is not None:
        with args.keep_ranges.open() as f:
            keep_ranges = json.load(f)

    builder = tfds.builder_from_directory(str(args.subset_builder_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    episode_count = 0
    missing_keep_ranges = 0
    empty_keep_ranges = 0
    kept_frame_count = 0
    first_keys = []

    for episode in dataset:
        if args.max_episodes is not None and episode_count >= args.max_episodes:
            break
        first_step = _first_step(episode["steps"])
        _validate_step_schema(first_step, args.fixed_prompt)

        key = _episode_key(episode)
        if len(first_keys) < 5:
            first_keys.append(key)

        if keep_ranges is not None:
            ranges = keep_ranges.get(key)
            if ranges is None:
                missing_keep_ranges += 1
            elif not ranges:
                empty_keep_ranges += 1
            if ranges:
                kept_frame_count += sum(max(0, end - start) for start, end in ranges)

        episode_count += 1

    if args.expected_episodes is not None and episode_count != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, scanned {episode_count}")
    if keep_ranges is not None and missing_keep_ranges:
        raise ValueError(f"{missing_keep_ranges} scanned episodes were missing from keep-ranges file")

    print(f"Builder dir: {args.subset_builder_dir}")
    print(f"TFDS name/version: {builder.info.name}/{builder.info.version}")
    print(f"Episodes scanned: {episode_count}")
    print(f"First episode keys: {first_keys}")
    if keep_ranges is not None:
        print(f"Episodes with empty keep ranges: {empty_keep_ranges}")
        print(f"Kept frames among scanned episodes: {kept_frame_count}")
    print("Validation passed.")


if __name__ == "__main__":
    main()
