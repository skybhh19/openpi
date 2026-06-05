"""Build a keep-ranges JSON for a DROID RLDS subset.

The subset must preserve original DROID episode metadata:
`recording_folderpath` and `file_path`. The output JSON is compatible with
`DroidRldsDataset(filter_dict_path=...)`.
"""

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


def _first_step(steps):
    for step in steps.take(1):
        return step
    raise ValueError("Encountered an episode with no steps")


def _check_fixed_prompt(first_step: dict[str, Any], fixed_prompt: str) -> None:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if key not in first_step:
            continue
        value = _decode_scalar_string(first_step[key])
        if value != fixed_prompt:
            raise ValueError(f"{key}={value!r}, expected {fixed_prompt!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-builder-dir", type=Path, required=True)
    parser.add_argument(
        "--full-keep-ranges",
        type=Path,
        default=Path("examples/droid/droid_refinement/keep_ranges_1_0_1.json"),
    )
    parser.add_argument("--output-keep-ranges", type=Path, required=True)
    parser.add_argument("--output-episode-map", type=Path, required=True)
    parser.add_argument("--fixed-prompt", default="Put the pen in the cup")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow_datasets as tfds

    with args.full_keep_ranges.open() as f:
        full_keep_ranges = json.load(f)

    builder = tfds.builder_from_directory(str(args.subset_builder_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    subset_keep_ranges: dict[str, list[list[int]]] = {}
    episode_map: list[dict[str, Any]] = []
    missing_keep_ranges = []
    empty_keep_ranges = 0

    for episode_index, episode in enumerate(dataset):
        if args.max_episodes is not None and episode_index >= args.max_episodes:
            break

        metadata = episode["episode_metadata"]
        recording_folderpath = _decode_scalar_string(metadata["recording_folderpath"])
        file_path = _decode_scalar_string(metadata["file_path"])
        episode_key = f"{recording_folderpath}--{file_path}"

        _check_fixed_prompt(_first_step(episode["steps"]), args.fixed_prompt)

        ranges = full_keep_ranges.get(episode_key)
        if ranges is None:
            missing_keep_ranges.append(episode_key)
            ranges = []
        if not ranges:
            empty_keep_ranges += 1

        subset_keep_ranges[episode_key] = ranges
        episode_map.append(
            {
                "subset_episode_index": episode_index,
                "recording_folderpath": recording_folderpath,
                "file_path": file_path,
                "episode_key": episode_key,
                "keep_range_count": len(ranges),
                "kept_frame_count": sum(max(0, end - start) for start, end in ranges),
            }
        )

    if missing_keep_ranges:
        print(f"WARNING: {len(missing_keep_ranges)} episodes were missing from full keep ranges")
    print(f"Subset episodes scanned: {len(episode_map)}")
    print(f"Episodes with empty keep ranges: {empty_keep_ranges}")
    print(f"Total kept frames: {sum(item['kept_frame_count'] for item in episode_map)}")

    args.output_keep_ranges.parent.mkdir(parents=True, exist_ok=True)
    with args.output_keep_ranges.open("w") as f:
        json.dump(subset_keep_ranges, f)

    args.output_episode_map.parent.mkdir(parents=True, exist_ok=True)
    with args.output_episode_map.open("w") as f:
        json.dump(
            {
                "subset_builder_dir": str(args.subset_builder_dir),
                "full_keep_ranges": str(args.full_keep_ranges),
                "fixed_prompt": args.fixed_prompt,
                "missing_keep_range_count": len(missing_keep_ranges),
                "missing_keep_ranges": missing_keep_ranges,
                "episodes": episode_map,
            },
            f,
            indent=2,
        )

    print(f"Wrote keep ranges: {args.output_keep_ranges}")
    print(f"Wrote episode map: {args.output_episode_map}")


if __name__ == "__main__":
    main()
