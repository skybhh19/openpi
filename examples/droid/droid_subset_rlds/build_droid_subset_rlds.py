"""Build a small DROID RLDS subset by copying selected serialized TFRecord episodes.

The output keeps the original DROID TFDS/RLDS schema. By default, it only overwrites
the three language instruction fields so training can use a fixed downstream prompt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
from typing import Any

LANGUAGE_FIELD_NAMES = {
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
}


def _load_index(index_jsonl: Path, max_episodes: int | None) -> list[dict[str, Any]]:
    rows = []
    with index_jsonl.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_episodes is not None and len(rows) >= max_episodes:
                break
    if not rows:
        raise ValueError(f"No episodes found in {index_jsonl}")
    return rows


def _source_tfrecord_path(source_builder_dir: Path, source_tfrecord: str) -> Path:
    path = source_builder_dir / source_tfrecord
    if not path.exists():
        raise FileNotFoundError(f"Missing source TFRecord: {path}")
    return path


def _patch_example_language(raw: bytes, fixed_prompt: str, tf) -> tuple[bytes, int]:
    """Patch language fields in a serialized tf.train.Example or SequenceExample."""
    prompt_bytes = fixed_prompt.encode("utf-8")

    example = tf.train.Example.FromString(raw)
    patched = 0
    for key, feature in example.features.feature.items():
        if key.split("/")[-1] not in LANGUAGE_FIELD_NAMES:
            continue
        if feature.WhichOneof("kind") != "bytes_list":
            continue
        count = len(feature.bytes_list.value)
        if count:
            del feature.bytes_list.value[:]
            feature.bytes_list.value.extend([prompt_bytes] * count)
            patched += count
    if patched:
        return example.SerializeToString(), patched

    sequence_example = tf.train.SequenceExample.FromString(raw)
    for key, feature_list in sequence_example.feature_lists.feature_list.items():
        if key.split("/")[-1] not in LANGUAGE_FIELD_NAMES:
            continue
        for feature in feature_list.feature:
            if feature.WhichOneof("kind") != "bytes_list":
                continue
            count = len(feature.bytes_list.value)
            if count:
                del feature.bytes_list.value[:]
                feature.bytes_list.value.extend([prompt_bytes] * count)
                patched += count
    if patched:
        return sequence_example.SerializeToString(), patched

    return raw, 0


def _copy_metadata(
    source_builder_dir: Path,
    output_builder_dir: Path,
    shard_lengths: list[int],
    dataset_name: str,
    dataset_version: str,
):
    features_src = source_builder_dir / "features.json"
    dataset_info_src = source_builder_dir / "dataset_info.json"
    if not features_src.exists() or not dataset_info_src.exists():
        raise FileNotFoundError(f"Expected features.json and dataset_info.json in {source_builder_dir}")

    shutil.copy2(features_src, output_builder_dir / "features.json")

    with dataset_info_src.open() as f:
        dataset_info = json.load(f)

    dataset_info["name"] = dataset_name
    dataset_info["version"] = dataset_version
    output_files = sorted(output_builder_dir.glob(f"{dataset_name}-train.tfrecord-*"))
    num_bytes = sum(path.stat().st_size for path in output_files)

    source_train_split = next(split for split in dataset_info["splits"] if split["name"] == "train")
    train_split = dict(source_train_split)
    train_split["numBytes"] = str(num_bytes)
    train_split["shardLengths"] = [str(length) for length in shard_lengths]
    dataset_info["splits"] = [train_split]

    with (output_builder_dir / "dataset_info.json").open("w") as f:
        json.dump(dataset_info, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-jsonl", type=Path, default=Path("examples/droid/droid_pen_cup_yes_index.jsonl"))
    parser.add_argument("--source-builder-dir", type=Path, default=Path("/iliad2/group/datasets/droid/1.0.1"))
    parser.add_argument("--output-builder-dir", type=Path, required=True)
    parser.add_argument("--fixed-prompt", default="Put the pen in the cup")
    parser.add_argument("--dataset-name", default="droid")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--examples-per-shard", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.examples_per_shard <= 0:
        raise ValueError("--examples-per-shard must be positive")

    rows = _load_index(args.index_jsonl, args.max_episodes)
    selected_by_shard: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        selected_by_shard[row["source_tfrecord"]].add(int(row["episode_in_shard"]))

    duplicate_count = len(rows) - sum(len(v) for v in selected_by_shard.values())
    total_selected = sum(len(v) for v in selected_by_shard.values())
    num_output_shards = math.ceil(total_selected / args.examples_per_shard)
    dataset_version = args.dataset_version or args.output_builder_dir.name
    print(f"Selected rows: {len(rows)} ({duplicate_count} duplicates)")
    print(f"Unique selected episodes: {total_selected}")
    print(f"Source shards touched: {len(selected_by_shard)}")
    print(f"Output shards: {num_output_shards}")
    print(f"Dataset name/version: {args.dataset_name}/{dataset_version}")
    print(f"Fixed prompt: {args.fixed_prompt!r}")

    if args.dry_run:
        return

    if args.output_builder_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_builder_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_builder_dir)
    args.output_builder_dir.mkdir(parents=True)

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf

    writers: list[Any] = []
    shard_lengths = [0] * num_output_shards
    for shard_idx in range(num_output_shards):
        path = args.output_builder_dir / (
            f"{args.dataset_name}-train.tfrecord-{shard_idx:05d}-of-{num_output_shards:05d}"
        )
        writers.append(tf.io.TFRecordWriter(str(path)))

    manifest = {
        "source_builder_dir": str(args.source_builder_dir),
        "index_jsonl": str(args.index_jsonl),
        "fixed_prompt": args.fixed_prompt,
        "dataset_name": args.dataset_name,
        "dataset_version": dataset_version,
        "examples_per_shard": args.examples_per_shard,
        "selected_episode_count": total_selected,
        "source_shard_count": len(selected_by_shard),
        "episodes": [],
    }

    written = 0
    language_patch_count = 0
    try:
        for source_tfrecord in sorted(selected_by_shard):
            wanted = selected_by_shard[source_tfrecord]
            found: set[int] = set()
            source_path = _source_tfrecord_path(args.source_builder_dir, source_tfrecord)
            print(f"Reading {source_path.name}: selecting {len(wanted)} episodes")
            dataset = tf.data.TFRecordDataset([str(source_path)])
            for episode_in_shard, raw_tensor in enumerate(dataset):
                if episode_in_shard not in wanted:
                    continue
                raw, patched = _patch_example_language(raw_tensor.numpy(), args.fixed_prompt, tf)
                if patched == 0:
                    raise ValueError(
                        f"No language instruction feature was patched for {source_tfrecord}:{episode_in_shard}"
                    )
                output_shard = written // args.examples_per_shard
                writers[output_shard].write(raw)
                shard_lengths[output_shard] += 1
                written += 1
                language_patch_count += patched
                found.add(episode_in_shard)
                manifest["episodes"].append(
                    {
                        "source_tfrecord": source_tfrecord,
                        "episode_in_shard": episode_in_shard,
                        "output_shard": output_shard,
                    }
                )
                if found == wanted:
                    break
            missing = sorted(wanted - found)
            if missing:
                raise ValueError(f"Missing selected episodes in {source_tfrecord}: {missing}")
    finally:
        for writer in writers:
            writer.close()

    if written != total_selected:
        raise RuntimeError(f"Expected to write {total_selected} episodes, wrote {written}")

    _copy_metadata(args.source_builder_dir, args.output_builder_dir, shard_lengths, args.dataset_name, dataset_version)
    with (args.output_builder_dir / "subset_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {written} episodes to {args.output_builder_dir}")
    print(f"Patched {language_patch_count} language feature values")
    print(f"Shard lengths: {shard_lengths}")


if __name__ == "__main__":
    main()
