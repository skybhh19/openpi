"""Focused provenance tests for Threading_D06_Hard subset filters."""

import json
from pathlib import Path

import h5py
import pytest

from examples.robomimic.threading_d06_hard_joint_1000 import build_filtering_keys as filtering
from examples.robomimic.threading_d06_hard_joint_1000 import convert_robomimic_data_to_lerobot as conversion
from examples.robomimic.threading_d06_hard_joint_1000.threading_d06_hard_1000_conversion_test import _env_args


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_path = tmp_path / "source.hdf5"
    lengths = [2, 3, 4, 5]
    demos = [f"demo_{index}" for index in range(1, 5)]
    with h5py.File(source_path, "w") as source:
        data = source.create_group("data")
        data.attrs["env_args"] = json.dumps(_env_args())
        for name, length in zip(demos, lengths, strict=True):
            data.create_group(name).create_dataset("actions", shape=(length, 8), dtype="f4")
        mask = source.create_group("mask")
        mask.create_dataset("partial", data=[name.encode() for name in demos[:2]])
        mask.create_dataset("full", data=[name.encode() for name in demos[2:]])

    dataset_path = tmp_path / "dataset"
    meta = dataset_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 4, "total_frames": 14, "fps": 20}))
    episode_rows = [
        {"episode_index": index, "tasks": [conversion.DEFAULT_TASK_PROMPT], "length": length}
        for index, length in enumerate(lengths)
    ]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episode_rows))
    source_stat = source_path.stat()
    manifest = {
        "format_version": 1,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_env_name": conversion.EXPECTED_ENV_NAME,
        "source_env_version": conversion.EXPECTED_ENV_VERSION,
        "repo_id": conversion.DEFAULT_REPO_ID,
        "task_prompt": conversion.DEFAULT_TASK_PROMPT,
        "fps": 20,
        "total_episodes": 4,
        "total_frames": 14,
        "episodes": [
            {"episode_index": index, "source_demo": name, "length": length}
            for index, (name, length) in enumerate(zip(demos, lengths, strict=True))
        ],
    }
    (meta / "robomimic_source_manifest.json").write_text(json.dumps(manifest))
    return source_path, dataset_path


def test_filter_mapping_is_derived_from_masks_and_lengths(tmp_path):
    source_path, dataset_path = _write_fixture(tmp_path)
    demos, indices, frame_totals = filtering.validate_mapping(source_path, dataset_path)

    assert demos == ["demo_1", "demo_2", "demo_3", "demo_4"]
    assert indices == {"full": [2, 3], "partial": [0, 1]}
    assert frame_totals == {"full": 9, "partial": 5}


def test_filter_mapping_rejects_task_metadata_drift(tmp_path):
    source_path, dataset_path = _write_fixture(tmp_path)
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0]["tasks"] = ["different task"]
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="task/length"):
        filtering.validate_mapping(source_path, dataset_path)


def test_filter_mapping_rejects_overlapping_masks(tmp_path):
    source_path, dataset_path = _write_fixture(tmp_path)
    with h5py.File(source_path, "r+") as source:
        del source["mask/full"]
        source["mask"].create_dataset("full", data=[b"demo_2", b"demo_3", b"demo_4"])

    manifest_path = dataset_path / "meta" / "robomimic_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    stat = source_path.stat()
    manifest["source_size_bytes"] = stat.st_size
    manifest["source_mtime_ns"] = stat.st_mtime_ns
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        filtering.validate_mapping(source_path, dataset_path)
