import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

filtering = importlib.import_module("examples.robomimic.threading_d05_joint_v3.build_filtering_keys")
conversion = importlib.import_module("examples.robomimic.threading_d05_joint_v3.convert_robomimic_data_to_lerobot")
conversion_test = importlib.import_module(
    "examples.robomimic.threading_d05_joint_v3.threading_d05_joint_v3_conversion_test"
)
env_args_fixture = conversion_test.env_args_fixture


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_path = tmp_path / "source.hdf5"
    lengths = {"demo_1": 2, "demo_2": 3, "demo_3": 4}
    with h5py.File(source_path, "w") as source:
        data = source.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args_fixture())
        for name, length in lengths.items():
            demo = data.create_group(name)
            demo.create_dataset("actions", shape=(length, 8), dtype=np.float64)
        masks = source.create_group("mask")
        masks.create_dataset("partial", data=np.asarray([b"demo_1", b"demo_3"]))
        masks.create_dataset("full", data=np.asarray([b"demo_2"]))

    dataset_path = tmp_path / "dataset"
    meta = dataset_path / "meta"
    meta.mkdir(parents=True)
    manifest = conversion.build_source_manifest(
        source_path=source_path,
        repo_id=conversion.DEFAULT_REPO_ID,
        env_args=env_args_fixture(),
        fps=20,
        frame_counts=lengths,
    )
    (meta / "robomimic_source_manifest.json").write_text(json.dumps(manifest))
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3, "total_frames": 9, "fps": 20}))
    episodes = [
        {"episode_index": index, "tasks": [conversion.DEFAULT_TASK_PROMPT], "length": length}
        for index, length in enumerate(lengths.values())
    ]
    (meta / "episodes.jsonl").write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
    return source_path, dataset_path


def test_filter_mapping_uses_source_masks_and_exact_frame_totals(tmp_path: Path):
    source_path, dataset_path = _write_fixture(tmp_path)
    demos, indices, frame_totals = filtering.validate_mapping(source_path, dataset_path)
    assert demos == ["demo_1", "demo_2", "demo_3"]
    assert indices == {"full": [1], "partial": [0, 2]}
    assert frame_totals == {"full": 3, "partial": 6}


def test_filter_mapping_rejects_task_or_provenance_mismatch(tmp_path: Path):
    source_path, dataset_path = _write_fixture(tmp_path)
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    episodes[0]["tasks"] = ["wrong task"]
    episodes_path.write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
    with pytest.raises(ValueError, match="task/length"):
        filtering.validate_mapping(source_path, dataset_path)


def test_filter_mapping_rejects_noncovering_labels(tmp_path: Path):
    source_path, dataset_path = _write_fixture(tmp_path)
    with h5py.File(source_path, "r+") as source:
        del source["mask/partial"]
        source["mask"].create_dataset("partial", data=np.asarray([b"demo_1"]))
    # Updating the source changes its size, which the manifest catches before mask coverage.
    manifest_path = dataset_path / "meta" / "robomimic_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_size_bytes"] = source_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover every"):
        filtering.validate_mapping(source_path, dataset_path)
