"""Build checked LeRobot episode filters from Threading_D05 HDF5 masks."""

import importlib
import json
from pathlib import Path
import sys

import h5py
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
conversion = importlib.import_module("examples.robomimic.threading_d05_joint.convert_robomimic_data_to_lerobot")

DEFAULT_DATASET_PATH = Path("/iris/u/tiangao/lerobot_datasets/local/robomimic_threading_d05_joint_v2_256")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "lerobot_filtering_keys"
MASK_TO_OUTPUT_LABEL = {"full": "full_only", "partial": "partial_only"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_mapping(source_path: Path, dataset_path: Path) -> tuple[list[str], dict[str, list[int]], dict[str, int]]:
    """Validate provenance and return source demos, mask indices, and frame totals."""
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    target_episodes = _read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    manifest = json.loads((dataset_path / "meta" / "robomimic_source_manifest.json").read_text())

    with h5py.File(source_path, "r") as source:
        env_args = json.loads(source["data"].attrs["env_args"])
        conversion.validate_environment_metadata(env_args)
        demos = conversion.selected_demo_names(source, "all")
        source_lengths = [source[f"data/{name}/actions"].shape[0] for name in demos]
        masks = {
            name: [value.decode() if isinstance(value, bytes) else str(value) for value in source[f"mask/{name}"][()]]
            for name in MASK_TO_OUTPUT_LABEL
        }

    expected_manifest_episodes = [
        {"episode_index": index, "source_demo": name, "length": source_lengths[index]}
        for index, name in enumerate(demos)
    ]
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported or missing source-manifest format_version")
    if Path(manifest.get("source_path", "")).resolve() != source_path.resolve():
        raise ValueError("Converted dataset source_path does not match the requested HDF5")
    if manifest.get("source_size_bytes") != source_path.stat().st_size:
        raise ValueError("Converted dataset source size does not match the requested HDF5")
    if manifest.get("source_env_name") != conversion.EXPECTED_ENV_NAME:
        raise ValueError("Converted dataset has the wrong source environment")
    if manifest.get("repo_id") != conversion.DEFAULT_REPO_ID:
        raise ValueError("Converted dataset has the wrong repo_id")
    if manifest.get("episodes") != expected_manifest_episodes:
        raise ValueError("Converted episode order/lengths differ from the source HDF5")

    expected_target_episodes = [
        {"episode_index": index, "tasks": [conversion.DEFAULT_TASK_PROMPT], "length": length}
        for index, length in enumerate(source_lengths)
    ]
    if target_episodes != expected_target_episodes:
        raise ValueError("LeRobot episodes.jsonl is renumbered or has mismatched task/length metadata")
    if info.get("total_episodes") != len(demos) or info.get("total_frames") != sum(source_lengths):
        raise ValueError("LeRobot info.json totals do not match the source HDF5")
    if info.get("fps") != 20:
        raise ValueError(f"Expected 20 Hz LeRobot data, got {info.get('fps')!r}")

    all_demos = set(demos)
    full_set, partial_set = set(masks["full"]), set(masks["partial"])
    if full_set & partial_set or full_set | partial_set != all_demos:
        raise ValueError("HDF5 full/partial masks must be disjoint and cover all demos")
    if any(len(values) != len(set(values)) for values in masks.values()):
        raise ValueError("HDF5 masks contain duplicate demo names")

    demo_to_index = {name: index for index, name in enumerate(demos)}
    indices = {name: sorted(demo_to_index[demo] for demo in values) for name, values in masks.items()}
    frame_totals = {name: sum(source_lengths[index] for index in selected) for name, selected in indices.items()}
    return demos, indices, frame_totals


def main(
    data_path: str = str(conversion.DEFAULT_DATA_PATH),
    *,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    """Validate the all-200 conversion and write full_only/partial_only keys."""
    source_path = Path(data_path).expanduser()
    target_path = Path(dataset_path).expanduser()
    destination = Path(output_dir).expanduser()
    demos, indices, frame_totals = validate_mapping(source_path, target_path)
    if len(demos) != 200:
        raise ValueError(f"Expected all 200 source demos, got {len(demos)}")

    destination.mkdir(parents=True, exist_ok=True)
    for mask_name, output_label in MASK_TO_OUTPUT_LABEL.items():
        output_path = destination / f"threading_d05_joint_v2_256_{output_label}_episode_indices.json"
        output_path.write_text(json.dumps(indices[mask_name], indent=2) + "\n")
        print(f"Wrote {output_path}: {len(indices[mask_name])} episodes / {frame_totals[mask_name]} frames")


if __name__ == "__main__":
    tyro.cli(main)
