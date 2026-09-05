"""Build checked LeRobot episode filters from Threading_D05 v3 HDF5 masks."""

import importlib
import json
from pathlib import Path
import sys

import h5py
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
conversion = importlib.import_module("examples.robomimic.threading_d05_joint_v3.convert_robomimic_data_to_lerobot")

DEFAULT_DATASET_PATH = Path("/iris/u/tiangao/lerobot_datasets") / conversion.DEFAULT_REPO_ID
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "lerobot_filtering_keys"
MASK_TO_OUTPUT_LABEL = {"full": "full_only", "partial": "partial_only"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decode_mask(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_mapping(source_path: Path, dataset_path: Path) -> tuple[list[str], dict[str, list[int]], dict[str, int]]:
    """Validate source provenance and return exact mask indices/frame totals."""
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    target_episodes = _read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    manifest = json.loads((dataset_path / "meta" / "robomimic_source_manifest.json").read_text())

    with h5py.File(source_path, "r") as source:
        if "env_args" not in source["data"].attrs:
            raise ValueError("Source HDF5 is missing environment metadata")
        env_args = json.loads(source["data"].attrs["env_args"])
        fps = conversion.validate_environment_metadata(env_args)
        demos = conversion.selected_demo_names(source, "all")
        source_lengths = [int(source[f"data/{name}/actions"].shape[0]) for name in demos]
        missing_masks = [name for name in MASK_TO_OUTPUT_LABEL if f"mask/{name}" not in source]
        if missing_masks:
            raise ValueError(f"Source HDF5 is missing required label masks: {missing_masks}")
        masks = {name: _decode_mask(source[f"mask/{name}"][()]) for name in MASK_TO_OUTPUT_LABEL}

    frame_counts = dict(zip(demos, source_lengths, strict=True))
    expected_manifest = conversion.build_source_manifest(
        source_path=source_path,
        repo_id=conversion.DEFAULT_REPO_ID,
        env_args=env_args,
        fps=fps,
        frame_counts=frame_counts,
    )
    if manifest != expected_manifest:
        raise ValueError("Converted source manifest differs from the requested HDF5, repo, task, order, or lengths")

    expected_target_episodes = [
        {"episode_index": index, "tasks": [conversion.DEFAULT_TASK_PROMPT], "length": length}
        for index, length in enumerate(source_lengths)
    ]
    if target_episodes != expected_target_episodes:
        raise ValueError("LeRobot episodes.jsonl is renumbered or has mismatched task/length metadata")

    expected_totals = (len(demos), sum(source_lengths), fps)
    target_totals = (info.get("total_episodes"), info.get("total_frames"), info.get("fps"))
    if target_totals != expected_totals:
        raise ValueError(f"LeRobot totals/fps {target_totals} do not match source {expected_totals}")

    all_demos = set(demos)
    full_set, partial_set = set(masks["full"]), set(masks["partial"])
    if any(len(values) != len(set(values)) for values in masks.values()):
        raise ValueError("HDF5 full/partial masks contain duplicate demo names")
    unknown = (full_set | partial_set) - all_demos
    if unknown:
        raise ValueError(f"HDF5 masks reference unknown demos: {sorted(unknown)[:5]}")
    if full_set & partial_set or full_set | partial_set != all_demos:
        raise ValueError("HDF5 full/partial masks must be disjoint and cover every converted demo")

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
    """Validate the complete conversion and write full_only/partial_only keys."""
    source_path = Path(data_path).expanduser()
    target_path = Path(dataset_path).expanduser()
    destination = Path(output_dir).expanduser()
    demos, indices, frame_totals = validate_mapping(source_path, target_path)

    destination.mkdir(parents=True, exist_ok=True)
    for mask_name, output_label in MASK_TO_OUTPUT_LABEL.items():
        output_path = destination / f"threading_d05_joint_v3_256_{output_label}_episode_indices.json"
        output_path.write_text(json.dumps(indices[mask_name], indent=2) + "\n")
        print(f"Wrote {output_path}: {len(indices[mask_name])} episodes / {frame_totals[mask_name]} frames")
    print(f"Validated complete source coverage: {len(demos)} episodes / {sum(frame_totals.values())} frames")


if __name__ == "__main__":
    tyro.cli(main)
