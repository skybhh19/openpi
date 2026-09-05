"""Build checked LeRobot episode filters from the 1,000-demo D06-hard masks."""

import importlib
import json
from pathlib import Path
import sys

import h5py
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
conversion = importlib.import_module(
    "examples.robomimic.threading_d06_hard_joint_1000.convert_robomimic_data_to_lerobot"
)

DEFAULT_DATASET_PATH = Path("/iris/u/tiangao/lerobot_datasets/local/robomimic_threading_d06_hard_joint_1000_256")
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
        fps = conversion.validate_environment_metadata(env_args)
        demos = conversion.selected_demo_names(source, "all")
        source_lengths = [int(source[f"data/{name}/actions"].shape[0]) for name in demos]
        masks = {
            name: [value.decode() if isinstance(value, bytes) else str(value) for value in source[f"mask/{name}"][()]]
            for name in MASK_TO_OUTPUT_LABEL
        }

    source_stat = source_path.stat()
    expected_manifest_episodes = [
        {"episode_index": index, "source_demo": name, "length": source_lengths[index]}
        for index, name in enumerate(demos)
    ]
    expected_manifest_fields = {
        "format_version": 1,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_env_name": conversion.EXPECTED_ENV_NAME,
        "source_env_version": conversion.EXPECTED_ENV_VERSION,
        "repo_id": conversion.DEFAULT_REPO_ID,
        "task_prompt": conversion.DEFAULT_TASK_PROMPT,
        "fps": fps,
        "total_episodes": len(demos),
        "total_frames": sum(source_lengths),
        "episodes": expected_manifest_episodes,
    }
    if manifest != expected_manifest_fields:
        differing = sorted(
            key
            for key in set(manifest) | set(expected_manifest_fields)
            if manifest.get(key) != expected_manifest_fields.get(key)
        )
        raise ValueError(f"Converted source manifest differs from the requested HDF5/dataset: {differing}")

    expected_target_episodes = [
        {"episode_index": index, "tasks": [conversion.DEFAULT_TASK_PROMPT], "length": length}
        for index, length in enumerate(source_lengths)
    ]
    if target_episodes != expected_target_episodes:
        raise ValueError("LeRobot episodes.jsonl is renumbered or has mismatched task/length metadata")
    if info.get("total_episodes") != len(demos) or info.get("total_frames") != sum(source_lengths):
        raise ValueError("LeRobot info.json totals do not match the source HDF5")
    if info.get("fps") != fps:
        raise ValueError(f"Expected {fps} Hz LeRobot data, got {info.get('fps')!r}")

    all_demos = set(demos)
    full_set, partial_set = set(masks["full"]), set(masks["partial"])
    for name, values in masks.items():
        if len(values) != len(set(values)):
            raise ValueError(f"HDF5 mask/{name} contains duplicate demo names")
        unknown = set(values) - all_demos
        if unknown:
            raise ValueError(f"HDF5 mask/{name} references unknown demos: {sorted(unknown)[:5]}")
        if not values:
            raise ValueError(f"HDF5 mask/{name} is empty")
    if full_set & partial_set:
        raise ValueError("HDF5 full/partial masks overlap")
    if full_set | partial_set != all_demos:
        raise ValueError("HDF5 full/partial masks do not cover all demos")

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
    if len(demos) != conversion.EXPECTED_SOURCE_EPISODES:
        raise ValueError(f"Expected all {conversion.EXPECTED_SOURCE_EPISODES} source demos, got {len(demos)}")
    if sum(frame_totals.values()) != conversion.EXPECTED_SOURCE_FRAMES:
        raise ValueError("Full/partial filtered frame totals do not cover the complete source")

    destination.mkdir(parents=True, exist_ok=True)
    for mask_name, output_label in MASK_TO_OUTPUT_LABEL.items():
        output_path = destination / f"threading_d06_hard_joint_1000_256_{output_label}_episode_indices.json"
        output_path.write_text(json.dumps(indices[mask_name], indent=2) + "\n")
        print(f"Wrote {output_path}: {len(indices[mask_name])} episodes / {frame_totals[mask_name]} frames")


if __name__ == "__main__":
    tyro.cli(main)
