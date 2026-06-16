"""Build LeRobot episode filters from human labels for the wrench-on-hook task."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ANNOTATIONS_CSV = Path(__file__).with_name("wrench_on_hook_06132026_annotations.csv")
OUTPUT_DIR = Path(__file__).with_name("lerobot_filtering_keys")
RAW_DATA_DIR = Path("/iris/u/tiangao/projects/droid/data/success/2026-06-13")
LEROBOT_DATASET_DIR = Path("/iliad/u/tiangao/lerobot_datasets/skybhh19/droid_wrench_on_hook")

OBSERVABILITY_RANKS = {"full": 1, "partial": 0}
OPTIMALITY_RANKS = {"better": 2, "okay": 1, "worse": 0}
LABEL_PCTS = (25, 50, 75)
TOTAL_EPISODES = 98


def _load_annotations(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != TOTAL_EPISODES:
        raise ValueError(f"Expected {TOTAL_EPISODES} annotation rows, found {len(rows)} in {csv_path}")
    return rows


def _load_raw_episode_paths(raw_data_dir: Path) -> list[Path]:
    paths = sorted(raw_data_dir.glob("*/trajectory.h5"))
    if len(paths) != TOTAL_EPISODES:
        raise ValueError(f"Expected {TOTAL_EPISODES} raw episodes, found {len(paths)} in {raw_data_dir}")
    return paths


def _assert_label_matches_episode(row: dict[str, str], raw_episode_path: Path) -> int:
    ep_idx = int(row["ep_idx"])
    episode_name = row["episode"]
    raw_episode_name = raw_episode_path.parent.name
    if episode_name != raw_episode_name:
        raise ValueError(
            f"CSV episode mismatch for ep_idx={ep_idx}: row has {episode_name!r}, "
            f"raw path has {raw_episode_name!r}"
        )
    return ep_idx


def _validate_annotations(rows: list[dict[str, str]], raw_paths: list[Path], lerobot_dataset_dir: Path) -> None:
    seen_indices = set()
    for expected_idx, (row, raw_episode_path) in enumerate(zip(rows, raw_paths, strict=True)):
        ep_idx = _assert_label_matches_episode(row, raw_episode_path)
        if ep_idx != expected_idx:
            raise ValueError(f"CSV ep_idx mismatch: row position {expected_idx} has ep_idx={ep_idx}")
        if ep_idx in seen_indices:
            raise ValueError(f"Duplicate ep_idx in annotations: {ep_idx}")
        seen_indices.add(ep_idx)

        episode_file = lerobot_dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not episode_file.exists():
            raise FileNotFoundError(f"Missing LeRobot episode file for ep_idx={ep_idx}: {episode_file}")

        if row["observability"] not in OBSERVABILITY_RANKS:
            raise ValueError(f"Unknown observability label for ep_idx={ep_idx}: {row['observability']!r}")
        if row["optimality"] not in OPTIMALITY_RANKS:
            raise ValueError(f"Unknown optimality label for ep_idx={ep_idx}: {row['optimality']!r}")


def _target_count(pct: int, total: int) -> int:
    return int(total * pct / 100 + 0.5)


def _select_by_rank(rows: list[dict[str, str]], label_key: str, ranks: dict[str, int], pct: int) -> list[int]:
    target_count = _target_count(pct, len(rows))
    ranked_rows = sorted(rows, key=lambda row: (-ranks[row[label_key]], int(row["ep_idx"])))
    return [int(row["ep_idx"]) for row in ranked_rows[:target_count]]


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def build_filters(
    annotations_csv: Path,
    raw_data_dir: Path,
    lerobot_dataset_dir: Path,
    output_dir: Path,
    check_only: bool,
) -> dict[str, object]:
    rows = _load_annotations(annotations_csv)
    raw_paths = _load_raw_episode_paths(raw_data_dir)
    _validate_annotations(rows, raw_paths, lerobot_dataset_dir)

    filters: dict[str, list[int]] = {}
    for pct in LABEL_PCTS:
        filters[f"observabilitypct{pct}"] = _select_by_rank(rows, "observability", OBSERVABILITY_RANKS, pct)
        filters[f"optimalitypct{pct}"] = _select_by_rank(rows, "optimality", OPTIMALITY_RANKS, pct)

    manifest = {
        "annotations_csv": str(annotations_csv),
        "raw_data_dir": str(raw_data_dir),
        "lerobot_dataset_dir": str(lerobot_dataset_dir),
        "total_episodes": len(rows),
        "selection": {
            "human_label_percentages": list(LABEL_PCTS),
            "human_label_target_counts": {str(pct): _target_count(pct, len(rows)) for pct in LABEL_PCTS},
            "tie_break": "ascending ep_idx within the same label",
            "observability_rank": ["full", "partial"],
            "optimality_rank": ["better", "okay", "worse"],
        },
        "label_counts": {
            "observability": dict(Counter(row["observability"] for row in rows)),
            "optimality": dict(Counter(row["optimality"] for row in rows)),
        },
        "filters": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in filters.items():
        label_key = "observability" if name.startswith("observability") else "optimality"
        filename = f"wrench_on_hook_{name}_episode_indices.json"
        label_counts = Counter(rows[index][label_key] for index in indices)
        manifest["filters"][filename] = {
            "count": len(indices),
            "label_counts": dict(label_counts),
            "episode_indices": indices,
        }
        if not check_only:
            _write_json(output_dir / filename, indices)

    if not check_only:
        _write_json(output_dir / "wrench_on_hook_human_label_filters_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, default=ANNOTATIONS_CSV)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--lerobot-dataset-dir", type=Path, default=LEROBOT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = build_filters(
        annotations_csv=args.annotations_csv,
        raw_data_dir=args.raw_data_dir,
        lerobot_dataset_dir=args.lerobot_dataset_dir,
        output_dir=args.output_dir,
        check_only=args.check_only,
    )
    print(json.dumps(manifest["filters"], indent=2))


if __name__ == "__main__":
    main()
