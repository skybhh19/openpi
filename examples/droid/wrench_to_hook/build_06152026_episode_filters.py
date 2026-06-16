"""Build LeRobot episode filters for the June 15 wrench-on-hook dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


ANNOTATIONS_CSV = Path(__file__).with_name("observability_scores_2026-06-15.csv")
OUTPUT_DIR = Path(__file__).with_name("lerobot_filtering_keys")
RAW_DATA_DIR = Path("/iris/u/tiangao/projects/droid/data/success/2026-06-15")
LEROBOT_DATASET_DIR = Path("/iliad/u/tiangao/lerobot_datasets/skybhh19/droid_wrench_on_hook_06152026")

DATASET_NAME = "skybhh19/droid_wrench_on_hook_06152026"
FILE_PREFIX = "wrench_on_hook_06152026"
FILTER_PCTS = (25, 50, 75)
RANDOM_SEED = 42
TOTAL_EPISODES = 96


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


def _target_count(pct: int, total: int) -> int:
    return int(total * pct / 100 + 0.5)


def _validate_annotations(rows: list[dict[str, str]], raw_paths: list[Path], lerobot_dataset_dir: Path) -> None:
    seen_indices = set()
    for expected_idx, (row, raw_episode_path) in enumerate(zip(rows, raw_paths, strict=True)):
        ep_idx = int(row["ep_idx"])
        if ep_idx != expected_idx:
            raise ValueError(f"CSV ep_idx mismatch: row position {expected_idx} has ep_idx={ep_idx}")
        if ep_idx in seen_indices:
            raise ValueError(f"Duplicate ep_idx in annotations: {ep_idx}")
        seen_indices.add(ep_idx)

        if row["dataset"] != "wrench_on_hook":
            raise ValueError(f"Unexpected dataset for ep_idx={ep_idx}: {row['dataset']!r}")

        raw_episode_name = raw_episode_path.parent.name
        if row["episode"] != raw_episode_name:
            raise ValueError(
                f"CSV episode mismatch for ep_idx={ep_idx}: row has {row['episode']!r}, "
                f"raw path has {raw_episode_name!r}"
            )

        episode_file = lerobot_dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not episode_file.exists():
            raise FileNotFoundError(f"Missing LeRobot episode file for ep_idx={ep_idx}: {episode_file}")

        try:
            observability = int(row["observability"])
        except ValueError as exc:
            raise ValueError(f"Invalid observability score for ep_idx={ep_idx}: {row['observability']!r}") from exc
        if observability not in (1, 2, 3):
            raise ValueError(f"Observability score must be 1, 2, or 3 for ep_idx={ep_idx}: {observability}")


def _select_by_observability(rows: list[dict[str, str]], pct: int) -> list[int]:
    target_count = _target_count(pct, len(rows))
    ranked_rows = sorted(rows, key=lambda row: (-int(row["observability"]), int(row["ep_idx"])))
    return [int(row["ep_idx"]) for row in ranked_rows[:target_count]]


def _select_random(total: int, pct: int, seed: int) -> list[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[: _target_count(pct, total)]


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def build_filters(
    annotations_csv: Path,
    raw_data_dir: Path,
    lerobot_dataset_dir: Path,
    output_dir: Path,
    check_only: bool,
    random_seed: int,
) -> dict[str, object]:
    rows = _load_annotations(annotations_csv)
    raw_paths = _load_raw_episode_paths(raw_data_dir)
    _validate_annotations(rows, raw_paths, lerobot_dataset_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    observability_manifest = {
        "dataset": DATASET_NAME,
        "annotations_csv": str(annotations_csv),
        "raw_data_dir": str(raw_data_dir),
        "lerobot_dataset_dir": str(lerobot_dataset_dir),
        "total_episodes": len(rows),
        "selection": {
            "percentages": list(FILTER_PCTS),
            "target_counts": {str(pct): _target_count(pct, len(rows)) for pct in FILTER_PCTS},
            "rank": "higher numeric observability is better",
            "tie_break": "ascending ep_idx within the same score",
        },
        "label_counts": {"observability": dict(Counter(row["observability"] for row in rows))},
        "filters": {},
    }

    random_manifest = {
        "dataset": DATASET_NAME,
        "seed": random_seed,
        "total_episodes": len(rows),
        "policy": "shuffle LeRobot episode indices once per percentage with seed, then take floor(total_episodes * pct + 0.5) episodes",
        "filters": {},
    }

    for pct in FILTER_PCTS:
        observability_indices = _select_by_observability(rows, pct)
        observability_filename = f"{FILE_PREFIX}_observabilitypct{pct}_episode_indices.json"
        observability_manifest["filters"][observability_filename] = {
            "count": len(observability_indices),
            "label_counts": dict(Counter(rows[index]["observability"] for index in observability_indices)),
            "episode_indices": observability_indices,
        }
        if not check_only:
            _write_json(output_dir / observability_filename, observability_indices)

        random_indices = _select_random(len(rows), pct, random_seed)
        random_filename = f"{FILE_PREFIX}_randompct{pct}_episode_indices.json"
        random_manifest["filters"][random_filename] = {
            "percentage": pct / 100,
            "episode_count": len(random_indices),
            "episode_indices_path": str(output_dir / random_filename),
            "episode_indices": random_indices,
        }
        if not check_only:
            _write_json(output_dir / random_filename, random_indices)

    manifest = {
        "observability": observability_manifest,
        "random": random_manifest,
    }
    if not check_only:
        _write_json(output_dir / f"{FILE_PREFIX}_observability_filters_manifest.json", observability_manifest)
        _write_json(output_dir / f"{FILE_PREFIX}_random_filters_manifest.json", random_manifest)
        _write_json(output_dir / f"{FILE_PREFIX}_filters_manifest.json", manifest)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, default=ANNOTATIONS_CSV)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--lerobot-dataset-dir", type=Path, default=LEROBOT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = build_filters(
        annotations_csv=args.annotations_csv,
        raw_data_dir=args.raw_data_dir,
        lerobot_dataset_dir=args.lerobot_dataset_dir,
        output_dir=args.output_dir,
        check_only=args.check_only,
        random_seed=args.random_seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
