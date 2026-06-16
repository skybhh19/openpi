"""Build LeRobot episode filters from human labels for the pen-in-blue-cup task."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ANNOTATIONS_CSV = Path(__file__).with_name("pen_in_cup_06102026_annotations.csv")
OUTPUT_DIR = Path(__file__).with_name("lerobot_filtering_keys")
RAW_DATA_DIR = Path("/iris/u/tiangao/projects/droid/data/success/2026-06-10")
LEROBOT_DATASET_DIR = Path("/iliad/u/tiangao/lerobot_datasets/skybhh19/droid_pen_in_blue_cup")

OBSERVABILITY_RANKS = {"full": 1, "partial": 0}
OPTIMALITY_RANKS = {"better": 2, "okay": 1, "worse": 0}
LABEL_PCTS = (25, 50, 75)
SCORE_PCTS = (25, 50, 75)
SCORE_COLUMNS = {
    "rank_h_wrist_ext_instant_abs": "rank H wrist+ext w/ instant_abs",
    "rank_h_robot_wrist_ext_instant_abs": "rank H robot+wrist_ext w/ instant_abs",
    "rank_nll_robot_wrist_ext_minus_robot_instant_abs": "rank NLL robot+wrist+ext - robot w/ instant_abs",
    "rank_nll_robot_wrist_ext_minus_action_prior_instant_abs": (
        "rank NLL robot+wrist_ext - action prior w/ instant_abs"
    ),
}
TOTAL_EPISODES = 106


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

    videos = json.loads(row["videos"])
    expected_video_name = f"{ep_idx:03d}_{episode_name}.mp4"
    if not videos or Path(videos[0]).name != expected_video_name:
        raise ValueError(
            f"CSV video mismatch for ep_idx={ep_idx}: expected basename {expected_video_name!r}, got {videos!r}"
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
        for score_column in SCORE_COLUMNS.values():
            try:
                float(row[score_column])
            except ValueError as exc:
                raise ValueError(f"Invalid score for ep_idx={ep_idx}, column {score_column!r}: {row[score_column]!r}") from exc


def _target_count(pct: int, total: int) -> int:
    return round(total * pct / 100)


def _select_by_rank(rows: list[dict[str, str]], label_key: str, ranks: dict[str, int], pct: int) -> list[int]:
    target_count = _target_count(pct, len(rows))
    ranked_rows = sorted(rows, key=lambda row: (-ranks[row[label_key]], int(row["ep_idx"])))
    return [int(row["ep_idx"]) for row in ranked_rows[:target_count]]


def _select_by_score(rows: list[dict[str, str]], score_column: str, pct: int) -> list[int]:
    target_count = _target_count(pct, len(rows))
    ranked_rows = sorted(rows, key=lambda row: (-float(row[score_column]), int(row["ep_idx"])))
    return [int(row["ep_idx"]) for row in ranked_rows[:target_count]]


def _score_summary(rows: list[dict[str, str]], indices: list[int], score_column: str) -> dict[str, float]:
    values = [float(rows[index][score_column]) for index in indices]
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


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
    for slug, score_column in SCORE_COLUMNS.items():
        for pct in SCORE_PCTS:
            filters[f"{slug}pct{pct}"] = _select_by_score(rows, score_column, pct)

    manifest = {
        "annotations_csv": str(annotations_csv),
        "raw_data_dir": str(raw_data_dir),
        "lerobot_dataset_dir": str(lerobot_dataset_dir),
        "total_episodes": len(rows),
        "selection": {
            "human_label_percentages": list(LABEL_PCTS),
            "score_percentages": list(SCORE_PCTS),
            "human_label_target_counts": {str(pct): _target_count(pct, len(rows)) for pct in LABEL_PCTS},
            "score_target_counts": {str(pct): _target_count(pct, len(rows)) for pct in SCORE_PCTS},
            "tie_break": "ascending ep_idx within the same label or score",
            "observability_rank": ["full", "partial"],
            "optimality_rank": ["better", "okay", "worse"],
            "score_rank": "higher score is better",
            "score_columns": SCORE_COLUMNS,
        },
        "label_counts": {
            "observability": dict(Counter(row["observability"] for row in rows)),
            "optimality": dict(Counter(row["optimality"] for row in rows)),
        },
        "filters": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in filters.items():
        filename = f"pen_in_blue_cup_{name}_episode_indices.json"
        if name.startswith("observability") or name.startswith("optimality"):
            label_key = "observability" if name.startswith("observability") else "optimality"
            label_counts = Counter(rows[index][label_key] for index in indices)
            manifest["filters"][filename] = {
                "count": len(indices),
                "label_counts": dict(label_counts),
                "episode_indices": indices,
            }
        else:
            score_slug = next(slug for slug in SCORE_COLUMNS if name.startswith(slug))
            score_column = SCORE_COLUMNS[score_slug]
            manifest["filters"][filename] = {
                "count": len(indices),
                "score_column": score_column,
                "score_summary": _score_summary(rows, indices, score_column),
                "episode_indices": indices,
            }
        if not check_only:
            _write_json(output_dir / filename, indices)

    if not check_only:
        _write_json(output_dir / "pen_in_blue_cup_human_label_filters_manifest.json", manifest)
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
