"""Build LeRobot episode filters for a DROID pen-in-blue-cup dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ANNOTATIONS_CSV = Path(__file__).with_name("pen_in_cup_0727_labels.csv")
OUTPUT_DIR = Path(__file__).with_name("lerobot_filtering_keys")
RAW_DATA_DIR = Path("/iris/u/tiangao/pen_in_cup_0727")
LEROBOT_DATASET_DIR = Path("/iliad/u/tiangao/lerobot_datasets/skybhh19/droid_pen_in_blue_cup_07272026")

DATASET_NAME = "skybhh19/droid_pen_in_blue_cup_07272026"
FILE_PREFIX = "pen_in_blue_cup_07272026"
LABEL_FILTER_NAME = "observability"
FILTER_PCTS = (25, 50, 75)
RANDOM_SEED = 42


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _infer_raw_data_dir_from_source_videos(rows: list[dict[str, str]]) -> Path | None:
    candidate_dirs = set()
    for row in rows:
        source_video = row.get("source_video")
        if not source_video:
            continue

        source_video_path = Path(source_video)
        if len(source_video_path.parents) < 4:
            continue

        episode_dir = source_video_path.parents[2]
        if episode_dir.name == row.get("episode"):
            candidate_dirs.add(source_video_path.parents[3])

    if len(candidate_dirs) == 1:
        return next(iter(candidate_dirs))
    return None


def _load_raw_episode_paths(raw_data_dir: Path, rows: list[dict[str, str]]) -> tuple[list[Path], Path]:
    paths = sorted(raw_data_dir.glob("*/trajectory.h5"))
    if not paths:
        inferred_raw_data_dir = _infer_raw_data_dir_from_source_videos(rows)
        if inferred_raw_data_dir is not None:
            paths = sorted(inferred_raw_data_dir.glob("*/trajectory.h5"))
            if paths:
                return paths, inferred_raw_data_dir
        raise FileNotFoundError(f"No trajectory.h5 files found under {raw_data_dir}")
    return paths, raw_data_dir


def _load_lerobot_info(lerobot_dataset_dir: Path) -> dict[str, Any]:
    info_path = lerobot_dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def _target_count(pct: int, total: int) -> int:
    return int(total * pct / 100 + 0.5)


def _infer_label_column(rows: list[dict[str, str]], requested_label_column: str | None) -> str:
    if not rows:
        raise ValueError("Annotation CSV is empty.")
    if requested_label_column is not None:
        if requested_label_column not in rows[0]:
            raise ValueError(f"Requested label column {requested_label_column!r} is not in the CSV.")
        return requested_label_column
    for label_column in ("observability_score", "observability", "score"):
        if label_column in rows[0]:
            return label_column
    raise ValueError(
        "Could not infer a human-label column. Pass --label-column; common columns are "
        "'observability_score', 'observability', and 'score'."
    )


def _declared_episode_index(row: dict[str, str], expected_idx: int) -> int:
    if row.get("ep_idx"):
        declared_idx = int(row["ep_idx"])
        if declared_idx != expected_idx:
            raise ValueError(
                f"CSV episode index mismatch for {row.get('episode')!r}: row has ep_idx={declared_idx}, "
                f"raw sorted order gives ep_idx={expected_idx}."
            )
        return declared_idx

    if row.get("index"):
        declared_idx = int(row["index"])
        if declared_idx == expected_idx:
            return declared_idx
        if declared_idx == expected_idx + 1:
            return expected_idx
        raise ValueError(
            f"CSV episode index mismatch for {row.get('episode')!r}: row has index={declared_idx}, "
            f"raw sorted order gives ep_idx={expected_idx}."
        )

    return expected_idx


def _validate_source_video(row: dict[str, str], episode_name: str) -> None:
    source_video = row.get("source_video")
    if not source_video:
        return

    source_video_path = Path(source_video)
    if len(source_video_path.parents) < 3:
        raise ValueError(f"Invalid source_video for {episode_name!r}: {source_video!r}")
    source_episode = source_video_path.parents[2].name
    if source_episode != episode_name:
        raise ValueError(
            f"CSV source_video mismatch for {episode_name!r}: video path points to {source_episode!r}."
        )


def _validate_and_index_rows(
    rows: list[dict[str, str]],
    raw_paths: list[Path],
    lerobot_dataset_dir: Path,
    label_column: str,
) -> list[dict[str, str]]:
    if len(rows) != len(raw_paths):
        raise ValueError(f"Expected {len(raw_paths)} annotation rows, found {len(rows)}.")

    episode_to_index = {path.parent.name: index for index, path in enumerate(raw_paths)}
    indexed_rows: list[dict[str, str]] = []
    seen_indices = set()
    for csv_position, row in enumerate(rows):
        if "episode" not in row:
            raise ValueError("Annotation CSV must contain an 'episode' column.")
        episode_name = row["episode"]
        if episode_name not in episode_to_index:
            raise ValueError(f"CSV row {csv_position} has unknown episode {episode_name!r}.")

        expected_idx = episode_to_index[episode_name]
        ep_idx = _declared_episode_index(row, expected_idx)
        if ep_idx in seen_indices:
            raise ValueError(f"Duplicate episode index in annotations: {ep_idx}")
        seen_indices.add(ep_idx)

        raw_episode_dir = raw_paths[ep_idx].parent
        if raw_episode_dir.name != episode_name:
            raise ValueError(
                f"CSV episode mismatch for ep_idx={ep_idx}: row has {episode_name!r}, "
                f"raw path has {raw_episode_dir.name!r}."
            )
        _validate_source_video(row, episode_name)

        episode_file = lerobot_dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not episode_file.exists():
            raise FileNotFoundError(f"Missing LeRobot episode file for ep_idx={ep_idx}: {episode_file}")

        try:
            float(row[label_column])
        except ValueError as exc:
            raise ValueError(f"Invalid human label for ep_idx={ep_idx}: {row[label_column]!r}") from exc

        row = row.copy()
        row["ep_idx"] = str(ep_idx)
        indexed_rows.append(row)

    expected_indices = set(range(len(raw_paths)))
    if seen_indices != expected_indices:
        missing = sorted(expected_indices - seen_indices)
        raise ValueError(f"Annotation CSV is missing LeRobot episode indices: {missing[:10]}")

    return sorted(indexed_rows, key=lambda row: int(row["ep_idx"]))


def _rank_by_label_with_random_ties(
    rows: list[dict[str, str]], label_column: str, seed: int
) -> list[dict[str, str]]:
    rows_by_score: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_score.setdefault(float(row[label_column]), []).append(row)

    rng = random.Random(seed)
    ranked_rows: list[dict[str, str]] = []
    for score in sorted(rows_by_score, reverse=True):
        score_rows = rows_by_score[score].copy()
        rng.shuffle(score_rows)
        ranked_rows.extend(score_rows)
    return ranked_rows


def _select_by_label(rows: list[dict[str, str]], label_column: str, pct: int, seed: int) -> list[int]:
    target_count = _target_count(pct, len(rows))
    ranked_rows = _rank_by_label_with_random_ties(rows, label_column, seed)
    return [int(row["ep_idx"]) for row in ranked_rows[:target_count]]


def _select_random(total: int, pct: int, seed: int) -> list[int]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[: _target_count(pct, total)]


def _label_summary(rows: list[dict[str, str]], indices: list[int], label_column: str) -> dict[str, float]:
    values = [float(rows[index][label_column]) for index in indices]
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
    file_prefix: str,
    dataset_name: str,
    label_column: str | None,
    label_filter_name: str,
    filter_pcts: tuple[int, ...],
    random_seed: int,
    check_only: bool,
) -> dict[str, object]:
    rows = _load_csv_rows(annotations_csv)
    raw_paths, raw_data_dir = _load_raw_episode_paths(raw_data_dir, rows)
    lerobot_info = _load_lerobot_info(lerobot_dataset_dir)
    total_episodes = int(lerobot_info["total_episodes"])
    if total_episodes != len(raw_paths):
        raise ValueError(
            f"Raw episode count ({len(raw_paths)}) does not match LeRobot total_episodes ({total_episodes})."
        )

    label_column = _infer_label_column(rows, label_column)
    rows = _validate_and_index_rows(rows, raw_paths, lerobot_dataset_dir, label_column)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_counts = {str(pct): _target_count(pct, len(rows)) for pct in filter_pcts}
    label_counts = Counter(row[label_column] for row in rows)

    label_manifest = {
        "dataset": dataset_name,
        "annotations_csv": str(annotations_csv),
        "raw_data_dir": str(raw_data_dir),
        "lerobot_dataset_dir": str(lerobot_dataset_dir),
        "total_episodes": len(rows),
        "label_column": label_column,
        "filter_name": label_filter_name,
        "seed": random_seed,
        "selection": {
            "percentages": list(filter_pcts),
            "target_counts": target_counts,
            "rank": f"higher numeric {label_column} is better",
            "tie_break": "random within the same score using seed",
        },
        "label_counts": {label_column: dict(label_counts)},
        "filters": {},
    }
    random_manifest = {
        "dataset": dataset_name,
        "seed": random_seed,
        "total_episodes": len(rows),
        "selection": {
            "percentages": list(filter_pcts),
            "target_counts": target_counts,
            "policy": (
                "shuffle LeRobot episode indices once per percentage with seed, then take "
                "floor(total_episodes * pct + 0.5) episodes"
            ),
        },
        "filters": {},
    }

    for pct in filter_pcts:
        label_indices = _select_by_label(rows, label_column, pct, random_seed)
        label_filename = f"{file_prefix}_{label_filter_name}pct{pct}_episode_indices.json"
        label_manifest["filters"][label_filename] = {
            "count": len(label_indices),
            "label_counts": dict(Counter(rows[index][label_column] for index in label_indices)),
            "label_summary": _label_summary(rows, label_indices, label_column),
            "episode_indices": label_indices,
        }
        if not check_only:
            _write_json(output_dir / label_filename, label_indices)

        random_indices = _select_random(len(rows), pct, random_seed)
        random_filename = f"{file_prefix}_randompct{pct}_episode_indices.json"
        random_manifest["filters"][random_filename] = {
            "count": len(random_indices),
            "episode_indices": random_indices,
        }
        if not check_only:
            _write_json(output_dir / random_filename, random_indices)

    manifest = {
        label_filter_name: label_manifest,
        "random": random_manifest,
    }
    if not check_only:
        _write_json(output_dir / f"{file_prefix}_{label_filter_name}_filters_manifest.json", label_manifest)
        _write_json(output_dir / f"{file_prefix}_random_filters_manifest.json", random_manifest)
        _write_json(output_dir / f"{file_prefix}_filters_manifest.json", manifest)

    return manifest


def _parse_filter_pcts(raw: str) -> tuple[int, ...]:
    pcts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not pcts:
        raise argparse.ArgumentTypeError("Expected at least one percentage.")
    for pct in pcts:
        if pct <= 0 or pct > 100:
            raise argparse.ArgumentTypeError(f"Filter percentage must be in 1..100, got {pct}.")
    return pcts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, default=ANNOTATIONS_CSV)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--lerobot-dataset-dir", type=Path, default=LEROBOT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--file-prefix", default=FILE_PREFIX)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--label-filter-name", default=LABEL_FILTER_NAME)
    parser.add_argument("--filter-pcts", type=_parse_filter_pcts, default=FILTER_PCTS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = build_filters(
        annotations_csv=args.annotations_csv,
        raw_data_dir=args.raw_data_dir,
        lerobot_dataset_dir=args.lerobot_dataset_dir,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        label_filter_name=args.label_filter_name,
        filter_pcts=args.filter_pcts,
        random_seed=args.random_seed,
        check_only=args.check_only,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
