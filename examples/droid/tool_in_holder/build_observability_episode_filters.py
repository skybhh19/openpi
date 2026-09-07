"""Build human-observability LeRobot filters using trajectory names, never CSV ep_idx."""

from collections import Counter
import csv
import json
from pathlib import Path
import random

import tyro


def _target_count(total: int, percentage: int) -> int:
    return int(total * percentage / 100 + 0.5)


def main(
    annotations_csv: str = "/iris/u/tiangao/tool_holder_combined_300_scores.csv",
    *,
    dataset_dir: str = ("/iris/u/tiangao/lerobot_datasets/skybhh19/droid_tool_in_holder_08272026_jointpos"),
    output_dir: str = "examples/droid/tool_in_holder/lerobot_filtering_keys",
    file_prefix: str = "tool_in_holder_08272026_jointpos",
    percentages: tuple[int, ...] = (80, 60, 50, 40),
    seed: int = 42,
) -> None:
    csv_path = Path(annotations_csv).expanduser().resolve()
    dataset_path = Path(dataset_dir).expanduser().resolve()
    destination = Path(output_dir)
    rows = list(csv.DictReader(csv_path.open(newline="")))
    manifest_path = dataset_path / "meta" / "droid_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]

    rows_by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["episode"]
        if name in rows_by_name:
            raise ValueError(f"Duplicate CSV episode name: {name}")
        score = row["observability_score"].strip()
        if score not in {"1", "2", "3"}:
            raise ValueError(f"Episode {name} has invalid observability_score={score!r}")
        rows_by_name[name] = row

    manifest_names = [Path(episode["raw_episode_dir"]).name for episode in episodes]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("Source manifest contains duplicate episode directory names")
    missing = sorted(set(manifest_names) - set(rows_by_name))
    extra = sorted(set(rows_by_name) - set(manifest_names))
    if missing or extra:
        raise ValueError(f"CSV/manifest name mismatch: missing={missing[:10]}, extra={extra[:10]}")

    matched = []
    for episode in episodes:
        name = Path(episode["raw_episode_dir"]).name
        row = rows_by_name[name]
        matched.append(
            {
                "episode_index": int(episode["episode_index"]),
                "episode": name,
                "observability_score": int(row["observability_score"]),
                "trajectory_path": episode["trajectory_path"],
                "csv_ep_idx": int(row["ep_idx"]) if row.get("ep_idx") else None,
            }
        )
    if [item["episode_index"] for item in matched] != list(range(len(matched))):
        raise ValueError("Manifest episode indices are not contiguous and ordered")

    rng = random.Random(seed)
    ranked = []
    for score in (3, 2, 1):
        tied = [item for item in matched if item["observability_score"] == score]
        rng.shuffle(tied)
        ranked.extend(tied)

    destination.mkdir(parents=True, exist_ok=True)
    output_manifest = {
        "format_version": 1,
        "dataset": manifest["repo_id"],
        "annotations_csv": str(csv_path),
        "source_manifest": str(manifest_path),
        "join_key": "CSV episode equals basename(manifest raw_episode_dir); CSV ep_idx is not used",
        "seed": seed,
        "total_episodes": len(matched),
        "label_counts": dict(sorted(Counter(item["observability_score"] for item in matched).items())),
        "filters": {},
        "episode_labels": matched,
    }
    for percentage in percentages:
        if not 0 < percentage <= 100:
            raise ValueError(f"Invalid percentage: {percentage}")
        count = _target_count(len(ranked), percentage)
        selected_items = ranked[:count]
        selected_indices = sorted(item["episode_index"] for item in selected_items)
        filename = f"{file_prefix}_observabilitypct{percentage}_episode_indices.json"
        (destination / filename).write_text(json.dumps(selected_indices, indent=2) + "\n")
        output_manifest["filters"][filename] = {
            "percentage": percentage,
            "count": count,
            "label_counts": dict(sorted(Counter(item["observability_score"] for item in selected_items).items())),
            "episode_indices": selected_indices,
        }
        print(
            f"{percentage}%: {count} episodes, "
            f"labels={output_manifest['filters'][filename]['label_counts']} -> {destination / filename}"
        )

    output_path = destination / f"{file_prefix}_observability_filters_manifest.json"
    output_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(f"Wrote audited name-based mapping to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
