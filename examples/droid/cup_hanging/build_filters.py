"""Join labels by raw episode name and build identical filters for both action spaces."""

from collections import Counter
import csv
import json
from pathlib import Path
import random

ROOT = Path("/iris/u/tiangao/lerobot_datasets/skybhh19")
CSV_PATH = Path("/iliad2/u/tiangao/projects/openpi/observability_labeler_scores_cup-hang-2026-09-13.csv")
OUTPUT = Path(__file__).parent / "lerobot_filtering_keys"


def main():
    manifests = [
        json.loads((ROOT / f"droid_cup_hanging_09132026_{space}/meta/droid_source_manifest.json").read_text())
        for space in ("jointpos", "jointvel")
    ]
    if manifests[0]["episodes"] != manifests[1]["episodes"]:
        raise ValueError("The two action datasets have different episode/frame mappings")
    episodes = manifests[0]["episodes"]
    with CSV_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_name = {row["episode"].strip(): row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != {Path(e["raw_episode_dir"]).name for e in episodes}:
        raise ValueError("Labels must match source episode names uniquely and exactly")
    matched = []
    for episode in episodes:
        name = Path(episode["raw_episode_dir"]).name
        score = int(by_name[name]["observability_score"])
        if score not in (1, 2):
            raise ValueError(f"Invalid label {score}: {name}")
        matched.append({**episode, "score": score, "episode_name": name})
    order = list(range(len(episodes)))
    random.Random(42).shuffle(order)
    rng = random.Random(42)
    ranked = []
    for score in (2, 1):
        tied = [e["episode_index"] for e in matched if e["score"] == score]
        rng.shuffle(tied)
        ranked.extend(tied)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {
        "datasets": [m["repo_id"] for m in manifests],
        "annotations_csv": str(CSV_PATH),
        "join": "CSV episode == basename(raw_episode_dir); CSV ep_idx is ignored",
        "seed": 42,
        "label_counts": dict(Counter(e["score"] for e in matched)),
        "episode_labels": matched,
        "filters": {},
    }
    for kind, pct in (("random", 60), ("observability", 60), ("observability", 30)):
        count = int(len(episodes) * pct / 100 + 0.5)
        indices = sorted((order if kind == "random" else ranked)[:count])
        name = f"cup_hanging_09132026_{kind}pct{pct}_episode_indices.json"
        (OUTPUT / name).write_text(json.dumps(indices, indent=2) + "\n")
        result["filters"][name] = {
            "episode_indices": indices,
            "count": count,
            "label_counts": dict(Counter(matched[i]["score"] for i in indices)),
        }
        print(name, result["filters"][name], flush=True)
    (OUTPUT / "cup_hanging_09132026_filters_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
