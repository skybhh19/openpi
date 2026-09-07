"""Build deterministic, nested random LeRobot episode subsets."""

import json
from pathlib import Path
import random

import tyro


def main(
    dataset_dir: str,
    *,
    output_dir: str = "examples/droid/tool_in_holder/lerobot_filtering_keys",
    file_prefix: str = "tool_in_holder_08272026_jointpos",
    percentages: tuple[int, ...] = (80, 60, 40),
    seed: int = 42,
) -> None:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    manifest = json.loads((dataset_path / "meta" / "droid_source_manifest.json").read_text())
    total_episodes = int(info["total_episodes"])
    if manifest["total_episodes"] != total_episodes:
        raise ValueError("LeRobot metadata and source manifest disagree on episode count")
    expected_indices = list(range(total_episodes))
    if [episode["episode_index"] for episode in manifest["episodes"]] != expected_indices:
        raise ValueError("Source manifest episode indices are not contiguous and ordered")

    shuffled = expected_indices.copy()
    random.Random(seed).shuffle(shuffled)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "total_episodes": total_episodes, "subsets": {}}
    for percentage in sorted(set(percentages), reverse=True):
        if not 0 < percentage <= 100:
            raise ValueError(f"Invalid percentage: {percentage}")
        count = round(total_episodes * percentage / 100)
        selected = sorted(shuffled[:count])
        filename = f"{file_prefix}_randompct{percentage}_episode_indices.json"
        (destination / filename).write_text(json.dumps(selected, indent=2) + "\n")
        summary["subsets"][str(percentage)] = {"episodes": count, "filename": filename}
        print(f"{percentage}%: {count} episodes -> {destination / filename}")
    (destination / f"{file_prefix}_random_split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)
