"""Render the actual selected training frames, exterior and wrist side by side at 15 FPS."""

import io
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def main():
    dataset = Path("/iris/u/tiangao/lerobot_datasets/skybhh19/droid_cup_hanging_09132026_jointpos")
    filters = Path(__file__).parent / "lerobot_filtering_keys"
    audit = json.loads((filters / "cup_hanging_09132026_filters_manifest.json").read_text())
    indices = json.loads((filters / "cup_hanging_09132026_observabilitypct30_episode_indices.json").read_text())
    destination = Path(__file__).resolve().parents[3] / "tmp/cup_hanging_0913_observability30"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "observability30_exterior_wrist.mp4"
    if output.exists():
        raise FileExistsError(output)
    command = [
        "ffmpeg",
        "-nostdin",
        "-n",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x400",
        "-r",
        "15",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    segments = []
    written = 0
    try:
        for index in indices:
            episode = audit["episode_labels"][index]
            columns = ["exterior_image_1_left", "wrist_image_left"]
            rows = pq.read_table(dataset / f"data/chunk-000/episode_{index:06d}.parquet", columns=columns).to_pylist()
            start = written
            for row in rows:
                canvas = np.zeros((400, 1280, 3), dtype=np.uint8)
                for view, key in enumerate(columns):
                    rgb = np.asarray(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"))
                    canvas[40:, view * 640 : (view + 1) * 640] = cv2.resize(rgb, (640, 360))
                title = f"ep={index} score={episode['score']} {episode['episode_name']} | exterior 31078156 / wrist 17471093"
                cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
                written += 1
            segments.append(
                {
                    "episode_index": index,
                    "score": episode["score"],
                    "trajectory_path": episode["trajectory_path"],
                    "start_frame": start,
                    "end_frame_exclusive": written,
                    "start_seconds": start / 15,
                }
            )
            print(f"Rendered episode {index}: {len(rows)} frames", flush=True)
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg exited with {code}")
    (destination / "render_manifest.json").write_text(
        json.dumps(
            {
                "output": str(output),
                "fps": 15,
                "frames": written,
                "episodes": segments,
                "source": "Converted training frames; same observations in both pipelines",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {output}: {written} frames", flush=True)


if __name__ == "__main__":
    main()
