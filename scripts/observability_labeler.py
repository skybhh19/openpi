"""Small local web UI for scoring DROID episode observability."""

from __future__ import annotations

import argparse
from contextlib import suppress
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qs
from urllib.parse import urlparse

import cv2

DEFAULT_DATA_ROOT = Path("/iris/u/tiangao/projects/droid/data/success/2026-06-15")
DEFAULT_CSV = Path("observability_scores_2026-06-15.csv")
FIELDNAMES = [
    "ep_idx",
    "episode",
    "observability_score",
    "updated_at",
    "camera_0_video",
    "camera_1_video",
    "episode_path",
    "total_frames",
]
CHUNK_SIZE = 1024 * 1024
UNKNOWN_VIDEO_INFO = None


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int


UNKNOWN_VIDEO_INFO = VideoInfo(frame_count=0, fps=30.0, width=0, height=0)
THIRD_PERSON_CAMERA_IDS = {"23404442", "31078156"}
WRIST_CAMERA_IDS = {"17471093"}


@dataclass(frozen=True)
class Episode:
    ep_idx: int
    episode: str
    episode_path: str
    videos: tuple[str, ...]
    video_infos: tuple[VideoInfo, ...]
    observability_score: str = ""
    updated_at: str = ""
    source_date: str = ""
    optimality: str = ""
    note: str = ""


def parse_episode_time(path: Path) -> tuple[datetime, str]:
    for date_format in ("%a_%b_%d_%H:%M:%S_%Y", "%a_%b__%d_%H:%M:%S_%Y"):
        try:
            return datetime.strptime(path.name, date_format).replace(tzinfo=UTC), path.name
        except ValueError:
            pass
    return datetime.max.replace(tzinfo=UTC), path.name


def camera_sort_key(path: Path) -> tuple[int, str]:
    camera_id = path.stem
    if camera_id in THIRD_PERSON_CAMERA_IDS:
        return 0, path.name
    if camera_id in WRIST_CAMERA_IDS:
        return 2, path.name
    return 1, path.name


def discover_episodes(data_root: Path, saved_rows: dict[str, dict[str, str]]) -> list[Episode]:
    episode_dirs = [path for path in data_root.iterdir() if path.is_dir()]
    episodes: list[Episode] = []
    for ep_idx, episode_dir in enumerate(sorted(episode_dirs, key=parse_episode_time)):
        video_paths = tuple(sorted((episode_dir / "recordings" / "MP4").glob("*.mp4"), key=camera_sort_key))
        videos = tuple(str(path) for path in video_paths)
        saved = saved_rows.get(episode_dir.name, {})
        video_infos = saved_video_infos(saved, len(video_paths))
        episodes.append(
            Episode(
                ep_idx=ep_idx,
                episode=episode_dir.name,
                episode_path=str(episode_dir),
                videos=videos,
                video_infos=video_infos,
                observability_score=saved.get("observability_score", ""),
                updated_at=saved.get("updated_at", ""),
            )
        )
    return episodes


def discover_annotation_episodes(data_root: Path, rows: list[dict[str, str]]) -> list[Episode]:
    episodes: list[Episode] = []
    for row_idx, row in enumerate(rows):
        source_date = row.get("source_date", "")
        episode_dir = data_root / source_date / row["episode"] if source_date else data_root / row["episode"]
        video_paths = tuple(sorted((episode_dir / "recordings" / "MP4").glob("*.mp4"), key=camera_sort_key))
        videos = tuple(str(path) for path in video_paths)
        video_infos = saved_video_infos(row, len(video_paths))
        episodes.append(
            Episode(
                ep_idx=row_idx,
                episode=row["episode"],
                episode_path=str(episode_dir),
                videos=videos,
                video_infos=video_infos,
                observability_score=row.get("observability", ""),
                source_date=source_date,
                optimality=row.get("optimality", ""),
                note=row.get("note", ""),
            )
        )
    return episodes


def filter_episodes(episodes: list[Episode], episode_name_regex: str | None) -> list[Episode]:
    if not episode_name_regex:
        return episodes

    pattern = re.compile(episode_name_regex)
    filtered = [episode for episode in episodes if pattern.search(episode.episode)]
    return [
        Episode(
            ep_idx=ep_idx,
            episode=episode.episode,
            episode_path=episode.episode_path,
            videos=episode.videos,
            video_infos=episode.video_infos,
            observability_score=episode.observability_score,
            updated_at=episode.updated_at,
            source_date=episode.source_date,
            optimality=episode.optimality,
            note=episode.note,
        )
        for ep_idx, episode in enumerate(filtered)
    ]


def saved_video_infos(row: dict[str, str], video_count: int) -> tuple[VideoInfo, ...]:
    frame_count_text = row.get("total_frames") or row.get("frame_count") or ""
    try:
        frame_count = int(float(frame_count_text))
    except ValueError:
        frame_count = 0
    if frame_count <= 0:
        return tuple(UNKNOWN_VIDEO_INFO for _ in range(video_count))

    fps_text = row.get("fps") or ""
    try:
        fps = float(fps_text)
    except ValueError:
        fps = 60.0
    return tuple(VideoInfo(frame_count=frame_count, fps=fps, width=1280, height=720) for _ in range(video_count))


def hydrate_video_infos(episodes: list[Episode]) -> list[Episode]:
    return [
        Episode(
            ep_idx=episode.ep_idx,
            episode=episode.episode,
            episode_path=episode.episode_path,
            videos=episode.videos,
            video_infos=tuple(
                info if info.frame_count > 0 else read_video_info(Path(video_path))
                for info, video_path in zip(episode.video_infos, episode.videos, strict=False)
            ),
            observability_score=episode.observability_score,
            updated_at=episode.updated_at,
            source_date=episode.source_date,
            optimality=episode.optimality,
            note=episode.note,
        )
        for episode in episodes
    ]


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return VideoInfo(
            frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            width=max(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=max(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    finally:
        cap.release()


def read_saved_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}

    with csv_path.open(newline="") as csv_file:
        return {row["episode"]: row for row in csv.DictReader(csv_file) if row.get("episode")}


def row_for_episode(episode: Episode) -> dict[str, str]:
    videos = list(episode.videos)
    frame_counts = [info.frame_count for info in episode.video_infos if info.frame_count > 0]
    return {
        "ep_idx": str(episode.ep_idx),
        "episode": episode.episode,
        "observability_score": episode.observability_score,
        "updated_at": episode.updated_at,
        "camera_0_video": videos[0] if len(videos) > 0 else "",
        "camera_1_video": videos[1] if len(videos) > 1 else "",
        "episode_path": episode.episode_path,
        "total_frames": str(min(frame_counts)) if frame_counts else "",
    }


def write_rows(csv_path: Path, episodes: list[Episode]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=csv_path.parent, newline="") as tmp_file:
        writer = csv.DictWriter(tmp_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for episode in episodes:
            writer.writerow(row_for_episode(episode))
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(csv_path)


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2).encode()


class LabelerApp:
    def __init__(
        self,
        data_root: Path,
        csv_path: Path,
        *,
        annotation_csv: bool = False,
        episode_name_regex: str | None = None,
    ) -> None:
        self.data_root = data_root.resolve()
        self.csv_path = csv_path.resolve()
        self.annotation_csv = annotation_csv
        self.annotation_rows: list[dict[str, str]] = []
        self.annotation_fieldnames: list[str] = []
        if annotation_csv:
            with self.csv_path.open(newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                self.annotation_fieldnames = list(reader.fieldnames or [])
                self.annotation_rows = list(reader)
            self.episodes = hydrate_video_infos(filter_episodes(
                discover_annotation_episodes(self.data_root, self.annotation_rows),
                episode_name_regex,
            ))
        else:
            self.episodes = hydrate_video_infos(filter_episodes(
                discover_episodes(self.data_root, read_saved_rows(self.csv_path)),
                episode_name_regex,
            ))
            write_rows(self.csv_path, self.episodes)

    def save_score(self, ep_idx: int, score: str) -> Episode:
        if score not in {"1", "2", "3"}:
            raise ValueError("score must be 1, 2, or 3")
        if ep_idx < 0 or ep_idx >= len(self.episodes):
            raise IndexError("episode not found")
        episode = self.episodes[ep_idx]
        if len(episode.videos) < 2:
            raise ValueError("episode does not have two MP4 videos")

        updated = Episode(
            ep_idx=episode.ep_idx,
            episode=episode.episode,
            episode_path=episode.episode_path,
            videos=episode.videos,
            video_infos=episode.video_infos,
            observability_score=score,
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            source_date=episode.source_date,
            optimality=episode.optimality,
            note=episode.note,
        )
        self.episodes[ep_idx] = updated
        if self.annotation_csv:
            self.annotation_rows[ep_idx]["observability"] = score
            self.write_annotation_rows()
        else:
            write_rows(self.csv_path, self.episodes)
        return updated

    def write_annotation_rows(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=self.csv_path.parent, newline="") as tmp_file:
            writer = csv.DictWriter(tmp_file, fieldnames=self.annotation_fieldnames)
            writer.writeheader()
            writer.writerows(self.annotation_rows)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(self.csv_path)

    def payload(self) -> dict[str, object]:
        scored = sum(1 for episode in self.episodes if episode.observability_score)
        return {
            "csv_path": str(self.csv_path),
            "data_root": str(self.data_root),
            "annotation_csv": self.annotation_csv,
            "episode_count": len(self.episodes),
            "scored_count": scored,
            "episodes": [self.episode_payload(episode) for episode in self.episodes],
        }

    def episode_payload(self, episode: Episode) -> dict[str, object]:
        payload = asdict(episode)
        payload["video_urls"] = [
            f"/video?episode={episode.ep_idx}&camera={camera_idx}" for camera_idx in range(len(episode.videos))
        ]
        payload["has_two_videos"] = len(episode.videos) == 2
        return payload


class LabelerHandler(BaseHTTPRequestHandler):
    server: LabelerServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTTPStatus.OK, INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/api/episodes":
            self.send_json(HTTPStatus.OK, self.server.app.payload())
        elif parsed.path == "/video":
            self.serve_video(parse_qs(parsed.query))
        elif parsed.path == "/frame":
            self.serve_frame(parse_qs(parsed.query))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/score":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            episode = self.server.app.save_score(int(payload["episode"]), str(payload["score"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except IndexError as exc:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return

        self.send_json(
            HTTPStatus.OK,
            {
                "episode": self.server.app.episode_payload(episode),
                "csv_path": str(self.server.app.csv_path),
                "scored_count": sum(1 for item in self.server.app.episodes if item.observability_score),
            },
        )

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/video":
            self.serve_video(parse_qs(parsed.query), head_only=True)
        elif parsed.path == "/frame":
            self.serve_frame(parse_qs(parsed.query), head_only=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def serve_video(self, query: dict[str, list[str]], *, head_only: bool = False) -> None:
        try:
            video_path = self.resolve_video_path(query)
        except (IndexError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if not video_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        size = video_path.stat().st_size
        start, end, status = self.parse_range(size)
        if start >= size or end >= size or start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if head_only:
            return

        with video_path.open("rb") as video_file:
            video_file.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = video_file.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def serve_frame(self, query: dict[str, list[str]], *, head_only: bool = False) -> None:
        try:
            video_path = self.resolve_video_path(query)
            frame_idx = max(0, int(query.get("frame", ["0"])[0]))
        except (IndexError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if frame_count:
                frame_idx = min(frame_idx, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            image_bytes = encoded.tobytes()
        finally:
            cap.release()

        self.send_bytes(
            HTTPStatus.OK,
            b"" if head_only else image_bytes,
            "image/jpeg",
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Content-Length": str(len(image_bytes)),
            },
        )

    def resolve_video_path(self, query: dict[str, list[str]]) -> Path:
        ep_idx = int(query.get("episode", [""])[0])
        camera_idx = int(query.get("camera", [""])[0])
        episode = self.server.app.episodes[ep_idx]
        video_path = Path(episode.videos[camera_idx])
        return video_path

    def parse_range(self, size: int) -> tuple[int, int, HTTPStatus]:
        range_header = self.headers.get("Range")
        if not range_header:
            return 0, size - 1, HTTPStatus.OK

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            return 0, size - 1, HTTPStatus.OK

        start_text, end_text = match.groups()
        if not start_text and end_text:
            suffix = int(end_text)
            return max(size - suffix, 0), size - 1, HTTPStatus.PARTIAL_CONTENT

        start = int(start_text or 0)
        end = int(end_text) if end_text else size - 1
        return start, min(end, size - 1), HTTPStatus.PARTIAL_CONTENT

    def send_json(self, status: HTTPStatus, payload: object) -> None:
        self.send_bytes(status, json_bytes(payload), "application/json")

    def send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if not headers or "Cache-Control" not in headers:
            self.send_header("Cache-Control", "no-store, max-age=0")
        if not headers or "Content-Length" not in headers:
            self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")


class LabelerServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], app: LabelerApp) -> None:
        super().__init__(server_address, LabelerHandler)
        self.app = app


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Observability Scoring</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #171a1f;
      --muted: #667085;
      --line: #d7dce3;
      --accent: #0f766e;
      --accent-ink: #ffffff;
      --warn: #a15c07;
      --missing: #f9efe1;
      --shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }

    .shell {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      border-right: 1px solid var(--line);
      background: #fbfbfc;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }

    .side-head {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .title {
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 750;
      letter-spacing: 0;
    }

    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }

    .episodes {
      overflow: auto;
      padding: 8px;
    }

    .episode-button {
      width: 100%;
      display: grid;
      grid-template-columns: 34px 1fr auto;
      gap: 8px;
      align-items: center;
      margin: 0 0 6px;
      padding: 9px 10px;
      text-align: left;
      border-radius: 8px;
    }

    .episode-button.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }

    .episode-button.missing {
      background: var(--missing);
    }

    .idx {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    .name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }

    .pill {
      min-width: 28px;
      border-radius: 999px;
      background: #e8f3f1;
      color: #0b5f58;
      font-size: 12px;
      font-weight: 750;
      text-align: center;
      padding: 2px 7px;
    }

    .pill.empty {
      background: #e7e9ee;
      color: #77808f;
    }

    .pill.missing {
      background: #f6dcb8;
      color: var(--warn);
    }

    .main {
      min-width: 0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }

    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      min-height: 74px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }

    .episode-title {
      margin: 0 0 4px;
      font-size: 20px;
      font-weight: 760;
      overflow-wrap: anywhere;
      letter-spacing: 0;
    }

    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .nav-button {
      min-width: 74px;
      height: 38px;
      padding: 0 12px;
    }

    .score-group {
      display: grid;
      grid-template-columns: repeat(3, 54px);
      gap: 8px;
      margin-left: 6px;
    }

    .score-button {
      height: 42px;
      font-size: 18px;
      font-weight: 800;
    }

    .score-button.selected,
    .score-button:hover:not(:disabled) {
      background: var(--accent);
      border-color: var(--accent);
      color: var(--accent-ink);
    }

    .content {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 18px 20px 20px;
      min-height: 0;
    }

    .videos {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      min-height: 0;
    }

    .video-panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .camera-label {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .frame-img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #111318;
      object-fit: contain;
    }

    .player-controls {
      display: grid;
      grid-template-columns: 90px minmax(0, 1fr) 110px;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      box-shadow: var(--shadow);
    }

    .player-button {
      height: 38px;
      font-weight: 720;
    }

    .frame-slider {
      width: 100%;
      min-width: 0;
      accent-color: var(--accent);
    }

    .frame-label {
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }

    .missing-panel {
      display: grid;
      place-items: center;
      min-height: 52vh;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--warn);
      font-weight: 700;
      box-shadow: var(--shadow);
    }

    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    @media (max-width: 940px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .sidebar {
        min-height: 0;
        max-height: 220px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .episodes {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
      }

      .episode-button {
        margin: 0;
      }

      .topbar {
        grid-template-columns: 1fr;
      }

      .toolbar {
        justify-content: flex-start;
      }

      .videos {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="side-head">
        <h1 class="title">Observability Scoring</h1>
        <div class="meta" id="progress">Loading...</div>
        <div class="meta" id="csvPath"></div>
      </div>
      <div class="episodes" id="episodeList"></div>
    </aside>
    <main class="main">
      <header class="topbar">
        <div>
          <h2 class="episode-title" id="episodeTitle">Loading...</h2>
          <div class="meta" id="episodeMeta"></div>
        </div>
        <div class="toolbar">
          <button class="nav-button" id="prevButton" type="button">Prev</button>
          <button class="nav-button" id="nextButton" type="button">Next</button>
          <div class="score-group" aria-label="Observability score">
            <button class="score-button" data-score="1" type="button">1</button>
            <button class="score-button" data-score="2" type="button">2</button>
            <button class="score-button" data-score="3" type="button">3</button>
          </div>
        </div>
      </header>
      <section class="content">
        <div class="videos" id="videos"></div>
        <div class="player-controls">
          <button class="player-button" id="playButton" type="button">Play</button>
          <input class="frame-slider" id="frameSlider" type="range" min="0" max="0" value="0" step="1" />
          <div class="frame-label" id="frameLabel">0 / 0</div>
        </div>
        <div class="status" id="status"></div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      episodes: [],
      index: 0,
      csvPath: "",
      scoredCount: 0,
      saving: false,
      playing: false,
      frame: 0,
      timer: null,
      renderedIndex: null,
      cacheToken: String(Date.now()),
    };

    const episodeList = document.getElementById("episodeList");
    const episodeTitle = document.getElementById("episodeTitle");
    const episodeMeta = document.getElementById("episodeMeta");
    const progress = document.getElementById("progress");
    const csvPath = document.getElementById("csvPath");
    const videos = document.getElementById("videos");
    const status = document.getElementById("status");
    const prevButton = document.getElementById("prevButton");
    const nextButton = document.getElementById("nextButton");
    const playButton = document.getElementById("playButton");
    const frameSlider = document.getElementById("frameSlider");
    const frameLabel = document.getElementById("frameLabel");
    const scoreButtons = [...document.querySelectorAll(".score-button")];
    const DISPLAY_FPS = 15;
    const PLAYBACK_RATE = 1.0;

    async function loadEpisodes() {
      const response = await fetch("/api/episodes");
      const payload = await response.json();
      state.episodes = payload.episodes;
      state.csvPath = payload.csv_path;
      state.scoredCount = payload.scored_count;
      render();
    }

    function currentEpisode() {
      return state.episodes[state.index];
    }

    function render() {
      renderSidebar();
      renderEpisode();
      renderProgress();
    }

    function renderProgress() {
      progress.textContent = `${state.scoredCount} / ${state.episodes.length} scored`;
      csvPath.textContent = state.csvPath;
    }

    function renderSidebar() {
      episodeList.replaceChildren();
      state.episodes.forEach((episode, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "episode-button";
        if (index === state.index) button.classList.add("active");
        if (!episode.has_two_videos) button.classList.add("missing");
        button.addEventListener("click", () => {
          state.index = index;
          render();
        });

        const idx = document.createElement("span");
        idx.className = "idx";
        idx.textContent = String(episode.ep_idx + 1).padStart(3, "0");

        const name = document.createElement("span");
        name.className = "name";
        name.textContent = episode.episode;

        const pill = document.createElement("span");
        pill.className = "pill";
        if (episode.observability_score) {
          pill.textContent = episode.observability_score;
        } else if (!episode.has_two_videos) {
          pill.classList.add("missing");
          pill.textContent = "MP4";
        } else {
          pill.classList.add("empty");
          pill.textContent = "-";
        }

        button.append(idx, name, pill);
        episodeList.append(button);
      });
    }

    function renderEpisode() {
      const episode = currentEpisode();
      if (!episode) return;
      if (state.renderedIndex !== state.index) {
        stopPlayback();
        state.frame = 0;
        state.renderedIndex = state.index;
      }

      episodeTitle.textContent = episode.episode;
      const metaBits = [`${episode.ep_idx + 1} of ${state.episodes.length}`];
      if (episode.source_date) metaBits.push(episode.source_date);
      if (episode.optimality) metaBits.push(`optimality ${episode.optimality}`);
      episodeMeta.textContent = metaBits.join(" - ");
      prevButton.disabled = state.index === 0;
      nextButton.disabled = state.index === state.episodes.length - 1;
      playButton.disabled = !episode.has_two_videos;
      playButton.textContent = state.playing ? "Pause" : "Play";
      frameSlider.disabled = !episode.has_two_videos;
      scoreButtons.forEach((button) => {
        const selected = button.dataset.score === episode.observability_score;
        button.classList.toggle("selected", selected);
        button.disabled = !episode.has_two_videos || state.saving;
      });

      videos.replaceChildren();
      if (!episode.has_two_videos) {
        const panel = document.createElement("div");
        panel.className = "missing-panel";
        panel.textContent = "Missing MP4 videos";
        videos.append(panel);
        status.textContent = episode.episode_path;
        frameSlider.max = 0;
        frameSlider.value = 0;
        frameLabel.textContent = "0 / 0";
        return;
      }

      const maxFrame = getMaxFrame(episode);
      state.frame = Math.min(state.frame, maxFrame);
      frameSlider.max = maxFrame;
      frameSlider.value = state.frame;
      frameLabel.textContent = `${state.frame + 1} / ${maxFrame + 1}`;

      episode.videos.slice(0, 2).forEach((path, cameraIndex) => {
        const panel = document.createElement("article");
        panel.className = "video-panel";

        const label = document.createElement("div");
        label.className = "camera-label";
        label.innerHTML = `<span>${cameraLabel(path, cameraIndex)}</span><span>${path.split("/").pop()}</span>`;

        const image = document.createElement("img");
        image.className = "frame-img";
        image.alt = `Camera ${cameraIndex + 1}`;
        image.src = frameUrl(episode, cameraIndex, state.frame);

        panel.append(label, image);
        videos.append(panel);
      });
      status.textContent = episode.updated_at ? `Saved ${episode.updated_at}` : "";
    }

    function getMaxFrame(episode) {
      const counts = episode.video_infos.slice(0, 2).map((info) => info.frame_count).filter((count) => count > 0);
      if (!counts.length) return 0;
      return Math.max(0, Math.min(...counts) - 1);
    }

    function cameraLabel(path, cameraIndex) {
      const fileName = path.split("/").pop() || "";
      const cameraId = fileName.split(".")[0];
      if (cameraId === "23404442" || cameraId === "31078156") return "Third-person";
      if (cameraId === "17471093") return "Wrist";
      return `Camera ${cameraIndex + 1}`;
    }

    function getFrameStep(episode) {
      const fps = episode.video_infos[0]?.fps || 30;
      return Math.max(1, Math.round((fps * PLAYBACK_RATE) / DISPLAY_FPS));
    }

    function frameUrl(episode, cameraIndex, frame) {
      return `/frame?episode=${episode.ep_idx}&camera=${cameraIndex}&frame=${frame}&v=${state.cacheToken}`;
    }

    function updateFrames() {
      const episode = currentEpisode();
      if (!episode || !episode.has_two_videos) return;
      const maxFrame = getMaxFrame(episode);
      state.frame = Math.min(state.frame, maxFrame);
      frameSlider.value = state.frame;
      frameLabel.textContent = `${state.frame + 1} / ${maxFrame + 1}`;
      [...document.querySelectorAll(".frame-img")].forEach((image, cameraIndex) => {
        image.src = frameUrl(episode, cameraIndex, state.frame);
      });
    }

    function startPlayback() {
      const episode = currentEpisode();
      if (!episode || !episode.has_two_videos || state.playing) return;
      state.playing = true;
      playButton.textContent = "Pause";
      const step = getFrameStep(episode);
      state.timer = window.setInterval(() => {
        const maxFrame = getMaxFrame(currentEpisode());
        if (state.frame >= maxFrame) {
          stopPlayback();
          return;
        }
        state.frame = Math.min(maxFrame, state.frame + step);
        updateFrames();
      }, 1000 / DISPLAY_FPS);
    }

    function stopPlayback() {
      state.playing = false;
      if (state.timer) {
        window.clearInterval(state.timer);
        state.timer = null;
      }
      if (playButton) playButton.textContent = "Play";
    }

    async function saveScore(score) {
      const episode = currentEpisode();
      if (!episode || !episode.has_two_videos || state.saving) return;

      state.saving = true;
      renderEpisode();
      status.textContent = "Saving...";

      const response = await fetch("/api/score", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({episode: episode.ep_idx, score}),
      });
      const payload = await response.json();
      state.saving = false;

      if (!response.ok) {
        status.textContent = payload.error || "Save failed";
        renderEpisode();
        return;
      }

      const wasUnscored = !state.episodes[state.index].observability_score;
      state.episodes[state.index] = payload.episode;
      state.scoredCount = payload.scored_count;
      status.textContent = `Saved ${payload.episode.updated_at}`;
      render();
      if (wasUnscored && state.index < state.episodes.length - 1) {
        state.index += 1;
        render();
      }
    }

    prevButton.addEventListener("click", () => {
      state.index = Math.max(0, state.index - 1);
      render();
    });

    nextButton.addEventListener("click", () => {
      state.index = Math.min(state.episodes.length - 1, state.index + 1);
      render();
    });

    scoreButtons.forEach((button) => {
      button.addEventListener("click", () => saveScore(button.dataset.score));
    });

    playButton.addEventListener("click", () => {
      if (state.playing) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    frameSlider.addEventListener("input", () => {
      stopPlayback();
      state.frame = Number(frameSlider.value);
      updateFrames();
    });

    window.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      if (["1", "2", "3"].includes(event.key)) saveScore(event.key);
      if (event.key === "ArrowLeft") prevButton.click();
      if (event.key === "ArrowRight") nextButton.click();
      if (event.key === " ") {
        event.preventDefault();
        playButton.click();
      }
    });

    loadEpisodes().catch((error) => {
      episodeTitle.textContent = "Load failed";
      status.textContent = String(error);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--annotation-csv", type=Path)
    parser.add_argument("--episode-name-regex")
    parser.add_argument("--base-data-root", type=Path, default=Path("/iris/u/tiangao/projects/droid/data/success"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.annotation_csv or args.csv
    data_root = args.base_data_root if args.annotation_csv else args.data_root
    app = LabelerApp(
        data_root,
        csv_path,
        annotation_csv=args.annotation_csv is not None,
        episode_name_regex=args.episode_name_regex,
    )
    server = LabelerServer((args.host, args.port), app)
    print(f"Found {len(app.episodes)} episodes")
    print(f"Writing scores to {app.csv_path}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
