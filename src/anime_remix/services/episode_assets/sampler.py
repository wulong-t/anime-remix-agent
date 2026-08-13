"""FFmpeg-based frame sampling for episode reference extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.errors import InputValidationError, MediaProbeError

_PTS_PATTERN = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_MAX_SAMPLE_WIDTH = 1280


@dataclass(frozen=True)
class SampledFrame:
    """One extracted frame with its source timestamp and fingerprint."""

    timestamp_seconds: float
    path: Path
    sha256: str
    width: int
    height: int


def _scene_timestamps(
    toolkit: FFmpegToolkit,
    video: Path,
    *,
    threshold: float,
    timeout: int,
) -> list[float]:
    args = [
        toolkit.ffmpeg or "ffmpeg",
        "-v",
        "info",
        "-i",
        str(video),
        "-vf",
        f"select='gt(scene,{threshold:.3f})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    completed = toolkit.run(
        args,
        timeout=timeout,
        error_cls=MediaProbeError,
        stage="scene_scan",
    )
    return [float(match) for match in _PTS_PATTERN.findall(completed.stderr)]


def _interval_timestamps(duration: float, max_frames: int) -> list[float]:
    if duration <= 0 or max_frames <= 0:
        return []
    step = duration / max_frames
    return [round(step * (index + 0.5), 3) for index in range(max_frames)]


def _apply_min_gap(timestamps: list[float], min_gap_seconds: float) -> list[float]:
    result: list[float] = []
    last = float("-inf")
    for timestamp in sorted(timestamps):
        if timestamp - last >= min_gap_seconds:
            result.append(timestamp)
            last = timestamp
    return result


def _spread_timestamps(timestamps: list[float], max_frames: int) -> list[float]:
    """Pick up to ``max_frames`` timestamps evenly across the full list."""

    if not timestamps or max_frames < 1:
        return []
    if len(timestamps) <= max_frames:
        return list(timestamps)
    step = (len(timestamps) - 1) / max(max_frames - 1, 1)
    return [timestamps[round(index * step)] for index in range(max_frames)]


def _extract_frame(
    toolkit: FFmpegToolkit,
    video: Path,
    out_dir: Path,
    *,
    timestamp: float,
    index: int,
    timeout: int,
) -> SampledFrame:
    out_path = out_dir / f"frame_{index:04d}.png"
    candidates = [timestamp, max(0.0, round(timestamp - 0.05, 3))]
    for candidate in candidates:
        args = [
            toolkit.ffmpeg or "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{candidate:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({_MAX_SAMPLE_WIDTH},iw)':-2",
            "-an",
            "-y",
            str(out_path),
        ]
        toolkit.run(
            args,
            timeout=timeout,
            error_cls=MediaProbeError,
            stage=f"frame_extract_{index}",
        )
        if out_path.is_file():
            break
    if not out_path.is_file():
        raise MediaProbeError(
            f"frame extraction produced no file at {out_path}",
            actual=str(out_path),
        )
    try:
        with Image.open(out_path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise MediaProbeError(
            f"extracted frame is not a readable image: {out_path}",
            actual=str(exc),
        ) from exc
    return SampledFrame(
        timestamp_seconds=round(timestamp, 3),
        path=out_path,
        sha256=hashlib.sha256(out_path.read_bytes()).hexdigest(),
        width=width,
        height=height,
    )


def sample_frames(
    video: Path,
    out_dir: Path,
    *,
    max_frames: int = 40,
    scene_threshold: float = 0.30,
    min_gap_seconds: float = 1.0,
    toolkit: FFmpegToolkit | None = None,
    timeout: int = 300,
) -> list[SampledFrame]:
    """Sample distinct frames from one video using scene-cut detection."""

    if not video.is_file():
        raise InputValidationError(f"video file does not exist: {video}")
    toolkit = toolkit or FFmpegToolkit()
    out_dir.mkdir(parents=True, exist_ok=True)
    _duration, timestamps = plan_timestamps(
        video,
        max_frames=max_frames,
        scene_threshold=scene_threshold,
        min_gap_seconds=min_gap_seconds,
        toolkit=toolkit,
        timeout=timeout,
    )
    return [
        _extract_frame(
            toolkit,
            video,
            out_dir,
            timestamp=timestamp,
            index=index,
            timeout=timeout,
        )
        for index, timestamp in enumerate(timestamps, start=1)
    ]


def plan_timestamps(
    video: Path,
    *,
    max_frames: int = 40,
    scene_threshold: float = 0.30,
    min_gap_seconds: float = 1.0,
    toolkit: FFmpegToolkit | None = None,
    timeout: int = 300,
) -> tuple[float, list[float]]:
    """Return (duration, selected timestamps) without writing any frame file."""

    if not video.is_file():
        raise InputValidationError(f"video file does not exist: {video}")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise InputValidationError("max_frames must be a positive integer")
    if isinstance(scene_threshold, bool) or not isinstance(scene_threshold, (int, float)):
        raise InputValidationError("scene_threshold must be a number")
    if not 0.0 <= float(scene_threshold) <= 1.0:
        raise InputValidationError("scene_threshold must be within 0..1")
    if isinstance(min_gap_seconds, bool) or not isinstance(min_gap_seconds, (int, float)):
        raise InputValidationError("min_gap_seconds must be a number")
    if min_gap_seconds < 0:
        raise InputValidationError("min_gap_seconds must be non-negative")
    toolkit = toolkit or FFmpegToolkit()
    duration = toolkit.probe_duration(video)
    scene_ts = _apply_min_gap(
        _scene_timestamps(toolkit, video, threshold=float(scene_threshold), timeout=timeout),
        float(min_gap_seconds),
    )
    fallback_floor = min(4, max_frames)
    if len(scene_ts) >= fallback_floor:
        timestamps = _spread_timestamps(scene_ts, max_frames)
    else:
        timestamps = _interval_timestamps(duration, max_frames)
    clamped: list[float] = []
    for timestamp in timestamps:
        value = min(timestamp, max(0.0, duration - 0.1))
        if not clamped or abs(value - clamped[-1]) >= 0.05:
            clamped.append(round(value, 3))
    return duration, clamped
