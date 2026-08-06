"""Generate copyright-free synthetic demo clips (24 fps, BT.709, video-only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        sys.exit("ffmpeg not found on PATH")
    return binary


def _encode(
    ffmpeg: str,
    source: str,
    output: Path,
    *,
    duration: int,
) -> None:
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"{source}=size=1280x720:rate=24:duration={duration}",
        "-vf",
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-bf",
        "0",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-video_track_timescale",
        "48000",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-chroma_sample_location",
        "left",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "mp4",
        str(output),
    ]
    subprocess.run(args, check=True, shell=False)


def main() -> None:
    demo_dir = Path(__file__).resolve().parent
    clips_dir = demo_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    specs = [
        ("clip_001", "testsrc2", 4),
        ("clip_002", "smptebars", 5),
        ("clip_003", "testsrc", 3),
        ("clip_004", "rgbtestsrc", 6),
    ]
    for clip_id, source, duration in specs:
        _encode(ffmpeg, source, clips_dir / f"{clip_id}.mp4", duration=duration)
        print(f"generated {clip_id}.mp4 ({duration * 24} frames)")


if __name__ == "__main__":
    main()

