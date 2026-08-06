"""Generate synthetic smoke media + hand-written timeline with real fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise SystemExit("ffmpeg not found on PATH")
    return binary


def _encode_clip(ffmpeg: str, output: Path, *, frames: int) -> None:
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=24:duration={frames / 24:.6f}",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirement(
    shot_id: str,
    order: int,
    source_text: str,
    action: str,
    target_frames: int,
    dialogue: str | None = None,
) -> dict:
    return {
        "id": shot_id,
        "order": order,
        "source_text": source_text,
        "characters": [],
        "location_id": None,
        "location_name": None,
        "action": action,
        "target_frames": target_frames,
        "dialogue": dialogue,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    clips_dir = output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()

    source_a = clips_dir / "source_a.mp4"
    source_b = clips_dir / "source_b.mp4"
    _encode_clip(ffmpeg, source_a, frames=96)
    _encode_clip(ffmpeg, source_b, frames=120)

    items = [
        {
            "shot_id": "shot_001",
            "order": 1,
            "requirement": _requirement(
                "shot_001",
                1,
                "林夏独自站在学校天台。",
                "独自站立",
                72,
            ),
            "strategy": "clip",
            "source_asset_id": "source_a",
            "source_path": "clips/source_a.mp4",
            "source_size_bytes": source_a.stat().st_size,
            "source_sha256": _sha256(source_a),
            "source_in_frame": 12,
            "source_frame_count": 72,
            "target_frames": 72,
            "score": None,
            "reason_code": "center_trim",
            "reason": "center_trim",
        },
        {
            "shot_id": "shot_002",
            "order": 2,
            "requirement": _requirement(
                "shot_002",
                2,
                "画面切入纯黑。",
                "纯黑",
                36,
            ),
            "strategy": "placeholder",
            "source_asset_id": None,
            "source_path": None,
            "source_size_bytes": None,
            "source_sha256": None,
            "source_in_frame": 0,
            "source_frame_count": 0,
            "target_frames": 36,
            "score": None,
            "reason_code": "no_candidate",
            "reason": "no candidate passed gates",
        },
    ]
    timeline = {
        "schema_version": "1.9",
        "path_base": "timeline_dir",
        "render_profile": {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "video_codec": "libx264",
            "pixel_format": "yuv420p",
            "video_preset": "medium",
            "video_crf": 20,
            "max_b_frames": 0,
            "gop_frames": 48,
            "video_track_timescale": 48000,
            "audio_codec": "aac",
            "audio_bitrate_kbps": 128,
            "audio_sample_rate": 48000,
            "audio_channels": 2,
        },
        "items": items,
    }
    timeline_path = output / "timeline.json"
    timeline_path.write_text(
        __import__("json").dumps(timeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {timeline_path}")
    print(f"wrote {source_a} ({source_a.stat().st_size} bytes)")
    print(f"wrote {source_b} ({source_b.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
