"""Generate copyright-free synthetic clips for the aliases demo.

The demo script uses only aliases (no canonical character/location names), so
parsed_script.json / timeline.json must contain canonical ID/name output.
"""

from __future__ import annotations

import json
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
    frames: int,
) -> None:
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"{source}=size=1280x720:rate=24:duration={frames / 24:.6f}",
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
        ("clip_001", "testsrc2", 96, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "独自站立", "林夏独自站在学校天台。"),
        ("clip_002", "smptebars", 120, "char_lu_chen", "陆辰", "loc_classroom", "教室", "沉默注视", "陆辰在教室沉默注视窗外。"),
        ("clip_003", "testsrc", 72, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "转身离开", "林夏在天台转身离开。"),
    ]
    clips = []
    for clip_id, source, frames, char_id, name, loc_id, location, action, description in specs:
        path = clips_dir / f"{clip_id}.mp4"
        _encode(ffmpeg, source, path, frames=frames)
        clips.append(
            {
                "id": clip_id,
                "path": f"clips/{clip_id}.mp4",
                "characters": [{"id": char_id, "name": name}],
                "location_id": loc_id,
                "location_name": location,
                "action": action,
                "description": description,
            }
        )
        print(f"generated {clip_id}.mp4 ({frames} frames)")

    (demo_dir / "script.md").write_text(
        "“天台的风很大。”小夏独自站在学校楼顶，望着远方的城市。\n\n"
        "阿辰在课堂上沉默注视窗外，没有说一句话。\n\n"
        "“走吧，要上课了。”小夏转身离开学校楼顶，阿辰跟在身后。",
        encoding="utf-8",
    )
    (demo_dir / "clips.json").write_text(
        json.dumps(
            {"schema_version": "1.9", "clips": clips},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (demo_dir / "aliases.json").write_text(
        json.dumps(
            {
                "schema_version": "1.9",
                "character_aliases": [
                    {"target_id": "char_lin_xia", "aliases": ["小夏"]},
                    {"target_id": "char_lu_chen", "aliases": ["阿辰"]},
                ],
                "location_aliases": [
                    {"target_id": "loc_school_rooftop", "aliases": ["楼顶"]},
                    {"target_id": "loc_classroom", "aliases": ["课堂"]},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {demo_dir / 'script.md'}")
    print(f"wrote {demo_dir / 'clips.json'}")
    print(f"wrote {demo_dir / 'aliases.json'}")


if __name__ == "__main__":
    main()
