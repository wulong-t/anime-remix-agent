"""Generate copyright-free synthetic clips for the semantic (3B-2) demo.

Three shots:
- shot A: requirement has emotion=sad; clip_a1 (sad) beats clip_a2 (calm)
  even though all other scoring factors are identical.
- shot B: requirement has shot_scale=wide; clip_b2 (wide) beats clip_b1
  (medium) with identical other factors.
- shot C: no emotion / shot_scale keywords; both requirement fields stay
  null and the old four-dimension logic still selects clip_c.
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
        # (id, lavfi source, frames, char_id, name, loc_id, loc_name, action, description, emotion, shot_scale)
        (
            "clip_a1",
            "smptebars",
            96,
            "char_lin_xia",
            "林夏",
            "loc_school_rooftop",
            "学校天台",
            "独自站立",
            "林夏独自站在学校天台。",
            "sad",
            "wide",
        ),
        (
            "clip_a2",
            "testsrc2",
            96,
            "char_lin_xia",
            "林夏",
            "loc_school_rooftop",
            "学校天台",
            "独自站立",
            "林夏独自站在学校天台。",
            "calm",
            "medium",
        ),
        (
            "clip_b1",
            "testsrc",
            96,
            "char_lu_chen",
            "陆辰",
            "loc_classroom",
            "教室",
            "沉默注视",
            "陆辰在教室沉默注视窗外。",
            None,
            "medium",
        ),
        (
            "clip_b2",
            "rgbtestsrc",
            96,
            "char_lu_chen",
            "陆辰",
            "loc_classroom",
            "教室",
            "沉默注视",
            "陆辰在教室沉默注视窗外。",
            None,
            "wide",
        ),
        (
            "clip_c",
            "smptebars",
            72,
            "char_lin_xia",
            "林夏",
            "loc_school_rooftop",
            "学校天台",
            "独自站立",
            "林夏独自站在学校天台。",
            None,
            None,
        ),
    ]
    clips = []
    for (
        clip_id,
        source,
        frames,
        char_id,
        name,
        loc_id,
        location,
        action,
        description,
        emotion,
        shot_scale,
    ) in specs:
        path = clips_dir / f"{clip_id}.mp4"
        _encode(ffmpeg, source, path, frames=frames)
        entry: dict[str, object] = {
            "id": clip_id,
            "path": f"clips/{clip_id}.mp4",
            "characters": [{"id": char_id, "name": name}],
            "location_id": loc_id,
            "location_name": location,
            "action": action,
            "description": description,
        }
        if emotion is not None:
            entry["emotion"] = emotion
        if shot_scale is not None:
            entry["shot_scale"] = shot_scale
        clips.append(entry)
        print(f"generated {clip_id}.mp4 ({frames} frames)")

    (demo_dir / "script.md").write_text(
        "林夏难过地独自站在学校天台，望着远方的城市。\n\n"
        "陆辰在教室沉默注视窗外，镜头里能看到远景。\n\n"
        "林夏站在学校天台，独自望着远方。",
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
    print(f"wrote {demo_dir / 'script.md'}")
    print(f"wrote {demo_dir / 'clips.json'}")


if __name__ == "__main__":
    main()
