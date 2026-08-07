"""Workflow tests for 3B-2 emotion / shot_scale (AGENTS.md v1.12 section 18.8)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.models import RenderProfile
from anime_remix.json_io import sha256_file
from anime_remix.workflows.build_workflow import build
from anime_remix.workflows.render_workflow import render_timeline

OLD_ALIASES_OUTPUT_SHA256 = (
    "f70d0187d3ff0427f7aaeb55778df492693688d3e8256c76daff2d45efc22a0e"
)


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


def _write_semantic_demo(root: Path) -> None:
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    specs = [
        (
            "clip_a1",
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
        _encode_clip(ffmpeg, path, frames=frames)
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
    (root / "clips.json").write_text(
        json.dumps(
            {"schema_version": "1.9", "clips": clips},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "script.md").write_text(
        "林夏难过地独自站在学校天台，望着远方的城市。\n\n"
        "陆辰在教室沉默注视窗外，镜头里能看到远景。\n\n"
        "林夏站在学校天台，独自望着远方。",
        encoding="utf-8",
    )


def _write_plain_demo(root: Path) -> None:
    """A demo whose script has no emotion / shot_scale keywords."""

    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    specs = [
        (
            "clip_001",
            96,
            "char_lin_xia",
            "林夏",
            "loc_school_rooftop",
            "学校天台",
            "独自站立",
            "林夏独自站在学校天台。",
        ),
        (
            "clip_002",
            96,
            "char_lu_chen",
            "陆辰",
            "loc_classroom",
            "教室",
            "沉默注视",
            "陆辰在教室沉默注视窗外。",
        ),
        (
            "clip_003",
            72,
            "char_lin_xia",
            "林夏",
            "loc_school_rooftop",
            "学校天台",
            "转身离开",
            "林夏在天台转身离开。",
        ),
    ]
    clips = []
    for (
        clip_id,
        frames,
        char_id,
        name,
        loc_id,
        location,
        action,
        description,
    ) in specs:
        path = clips_dir / f"{clip_id}.mp4"
        _encode_clip(ffmpeg, path, frames=frames)
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
    (root / "clips.json").write_text(
        json.dumps(
            {"schema_version": "1.9", "clips": clips},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "script.md").write_text(
        "林夏独自站在学校天台。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "林夏转身离开学校天台。",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def semantic_demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("semantic-demo")
    _write_semantic_demo(root)
    return {
        "root": root,
        "script": root / "script.md",
        "clips": root / "clips.json",
    }


@pytest.fixture(scope="module")
def plain_demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("plain-demo")
    _write_plain_demo(root)
    return {
        "root": root,
        "script": root / "script.md",
        "clips": root / "clips.json",
    }


def test_build_saves_semantic_metadata_and_selection(
    semantic_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "semantic-001"
    build(
        script_path=semantic_demo["script"],
        clips_path=semantic_demo["clips"],
        output_dir=target,
    )
    parsed = json.loads(
        (target / "parsed_script.json").read_text(encoding="utf-8")
    )
    assert parsed["shots"][0]["emotion"] == "sad"
    assert parsed["shots"][0]["shot_scale"] is None
    assert parsed["shots"][1]["emotion"] is None
    assert parsed["shots"][1]["shot_scale"] == "wide"
    assert parsed["shots"][2]["emotion"] is None
    assert parsed["shots"][2]["shot_scale"] is None

    retrieval = json.loads(
        (target / "retrieval_results.json").read_text(encoding="utf-8")
    )
    first = retrieval["shots"][0]
    assert first["selected"]["selected_asset_id"] == "clip_a1"
    assert first["top_3"][0]["score"]["emotion"] == "1.000000"
    second = retrieval["shots"][1]
    assert second["selected"]["selected_asset_id"] == "clip_b2"
    assert second["top_3"][0]["score"]["shot_scale"] == "1.000000"
    third = retrieval["shots"][2]
    assert third["selected"]["selected_asset_id"] == "clip_c"
    assert third["top_3"][0]["score"]["emotion"] is None
    assert third["top_3"][0]["score"]["shot_scale"] is None

    timeline = json.loads(
        (target / "timeline.json").read_text(encoding="utf-8")
    )
    items = timeline["items"]
    assert items[0]["requirement"]["emotion"] == "sad"
    assert items[1]["requirement"]["shot_scale"] == "wide"
    assert items[2]["requirement"]["emotion"] is None
    assert items[2]["requirement"]["shot_scale"] is None
    assert items[0]["source_asset_id"] == "clip_a1"
    assert items[0]["source_in_frame"] == 12
    assert items[0]["source_frame_count"] == 72
    assert items[1]["source_asset_id"] == "clip_b2"
    assert items[1]["source_in_frame"] == 12
    assert items[1]["source_frame_count"] == 72
    assert items[2]["source_asset_id"] == "clip_c"
    assert items[2]["source_in_frame"] == 0
    assert items[2]["source_frame_count"] == 72


def test_independent_rerender_without_inputs_is_byte_identical(
    semantic_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "semantic-rerender"
    build(
        script_path=semantic_demo["script"],
        clips_path=semantic_demo["clips"],
        output_dir=target,
    )
    timeline = json.loads(
        (target / "timeline.json").read_text(encoding="utf-8")
    )
    total_frames = sum(item["target_frames"] for item in timeline["items"])
    backups = [
        semantic_demo["script"].with_suffix(".bak"),
        semantic_demo["clips"].with_suffix(".bak"),
        target / "retrieval_results.json.bak",
    ]
    try:
        semantic_demo["script"].rename(backups[0])
        semantic_demo["clips"].rename(backups[1])
        (target / "retrieval_results.json").rename(backups[2])
        rerendered = tmp_path / "semantic-rerendered.mp4"
        render_timeline(
            target / "timeline.json",
            rerendered,
            toolkit=FFmpegToolkit(),
        )
        assert rerendered.exists()
        FFmpegToolkit().validate_final(
            rerendered,
            total_frames=total_frames,
            profile=RenderProfile(),
        )
        assert sha256_file(target / "output.mp4") == sha256_file(rerendered)
    finally:
        backups[0].rename(semantic_demo["script"])
        backups[1].rename(semantic_demo["clips"])
        backups[2].rename(target / "retrieval_results.json")


def test_renderer_ignores_semantic_metadata(
    plain_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    # Same media, same script (no semantic keywords), one clips.json with
    # emotion / shot_scale metadata and one without: selection and output
    # bytes must be identical.
    clips_dir = plain_demo["root"] / "clips"
    original = json.loads(
        plain_demo["clips"].read_text(encoding="utf-8")
    )
    with_metadata = {
        "schema_version": "1.9",
        "clips": [
            {
                **entry,
                "emotion": "sad",
                "shot_scale": "wide",
            }
            for entry in original["clips"]
        ],
    }
    clips_with = plain_demo["root"] / "clips-with-metadata.json"
    clips_with.write_text(
        json.dumps(with_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assert clips_dir.exists()

    target_without = tmp_path / "runs" / "plain-without"
    target_with = tmp_path / "runs" / "plain-with"
    build(
        script_path=plain_demo["script"],
        clips_path=plain_demo["clips"],
        output_dir=target_without,
    )
    build(
        script_path=plain_demo["script"],
        clips_path=clips_with,
        output_dir=target_with,
    )
    timeline_without = json.loads(
        (target_without / "timeline.json").read_text(encoding="utf-8")
    )
    timeline_with = json.loads(
        (target_with / "timeline.json").read_text(encoding="utf-8")
    )
    assert [
        (item["source_asset_id"], item["source_in_frame"], item["source_frame_count"])
        for item in timeline_without["items"]
    ] == [
        (item["source_asset_id"], item["source_in_frame"], item["source_frame_count"])
        for item in timeline_with["items"]
    ]
    assert sha256_file(target_without / "output.mp4") == sha256_file(
        target_with / "output.mp4"
    )


def test_old_aliases_demo_output_sha256_unchanged(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    aliases_demo = repo_root / "demo" / "aliases"
    script = aliases_demo / "script.md"
    clips = aliases_demo / "clips.json"
    aliases = aliases_demo / "aliases.json"
    assert script.is_file() and clips.is_file() and aliases.is_file()

    target = tmp_path / "runs" / "aliases-golden-3b2"
    build(
        script_path=script,
        clips_path=clips,
        aliases_path=aliases,
        output_dir=target,
    )
    output_hash = sha256_file(target / "output.mp4")
    assert output_hash == OLD_ALIASES_OUTPUT_SHA256

    timeline = json.loads(
        (target / "timeline.json").read_text(encoding="utf-8")
    )
    assert [
        (
            item["source_asset_id"],
            item["source_in_frame"],
            item["source_frame_count"],
        )
        for item in timeline["items"]
    ] == [
        ("clip_001", 12, 72),
        ("clip_002", 24, 72),
        ("clip_003", 0, 72),
    ]
    for item in timeline["items"]:
        score = item["score"]
        assert score is not None
        assert score["active_weights"] == {
            "action": "0.450000",
            "character": "0.250000",
            "duration": "0.150000",
            "location": "0.150000",
        }
    assert [item["score"]["total"] for item in timeline["items"]] == [
        "0.710000",
        "0.673448",
        "0.618889",
    ]

    total_frames = sum(item["target_frames"] for item in timeline["items"])
    rerendered = tmp_path / "aliases-golden-3b2-rerendered.mp4"
    render_timeline(
        target / "timeline.json",
        rerendered,
        toolkit=FFmpegToolkit(),
    )
    assert sha256_file(rerendered) == OLD_ALIASES_OUTPUT_SHA256
    FFmpegToolkit().validate_final(
        rerendered,
        total_frames=total_frames,
        profile=RenderProfile(),
    )


def test_old_demo_and_freeze_still_work(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("", "freeze"):
        demo_dir = repo_root / "demo" / name
        script = demo_dir / "script.md"
        clips = demo_dir / "clips.json"
        target = tmp_path / "runs" / f"regression-{name or 'base'}"
        build(
            script_path=script,
            clips_path=clips,
            output_dir=target,
        )
        assert (target / "output.mp4").is_file()
        timeline = json.loads(
            (target / "timeline.json").read_text(encoding="utf-8")
        )
        total_frames = sum(
            item["target_frames"] for item in timeline["items"]
        )
        FFmpegToolkit().validate_final(
            target / "output.mp4",
            total_frames=total_frames,
            profile=RenderProfile(),
        )
