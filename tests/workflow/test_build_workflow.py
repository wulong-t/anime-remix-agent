from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.models import RenderProfile
from anime_remix.errors import PublicationError, UnsafePathError
from anime_remix.json_io import sha256_file
from anime_remix.workflows.build_workflow import build
from anime_remix.workflows.render_workflow import render_timeline


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


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("demo")
    clips_dir = root / "clips"
    clips_dir.mkdir()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    specs = [
        ("clip_001", 96, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "独自站立", "林夏独自站在学校天台。"),
        ("clip_002", 120, "char_lu_chen", "陆辰", "loc_classroom", "教室", "沉默注视", "陆辰在教室沉默注视窗外。"),
        ("clip_003", 72, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "转身离开", "林夏在天台转身离开。"),
    ]
    clips = []
    for clip_id, frames, char_id, name, loc_id, location, action, description in specs:
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
    (root / "script.md").write_text(
        "“天台的风很大。”林夏独自站在学校天台。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "林夏转身离开学校天台。",
        encoding="utf-8",
    )
    (root / "clips.json").write_text(
        json.dumps({"schema_version": "1.9", "clips": clips}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"root": root, "script": root / "script.md", "clips": root / "clips.json"}


def test_build_success_and_artifacts(demo: dict[str, Path], tmp_path: Path) -> None:
    target = tmp_path / "runs" / "demo-001"
    result = build(
        script_path=demo["script"],
        clips_path=demo["clips"],
        output_dir=target,
    )
    assert target.exists()
    for name in (
        ".anime-remix-run",
        "run_manifest.json",
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
        "render.log",
        "output.mp4",
    ):
        assert (target / name).exists(), name
    manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["output_sha256"]
    marker = json.loads((target / ".anime-remix-run").read_text(encoding="utf-8"))
    assert marker["managed_entries"] == sorted(marker["managed_entries"])
    timeline = json.loads((target / "timeline.json").read_text(encoding="utf-8"))
    total = sum(item["target_frames"] for item in timeline["items"])
    toolkit = FFmpegToolkit()
    toolkit.validate_final(target / "output.mp4", total_frames=total, profile=RenderProfile())
    assert result == target


def test_independent_rerender_without_script_or_clips(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "demo-002"
    build(
        script_path=demo["script"],
        clips_path=demo["clips"],
        output_dir=target,
    )
    script_backup = demo["root"] / "script.md.bak"
    clips_backup = demo["root"] / "clips.json.bak"
    demo["script"].rename(script_backup)
    demo["clips"].rename(clips_backup)
    try:
        rerendered = tmp_path / "rerendered.mp4"
        render_timeline(
            target / "timeline.json",
            rerendered,
            toolkit=FFmpegToolkit(),
        )
        assert rerendered.exists()
    finally:
        script_backup.rename(demo["script"])
        clips_backup.rename(demo["clips"])


def test_target_exists_fails_without_modification(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "exists"
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(PublicationError):
        build(
            script_path=demo["script"],
            clips_path=demo["clips"],
            output_dir=target,
        )
    assert sentinel.read_text(encoding="utf-8") == "original"


def test_preflight_failure_creates_no_staging(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    clips_path = tmp_path / "broken.json"
    clips_path.write_text(
        json.dumps(
            {
                "schema_version": "1.9",
                "clips": [
                    {
                        "id": "clip_001",
                        "path": "missing.mp4",
                        "characters": [{"id": "c", "name": "林夏"}],
                        "action": "x",
                        "description": "y",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    target = tmp_path / "runs" / "preflight"
    with pytest.raises(UnsafePathError):
        build(
            script_path=demo["script"],
            clips_path=clips_path,
            output_dir=target,
        )
    assert not target.exists()
    assert not (tmp_path / "runs" / ".preflight.staging").exists()


def test_manifest_core_artifact_sha256_stable_definition(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "demo-core"
    build(
        script_path=demo["script"],
        clips_path=demo["clips"],
        output_dir=target,
    )
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    for name in (
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
    ):
        digest.update((target / name).read_bytes())
    assert manifest["core_artifact_sha256"] == digest.hexdigest()


def test_manifest_selected_sources_real_sha256_sorted(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "demo-sources"
    build(
        script_path=demo["script"],
        clips_path=demo["clips"],
        output_dir=target,
    )
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (target / "timeline.json").read_text(encoding="utf-8")
    )
    selected = {
        item["source_asset_id"]
        for item in timeline["items"]
        if item["source_asset_id"] is not None
    }
    assert selected
    assert set(manifest["selected_source_sha256"]) == selected
    assert list(manifest["selected_source_sha256"]) == sorted(selected)
    for item in timeline["items"]:
        asset_id = item["source_asset_id"]
        if asset_id is None:
            continue
        source = (target / item["source_path"]).resolve()
        assert manifest["selected_source_sha256"][asset_id] == sha256_file(
            source
        )


def test_build_creates_missing_parent_dirs(
    demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "a" / "b" / "demo-parents"
    build(
        script_path=demo["script"],
        clips_path=demo["clips"],
        output_dir=target,
    )
    assert target.exists()
    assert (target / "output.mp4").exists()
