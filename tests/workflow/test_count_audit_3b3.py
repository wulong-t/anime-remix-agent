"""Workflow tests for 3B-3 count_frames audit and manifest hardening."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import RunStatus
from anime_remix.errors import RenderError, UnsupportedMediaError
from anime_remix.json_io import sha256_file
from anime_remix.workflows import build_workflow
from anime_remix.workflows.build_workflow import _build_manifest, build
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


def _write_clips_json(path: Path, clips: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {"schema_version": "1.9", "clips": clips},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _clip_entry(
    clip_id: str,
    media_name: str,
    *,
    char_id: str,
    name: str,
    loc_id: str,
    loc_name: str,
    action: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "path": f"clips/{media_name}",
        "characters": [{"id": char_id, "name": name}],
        "location_id": loc_id,
        "location_name": loc_name,
        "action": action,
        "description": description,
    }


@pytest.fixture(scope="module")
def mixed_demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Demo with a repeated clip, a freeze clip and a placeholder shot."""

    root = tmp_path_factory.mktemp("count-mixed")
    clips_dir = root / "clips"
    clips_dir.mkdir()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    _encode_clip(ffmpeg, clips_dir / "clip_a.mp4", frames=96)
    _encode_clip(ffmpeg, clips_dir / "clip_b.mp4", frames=96)
    _encode_clip(ffmpeg, clips_dir / "clip_c.mp4", frames=30)
    _write_clips_json(
        root / "clips.json",
        [
            _clip_entry(
                "clip_a",
                "clip_a.mp4",
                char_id="char_lin_xia",
                name="林夏",
                loc_id="loc_school_rooftop",
                loc_name="学校天台",
                action="独自站立",
                description="林夏独自站在学校天台。",
            ),
            _clip_entry(
                "clip_b",
                "clip_b.mp4",
                char_id="char_lu_chen",
                name="陆辰",
                loc_id="loc_classroom",
                loc_name="教室",
                action="沉默注视",
                description="陆辰在教室沉默注视窗外。",
            ),
            _clip_entry(
                "clip_c",
                "clip_c.mp4",
                char_id="char_lu_chen",
                name="陆辰",
                loc_id="loc_garden",
                loc_name="花园",
                action="独自站立",
                description="陆辰独自站在花园里。",
            ),
        ],
    )
    (root / "script.md").write_text(
        "林夏独自站在学校天台，望着远方。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "陆辰独自站在花园里，望着远方。\n\n"
        "林夏独自站在学校天台，望着远方。\n\n"
        "路人在街道上骑车。",
        encoding="utf-8",
    )
    return {"root": root, "script": root / "script.md", "clips": root / "clips.json"}


@pytest.fixture(scope="module")
def placeholder_demo(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("count-placeholder")
    clips_dir = root / "clips"
    clips_dir.mkdir()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    _encode_clip(ffmpeg, clips_dir / "clip_p.mp4", frames=96)
    _write_clips_json(
        root / "clips.json",
        [
            _clip_entry(
                "clip_p",
                "clip_p.mp4",
                char_id="char_passerby",
                name="路人",
                loc_id="loc_street",
                loc_name="街道",
                action="骑车",
                description="路人在街道上骑车。",
            )
        ],
    )
    (root / "script.md").write_text(
        "林夏独自站在学校天台。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "林夏在天台转身离开。",
        encoding="utf-8",
    )
    return {"root": root, "script": root / "script.md", "clips": root / "clips.json"}


@pytest.fixture(scope="module")
def same_file_demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("count-same-file")
    clips_dir = root / "clips"
    clips_dir.mkdir()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    _encode_clip(ffmpeg, clips_dir / "clip_shared.mp4", frames=96)
    _write_clips_json(
        root / "clips.json",
        [
            _clip_entry(
                "clip_x",
                "clip_shared.mp4",
                char_id="char_lin_xia",
                name="林夏",
                loc_id="loc_school_rooftop",
                loc_name="学校天台",
                action="独自站立",
                description="林夏独自站在学校天台。",
            ),
            _clip_entry(
                "clip_y",
                "clip_shared.mp4",
                char_id="char_lu_chen",
                name="陆辰",
                loc_id="loc_classroom",
                loc_name="教室",
                action="沉默注视",
                description="陆辰在教室沉默注视窗外。",
            ),
        ],
    )
    (root / "script.md").write_text(
        "林夏独自站在学校天台。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "林夏独自站在学校天台。",
        encoding="utf-8",
    )
    return {"root": root, "script": root / "script.md", "clips": root / "clips.json"}


def _expected_audit() -> dict[str, dict[str, int]]:
    return {
        "clip_a": {"metadata_nb_frames": 96, "counted_nb_frames": 96},
        "clip_b": {"metadata_nb_frames": 96, "counted_nb_frames": 96},
        "clip_c": {"metadata_nb_frames": 30, "counted_nb_frames": 30},
    }


def test_build_real_count_audit_and_manifest(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "count-mixed"
    build(
        script_path=mixed_demo["script"],
        clips_path=mixed_demo["clips"],
        output_dir=target,
    )
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selected_source_frame_audit"] == _expected_audit()
    keys = list(manifest["selected_source_frame_audit"])
    assert keys == sorted(keys, key=lambda key: key.encode("utf-8"))

    member = manifest["core_artifact_member_sha256"]
    assert member is not None
    assert list(member) == [
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
    ]
    for name, digest in member.items():
        assert sha256_file(target / name) == digest

    digest = hashlib.sha256()
    for name in (
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
    ):
        digest.update((target / name).read_bytes())
    assert manifest["core_artifact_sha256"] == digest.hexdigest()

    retrieval = json.loads(
        (target / "retrieval_results.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (target / "timeline.json").read_text(encoding="utf-8")
    )
    timeline_by_shot = {item["shot_id"]: item for item in timeline["items"]}
    for shot in retrieval["shots"]:
        trace = shot["selection_trace"]
        final = trace["final_decision"]
        item = timeline_by_shot[shot["shot_id"]]
        assert final["selected_asset_id"] == item["source_asset_id"]
        assert final["selected_strategy"] == item["strategy"]
        assert final["reason_code"] == item["reason_code"]
        assert final["source_in_frame"] == item["source_in_frame"]
        assert final["source_frame_count"] == item["source_frame_count"]
        assert final["target_frames"] == item["target_frames"]

    assert timeline["items"][3]["source_asset_id"] == "clip_a"
    assert timeline["items"][4]["strategy"] == "placeholder"
    assert "clip_a" in manifest["selected_source_frame_audit"]


def test_real_ffprobe_count_calls_selected_unique_sorted(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FFmpegToolkit.count_source_frames
    calls: list[str] = []

    def recording(self: FFmpegToolkit, path: Path) -> int:
        calls.append(Path(path).name)
        return original(self, path)

    monkeypatch.setattr(FFmpegToolkit, "count_source_frames", recording)
    target = tmp_path / "runs" / "count-real"
    build(
        script_path=mixed_demo["script"],
        clips_path=mixed_demo["clips"],
        output_dir=target,
    )
    # clip_a selected twice but counted once; byte-order by asset_id; clip_c
    # is the freeze source; placeholder assets are never counted.
    assert calls == ["clip_a.mp4", "clip_b.mp4", "clip_c.mp4"]


def test_all_placeholder_audit_is_empty(
    placeholder_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "count-placeholder"
    build(
        script_path=placeholder_demo["script"],
        clips_path=placeholder_demo["clips"],
        output_dir=target,
    )
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selected_source_frame_audit"] == {}


def test_different_asset_ids_same_file_counted_twice(
    same_file_demo: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FFmpegToolkit.count_source_frames
    calls: list[str] = []

    def recording(self: FFmpegToolkit, path: Path) -> int:
        calls.append(Path(path).name)
        return original(self, path)

    monkeypatch.setattr(FFmpegToolkit, "count_source_frames", recording)
    target = tmp_path / "runs" / "count-same-file"
    build(
        script_path=same_file_demo["script"],
        clips_path=same_file_demo["clips"],
        output_dir=target,
    )
    assert calls == ["clip_shared.mp4", "clip_shared.mp4"]
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["selected_source_frame_audit"]) == {
        "clip_x",
        "clip_y",
    }


def test_count_mismatch_fails_before_staging_and_render(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_count(self: FFmpegToolkit, path: Path) -> int:
        return 97

    monkeypatch.setattr(FFmpegToolkit, "count_source_frames", fake_count)

    def fail_render(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("render must not run on count mismatch")

    monkeypatch.setattr(build_workflow, "render_timeline", fail_render)
    target = tmp_path / "runs" / "count-mismatch"
    with pytest.raises(UnsupportedMediaError) as excinfo:
        build(
            script_path=mixed_demo["script"],
            clips_path=mixed_demo["clips"],
            output_dir=target,
        )
    error = excinfo.value
    assert error.stage == "media_contract"
    assert error.asset_id == "clip_a"
    assert error.actual == {
        "metadata_nb_frames": 96,
        "counted_nb_frames": 97,
    }
    assert not target.exists()
    assert not (tmp_path / "runs" / ".count-mismatch.staging").exists()


def test_manifest_lifecycle_shapes(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    toolkit = FFmpegToolkit()
    audit = _expected_audit()
    running = _build_manifest(
        status=RunStatus.RUNNING,
        toolkit=toolkit,
        script_path=mixed_demo["script"],
        clips_path=mixed_demo["clips"],
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=None,
        selected_source_sha256={},
        selected_source_frame_audit=audit,
        core_artifact_member_sha256=None,
    )
    assert running["selected_source_frame_audit"] == audit
    assert running["core_artifact_member_sha256"] is None
    assert running["core_artifact_sha256"] is None

    succeeded = _build_manifest(
        status=RunStatus.SUCCEEDED,
        toolkit=toolkit,
        script_path=mixed_demo["script"],
        clips_path=mixed_demo["clips"],
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        selected_source_sha256={},
        core_artifact_sha256="a" * 64,
        selected_source_frame_audit=audit,
        core_artifact_member_sha256={
            "parsed_script.json": "b" * 64,
            "retrieval_results.json": "c" * 64,
            "timeline.json": "d" * 64,
        },
        output_sha256="e" * 64,
    )
    assert succeeded["selected_source_frame_audit"] == audit
    assert succeeded["core_artifact_member_sha256"] is not None
    assert list(succeeded["core_artifact_member_sha256"]) == [
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
    ]


def test_render_failure_failed_manifest_keeps_known_values(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_render(*args: Any, **kwargs: Any) -> None:
        raise RenderError("injected render failure")

    monkeypatch.setattr(build_workflow, "render_timeline", fail_render)
    target = tmp_path / "runs" / "count-render-fail"
    with pytest.raises(RenderError):
        build(
            script_path=mixed_demo["script"],
            clips_path=mixed_demo["clips"],
            output_dir=target,
        )
    assert not target.exists()
    staging = tmp_path / "runs" / ".count-render-fail.staging"
    assert staging.exists()
    failed = json.loads(
        (staging / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert failed["status"] == "failed"
    assert failed["selected_source_frame_audit"] == _expected_audit()
    member = failed["core_artifact_member_sha256"]
    assert member is not None
    for name, digest in member.items():
        assert sha256_file(staging / name) == digest
    digest = hashlib.sha256()
    for name in (
        "parsed_script.json",
        "retrieval_results.json",
        "timeline.json",
    ):
        digest.update((staging / name).read_bytes())
    assert failed["core_artifact_sha256"] == digest.hexdigest()


def test_independent_rerender_still_byte_identical(
    mixed_demo: dict[str, Path],
    tmp_path: Path,
) -> None:
    target = tmp_path / "runs" / "count-rerender"
    build(
        script_path=mixed_demo["script"],
        clips_path=mixed_demo["clips"],
        output_dir=target,
    )
    rerendered = tmp_path / "count-rerendered.mp4"
    render_timeline(
        target / "timeline.json",
        rerendered,
        toolkit=FFmpegToolkit(),
    )
    assert sha256_file(target / "output.mp4") == sha256_file(rerendered)
