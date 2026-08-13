"""G1-MK1-L integration: synthetic raw -> 121f normalization -> Timeline -> render."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import (
    ClipAsset,
    RenderProfile,
    ShotRequirement,
    Timeline,
    TimelineItem,
)
from anime_remix.errors import MediaProbeError, RenderError
from anime_remix.json_io import sha256_file
from anime_remix.workflows.render_workflow import render_timeline
from experiments.manual_keyframe_mvp import manual_keyframe_mvp as harness
from experiments.manual_keyframe_mvp.manual_keyframe_mvp import (
    FROZEN_SAMPLING,
    RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    TARGET_FRAMES,
    TOTAL_AUDIO_SAMPLES,
    HarnessError,
    cmd_finalize,
    cmd_inspect,
    cmd_package,
)


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    assert binary, "ffmpeg must be installed for media tests"
    return binary


def _encode_raw(output: Path, *, frames: int = 81) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x704:rate=16",
            "-frames:v",
            str(frames),
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
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-f",
            "mp4",
            str(output),
        ],
        check=True,
        shell=False,
    )


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    rng = random.Random(seed)
    for _ in range(height):
        raw.append(0)
        for _ in range(width):
            raw.extend(
                bytes(
                    (
                        rng.randrange(256),
                        rng.randrange(256),
                        rng.randrange(256),
                    )
                )
            )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_request_root(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "request-root"
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    k0 = inputs / "k0.png"
    k_end = inputs / "k_end.png"
    k0.write_bytes(_png_bytes(1280, 720, seed=11))
    k_end.write_bytes(_png_bytes(1280, 720, seed=22))
    k0_sha = _sha256(k0)
    k_end_sha = _sha256(k_end)
    for name, image, image_sha in (
        ("k0.provenance.json", k0, k0_sha),
        ("k_end.provenance.json", k_end, k_end_sha),
    ):
        asset = (
            "inputs/k0.png" if name.startswith("k0.") else "inputs/k_end.png"
        )
        (inputs / name).write_text(
            json.dumps(
                {
                    "asset": asset,
                    "sha256": image_sha,
                    "creation_method": "human-edited synthetic test",
                    "external_inputs": [],
                    "named_references": {
                        "artists": [],
                        "studios": [],
                        "series": [],
                        "characters": [],
                    },
                    "rights_basis": "original/authorized synthetic test",
                    "public_demo_allowed": False,
                    "notes": "synthetic test asset",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert image.is_file()
    request = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": "g1mk1-synthetic-001",
        "start_keyframe": "inputs/k0.png",
        "end_keyframe": "inputs/k_end.png",
        "start_provenance": "inputs/k0.provenance.json",
        "end_provenance": "inputs/k_end.provenance.json",
        "start_sha256": k0_sha,
        "end_sha256": k_end_sha,
        "subject_description": "An original 2D cel-animation woman",
        "scene_description": "A quiet observatory control room at dusk",
        "action": "turn head slightly to the right",
        "start_state": "near-frontal calm",
        "end_state": "three-quarter calm",
        "emotion": "calm",
        "shot_scale": "medium",
        "camera": "fixed",
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
    }
    (root / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    return root, {"k0_sha": k0_sha, "k_end_sha": k_end_sha}


def _approve(pending: Path) -> Path:
    data = json.loads(pending.read_text(encoding="utf-8"))
    for key in data["rights"]:
        data["rights"][key] = True
    for key in (
        "identity",
        "endpoint_pose",
        "body_camera_background",
        "style",
        "artifact",
    ):
        data["visual_review"][key] = "pass"
    data["visual_review"]["accept_borderline"] = False
    data["visual_review"]["overall"] = "approved"
    data["approved_at"] = "2026-08-09T12:00:00+08:00"
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return pending


def _receipt(
    raw: Path, package: Path, **overrides: object
) -> dict[str, object]:
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (package / "request.json").read_text(encoding="utf-8")
    )
    data = dict(FROZEN_SAMPLING)
    data.update(
        {
            "schema_version": RECEIPT_SCHEMA,
            "request_id": "g1mk1-synthetic-001",
            "request_sha256": manifest["request_sha256"],
            "package_manifest_sha256": _sha256(
                package / "package_manifest.json"
            ),
            "sampling_contract_sha256": _sha256(
                package / "sampling_contract.json"
            ),
            "start_sha256": request["start_sha256"],
            "end_sha256": request["end_sha256"],
            "status": "success",
            "raw_sha256": sha256_file(raw),
        }
    )
    data.update(overrides)
    return data


def _make_package(tmp_path: Path, name: str = "package") -> Path:
    request_root, _ = _make_request_root(tmp_path)
    inspection = tmp_path / f"{name}-inspection"
    cmd_inspect(request_root / "request.json", inspection)
    approval = _approve(inspection / "approval.json")
    package = tmp_path / name
    cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        package,
    )
    return package


def _staging_leftovers(parent: Path) -> list[Path]:
    return [
        p
        for p in parent.iterdir()
        if p.name.startswith(".staging-") or ".staging-" in p.name
    ]


def _assert_no_abs_paths_or_secrets(run: Path) -> None:
    for path in run.rglob("*"):
        if not path.is_file() or path.suffix in {".png", ".mp4"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "DEEPSEEK" not in text
        assert "API_KEY" not in text
        # render.log is written by the frozen existing renderer and records
        # the staging path; official JSON fields carry the relative-path rule.
        if path.name == "render.log":
            continue
        assert "://" not in text
        assert "C:" not in text
        assert "\\" not in text
        assert not text.lstrip().startswith("/")


def test_normalize_81f_16fps_1280x704_to_121f_24fps_1280x720(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    toolkit = FFmpegToolkit()
    generated = tmp_path / "generated_clip.mp4"
    toolkit.normalize_generated_source(
        raw, generated, target_frames=TARGET_FRAMES
    )
    info = toolkit._probe(generated, count_frames=True)
    streams = info["streams"]
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    assert len(videos) == 1
    assert audios == []
    video = videos[0]
    assert video["codec_name"] == "h264"
    assert video["profile"] == "High"
    assert (video["width"], video["height"]) == (1280, 720)
    assert video["r_frame_rate"] == "24/1"
    assert video["avg_frame_rate"] == "24/1"
    assert video["pix_fmt"] == "yuv420p"
    assert video["sample_aspect_ratio"] == "1:1"
    assert video["color_range"] == "tv"
    assert video["color_space"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert video["color_primaries"] == "bt709"
    assert video["field_order"] == "progressive"
    assert video["chroma_location"] == "left"
    assert int(video["nb_read_frames"]) == TARGET_FRAMES
    toolkit.validate_segment(
        generated, target_frames=TARGET_FRAMES, shot_id="generated"
    )


@pytest.mark.parametrize("target", [0, -5, True])
def test_normalization_rejects_nonpositive_target_frames(
    tmp_path: Path, target: object
) -> None:
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    toolkit = FFmpegToolkit()
    with pytest.raises(MediaProbeError):
        toolkit.normalize_generated_source(
            raw, tmp_path / "wrong.mp4", target_frames=target  # type: ignore[arg-type]
        )


def test_normalization_cannot_fabricate_121_from_5s_source(
    tmp_path: Path,
) -> None:
    # 80 frames @ 16fps span exactly 5.0s and yield 120 frames at 24fps;
    # the strict gate must reject a 121-frame claim instead of padding.
    raw = tmp_path / "raw-80.mp4"
    _encode_raw(raw, frames=80)
    toolkit = FFmpegToolkit()
    with pytest.raises(RenderError):
        toolkit.normalize_generated_source(
            raw,
            tmp_path / "cannot-fabricate.mp4",
            target_frames=TARGET_FRAMES,
        )


def test_normalization_rejects_missing_source(tmp_path: Path) -> None:
    toolkit = FFmpegToolkit()
    with pytest.raises(MediaProbeError):
        toolkit.normalize_generated_source(
            tmp_path / "missing.mp4",
            tmp_path / "out.mp4",
            target_frames=TARGET_FRAMES,
        )


def _one_item_timeline(
    timeline_dir: Path, source: Path
) -> Timeline:
    requirement = ShotRequirement(
        id="g1mk1-shot",
        order=1,
        source_text="Generated manual-keyframe shot",
        characters=[],
        location_id=None,
        location_name=None,
        action="turn head slightly to the right",
        target_frames=TARGET_FRAMES,
        dialogue=None,
        emotion=None,
        shot_scale=None,
    )
    item = TimelineItem(
        shot_id="g1mk1-shot",
        order=1,
        requirement=requirement,
        strategy=TimelineStrategy.CLIP,
        source_asset_id="g1mk1-generated-clip",
        source_path=str(source.relative_to(timeline_dir).as_posix()),
        source_size_bytes=source.stat().st_size,
        source_sha256=sha256_file(source),
        source_in_frame=0,
        source_frame_count=TARGET_FRAMES,
        target_frames=TARGET_FRAMES,
        score=None,
        reason_code="exact_length",
        reason="exact_length",
    )
    return Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )


def test_generated_clip_renders_one_item_timeline(tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    toolkit = FFmpegToolkit()
    generated = tmp_path / "generated_clip.mp4"
    toolkit.normalize_generated_source(
        raw, generated, target_frames=TARGET_FRAMES
    )
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(
        _one_item_timeline(tmp_path, generated).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.mp4"
    result = render_timeline(timeline_path, output, toolkit=FFmpegToolkit())
    toolkit.validate_final(result, total_frames=TARGET_FRAMES, profile=RenderProfile())
    final_info = toolkit._probe(result, count_frames=True)
    audio = [
        s
        for s in final_info["streams"]
        if s.get("codec_type") == "audio"
    ]
    assert len(audio) == 1
    assert audio[0]["codec_name"] == "aac"
    assert audio[0]["sample_rate"] == "48000"
    assert audio[0]["channels"] == 2
    assert toolkit.max_volume(result) <= -90


def test_finalize_end_to_end_synthetic(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    result = cmd_finalize(package, raw, receipt, run)
    assert result == run
    for name in (
        ".anime-remix-run",
        "generation_manifest.json",
        "request.json",
        "approval.json",
        "package_manifest.json",
        "remote_receipt.json",
        "raw_shot.mp4",
        "generated_clip.mp4",
        "timeline.json",
        "render.log",
        "output.mp4",
    ):
        assert (run / name).is_file(), name
    assert (run / "normalized").is_dir()

    manifest = json.loads(
        (run / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["request_id"] == "g1mk1-synthetic-001"
    assert manifest["failure_layer"] is None
    assert manifest["total_audio_samples"] == TOTAL_AUDIO_SAMPLES
    assert manifest["raw"]["probe"]["nb_read_frames"] == 81
    assert manifest["raw"]["probe"]["video_streams"] == 1
    assert manifest["raw"]["probe"]["width"] == 1280
    assert manifest["raw"]["probe"]["height"] == 704
    assert manifest["raw"]["probe"]["r_frame_rate"] == "16/1"
    assert manifest["normalized"]["probe"]["nb_read_frames"] == TARGET_FRAMES
    assert manifest["normalized"]["probe"]["video_streams"] == 1
    assert manifest["normalized"]["probe"]["width"] == 1280
    assert manifest["normalized"]["probe"]["height"] == 720
    assert manifest["normalized"]["probe"]["r_frame_rate"] == "24/1"
    assert manifest["final"]["probe"]["nb_read_frames"] == TARGET_FRAMES
    assert manifest["final"]["probe"]["video_streams"] == 1
    assert manifest["final"]["probe"]["audio_streams"] == 1
    assert manifest["final"]["probe"]["audio_time_base"] == "1/48000"
    assert (
        manifest["final"]["probe"]["audio_duration_ts"] == TOTAL_AUDIO_SAMPLES
    )
    assert manifest["total_audio_samples"] == TOTAL_AUDIO_SAMPLES
    for key in ("raw", "normalized", "final", "timeline", "clips", "render_log"):
        record = manifest[key]
        assert sha256_file(run / record["path"]) == record["sha256"]

    timeline = json.loads(
        (run / "timeline.json").read_text(encoding="utf-8")
    )
    assert timeline["schema_version"] == "1.9"
    assert len(timeline["items"]) == 1
    item = timeline["items"][0]
    assert item["strategy"] == "clip"
    assert item["source_path"] == "generated_clip.mp4"
    assert item["source_in_frame"] == 0
    assert item["source_frame_count"] == TARGET_FRAMES
    assert item["target_frames"] == TARGET_FRAMES
    assert item["reason_code"] == "exact_length"
    assert item["requirement"]["target_frames"] == TARGET_FRAMES
    assert item["requirement"]["emotion"] == "calm"
    assert item["requirement"]["shot_scale"] == "medium"

    clips = json.loads((run / "clips.json").read_text(encoding="utf-8"))
    assert clips["schema_version"] == "1.9"
    assert len(clips["clips"]) == 1
    clip = clips["clips"][0]
    assert clip["id"] == "g1mk1-synthetic-001"
    assert clip["path"] == "generated_clip.mp4"
    assert clip["characters"] == []
    assert clip["location_id"] is None
    assert clip["location_name"] is None
    assert clip["action"] == "turn head slightly to the right"
    assert clip["emotion"] == "calm"
    assert clip["shot_scale"] == "medium"
    request_data = json.loads(
        (run / "request.json").read_text(encoding="utf-8")
    )
    assert clip["description"] == harness.generated_clip_description(
        request_data
    )
    assert manifest["clips"]["path"] == "clips.json"
    assert sha256_file(run / "clips.json") == manifest["clips"]["sha256"]
    assert item["source_asset_id"] == clip["id"]
    assert item["source_path"] == clip["path"]

    clip_asset = ClipAsset.model_validate(clip)
    probed = FFmpegToolkit().probe_asset(run / clip["path"], clip_asset)
    assert probed.nb_frames == TARGET_FRAMES
    assert probed.fps_num == 24
    assert probed.fps_den == 1
    assert (probed.width, probed.height) == (1280, 720)

    FFmpegToolkit().validate_final(
        run / "output.mp4",
        total_frames=TARGET_FRAMES,
        profile=RenderProfile(),
    )
    _assert_no_abs_paths_or_secrets(run)
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_raw_wrong_frame_count(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw, frames=80)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt, run)
    assert exc.value.layer == "media_normalization"
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_receipt_raw_hash_mismatch(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(
            _receipt(raw, package, raw_sha256="f" * 64), indent=2
        ),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt, run)
    assert exc.value.layer == "media_normalization"
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_receipt_missing_frozen_params(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_data = _receipt(raw, package)
    del receipt_data["seed"]
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt, run)
    assert exc.value.layer == "evidence_incomplete"
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_existing_output(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(HarnessError, match="already exists"):
        cmd_finalize(package, raw, receipt, run)
    assert _staging_leftovers(tmp_path) == []


def _encode_raw_two_streams(output: Path, *, frames: int = 81) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x704:rate=16",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x704:rate=16",
            "-frames:v",
            str(frames),
            "-map",
            "0:v:0",
            "-map",
            "1:v:0",
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
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-f",
            "mp4",
            str(output),
        ],
        check=True,
        shell=False,
    )


def test_finalize_rejects_raw_with_two_video_streams(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw-two.mp4"
    _encode_raw_two_streams(raw)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt, run)
    assert exc.value.layer == "media_normalization"
    assert "video_streams" in str(exc.value)
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_raw_decode_errors(tmp_path: Path) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    data = raw.read_bytes()
    start = len(data) // 5
    corrupted = data[:start] + b"\x00" * 4096 + data[start + 4096 :]
    raw.write_bytes(corrupted)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt, run)
    assert exc.value.layer == "media_normalization"
    assert "decode_check" in str(exc.value)
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_receipt_bound_to_different_package(
    tmp_path: Path,
) -> None:
    package_a = _make_package(tmp_path, "package-a")
    package_b = _make_package(tmp_path, "package-b")
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt = tmp_path / "remote_receipt.json"
    receipt.write_text(
        json.dumps(_receipt(raw, package_a), indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError, match="package_manifest_sha256"):
        cmd_finalize(package_b, raw, receipt, run)
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_binds_package_member_to_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    original_approval = (package / "approval.json").read_bytes()
    real_write = harness._write_staged_bytes

    def replace_then_write(path, data):
        if Path(path).name == "approval.json":
            (package / "approval.json").write_bytes(
                b'{"replaced": true}\n'
            )
        real_write(path, data)

    monkeypatch.setattr(harness, "_write_staged_bytes", replace_then_write)
    run = cmd_finalize(package, raw, receipt_path, tmp_path / "run")
    assert (run / "approval.json").read_bytes() == original_approval
    manifest = json.loads(
        (run / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["approval_sha256"]
        == hashlib.sha256(original_approval).hexdigest()
    )
    assert _staging_leftovers(tmp_path) == []


def test_finalize_binds_sampling_receipt_to_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    original_receipt_bytes = json.dumps(
        _receipt(raw, package), indent=2
    ).encode("utf-8")
    receipt_path.write_bytes(original_receipt_bytes)
    real_write = harness._write_staged_bytes

    def replace_then_write(path, data):
        if Path(path).name == "remote_receipt.json":
            receipt_path.write_bytes(
                json.dumps(
                    _receipt(raw, package, raw_sha256="f" * 64), indent=2
                ).encode("utf-8")
            )
        real_write(path, data)

    monkeypatch.setattr(harness, "_write_staged_bytes", replace_then_write)
    run = cmd_finalize(package, raw, receipt_path, tmp_path / "run")
    assert (run / "remote_receipt.json").read_bytes() == original_receipt_bytes
    manifest = json.loads(
        (run / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["remote_receipt"]["sha256"]
        == hashlib.sha256(original_receipt_bytes).hexdigest()
    )
    assert _staging_leftovers(tmp_path) == []


def test_finalize_raw_copy_corruption_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    real_write = harness._write_staged_bytes

    def corrupt_raw(path, data):
        real_write(path, data)
        if Path(path).name == "raw_shot.mp4":
            Path(path).write_bytes(Path(path).read_bytes() + b"x")

    monkeypatch.setattr(harness, "_write_staged_bytes", corrupt_raw)
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt_path, run)
    assert exc.value.layer == "evidence_incomplete"
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_request_replaced_after_manifest_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2: manifest captured, request replaced before member capture -> fail."""

    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    original_request = (package / "request.json").read_bytes()
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def capture_then_replace(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "package_manifest.json" and counts[p.name] == 1:
            replaced = json.loads(original_request)
            replaced["subject_description"] = "REPLACED AFTER MANIFEST CAPTURE"
            (package / "request.json").write_bytes(
                json.dumps(replaced, indent=2).encode("utf-8")
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_replace)
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt_path, run)
    assert exc.value.layer == "evidence_incomplete"
    assert "request_sha256" in str(exc.value) or "mismatch" in str(exc.value)
    assert counts.get("package_manifest.json") == 1
    assert counts.get("request.json") == 1
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_contract_replaced_after_manifest_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2: sampling contract replaced after manifest capture -> fail."""

    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def capture_then_replace(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "package_manifest.json" and counts[p.name] == 1:
            (package / "sampling_contract.json").write_bytes(
                b'{"replaced": true}\n'
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_replace)
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt_path, run)
    assert exc.value.layer == "evidence_incomplete"
    assert "sampling_contract_sha256" in str(exc.value) or "mismatch" in str(
        exc.value
    )
    assert counts.get("package_manifest.json") == 1
    assert counts.get("sampling_contract.json") == 1
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_captures_each_package_member_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2: every package member, receipt and raw is captured exactly once."""

    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def counting_capture(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        return real_capture(p, layer)

    monkeypatch.setattr(harness, "_capture_bytes", counting_capture)
    run = cmd_finalize(package, raw, receipt_path, tmp_path / "run")
    assert run.is_dir()
    for name in (
        "package_manifest.json",
        "request.json",
        "k0.png",
        "k_end.png",
        "k0.provenance.json",
        "k_end.provenance.json",
        "inspection.json",
        "approval.json",
        "sampling_contract.json",
        "anisora_input.txt",
        "remote_receipt.json",
        "raw.mp4",
    ):
        assert counts.get(name) == 1, name
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_package_root_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: finalize rejects a symlink/junction/reparse package root."""

    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    real = harness._is_link_or_reparse

    def fake_is_link(path: Path) -> bool:
        return Path(path) == package or real(path)

    monkeypatch.setattr(harness, "_is_link_or_reparse", fake_is_link)
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt_path, run)
    assert exc.value.layer == "evidence_incomplete"
    assert "package root" in str(exc.value)
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []


def test_finalize_rejects_tampered_manifest_request_hash(
    tmp_path: Path,
) -> None:
    """B5: manifest header hash no longer matches the packaged member."""

    package = _make_package(tmp_path)
    raw = tmp_path / "raw.mp4"
    _encode_raw(raw)
    receipt_path = tmp_path / "remote_receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt(raw, package), indent=2), encoding="utf-8"
    )
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    run = tmp_path / "run"
    with pytest.raises(HarnessError) as exc:
        cmd_finalize(package, raw, receipt_path, run)
    assert exc.value.layer == "evidence_incomplete"
    assert "request_sha256" in str(exc.value)
    assert not run.exists()
    assert _staging_leftovers(tmp_path) == []
