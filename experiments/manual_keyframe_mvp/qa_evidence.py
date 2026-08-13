#!/usr/bin/env python
"""G1-MK1-R-PREP-L: evidence-only endpoint QA for the manual keyframe MVP.

``qa_evidence`` revalidates the exact package, raw sample and finalized run,
produces each endpoint reference canvas through FFmpeg using the same frozen
finalized-media transform semantics (scale-decrease + pad + BT.709 limited
yuv420p), compares them with ``generated_clip.mp4`` frames 0 and 120 in RGB,
and atomically publishes ``metrics.json``, a raw contact sheet, an endpoint
comparison sheet and ``artifacts.json``. All probes, frame extraction and
metrics consume only captured bytes staged from the validated package/raw/
finalized run; live paths are never reopened after the validated capture.

Boundary: metrics are evidence only and never decide PASS/FAIL; no capability
verdict is produced. No network, no new dependency, no directory discovery,
and no environment-variable values are emitted.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import ValidationError

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import Emotion, ShotScale
from anime_remix.domain.models import ClipsDocument
from anime_remix.errors import AnimeRemixError
from experiments.manual_keyframe_mvp import (
    manual_keyframe_mvp as harness,
)
from experiments.manual_keyframe_mvp.remote_sample import (
    validate_package,
)

QA_EVIDENCE_SCHEMA = "g1-mk1-qa-evidence-v1"
QA_ARTIFACTS_SCHEMA = "g1-mk1-qa-artifacts-v1"

RAW_CONTACT_SHEET_FRAMES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
ENDPOINT_ORDER = ["k0", "k_end", "raw_start", "raw_end"]
ENDPOINT_FRAME_INDICES = [0, 120]
CANVAS_SIZE = (1280, 720)
RAW_SIZE = (1280, 704)
CONTACT_CELL_SIZE = (424, 232)
RAW_SHEET_SIZE = (3 * 424, 3 * 232)  # 3x3 grid
ENDPOINT_SHEET_SIZE = (2 * 1280, 2 * 720)  # 2x2 grid
MAX_RGB = 255

_FINALIZED_REFERENCE_FILTER = (
    "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
    "setsar=1,"
    "format=yuv420p,"
    "setparams=range=limited:color_primaries=bt709:color_trc=bt709:"
    "colorspace=bt709:field_mode=prog"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _scale_fit(
    source_w: int, source_h: int, target_w: int, target_h: int
) -> tuple[int, int]:
    scale = min(target_w / source_w, target_h / source_h)
    scaled_w = max(1, int(source_w * scale))
    scaled_h = max(1, int(source_h * scale))
    scaled_w -= scaled_w % 2
    scaled_h -= scaled_h % 2
    if scaled_w < 2 or scaled_h < 2:
        raise harness.HarnessError(
            "media_normalization",
            f"cannot scale {source_w}x{source_h} onto {target_w}x{target_h}",
        )
    return scaled_w, scaled_h


def _nearest_resize(
    rgb: bytes, source_w: int, source_h: int, target_w: int, target_h: int
) -> bytes:
    if (source_w, source_h) == (target_w, target_h):
        return rgb
    out = bytearray(target_w * target_h * 3)
    for y in range(target_h):
        source_y = y * source_h // target_h
        source_row = source_y * source_w * 3
        target_row = y * target_w * 3
        for x in range(target_w):
            source_x = x * source_w // target_w
            source_index = source_row + source_x * 3
            target_index = target_row + x * 3
            out[target_index] = rgb[source_index]
            out[target_index + 1] = rgb[source_index + 1]
            out[target_index + 2] = rgb[source_index + 2]
    return bytes(out)


def _pad_canvas(
    rgb: bytes, scaled_w: int, scaled_h: int, target_w: int, target_h: int
) -> bytes:
    if scaled_w > target_w or scaled_h > target_h:
        raise harness.HarnessError(
            "media_normalization",
            f"scaled {scaled_w}x{scaled_h} exceeds {target_w}x{target_h}",
        )
    out = bytearray(target_w * target_h * 3)
    x0 = (target_w - scaled_w) // 2
    y0 = (target_h - scaled_h) // 2
    for y in range(scaled_h):
        source = y * scaled_w * 3
        target = (y0 + y) * target_w * 3 + x0 * 3
        out[target : target + scaled_w * 3] = rgb[
            source : source + scaled_w * 3
        ]
    return bytes(out)


def _to_canvas(
    source_w: int,
    source_h: int,
    rgb: bytes,
    target_w: int | None = None,
    target_h: int | None = None,
) -> bytes:
    if target_w is None or target_h is None:
        target_w, target_h = CANVAS_SIZE
    scaled_w, scaled_h = _scale_fit(source_w, source_h, target_w, target_h)
    resized = _nearest_resize(
        rgb, source_w, source_h, scaled_w, scaled_h
    )
    return _pad_canvas(resized, scaled_w, scaled_h, target_w, target_h)


def _png_chunk(ctype: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + ctype
        + data
        + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != width * height * 3:
        raise harness.HarnessError(
            "evidence_incomplete",
            f"PNG payload {len(rgb)} does not match {width}x{height}",
        )
    raw = bytearray()
    row = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * row : (y + 1) * row])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _ffmpeg_reference_canvas(
    toolkit: FFmpegToolkit, input_args: list[str]
) -> bytes:
    """Run the frozen finalized transform and extract one RGB24 frame (R2).

    The reference canvas goes through the exact finalized-media filter chain
    and is then converted to RGB24 by FFmpeg, so references and decoded
    ``generated_clip.mp4`` frames share the same byte-domain conversion path.
    """

    args = [
        toolkit.ffmpeg or "ffmpeg",
        "-v",
        "error",
        *input_args,
        "-vf",
        _FINALIZED_REFERENCE_FILTER,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        completed = subprocess.run(
            args, capture_output=True, check=False, timeout=180
        )
    except OSError as exc:
        raise harness.HarnessError(
            "media_normalization",
            f"reference canvas failed to start: {exc}",
        ) from exc
    if completed.returncode != 0:
        raise harness.HarnessError(
            "media_normalization",
            "reference canvas conversion failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:],
        )
    expected = CANVAS_SIZE[0] * CANVAS_SIZE[1] * 3
    if len(completed.stdout) != expected:
        raise harness.HarnessError(
            "media_normalization",
            f"reference canvas returned {len(completed.stdout)} bytes, "
            f"expected {expected}",
        )
    return completed.stdout


def _png_reference_canvas(
    toolkit: FFmpegToolkit, png_path: Path
) -> bytes:
    """Reference canvas for a staged, captured guide PNG (R2/R3)."""

    return _ffmpeg_reference_canvas(toolkit, ["-i", str(png_path)])


def _raw_rgb_reference_canvas(
    toolkit: FFmpegToolkit, rgb_path: Path
) -> bytes:
    """Reference canvas for a staged raw RGB24 frame (R2/R3)."""

    return _ffmpeg_reference_canvas(
        toolkit,
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{RAW_SIZE[0]}x{RAW_SIZE[1]}",
            "-i",
            str(rgb_path),
        ],
    )


def _extract_rgb_frames(
    toolkit: FFmpegToolkit,
    video: Path,
    indices: list[int],
    width: int,
    height: int,
) -> list[bytes]:
    """Extract exact frames as raw RGB24 via a deterministic select filter.

    ``select=not(mod(n,K))`` is used instead of an OR expression because
    ``+`` is treated as a filtergraph separator by newer FFmpeg builds.
    """

    if not indices:
        return []
    step = indices[1] - indices[0] if len(indices) > 1 else 1
    select_expr = f"not(mod(n\\,{step}))"
    args = [
        toolkit.ffmpeg or "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select={select_expr}",
        "-frames:v",
        str(len(indices)),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            timeout=180,
        )
    except OSError as exc:
        raise harness.HarnessError(
            "media_normalization", f"frame extraction failed to start: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise harness.HarnessError(
            "media_normalization",
            "frame extraction failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:],
        )
    frame_bytes = width * height * 3
    expected = len(indices) * frame_bytes
    if len(completed.stdout) != expected:
        raise harness.HarnessError(
            "media_normalization",
            f"frame extraction returned {len(completed.stdout)} bytes, "
            f"expected {expected}",
        )
    return [
        completed.stdout[i * frame_bytes : (i + 1) * frame_bytes]
        for i in range(len(indices))
    ]


def _rgb_metrics(a: bytes, b: bytes) -> dict[str, Any]:
    if len(a) != len(b):
        raise harness.HarnessError(
            "evidence_incomplete", "frame byte length mismatch"
        )
    total = len(a)
    mae = 0.0
    mse = 0.0
    for x, y in zip(a, b):
        difference = abs(x - y)
        mae += difference
        mse += difference * difference
    mae /= total
    mse /= total
    if mse == 0.0:
        psnr: float | str = "infinite"
    else:
        psnr = round(
            10.0 * math.log10((MAX_RGB * MAX_RGB) / mse), 6
        )
    return {
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "psnr": psnr,
    }


def _validate_generated_clip(
    toolkit: FFmpegToolkit, path: Path
) -> dict[str, Any]:
    """Enforce the full finalized generated-media contract (R7).

    One H.264 video stream, zero audio, 1280x720, exactly 121 counted
    frames, 24/1 CFR, yuv420p, SAR 1:1, BT.709 limited, progressive,
    chroma-left, plus a full decode check. Reuses the existing toolkit
    ``validate_segment`` finalized contract.
    """

    try:
        summary = harness._probe_summary(toolkit, path)
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc
    problems: list[str] = []
    if summary["video_streams"] != 1:
        problems.append(f"video_streams={summary['video_streams']}")
    if summary["video_codec"] != "h264":
        problems.append(f"codec={summary['video_codec']}")
    if (summary["width"], summary["height"]) != CANVAS_SIZE:
        problems.append(f"size={summary['width']}x{summary['height']}")
    if summary["nb_read_frames"] != 121:
        problems.append(f"nb_read_frames={summary['nb_read_frames']}")
    if summary["r_frame_rate"] != "24/1" or summary["avg_frame_rate"] != "24/1":
        problems.append(
            f"fps r={summary['r_frame_rate']} avg={summary['avg_frame_rate']}"
        )
    if summary["audio_streams"] != 0:
        problems.append(f"audio_streams={summary['audio_streams']}")
    if problems:
        raise harness.HarnessError(
            "media_normalization",
            "generated_clip contract: " + "; ".join(problems),
        )
    try:
        toolkit.verify_video_decodable(path)
        toolkit.validate_segment(
            path, target_frames=121, shot_id="g1mk1-generated-clip"
        )
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc
    return summary


def _validate_generation_bindings(
    gen: dict[str, Any],
    package_info: dict[str, Any],
    receipt_sha256: str,
    raw_sha256: str,
    generated_sha256: str,
    clips_sha256: str,
    timeline_sha256: str,
) -> None:
    info = package_info["info"]
    package_sha256 = package_info["package_manifest_sha256"]
    problems: list[str] = []
    if gen.get("schema_version") != harness.GENERATION_MANIFEST_SCHEMA:
        problems.append("generation_manifest schema_version mismatch")
    if gen.get("request_id") != info["request_id"]:
        problems.append("generation_manifest request_id mismatch")
    if gen.get("request_sha256") != info["request_sha256"]:
        problems.append("generation_manifest request_sha256 mismatch")
    package = gen.get("package")
    if (
        not isinstance(package, dict)
        or package.get("path") != "package_manifest.json"
        or package.get("sha256") != package_sha256
    ):
        problems.append("generation_manifest package binding mismatch")
    remote = gen.get("remote_receipt")
    if (
        not isinstance(remote, dict)
        or remote.get("path") != "remote_receipt.json"
        or remote.get("sha256") != receipt_sha256
    ):
        problems.append("generation_manifest remote_receipt binding mismatch")
    raw = gen.get("raw")
    if (
        not isinstance(raw, dict)
        or raw.get("path") != "raw_shot.mp4"
        or raw.get("sha256") != raw_sha256
    ):
        problems.append("generation_manifest raw binding mismatch")
    normalized = gen.get("normalized")
    if (
        not isinstance(normalized, dict)
        or normalized.get("path") != "generated_clip.mp4"
        or normalized.get("sha256") != generated_sha256
    ):
        problems.append("generation_manifest normalized binding mismatch")
    clips = gen.get("clips")
    if (
        not isinstance(clips, dict)
        or clips.get("path") != "clips.json"
        or clips.get("sha256") != clips_sha256
    ):
        problems.append("generation_manifest clips binding mismatch")
    timeline = gen.get("timeline")
    if (
        not isinstance(timeline, dict)
        or timeline.get("path") != "timeline.json"
        or timeline.get("sha256") != timeline_sha256
    ):
        problems.append("generation_manifest timeline binding mismatch")
    model_params = gen.get("model_params")
    contract_steps = package_info["contract"]["frozen_parameters"][
        "sample_steps"
    ]
    if not isinstance(model_params, dict):
        problems.append("generation_manifest model_params must be an object")
    else:
        steps = model_params.get("sample_steps")
        try:
            harness._validate_sample_steps(steps, "evidence_incomplete")
        except harness.HarnessError as exc:
            problems.append(
                f"generation_manifest model_params.sample_steps invalid: {exc}"
            )
        else:
            if steps != contract_steps:
                problems.append(
                    "generation_manifest model_params.sample_steps mismatch "
                    "(packaged contract)"
                )
    if problems:
        raise harness.HarnessError(
            "evidence_incomplete", "; ".join(problems)
        )


def _validate_clips_handoff(
    clips_bytes: bytes,
    timeline_bytes: bytes,
    package_info: dict[str, Any],
    generated_sha256: str,
    generated_size_bytes: int,
) -> None:
    """Validate the Generated ClipAsset handoff (G1-MK3-L).

    The exact ``clips.json`` bytes must form a strict existing ``ClipsDocument``
    with exactly one asset whose fields map only from request data, and the
    timeline must bind to that asset by id, path, and the captured
    ``generated_clip.mp4`` bytes (source SHA256 + byte length). No directory
    search: only the captured bytes are consumed.
    """

    info = package_info["info"]
    request = info["data"]
    problems: list[str] = []
    clips_data = harness._require_object(
        harness._load_json_bytes(
            clips_bytes, "clips", "evidence_incomplete"
        ),
        "clips",
        "evidence_incomplete",
    )
    try:
        document = ClipsDocument.model_validate(clips_data)
    except ValidationError as exc:
        raise harness.HarnessError(
            "evidence_incomplete",
            f"clips.json is not a strict ClipsDocument: {exc}",
        ) from exc
    if len(document.clips) != 1:
        problems.append(
            f"clips must contain exactly one asset, got {len(document.clips)}"
        )
    else:
        clip = document.clips[0]
        if clip.id != info["request_id"]:
            problems.append("clips asset id does not match request_id")
        if clip.path.as_posix() != "generated_clip.mp4":
            problems.append(
                "clips asset path must be generated_clip.mp4, got "
                f"{clip.path.as_posix()}"
            )
        if clip.characters:
            problems.append("clips asset characters must be []")
        if clip.location_id is not None or clip.location_name is not None:
            problems.append("clips asset location fields must be null")
        if clip.action != request.get("action"):
            problems.append(
                "clips asset action does not match request action"
            )
        if clip.description != harness.generated_clip_description(request):
            problems.append(
                "clips asset description is not the deterministic "
                "request-field mapping"
            )
        expected_emotion = (
            None
            if request.get("emotion") is None
            else Emotion(request["emotion"])
        )
        if clip.emotion != expected_emotion:
            problems.append(
                "clips asset emotion does not match request emotion"
            )
        expected_shot_scale = (
            None
            if request.get("shot_scale") is None
            else ShotScale(request["shot_scale"])
        )
        if clip.shot_scale != expected_shot_scale:
            problems.append(
                "clips asset shot_scale does not match request shot_scale"
            )
    timeline = harness._require_object(
        harness._load_json_bytes(
            timeline_bytes, "timeline", "evidence_incomplete"
        ),
        "timeline",
        "evidence_incomplete",
    )
    if timeline.get("schema_version") != "1.9":
        problems.append("timeline schema_version must be 1.9")
    items = timeline.get("items")
    if not isinstance(items, list) or len(items) != 1:
        problems.append("timeline must contain exactly one item")
    elif len(document.clips) == 1:
        item = items[0]
        clip = document.clips[0]
        if item.get("strategy") != "clip":
            problems.append("timeline item strategy must be clip")
        if item.get("source_asset_id") != clip.id:
            problems.append(
                "timeline source_asset_id does not match clips asset id"
            )
        if item.get("source_path") != clip.path.as_posix():
            problems.append(
                "timeline source_path does not match clips asset path"
            )
        if item.get("source_sha256") != generated_sha256:
            problems.append(
                "timeline source_sha256 does not match captured "
                "generated_clip.mp4 bytes"
            )
        if item.get("source_size_bytes") != generated_size_bytes:
            problems.append(
                "timeline source_size_bytes does not match captured "
                "generated_clip.mp4 byte length"
            )
        if item.get("source_frame_count") != harness.TARGET_FRAMES:
            problems.append("timeline source_frame_count must be 121")
        if item.get("target_frames") != harness.TARGET_FRAMES:
            problems.append("timeline target_frames must be 121")
    if problems:
        raise harness.HarnessError(
            "evidence_incomplete", "; ".join(problems)
        )


def _validate_finalized_run(
    finalized_run: Path,
    package_info: dict[str, Any],
    raw_sha256: str,
) -> dict[str, Any]:
    """Revalidate the exact finalized-run files and SHA bindings."""

    finalized = Path(finalized_run)
    if finalized.is_symlink() or not finalized.is_dir():
        raise harness.HarnessError(
            "input_contract", "finalized-run must be a regular directory"
        )
    expected_files = (
        "package_manifest.json",
        "remote_receipt.json",
        "generation_manifest.json",
        "raw_shot.mp4",
        "generated_clip.mp4",
        "clips.json",
        "timeline.json",
    )
    data: dict[str, bytes] = {}
    for name in expected_files:
        path = finalized / name
        if path.is_symlink():
            raise harness.HarnessError(
                "evidence_incomplete",
                f"finalized-run {name} must not be a symlink",
            )
        data[name] = harness._capture_bytes(path, "evidence_incomplete")
    info = package_info["info"]
    package_sha256 = package_info["package_manifest_sha256"]
    contract_sha256 = package_info["sampling_contract_sha256"]
    if harness._sha256_bytes(data["package_manifest.json"]) != package_sha256:
        raise harness.HarnessError(
            "evidence_incomplete",
            "finalized-run package_manifest.json does not match the package",
        )
    receipt = harness._validate_receipt(
        harness._load_json_bytes(
            data["remote_receipt.json"], "remote_receipt", "evidence_incomplete"
        ),
        info["request_id"],
        info["request_sha256"],
        package_sha256,
        contract_sha256,
        info["start_sha256"],
        info["end_sha256"],
        package_info["contract"]["frozen_parameters"]["sample_steps"],
    )
    if receipt["raw_sha256"] != raw_sha256:
        raise harness.HarnessError(
            "evidence_incomplete",
            "remote_receipt raw_sha256 does not match the raw file",
        )
    if harness._sha256_bytes(data["raw_shot.mp4"]) != raw_sha256:
        raise harness.HarnessError(
            "evidence_incomplete",
            "finalized-run raw_shot.mp4 does not match the raw file",
        )
    generated_sha256 = harness._sha256_bytes(data["generated_clip.mp4"])
    clips_sha256 = harness._sha256_bytes(data["clips.json"])
    timeline_sha256 = harness._sha256_bytes(data["timeline.json"])
    gen = harness._require_object(
        harness._load_json_bytes(
            data["generation_manifest.json"],
            "generation_manifest",
            "evidence_incomplete",
        ),
        "generation_manifest",
        "evidence_incomplete",
    )
    receipt_sha256 = harness._sha256_bytes(data["remote_receipt.json"])
    _validate_generation_bindings(
        gen,
        package_info,
        receipt_sha256,
        raw_sha256,
        generated_sha256,
        clips_sha256,
        timeline_sha256,
    )
    _validate_clips_handoff(
        data["clips.json"],
        data["timeline.json"],
        package_info,
        generated_sha256,
        len(data["generated_clip.mp4"]),
    )
    return {
        "receipt": receipt,
        "generation_manifest": gen,
        "generated_bytes": data["generated_clip.mp4"],
        "generated_sha256": generated_sha256,
    }


def _place_canvas(
    sheet: bytearray,
    sheet_width: int,
    canvas: bytes,
    col: int,
    row: int,
    cell_w: int,
    cell_h: int,
) -> None:
    for y in range(cell_h):
        source = y * cell_w * 3
        target = (
            (row * cell_h + y) * sheet_width * 3 + col * cell_w * 3
        )
        sheet[target : target + cell_w * 3] = canvas[
            source : source + cell_w * 3
        ]


def _raw_contact_sheet(frames: list[bytes]) -> tuple[bytes, tuple[int, int]]:
    cell_w, cell_h = CONTACT_CELL_SIZE
    sheet_w, sheet_h = RAW_SHEET_SIZE
    sheet = bytearray(sheet_w * sheet_h * 3)
    for index, frame in enumerate(frames):
        canvas = _to_canvas(
            RAW_SIZE[0], RAW_SIZE[1], frame, cell_w, cell_h
        )
        col = index % 3
        row = index // 3
        _place_canvas(sheet, sheet_w, canvas, col, row, cell_w, cell_h)
    return bytes(sheet), (sheet_w, sheet_h)


def _endpoint_comparison(canvases: list[bytes]) -> tuple[bytes, tuple[int, int]]:
    if len(canvases) != len(ENDPOINT_ORDER):
        raise harness.HarnessError(
            "evidence_incomplete",
            f"endpoint comparison needs {len(ENDPOINT_ORDER)} canvases",
        )
    cell_w, cell_h = CANVAS_SIZE
    sheet_w, sheet_h = ENDPOINT_SHEET_SIZE
    sheet = bytearray(sheet_w * sheet_h * 3)
    for index, canvas in enumerate(canvases):
        col = index % 2
        row = index // 2
        _place_canvas(sheet, sheet_w, canvas, col, row, cell_w, cell_h)
    return bytes(sheet), (sheet_w, sheet_h)


def _map_anime_error(layer: str, exc: AnimeRemixError) -> harness.HarnessError:
    return harness.HarnessError(layer, str(exc))


def cmd_qa(
    package: Path,
    raw: Path,
    finalized_run: Path,
    output: Path,
) -> Path:
    """Revalidate bindings, compute endpoint metrics and publish evidence."""

    output = Path(output)
    if output.exists():
        raise harness.HarnessError(
            "input_contract", f"qa output already exists: {output}"
        )
    package_info = validate_package(package)
    raw_bytes = harness._capture_bytes(raw, "media_normalization")
    raw_sha256 = harness._sha256_bytes(raw_bytes)
    finalized_bindings = _validate_finalized_run(
        finalized_run, package_info, raw_sha256
    )
    generated_bytes = finalized_bindings["generated_bytes"]
    generated_sha256 = finalized_bindings["generated_sha256"]
    k0_bytes = package_info["members"]["inputs/k0.png"]
    k_end_bytes = package_info["members"]["inputs/k_end.png"]
    if harness._sha256_bytes(k0_bytes) != package_info["info"]["start_sha256"]:
        raise harness.HarnessError(
            "evidence_incomplete",
            "captured K0 bytes do not match the validated package hash",
        )
    if harness._sha256_bytes(k_end_bytes) != package_info["info"]["end_sha256"]:
        raise harness.HarnessError(
            "evidence_incomplete",
            "captured K_end bytes do not match the validated package hash",
        )

    toolkit = FFmpegToolkit()
    try:
        toolkit.check_capabilities()
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc

    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.probe-staging-", dir=str(parent)
        )
    )
    try:
        probe_raw = probe_dir / "raw_probe.mp4"
        probe_raw.write_bytes(raw_bytes)
        probe_generated = probe_dir / "generated_probe.mp4"
        probe_generated.write_bytes(generated_bytes)
        probe_k0 = probe_dir / "k0.png"
        probe_k0.write_bytes(k0_bytes)
        probe_k_end = probe_dir / "k_end.png"
        probe_k_end.write_bytes(k_end_bytes)
        try:
            raw_summary = harness._preflight_raw(
                toolkit, probe_raw, raw_sha256
            )
            harness._decode_check_raw(toolkit, probe_raw)
        except (harness.HarnessError, AnimeRemixError) as exc:
            if isinstance(exc, harness.HarnessError):
                raise
            raise _map_anime_error("media_normalization", exc) from exc
        generated_summary = _validate_generated_clip(
            toolkit, probe_generated
        )
        raw_frames = _extract_rgb_frames(
            toolkit,
            probe_raw,
            RAW_CONTACT_SHEET_FRAMES,
            RAW_SIZE[0],
            RAW_SIZE[1],
        )
        generated_frames = _extract_rgb_frames(
            toolkit,
            probe_generated,
            ENDPOINT_FRAME_INDICES,
            CANVAS_SIZE[0],
            CANVAS_SIZE[1],
        )
        k0_canvas = _png_reference_canvas(toolkit, probe_k0)
        k_end_canvas = _png_reference_canvas(toolkit, probe_k_end)
        raw_start_path = probe_dir / "raw_start.rgb"
        raw_start_path.write_bytes(raw_frames[0])
        raw_end_path = probe_dir / "raw_end.rgb"
        raw_end_path.write_bytes(raw_frames[-1])
        raw_start_canvas = _raw_rgb_reference_canvas(
            toolkit, raw_start_path
        )
        raw_end_canvas = _raw_rgb_reference_canvas(
            toolkit, raw_end_path
        )
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    endpoints = {
        "k0_vs_frame0": {
            "frame_index": 0,
            "source": "inputs/k0.png",
            "canvas": list(CANVAS_SIZE),
            **_rgb_metrics(k0_canvas, generated_frames[0]),
        },
        "k_end_vs_frame120": {
            "frame_index": 120,
            "source": "inputs/k_end.png",
            "canvas": list(CANVAS_SIZE),
            **_rgb_metrics(k_end_canvas, generated_frames[1]),
        },
    }
    contact_sheet, contact_sheet_size = _raw_contact_sheet(raw_frames)
    endpoint_sheet, endpoint_sheet_size = _endpoint_comparison(
        [k0_canvas, k_end_canvas, raw_start_canvas, raw_end_canvas]
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent))
    )
    try:
        contact_path = staging / "raw_contact_sheet.png"
        endpoint_path = staging / "endpoint_comparison.png"
        _write_png(contact_path, *contact_sheet_size, contact_sheet)
        _write_png(endpoint_path, *endpoint_sheet_size, endpoint_sheet)
        metrics = {
            "schema_version": QA_EVIDENCE_SCHEMA,
            "request_id": package_info["info"]["request_id"],
            "package_manifest_sha256": package_info[
                "package_manifest_sha256"
            ],
            "raw_sha256": raw_sha256,
            "generated_clip_sha256": generated_sha256,
            "raw_probe": raw_summary,
            "generated_probe": generated_summary,
            "endpoints": endpoints,
            "contact_sheet_order": {
                "raw_contact_sheet.png": list(RAW_CONTACT_SHEET_FRAMES),
                "endpoint_comparison.png": list(ENDPOINT_ORDER),
            },
            "capability_verdict": None,
            "created_at": _utc_now(),
        }
        harness.dump_json_atomic(staging / "metrics.json", metrics)
        artifacts = {
            "schema_version": QA_ARTIFACTS_SCHEMA,
            "request_id": package_info["info"]["request_id"],
            "artifacts": {
                "raw_contact_sheet.png": {
                    "sha256": harness.sha256_file(contact_path),
                    "size_bytes": contact_path.stat().st_size,
                    "width": contact_sheet_size[0],
                    "height": contact_sheet_size[1],
                    "frames": list(RAW_CONTACT_SHEET_FRAMES),
                },
                "endpoint_comparison.png": {
                    "sha256": harness.sha256_file(endpoint_path),
                    "size_bytes": endpoint_path.stat().st_size,
                    "width": endpoint_sheet_size[0],
                    "height": endpoint_sheet_size[1],
                    "order": list(ENDPOINT_ORDER),
                },
            },
            "created_at": _utc_now(),
        }
        harness.dump_json_atomic(staging / "artifacts.json", artifacts)
        harness._publish_dir(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_evidence",
        description="G1-MK1-R evidence-only endpoint QA",
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--finalized-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        cmd_qa(args.package, args.raw, args.finalized_run, args.output)
    except harness.HarnessError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
