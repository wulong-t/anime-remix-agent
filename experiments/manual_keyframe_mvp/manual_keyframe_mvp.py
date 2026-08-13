#!/usr/bin/env python
"""G1-MK1-L: local deterministic harness for the manual keyframe MVP.

Subcommands:
  inspect   -- structural/media-technical checks, hashes and a pending
              approval template; never auto-approves.
  package   -- re-verifies request/provenance/images/inspection/approval with
              exact hashes and atomically publishes a self-contained package
              (inputs, request, provenance, inspection, approval,
              sampling_contract.json, anisora_input.txt).
  finalize  -- verifies the remote sampling receipt and raw, normalizes to the
              frozen 121-frame 1280x720 24fps contract, builds the one-item
              Timeline 1.9 clip item, renders with the existing renderer, and
              atomically publishes the run directory.

Boundary: this Worker's implementation and tests never read repository real
media (PNG/MP4/audio), never call a model, never use the network, and never
emit absolute paths, keys or environment values in outputs. When a user runs
inspect/package/finalize, the harness reads exactly the files the user
explicitly passes and authorizes; it never auto-discovers, auto-uploads or
reads unspecified media. Synthetic media is only produced by tests in pytest
temporary directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import Emotion, ShotScale, TimelineStrategy
from anime_remix.domain.models import (
    ClipAsset,
    ClipsDocument,
    RenderProfile,
    ShotRequirement,
    Timeline,
    TimelineItem,
)
from anime_remix.errors import AnimeRemixError
from anime_remix.json_io import dump_json_atomic, sha256_file
from anime_remix.workflows.render_workflow import render_timeline

REQUEST_SCHEMA = "g1-mk1-request-v1"
APPROVAL_SCHEMA = "g1-mk1-approval-v1"
INSPECTION_SCHEMA = "g1-mk1-inspection-v1"
PACKAGE_MANIFEST_SCHEMA = "g1-mk1-package-v1"
SAMPLING_CONTRACT_SCHEMA = "g1-mk1-sampling-contract-v1"
RECEIPT_SCHEMA = "g1-mk1-sampling-receipt-v1"
GENERATION_MANIFEST_SCHEMA = "g1-mk1-generation-manifest-v1"

TARGET_FRAMES = 121
TOTAL_AUDIO_SAMPLES = TARGET_FRAMES * 48000 // 24

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")
_WINDOWS_UNC = re.compile(r"^\\\\")
_URL_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_ANISORA_SEPARATORS = ("@@", "&&")
_EMOTIONS = {member.value for member in Emotion}
_SHOT_SCALES = {member.value for member in ShotScale}
_REQUEST_FIELDS = {
    "schema_version",
    "request_id",
    "start_keyframe",
    "end_keyframe",
    "start_provenance",
    "end_provenance",
    "start_sha256",
    "end_sha256",
    "subject_description",
    "scene_description",
    "action",
    "start_state",
    "end_state",
    "emotion",
    "shot_scale",
    "camera",
    "duration_seconds",
    "aspect_ratio",
}

MAX_PNG_BYTES = 25 * 1024 * 1024
MIN_CANVAS = 512
MAX_CANVAS = 4096
ASPECT_RATIO = 16.0 / 9.0
ASPECT_TOLERANCE = 0.005
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

PACKAGE_MEMBERS = (
    "request.json",
    "inputs/k0.png",
    "inputs/k_end.png",
    "inputs/k0.provenance.json",
    "inputs/k_end.provenance.json",
    "inspection.json",
    "approval.json",
    "sampling_contract.json",
    "anisora_input.txt",
)

PACKAGE_MANIFEST_FIELDS = {
    "schema_version",
    "request_id",
    "request_sha256",
    "start_sha256",
    "end_sha256",
    "inspection_sha256",
    "approval_sha256",
    "sampling_contract_sha256",
    "created_at",
    "inputs",
    "files",
}
PACKAGE_FILE_RECORD_FIELDS = {"sha256", "size_bytes"}

PROVENANCE_FIELDS = {
    "asset",
    "sha256",
    "creation_method",
    "external_inputs",
    "named_references",
    "rights_basis",
    "public_demo_allowed",
    "notes",
}
NAMED_REFERENCE_FIELDS = {"artists", "studios", "series", "characters"}
APPROVAL_FIELDS = {
    "schema_version",
    "request_id",
    "request_sha256",
    "start_sha256",
    "end_sha256",
    "rights",
    "visual_review",
    "approved_at",
}
APPROVAL_RIGHTS_FIELDS = {
    "start_owned_or_authorized",
    "end_owned_or_authorized",
    "no_prohibited_copyrighted_character",
    "public_demo_allowed",
}
APPROVAL_VISUAL_FIELDS = {
    "identity",
    "endpoint_pose",
    "body_camera_background",
    "style",
    "artifact",
    "accept_borderline",
    "overall",
}

FROZEN_SAMPLING: dict[str, Any] = {
    "provider": "IndexTeam / official Index-AniSora",
    "model": "AniSora V3.1 (Wan 14B)",
    "task": "i2v-14B",
    "dtype": "bfloat16 runtime shim",
    "size_argument": "1280*720",
    "observed_raw_canvas": "1280x704",
    "guide_positions": [0, 1],
    "raw_frame_count": 81,
    "raw_fps": 16,
    "seed": 4096,
    "sample_steps": 40,
    "sample_shift": 5,
    "sample_guide_scale": 5,
    "offload_model": True,
    "aesthetic_score": 5.5,
    "motion_score": 3.0,
    "valid_content_samples": 1,
}

SAMPLE_STEPS_DEFAULT = 40
SAMPLE_STEPS_MIN = 1
SAMPLE_STEPS_MAX = 100

RECEIPT_FIELDS = {
    "schema_version",
    "request_id",
    "request_sha256",
    "package_manifest_sha256",
    "sampling_contract_sha256",
    "start_sha256",
    "end_sha256",
    "raw_sha256",
    "status",
    *FROZEN_SAMPLING.keys(),
}

PROMPT_TEMPLATE = (
    "{subject_description}, {scene_description}. The camera is completely still "
    "and fixed. The character performs one simple action: {action}. The video "
    "starts at the start keyframe with state: {start_state}, and ends exactly at "
    "the end keyframe with state: {end_state}. Emotion: {emotion}; shot scale: "
    "{shot_scale}. Stop at the end keyframe pose; do not overshoot, do not turn "
    "to profile or back of head, do not continue rotating. Preserve the exact "
    "same character identity, hairstyle, hair color, face, clothing, colors, "
    "body, composition, background and clean 2D cel-animation style from start "
    "to end. No camera movement, no added characters, objects, text, logos or "
    "watermarks. aesthetic score: 5.5. motion score: 3.0. There is no text in "
    "the video."
)

TECHNICAL_RETRY_RULE = (
    "A technical failure before any valid content sample exists may be repaired "
    "and rerun with identical frozen content parameters; every attempt must be "
    "logged. No retry is permitted after one valid content sample exists."
)
CONTENT_RETRY_RULE = (
    "Never retry, change the prompt, change the seed, change sampling "
    "parameters, or change the model because the valid sample looks bad."
)


class HarnessError(Exception):
    """Harness error classified into the G1-MK1 failure taxonomy."""

    def __init__(self, layer: str, message: str) -> None:
        self.layer = layer
        super().__init__(f"{layer}: {message}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_sample_steps(
    value: Any, layer: str = "input_contract"
) -> int:
    """Strictly validate ``sample_steps`` as a real integer 1..100.

    ``bool`` is never accepted as an integer, and zero/negative/>100 are
    rejected at the validation layer before any runner invocation or output.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessError(
            layer,
            f"sample_steps must be a real integer 1..100, got "
            f"{type(value).__name__}",
        )
    if not (SAMPLE_STEPS_MIN <= value <= SAMPLE_STEPS_MAX):
        raise HarnessError(
            layer,
            f"sample_steps must be a real integer 1..100, got {value}",
        )
    return value


def _strict_eq(a: Any, b: Any) -> bool:
    """Recursive strict JSON type/value comparison (no bool==int coercion)."""

    if type(a) is not type(b):
        return False
    if isinstance(a, bool):
        return a is b
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_strict_eq(a[key], b[key]) for key in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_strict_eq(x, y) for x, y in zip(a, b))
    return a == b


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name == "nt":
        try:
            st = path.lstat()
        except OSError:
            return False
        return bool(
            getattr(st, "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    return False


def _reject_link_or_reparse(path: Path, field: str) -> None:
    if _is_link_or_reparse(path):
        raise HarnessError(
            "evidence_incomplete",
            f"{field} must not be a symlink/junction/reparse point: {path}",
        )


def _capture_bytes(path: Path, layer: str = "input_contract") -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise HarnessError(layer, f"cannot read {path}: {exc}") from exc


def _load_json_bytes(
    data_bytes: bytes, what: str, layer: str = "input_contract"
) -> Any:
    try:
        return json.loads(data_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HarnessError(layer, f"invalid JSON in {what}: {exc}") from exc


def _require_object(data: Any, what: str, layer: str = "input_contract") -> dict:
    if not isinstance(data, dict):
        raise HarnessError(layer, f"{what} must be a JSON object")
    return data


def _reject_unsafe_relative(
    raw: str, field: str, layer: str = "input_contract"
) -> None:
    if not raw:
        raise HarnessError(layer, f"{field} must not be empty")
    if _URL_PREFIX.match(raw) or "://" in raw:
        raise HarnessError(layer, f"{field} must be a relative path")
    if (
        Path(raw).is_absolute()
        or _WINDOWS_DRIVE.match(raw)
        or _WINDOWS_UNC.match(raw)
    ):
        raise HarnessError(layer, f"{field} must be a relative path")
    if any(part == ".." for part in Path(raw).parts):
        raise HarnessError(layer, f"{field} must not traverse parents")
    name = Path(raw).name.upper()
    if name in _DEVICE_NAMES or re.fullmatch(r"CON[0-9]|COM[0-9]|LPT[0-9]", name):
        raise HarnessError(layer, f"{field} is a device path")


def _validate_canonical_relative(
    raw: str, field: str, layer: str = "input_contract"
) -> None:
    if not raw:
        raise HarnessError(layer, f"{field} must not be empty")
    if "\\" in raw:
        raise HarnessError(
            layer,
            f"{field} must be a canonical forward-slash relative path",
        )
    _reject_unsafe_relative(raw, field, layer)
    if any(part == "." for part in Path(raw).parts):
        raise HarnessError(layer, f"{field} must not contain '.' components")


def _reject_symlink_components(
    base: Path, raw: str, field: str, layer: str = "input_contract"
) -> None:
    current = base
    for part in Path(raw).parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise HarnessError(
                layer,
                f"{field} contains a symlink/reparse component: {raw}",
            )


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _resolve_relative(base: Path, raw: str, field: str) -> Path:
    _reject_unsafe_relative(raw, field)
    _reject_symlink_components(base, raw, field)
    try:
        resolved = (base / raw).resolve(strict=True)
    except OSError as exc:
        raise HarnessError(
            "input_contract",
            f"{field} does not exist or is unreadable: {raw}",
        ) from exc
    if not _is_within(resolved, base):
        raise HarnessError(
            "input_contract",
            f"{field} escapes the request directory: {raw}",
        )
    if not resolved.is_file():
        raise HarnessError(
            "input_contract",
            f"{field} is not a regular file: {raw}",
        )
    return resolved


def _validate_text(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise HarnessError("input_contract", f"{field} must be a string")
    stripped = value.strip()
    if not stripped:
        raise HarnessError("input_contract", f"{field} must be non-empty")
    if CONTROL_RE.search(value):
        raise HarnessError(
            "input_contract",
            f"{field} must not contain control characters",
        )
    if "@@" in value or "&&" in value:
        raise HarnessError(
            "input_contract",
            f"{field} must not contain AniSora separators @@ or &&",
        )


# Strict RFC 3339 timestamp with mandatory offset:
#   YYYY-MM-DDTHH:MM:SS[.fraction](Z|±HH:MM)
# The regex pins the exact grammar (including offset hour 00-23 and minute
# 00-59) so datetime is used only to verify the values are a real date/time.
_RFC3339_OFFSET_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(\.\d+)?"
    r"(Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)


def _parse_rfc3339(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not _RFC3339_OFFSET_RE.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_request_bytes(
    data_bytes: bytes, request_path: Path
) -> dict[str, Any]:
    base = request_path.resolve().parent
    request_sha256 = _sha256_bytes(data_bytes)
    data = _require_object(_load_json_bytes(data_bytes, "request"), "request")
    problems: list[str] = []
    missing = sorted(_REQUEST_FIELDS - set(data))
    unknown = sorted(set(data) - _REQUEST_FIELDS)
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown fields: {', '.join(unknown)}")
    if data.get("schema_version") != REQUEST_SCHEMA:
        problems.append(f"schema_version must be {REQUEST_SCHEMA}")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
        problems.append("request_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
    for field in (
        "start_keyframe",
        "end_keyframe",
        "start_provenance",
        "end_provenance",
    ):
        if field not in data:
            problems.append(f"{field} is missing")
    if problems:
        raise HarnessError("input_contract", "; ".join(problems))

    start_path = _resolve_relative(base, data["start_keyframe"], "start_keyframe")
    end_path = _resolve_relative(base, data["end_keyframe"], "end_keyframe")
    start_prov = _resolve_relative(
        base, data["start_provenance"], "start_provenance"
    )
    end_prov = _resolve_relative(base, data["end_provenance"], "end_provenance")

    start_sha256 = data.get("start_sha256")
    end_sha256 = data.get("end_sha256")
    if not isinstance(start_sha256, str) or not SHA256_RE.fullmatch(start_sha256):
        problems.append("start_sha256 must be 64 lowercase hex characters")
    if not isinstance(end_sha256, str) or not SHA256_RE.fullmatch(end_sha256):
        problems.append("end_sha256 must be 64 lowercase hex characters")
    if (
        isinstance(start_sha256, str)
        and isinstance(end_sha256, str)
        and start_sha256 == end_sha256
    ):
        problems.append("start and end keyframe SHA256 must differ")
    for field in (
        "subject_description",
        "scene_description",
        "action",
        "start_state",
        "end_state",
    ):
        try:
            _validate_text(data.get(field), field)
        except HarnessError as exc:
            problems.append(str(exc))
    emotion = data.get("emotion")
    if emotion is not None and (
        not isinstance(emotion, str) or emotion not in _EMOTIONS
    ):
        problems.append(f"emotion must be null or one of {sorted(_EMOTIONS)}")
    shot_scale = data.get("shot_scale")
    if shot_scale is not None and (
        not isinstance(shot_scale, str) or shot_scale not in _SHOT_SCALES
    ):
        problems.append(
            f"shot_scale must be null or one of {sorted(_SHOT_SCALES)}"
        )
    if data.get("camera") != "fixed":
        problems.append("camera must be fixed")
    if data.get("duration_seconds") != 5 or not isinstance(
        data.get("duration_seconds"), int
    ):
        problems.append("duration_seconds must be 5")
    if data.get("aspect_ratio") != "16:9":
        problems.append("aspect_ratio must be 16:9")
    if problems:
        raise HarnessError("input_contract", "; ".join(problems))

    return {
        "base": base,
        "data": data,
        "request_id": request_id,
        "request_sha256": request_sha256,
        "start_path": start_path,
        "end_path": end_path,
        "start_provenance": start_prov,
        "end_provenance": end_prov,
        "start_sha256": start_sha256,
        "end_sha256": end_sha256,
    }


def _verify_guide_snapshot_hashes(
    info: dict[str, Any], start_bytes: bytes, end_bytes: bytes
) -> None:
    """Bind the request-declared guide hashes to the captured keyframe bytes."""

    problems: list[str] = []
    if _sha256_bytes(start_bytes) != info["start_sha256"]:
        problems.append(
            "start_sha256 does not match the captured start keyframe bytes"
        )
    if _sha256_bytes(end_bytes) != info["end_sha256"]:
        problems.append(
            "end_sha256 does not match the captured end keyframe bytes"
        )
    if problems:
        raise HarnessError("input_contract", "; ".join(problems))


def _bounded_zlib_decompress(
    compressed: bytes, expected: int
) -> tuple[bytes | None, str | None]:
    """Incrementally inflate IDAT with a hard output cap of expected+1 bytes."""

    if not compressed:
        return None, "IDAT payload is empty"
    decomp = zlib.decompressobj()
    out = bytearray()
    try:
        out += decomp.decompress(compressed, expected + 1)
        while decomp.unconsumed_tail and len(out) <= expected:
            out += decomp.decompress(
                decomp.unconsumed_tail, expected + 1 - len(out)
            )
        if len(out) <= expected:
            out += decomp.flush(expected + 1 - len(out))
    except zlib.error as exc:
        return None, f"IDAT is not decodable: {exc}"
    if len(out) > expected:
        return None, "IDAT decompresses beyond the expected IHDR payload size"
    if not decomp.eof:
        return None, "IDAT zlib stream is truncated (missing end marker)"
    if decomp.unused_data:
        return None, (
            "IDAT contains trailing compressed data after the zlib stream"
        )
    if len(out) != expected:
        return None, "IDAT payload length does not match IHDR"
    return bytes(out), None


def _png_details_bytes(data: bytes) -> dict[str, Any]:
    problems: list[str] = []
    size_bytes = len(data)
    if size_bytes > MAX_PNG_BYTES:
        problems.append(f"PNG exceeds {MAX_PNG_BYTES} bytes")
    info: dict[str, Any] = {"size_bytes": size_bytes, "problems": problems}
    if not data.startswith(_PNG_SIGNATURE):
        problems.append("not a PNG file")
        return info
    offset = len(_PNG_SIGNATURE)
    ihdr: dict[str, Any] | None = None
    idat_chunks: list[bytes] = []
    first = True
    seen_iend = False
    while offset < len(data):
        head = data[offset : offset + 8]
        if len(head) != 8:
            problems.append("truncated PNG chunk header")
            break
        length = int.from_bytes(head[:4], "big")
        ctype = head[4:8].decode("latin-1")
        offset += 8
        chunk_data = data[offset : offset + length]
        if len(chunk_data) != length:
            problems.append("truncated PNG chunk data")
            break
        offset += length
        crc = data[offset : offset + 4]
        if len(crc) != 4:
            problems.append("missing PNG chunk CRC")
            break
        offset += 4
        if zlib.crc32(head[4:8] + chunk_data) & 0xFFFFFFFF != int.from_bytes(
            crc, "big"
        ):
            problems.append(f"PNG chunk CRC mismatch: {ctype}")
        if first:
            if ctype != "IHDR":
                problems.append("first PNG chunk must be IHDR")
                break
            first = False
        if ctype == "IHDR":
            if ihdr is not None:
                problems.append("duplicate IHDR chunk")
            elif len(chunk_data) != 13:
                problems.append("invalid IHDR length")
            else:
                width = int.from_bytes(chunk_data[0:4], "big")
                height = int.from_bytes(chunk_data[4:8], "big")
                bit_depth = chunk_data[8]
                color_type = chunk_data[9]
                compression = chunk_data[10]
                filter_method = chunk_data[11]
                interlace = chunk_data[12]
                ihdr = {
                    "width": width,
                    "height": height,
                    "bit_depth": bit_depth,
                    "color_type": {2: "RGB", 6: "RGBA"}.get(color_type),
                }
                if color_type not in (2, 6):
                    problems.append(
                        f"color_type must be RGB(2) or RGBA(6), got {color_type}"
                    )
                if bit_depth not in (8, 16):
                    problems.append(
                        f"bit_depth must be 8 or 16, got {bit_depth}"
                    )
                if compression != 0 or filter_method != 0:
                    problems.append("compression/filter method must be 0")
                if interlace != 0:
                    problems.append("interlaced PNG is rejected")
                if width <= 0 or height <= 0:
                    problems.append(
                        "PNG canvas must have positive dimensions"
                    )
                elif not (MIN_CANVAS <= width <= MAX_CANVAS) or not (
                    MIN_CANVAS <= height <= MAX_CANVAS
                ):
                    problems.append(
                        f"PNG canvas {width}x{height} outside "
                        f"{MIN_CANVAS}..{MAX_CANVAS}"
                    )
        elif ctype == "IDAT":
            idat_chunks.append(chunk_data)
        elif ctype == "acTL":
            problems.append("animated PNG (acTL) is rejected")
        elif ctype == "IEND":
            seen_iend = True
            if length != 0:
                problems.append("IEND chunk must be empty")
            if offset < len(data):
                problems.append("trailing data after IEND is rejected")
            break
        else:
            if ctype[0].isupper() and ctype not in {"PLTE", "tRNS"}:
                problems.append(f"unknown critical PNG chunk: {ctype}")
    if not seen_iend:
        problems.append("missing IEND chunk")
    if ihdr is not None and not problems:
        channels = 3 if ihdr["color_type"] == "RGB" else 4
        bytes_per_sample = 2 if ihdr["bit_depth"] == 16 else 1
        stride = 1 + ihdr["width"] * channels * bytes_per_sample
        expected = ihdr["height"] * stride
        raw, decode_problem = _bounded_zlib_decompress(
            b"".join(idat_chunks), expected
        )
        if decode_problem is not None:
            problems.append(decode_problem)
        else:
            bad_rows: list[int] = []
            pos = 0
            for row in range(ihdr["height"]):
                if raw[pos] > 4:
                    bad_rows.append(row)
                    if len(bad_rows) >= 10:
                        break
                pos += stride
            if bad_rows:
                problems.append(
                    "IDAT scanline filter byte must be 0..4; invalid rows: "
                    + ", ".join(str(row) for row in bad_rows)
                )
    if ihdr is not None:
        info.update(ihdr)
    return info


def _aspect_error(width: int, height: int) -> float:
    return abs((width / height) - ASPECT_RATIO) / ASPECT_RATIO


def _validate_keyframes_bytes(
    start_bytes: bytes, end_bytes: bytes
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_info = _png_details_bytes(start_bytes)
    end_info = _png_details_bytes(end_bytes)
    problems: list[str] = []
    problems.extend(f"start: {p}" for p in start_info["problems"])
    problems.extend(f"end: {p}" for p in end_info["problems"])
    for name, info in (("start", start_info), ("end", end_info)):
        width = info.get("width")
        height = info.get("height")
        if width is not None and height is not None:
            if width <= 0 or height <= 0:
                problems.append(f"{name} canvas must have positive dimensions")
                continue
            if not (MIN_CANVAS <= width <= MAX_CANVAS) or not (
                MIN_CANVAS <= height <= MAX_CANVAS
            ):
                problems.append(
                    f"{name} canvas {width}x{height} outside {MIN_CANVAS}..{MAX_CANVAS}"
                )
            if _aspect_error(width, height) > ASPECT_TOLERANCE:
                problems.append(
                    f"{name} canvas {width}x{height} deviates more than 0.5% "
                    "from 16:9"
                )
    if start_info.get("width") != end_info.get("width") or start_info.get(
        "height"
    ) != end_info.get("height"):
        problems.append("start and end keyframe pixel dimensions must be identical")
    if problems:
        raise HarnessError("input_contract", "; ".join(problems))
    return start_info, end_info


def _validate_provenance_bytes(
    data: Any, image_sha256: str, expected_asset: str
) -> dict[str, Any]:
    data = _require_object(data, "provenance", "rights_blocked")
    problems: list[str] = []
    missing = sorted(PROVENANCE_FIELDS - set(data))
    unknown = sorted(set(data) - PROVENANCE_FIELDS)
    if missing:
        problems.append(f"provenance missing keys: {', '.join(missing)}")
    if unknown:
        problems.append(f"provenance unknown keys: {', '.join(unknown)}")
    if data.get("sha256") != image_sha256:
        problems.append("provenance sha256 does not match the keyframe")
    asset = data.get("asset")
    if not isinstance(asset, str):
        problems.append("provenance asset must be a string")
    else:
        try:
            _validate_canonical_relative(asset, "provenance asset", "rights_blocked")
        except HarnessError as exc:
            problems.append(str(exc))
        else:
            if asset != expected_asset:
                problems.append(
                    "provenance asset must equal the corresponding request "
                    "keyframe path"
                )
    for field in ("creation_method", "rights_basis"):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"provenance {field} must be a non-empty string")
        elif CONTROL_RE.search(value):
            problems.append(
                f"provenance {field} must not contain control characters"
            )
    if not isinstance(data.get("public_demo_allowed"), bool):
        problems.append("provenance public_demo_allowed must be a boolean")
    notes = data.get("notes")
    if not isinstance(notes, str) or CONTROL_RE.search(notes):
        problems.append("provenance notes must be a single-line string")
    external = data.get("external_inputs")
    if not isinstance(external, list) or any(
        not isinstance(item, str) or CONTROL_RE.search(item) for item in external
    ):
        problems.append(
            "provenance external_inputs must be a list of single-line strings"
        )
    named = data.get("named_references")
    if not isinstance(named, dict):
        problems.append("provenance named_references must be an object")
    else:
        named_missing = sorted(NAMED_REFERENCE_FIELDS - set(named))
        named_unknown = sorted(set(named) - NAMED_REFERENCE_FIELDS)
        if named_missing:
            problems.append(
                f"named_references missing keys: {', '.join(named_missing)}"
            )
        if named_unknown:
            problems.append(
                f"named_references unknown keys: {', '.join(named_unknown)}"
            )
        for key in sorted(NAMED_REFERENCE_FIELDS):
            value = named.get(key)
            if not isinstance(value, list) or any(
                not isinstance(item, str) or CONTROL_RE.search(item)
                for item in value
            ):
                problems.append(
                    f"named_references.{key} must be a list of single-line strings"
                )
    if problems:
        raise HarnessError("rights_blocked", "; ".join(problems))
    return {"sha256": image_sha256, "problems": []}


def _first_gate_problems(data: dict[str, Any]) -> list[str]:
    """Run-scoped first formal gate problems for the declared hashes.

    The first formal gate is no longer bound to a frozen media identity or a
    forbidden SHA: it becomes active for any request whose exact start/end
    keyframe paths and declared SHA256 values match the inspected bytes and
    whose images satisfy the existing image contract.  Byte/image binding is
    enforced before this state is computed (``_verify_guide_snapshot_hashes``
    and ``_validate_keyframes_bytes``); the only gate-level checks left here
    are that the declared hashes are well-formed and distinct.
    """

    problems: list[str] = []
    for field in ("start_sha256", "end_sha256"):
        value = data.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            problems.append(f"{field} must be 64 lowercase hex characters")
    start_sha256 = data.get("start_sha256")
    end_sha256 = data.get("end_sha256")
    if (
        isinstance(start_sha256, str)
        and isinstance(end_sha256, str)
        and start_sha256 == end_sha256
    ):
        problems.append("start and end keyframe SHA256 must differ")
    return problems


def _first_gate_state(
    data: dict[str, Any], start_info: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Compute the run-scoped first formal gate state.

    ``active`` is true for any request whose exact start/end paths and
    declared SHA256 values match the inspected bytes and whose images satisfy
    the existing PNG/canvas contract.  Callers run those checks before
    invoking this helper, so a valid inspection/package is always active; the
    result no longer depends on any hard-coded K0/K_end media identity.
    """

    problems = list(_first_gate_problems(data))
    if not all(
        start_info.get(key) is not None
        for key in ("width", "height", "color_type")
    ):
        problems.append(
            "start keyframe does not satisfy the PNG/canvas image contract"
        )
    return not problems, problems


def _build_prompt(request: dict[str, Any]) -> str:
    emotion = request.get("emotion") or "not specified"
    shot_scale = request.get("shot_scale") or "not specified"
    return PROMPT_TEMPLATE.format(
        subject_description=request["subject_description"].strip(),
        scene_description=request["scene_description"].strip(),
        action=request["action"].strip(),
        start_state=request["start_state"].strip(),
        end_state=request["end_state"].strip(),
        emotion=emotion,
        shot_scale=shot_scale,
    )


def _inspection_record(
    info: dict[str, Any],
    request_name: str,
    start_info: dict[str, Any],
    end_info: dict[str, Any],
    first_gate_active: bool,
    first_gate_problems: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": INSPECTION_SCHEMA,
        "request_id": info["request_id"],
        "request_file": request_name,
        "request_sha256": info["request_sha256"],
        "checked_at": _utc_now(),
        "status": "pending",
        "request_problems": [],
        "images": {
            "start_keyframe": {
                "path": info["data"]["start_keyframe"],
                "sha256": info["start_sha256"],
                "width": start_info["width"],
                "height": start_info["height"],
                "bit_depth": start_info["bit_depth"],
                "color_type": start_info["color_type"],
                "size_bytes": start_info["size_bytes"],
                "aspect_error_ratio": round(
                    _aspect_error(start_info["width"], start_info["height"]), 6
                ),
                "decodable": True,
                "problems": [],
            },
            "end_keyframe": {
                "path": info["data"]["end_keyframe"],
                "sha256": info["end_sha256"],
                "width": end_info["width"],
                "height": end_info["height"],
                "bit_depth": end_info["bit_depth"],
                "color_type": end_info["color_type"],
                "size_bytes": end_info["size_bytes"],
                "aspect_error_ratio": round(
                    _aspect_error(end_info["width"], end_info["height"]), 6
                ),
                "decodable": True,
                "problems": [],
            },
        },
        "provenance": {
            "start": {
                "path": info["data"]["start_provenance"],
                "sha256": info["start_sha256"],
                "problems": [],
            },
            "end": {
                "path": info["data"]["end_provenance"],
                "sha256": info["end_sha256"],
                "problems": [],
            },
        },
        "first_formal_gate": {
            "active": first_gate_active,
            "problems": first_gate_problems,
        },
        "pending_approval_template": "approval.json",
    }


def _pending_approval(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": APPROVAL_SCHEMA,
        "request_id": info["request_id"],
        "request_sha256": info["request_sha256"],
        "start_sha256": info["start_sha256"],
        "end_sha256": info["end_sha256"],
        "rights": {
            "start_owned_or_authorized": False,
            "end_owned_or_authorized": False,
            "no_prohibited_copyrighted_character": False,
            "public_demo_allowed": False,
        },
        "visual_review": {
            "identity": None,
            "endpoint_pose": None,
            "body_camera_background": None,
            "style": None,
            "artifact": None,
            "accept_borderline": None,
            "overall": "pending",
        },
        "approved_at": None,
    }


def cmd_inspect(request: Path, output: Path) -> Path:
    """Run structural/media-technical checks and write a pending approval."""

    request = Path(request)
    output = Path(output)
    if output.exists():
        raise HarnessError(
            "input_contract",
            f"inspection output already exists: {output}",
        )
    request_bytes = _capture_bytes(request)
    info = _validate_request_bytes(request_bytes, request)
    start_bytes = _capture_bytes(info["start_path"])
    end_bytes = _capture_bytes(info["end_path"])
    _verify_guide_snapshot_hashes(info, start_bytes, end_bytes)
    start_info, end_info = _validate_keyframes_bytes(start_bytes, end_bytes)
    first_gate_active, first_gate_problems = _first_gate_state(
        info["data"], start_info
    )
    if first_gate_problems:
        raise HarnessError("input_contract", "; ".join(first_gate_problems))
    start_prov_bytes = _capture_bytes(
        info["start_provenance"], "rights_blocked"
    )
    end_prov_bytes = _capture_bytes(info["end_provenance"], "rights_blocked")
    _validate_provenance_bytes(
        _load_json_bytes(start_prov_bytes, "provenance", "rights_blocked"),
        info["start_sha256"],
        info["data"]["start_keyframe"],
    )
    _validate_provenance_bytes(
        _load_json_bytes(end_prov_bytes, "provenance", "rights_blocked"),
        info["end_sha256"],
        info["data"]["end_keyframe"],
    )

    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent))
    )
    try:
        dump_json_atomic(
            staging / "inspection.json",
            _inspection_record(
                info,
                request.name,
                start_info,
                end_info,
                first_gate_active,
                first_gate_problems,
            ),
        )
        dump_json_atomic(staging / "approval.json", _pending_approval(info))
        _publish_dir(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _inspection_expected_record(
    info: dict[str, Any],
    request_name: str,
    start_bytes: bytes,
    end_bytes: bytes,
) -> dict[str, Any]:
    start_info, end_info = _validate_keyframes_bytes(start_bytes, end_bytes)
    first_gate_active, first_gate_problems = _first_gate_state(
        info["data"], start_info
    )
    return _inspection_record(
        info,
        request_name,
        start_info,
        end_info,
        first_gate_active,
        first_gate_problems,
    )


def _validate_inspection(
    data: Any,
    info: dict[str, Any],
    start_bytes: bytes,
    end_bytes: bytes,
    request_name: str,
) -> None:
    """Strictly validate inspection evidence against the captured snapshot.

    Every field except ``checked_at`` must equal the expected record rebuilt
    from the captured request and keyframe bytes; ``checked_at`` is validated
    separately as a single-line RFC 3339 timestamp with offset.
    """

    data = _require_object(data, "inspection", "approval_blocked")
    expected = _inspection_expected_record(
        info, request_name, start_bytes, end_bytes
    )
    problems: list[str] = []
    missing = sorted(set(expected) - set(data))
    unknown = sorted(set(data) - set(expected))
    if missing:
        problems.append(f"inspection missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"inspection unknown fields: {', '.join(unknown)}")
    for key, expected_value in expected.items():
        if key == "checked_at":
            continue
        if key not in data:
            continue
        if not _strict_eq(data[key], expected_value):
            if key == "images":
                problems.append(
                    "inspection images do not match the captured PNG evidence"
                )
            elif key == "provenance":
                problems.append(
                    "inspection provenance entries do not match the request "
                    "paths and keyframe hashes"
                )
            elif key == "first_formal_gate":
                problems.append(
                    "inspection first_formal_gate does not match the "
                    "recomputed gate"
                )
            else:
                problems.append(f"inspection {key} mismatch")
    checked_at = data.get("checked_at")
    if (
        not isinstance(checked_at, str)
        or CONTROL_RE.search(checked_at)
        or not _parse_rfc3339(checked_at)
    ):
        problems.append(
            "inspection checked_at must be a single-line RFC 3339 timestamp "
            "with offset"
        )
    if problems:
        raise HarnessError("approval_blocked", "; ".join(problems))


def _validate_approval(
    data: Any,
    request_id: str,
    request_sha256: str,
    start_sha256: str,
    end_sha256: str,
) -> None:
    data = _require_object(data, "approval", "approval_blocked")
    problems: list[str] = []
    missing = sorted(APPROVAL_FIELDS - set(data))
    unknown = sorted(set(data) - APPROVAL_FIELDS)
    if missing:
        problems.append(f"approval missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"approval unknown fields: {', '.join(unknown)}")
    if data.get("schema_version") != APPROVAL_SCHEMA:
        problems.append(f"approval schema_version must be {APPROVAL_SCHEMA}")
    if data.get("request_id") != request_id:
        problems.append("approval request_id mismatch")
    if data.get("request_sha256") != request_sha256:
        problems.append("approval request_sha256 mismatch")
    if data.get("start_sha256") != start_sha256:
        problems.append("approval start_sha256 mismatch")
    if data.get("end_sha256") != end_sha256:
        problems.append("approval end_sha256 mismatch")
    rights = data.get("rights")
    if not isinstance(rights, dict):
        problems.append("rights must be an object")
    else:
        rights_missing = sorted(APPROVAL_RIGHTS_FIELDS - set(rights))
        rights_unknown = sorted(set(rights) - APPROVAL_RIGHTS_FIELDS)
        if rights_missing:
            problems.append(
                f"rights missing fields: {', '.join(rights_missing)}"
            )
        if rights_unknown:
            problems.append(
                f"rights unknown fields: {', '.join(rights_unknown)}"
            )
        for key in (
            "start_owned_or_authorized",
            "end_owned_or_authorized",
            "no_prohibited_copyrighted_character",
        ):
            if rights.get(key) is not True:
                problems.append(f"rights.{key} must be true")
        if (
            rights.get("public_demo_allowed") is not True
            and rights.get("public_demo_allowed") is not False
        ):
            problems.append("rights.public_demo_allowed must be a boolean")
    visual = data.get("visual_review")
    if not isinstance(visual, dict):
        problems.append("visual_review must be an object")
    else:
        visual_missing = sorted(APPROVAL_VISUAL_FIELDS - set(visual))
        visual_unknown = sorted(set(visual) - APPROVAL_VISUAL_FIELDS)
        if visual_missing:
            problems.append(
                f"visual_review missing fields: {', '.join(visual_missing)}"
            )
        if visual_unknown:
            problems.append(
                f"visual_review unknown fields: {', '.join(visual_unknown)}"
            )
        for key in (
            "identity",
            "endpoint_pose",
            "body_camera_background",
            "style",
            "artifact",
        ):
            value = visual.get(key)
            if value not in ("pass", "borderline", "fail"):
                problems.append(f"visual_review.{key} must be pass|borderline|fail")
            elif value == "fail":
                problems.append(f"visual_review.{key}=fail is rejected")
            elif value == "borderline" and visual.get("accept_borderline") is not True:
                problems.append(
                    f"visual_review.{key}=borderline requires accept_borderline=true"
                )
        if (
            visual.get("accept_borderline") is not True
            and visual.get("accept_borderline") is not False
        ):
            problems.append("visual_review.accept_borderline must be a boolean")
        if visual.get("overall") != "approved":
            problems.append("visual_review.overall must be approved")
    approved_at = data.get("approved_at")
    if not isinstance(approved_at, str) or not _parse_rfc3339(approved_at):
        problems.append("approved_at must be an RFC 3339 timestamp with offset")
    if problems:
        raise HarnessError("approval_blocked", "; ".join(problems))


def _sampling_contract(
    info: dict[str, Any],
    first_gate_active: bool,
    first_gate_problems: list[str],
    sample_steps: int = SAMPLE_STEPS_DEFAULT,
) -> dict[str, Any]:
    prompt = _build_prompt(info["data"])
    input_line = f"{prompt}@@inputs/k0.png,inputs/k_end.png&&0,1"
    frozen_parameters = dict(FROZEN_SAMPLING)
    frozen_parameters["sample_steps"] = _validate_sample_steps(sample_steps)
    return {
        "schema_version": SAMPLING_CONTRACT_SCHEMA,
        "request_id": info["request_id"],
        "request_sha256": info["request_sha256"],
        "start_sha256": info["start_sha256"],
        "end_sha256": info["end_sha256"],
        "frozen_parameters": frozen_parameters,
        "guide_files": ["inputs/k0.png", "inputs/k_end.png"],
        "guide_sha256": {
            "inputs/k0.png": info["start_sha256"],
            "inputs/k_end.png": info["end_sha256"],
        },
        "guide_positions": [0, 1],
        "prompt_template": PROMPT_TEMPLATE,
        "resolved_prompt": prompt,
        "input_line": input_line,
        "technical_retry_rule": TECHNICAL_RETRY_RULE,
        "content_retry_rule": CONTENT_RETRY_RULE,
        "first_formal_gate": {
            "active": first_gate_active,
            "problems": list(first_gate_problems),
        },
    }


def _write_staged_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def cmd_package(
    request: Path,
    inspection: Path,
    approval: Path,
    output: Path,
    sample_steps: int = SAMPLE_STEPS_DEFAULT,
) -> Path:
    """Re-verify the full approval chain and atomically publish a package."""

    sample_steps = _validate_sample_steps(sample_steps, "input_contract")
    request = Path(request)
    inspection = Path(inspection)
    approval = Path(approval)
    output = Path(output)
    if output.exists():
        raise HarnessError(
            "input_contract",
            f"package output already exists: {output}",
        )
    request_bytes = _capture_bytes(request)
    info = _validate_request_bytes(request_bytes, request)
    start_bytes = _capture_bytes(info["start_path"])
    end_bytes = _capture_bytes(info["end_path"])
    _verify_guide_snapshot_hashes(info, start_bytes, end_bytes)
    start_info, _end_info = _validate_keyframes_bytes(start_bytes, end_bytes)
    first_gate_active, first_gate_problems = _first_gate_state(
        info["data"], start_info
    )
    if first_gate_problems:
        raise HarnessError("input_contract", "; ".join(first_gate_problems))
    start_prov_bytes = _capture_bytes(
        info["start_provenance"], "rights_blocked"
    )
    end_prov_bytes = _capture_bytes(info["end_provenance"], "rights_blocked")
    inspection_bytes = _capture_bytes(inspection, "approval_blocked")
    approval_bytes = _capture_bytes(approval, "approval_blocked")
    _validate_inspection(
        _load_json_bytes(inspection_bytes, "inspection", "approval_blocked"),
        info,
        start_bytes,
        end_bytes,
        request.name,
    )
    _validate_approval(
        _load_json_bytes(approval_bytes, "approval", "approval_blocked"),
        info["request_id"],
        info["request_sha256"],
        info["start_sha256"],
        info["end_sha256"],
    )
    _validate_provenance_bytes(
        _load_json_bytes(
            start_prov_bytes, "provenance", "rights_blocked"
        ),
        info["start_sha256"],
        info["data"]["start_keyframe"],
    )
    _validate_provenance_bytes(
        _load_json_bytes(
            end_prov_bytes, "provenance", "rights_blocked"
        ),
        info["end_sha256"],
        info["data"]["end_keyframe"],
    )

    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent))
    )
    try:
        staged: list[tuple[str, bytes]] = [
            ("request.json", request_bytes),
            ("inputs/k0.png", start_bytes),
            ("inputs/k_end.png", end_bytes),
            ("inputs/k0.provenance.json", start_prov_bytes),
            ("inputs/k_end.provenance.json", end_prov_bytes),
            ("inspection.json", inspection_bytes),
            ("approval.json", approval_bytes),
        ]
        for relative, data_bytes in staged:
            _write_staged_bytes(staging / relative, data_bytes)
        for relative, data_bytes in staged:
            if sha256_file(staging / relative) != _sha256_bytes(data_bytes):
                raise HarnessError(
                    "evidence_incomplete",
                    f"TOCTOU: {relative} changed during packaging",
                )
        contract = _sampling_contract(
            info,
            first_gate_active,
            first_gate_problems,
            sample_steps,
        )
        dump_json_atomic(staging / "sampling_contract.json", contract)
        (staging / "anisora_input.txt").write_bytes(
            (contract["input_line"] + "\n").encode("utf-8")
        )
        files: dict[str, Any] = {}
        for relative in PACKAGE_MEMBERS:
            path = staging / relative
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        manifest = {
            "schema_version": PACKAGE_MANIFEST_SCHEMA,
            "request_id": info["request_id"],
            "request_sha256": info["request_sha256"],
            "start_sha256": info["start_sha256"],
            "end_sha256": info["end_sha256"],
            "inspection_sha256": _sha256_bytes(inspection_bytes),
            "approval_sha256": _sha256_bytes(approval_bytes),
            "sampling_contract_sha256": sha256_file(
                staging / "sampling_contract.json"
            ),
            "created_at": _utc_now(),
            "inputs": {
                "start_keyframe": "inputs/k0.png",
                "end_keyframe": "inputs/k_end.png",
            },
            "files": files,
        }
        dump_json_atomic(staging / "package_manifest.json", manifest)
        _publish_dir(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _publish_dir(staging: Path, output: Path) -> None:
    if output.exists():
        raise HarnessError("input_contract", f"output already exists: {output}")
    try:
        os.replace(staging, output)
    except OSError:
        try:
            os.rename(staging, output)
        except OSError as exc:
            raise HarnessError(
                "evidence_incomplete",
                f"atomic publication failed: {exc}",
            ) from exc


def _validate_package_path_within(base: Path, raw: str, field: str) -> Path:
    _validate_canonical_relative(raw, field, "evidence_incomplete")
    _reject_symlink_components(
        base, raw, field, layer="evidence_incomplete"
    )
    try:
        resolved = (base / raw).resolve(strict=True)
    except OSError as exc:
        raise HarnessError(
            "evidence_incomplete",
            f"{field} missing or unreadable: {raw}",
        ) from exc
    if not _is_within(resolved, base):
        raise HarnessError(
            "evidence_incomplete",
            f"{field} escapes the package directory: {raw}",
        )
    return resolved


def _validate_package_manifest(
    manifest: Any, package_dir: Path
) -> dict[str, Any]:
    package_dir = Path(package_dir)
    _reject_link_or_reparse(package_dir, "package root")
    root = package_dir.resolve()
    manifest = _require_object(
        manifest, "package_manifest", "evidence_incomplete"
    )
    problems: list[str] = []
    missing = sorted(PACKAGE_MANIFEST_FIELDS - set(manifest))
    unknown = sorted(set(manifest) - PACKAGE_MANIFEST_FIELDS)
    if missing:
        problems.append(
            f"package_manifest missing fields: {', '.join(missing)}"
        )
    if unknown:
        problems.append(
            f"package_manifest unknown fields: {', '.join(unknown)}"
        )
    if manifest.get("schema_version") != PACKAGE_MANIFEST_SCHEMA:
        problems.append(
            f"package_manifest schema_version must be {PACKAGE_MANIFEST_SCHEMA}"
        )
    created_at = manifest.get("created_at")
    if (
        not isinstance(created_at, str)
        or CONTROL_RE.search(created_at)
        or not _parse_rfc3339(created_at)
    ):
        problems.append(
            "package_manifest created_at must be a single-line RFC 3339 "
            "timestamp with offset"
        )
    request_id = manifest.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(
        request_id
    ):
        problems.append(
            "package_manifest request_id must match "
            "[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
        )
    for key in (
        "request_sha256",
        "start_sha256",
        "end_sha256",
        "inspection_sha256",
        "approval_sha256",
        "sampling_contract_sha256",
    ):
        value = manifest.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            problems.append(
                f"package_manifest {key} must be 64 lowercase hex characters"
            )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        problems.append("package_manifest inputs must be an object")
    elif set(inputs) != {"start_keyframe", "end_keyframe"}:
        problems.append(
            "package_manifest inputs must contain exactly start_keyframe and "
            "end_keyframe"
        )
    else:
        if inputs.get("start_keyframe") != "inputs/k0.png":
            problems.append(
                "package_manifest inputs.start_keyframe must be inputs/k0.png"
            )
        if inputs.get("end_keyframe") != "inputs/k_end.png":
            problems.append(
                "package_manifest inputs.end_keyframe must be inputs/k_end.png"
            )
    files = manifest.get("files")
    if not isinstance(files, dict):
        problems.append("package_manifest files must be an object")
        raise HarnessError("evidence_incomplete", "; ".join(problems))
    keys = set(files)
    expected = set(PACKAGE_MEMBERS)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        problems.append(
            "package members must be exactly "
            f"{len(PACKAGE_MEMBERS)}: missing={missing} extra={extra}"
        )
    for relative, record in files.items():
        if not isinstance(record, dict):
            problems.append(f"package_manifest file record {relative} invalid")
            continue
        record_missing = sorted(PACKAGE_FILE_RECORD_FIELDS - set(record))
        record_unknown = sorted(set(record) - PACKAGE_FILE_RECORD_FIELDS)
        if record_missing:
            problems.append(
                f"package_manifest files.{relative} missing fields: "
                f"{', '.join(record_missing)}"
            )
        if record_unknown:
            problems.append(
                f"package_manifest files.{relative} unknown fields: "
                f"{', '.join(record_unknown)}"
            )
        sha = record.get("sha256")
        if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
            problems.append(
                f"package_manifest files.{relative} sha256 must be 64 "
                "lowercase hex characters"
            )
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            problems.append(
                f"package_manifest files.{relative} size_bytes must be a "
                "non-negative integer"
            )
        try:
            member = _validate_package_path_within(
                root, relative, f"package member {relative}"
            )
        except HarnessError as exc:
            problems.append(str(exc))
            continue
        if not member.is_file():
            problems.append(f"packaged member is not a regular file: {relative}")
    manifest_path = root / "package_manifest.json"
    if _is_link_or_reparse(manifest_path) or not manifest_path.is_file():
        problems.append(
            "package_manifest.json must be a regular file at the package root"
        )
    if "package_manifest.json" in files:
        problems.append("package_manifest.json must not be self-referenced")
    if problems:
        raise HarnessError("evidence_incomplete", "; ".join(problems))
    return manifest


def _capture_package_members(package_root: Path) -> dict[str, bytes]:
    """Capture every package member exactly once (no later disk re-reads)."""

    members: dict[str, bytes] = {}
    for relative in PACKAGE_MEMBERS:
        _reject_symlink_components(
            package_root,
            relative,
            f"package member {relative}",
            layer="evidence_incomplete",
        )
        members[relative] = _capture_bytes(
            package_root / relative, "evidence_incomplete"
        )
    return members


def _validate_package_manifest_bindings(
    manifest: dict[str, Any], members: dict[str, bytes]
) -> None:
    """Bind the captured manifest header and file records to member snapshots."""

    problems: list[str] = []
    header_members = {
        "request_sha256": "request.json",
        "start_sha256": "inputs/k0.png",
        "end_sha256": "inputs/k_end.png",
        "inspection_sha256": "inspection.json",
        "approval_sha256": "approval.json",
        "sampling_contract_sha256": "sampling_contract.json",
    }
    for key, member_name in header_members.items():
        if _sha256_bytes(members[member_name]) != manifest.get(key):
            problems.append(
                f"package_manifest {key} does not match packaged member "
                f"{member_name}"
            )
    for relative in PACKAGE_MEMBERS:
        record = manifest["files"][relative]
        if _sha256_bytes(members[relative]) != record["sha256"]:
            problems.append(f"packaged file hash mismatch: {relative}")
        if len(members[relative]) != record["size_bytes"]:
            problems.append(f"packaged file size mismatch: {relative}")
    if problems:
        raise HarnessError("evidence_incomplete", "; ".join(problems))


def _validate_sampling_contract(
    data: Any,
    info: dict[str, Any],
    first_gate_active: bool,
    first_gate_problems: list[str],
) -> dict[str, Any]:
    data = _require_object(data, "sampling_contract", "evidence_incomplete")
    frozen_parameters = data.get("frozen_parameters")
    if not isinstance(frozen_parameters, dict) or "sample_steps" not in frozen_parameters:
        raise HarnessError(
            "evidence_incomplete",
            "sampling_contract frozen_parameters.sample_steps must exist",
        )
    sample_steps = _validate_sample_steps(
        frozen_parameters["sample_steps"], "evidence_incomplete"
    )
    expected = _sampling_contract(
        info,
        first_gate_active,
        first_gate_problems,
        sample_steps,
    )
    problems: list[str] = []
    missing = sorted(set(expected) - set(data))
    unknown = sorted(set(data) - set(expected))
    if missing:
        problems.append(
            f"sampling_contract missing fields: {', '.join(missing)}"
        )
    if unknown:
        problems.append(
            f"sampling_contract unknown fields: {', '.join(unknown)}"
        )
    for key, expected_value in expected.items():
        if not _strict_eq(data.get(key), expected_value):
            problems.append(f"sampling_contract {key} mismatch")
    if problems:
        raise HarnessError("evidence_incomplete", "; ".join(problems))
    return data


def _validate_anisora_input_bytes(
    contract: dict[str, Any], input_bytes: bytes
) -> None:
    expected = (contract["input_line"] + "\n").encode("utf-8")
    if input_bytes != expected:
        raise HarnessError(
            "evidence_incomplete",
            "anisora_input.txt must equal input_line plus a single trailing "
            "newline",
        )


def _validate_receipt(
    data: Any,
    request_id: str,
    request_sha256: str,
    package_manifest_sha256: str,
    sampling_contract_sha256: str,
    start_sha256: str,
    end_sha256: str,
    sample_steps: int | None = None,
) -> dict[str, Any]:
    data = _require_object(data, "remote receipt", "evidence_incomplete")
    problems: list[str] = []
    missing = sorted(RECEIPT_FIELDS - set(data))
    unknown = sorted(set(data) - RECEIPT_FIELDS)
    if missing:
        problems.append(f"remote receipt missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"remote receipt unknown fields: {', '.join(unknown)}")
    if data.get("schema_version") != RECEIPT_SCHEMA:
        problems.append(f"remote receipt schema_version must be {RECEIPT_SCHEMA}")
    if data.get("request_id") != request_id:
        problems.append("remote receipt request_id mismatch")
    if data.get("request_sha256") != request_sha256:
        problems.append("remote receipt request_sha256 mismatch")
    if data.get("package_manifest_sha256") != package_manifest_sha256:
        problems.append("remote receipt package_manifest_sha256 mismatch")
    if data.get("sampling_contract_sha256") != sampling_contract_sha256:
        problems.append("remote receipt sampling_contract_sha256 mismatch")
    if data.get("start_sha256") != start_sha256:
        problems.append("remote receipt start_sha256 mismatch")
    if data.get("end_sha256") != end_sha256:
        problems.append("remote receipt end_sha256 mismatch")
    if data.get("status") != "success":
        problems.append("remote receipt status must be success")
    raw_sha256 = data.get("raw_sha256")
    if not isinstance(raw_sha256, str) or not SHA256_RE.fullmatch(raw_sha256):
        problems.append("remote receipt raw_sha256 must be 64 lowercase hex")
    receipt_steps = data.get("sample_steps")
    try:
        _validate_sample_steps(receipt_steps, "evidence_incomplete")
    except HarnessError as exc:
        problems.append(str(exc))
    expected_frozen = dict(FROZEN_SAMPLING)
    if sample_steps is not None:
        expected_frozen["sample_steps"] = _validate_sample_steps(
            sample_steps, "evidence_incomplete"
        )
    for key, expected in expected_frozen.items():
        if key == "offload_model":
            if data.get(key) is not True:
                problems.append(f"remote receipt {key} must be true")
        elif not _strict_eq(data.get(key), expected):
            problems.append(
                f"remote receipt {key} mismatch (frozen contract)"
            )
    if problems:
        raise HarnessError("evidence_incomplete", "; ".join(problems))
    return data


def _probe_summary(toolkit: FFmpegToolkit, path: Path) -> dict[str, Any]:
    info = toolkit._probe(path, count_frames=True)
    streams = info.get("streams") or []
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    video = videos[0] if videos else {}
    audio = audios[0] if audios else {}
    frames = video.get("nb_read_frames") or video.get("nb_frames")
    try:
        frame_count = int(frames) if frames is not None else None
    except (TypeError, ValueError):
        frame_count = None
    format_duration = (info.get("format") or {}).get("duration")
    return {
        "video_streams": len(videos),
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "r_frame_rate": video.get("r_frame_rate"),
        "avg_frame_rate": video.get("avg_frame_rate"),
        "nb_read_frames": frame_count,
        "pix_fmt": video.get("pix_fmt"),
        "sample_aspect_ratio": video.get("sample_aspect_ratio"),
        "color_range": video.get("color_range"),
        "color_space": video.get("color_space"),
        "color_transfer": video.get("color_transfer"),
        "color_primaries": video.get("color_primaries"),
        "field_order": video.get("field_order"),
        "chroma_location": video.get("chroma_location"),
        "audio_streams": len(audios),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": audio.get("sample_rate"),
        "audio_channels": audio.get("channels"),
        "audio_time_base": audio.get("time_base"),
        "audio_duration_ts": audio.get("duration_ts"),
        "duration_seconds": format_duration,
    }


def _preflight_raw(
    toolkit: FFmpegToolkit, raw: Path, expected_sha256: str
) -> dict[str, Any]:
    if raw.is_symlink() or not raw.is_file():
        raise HarnessError("media_normalization", "raw must be a regular file")
    if sha256_file(raw) != expected_sha256:
        raise HarnessError(
            "media_normalization",
            "raw SHA256 does not match the remote sampling receipt",
        )
    try:
        summary = _probe_summary(toolkit, raw)
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc
    problems: list[str] = []
    if summary["video_streams"] != 1:
        problems.append(f"raw video_streams={summary['video_streams']}")
    if summary["video_codec"] != "h264":
        problems.append(f"raw codec={summary['video_codec']}")
    if (summary["width"], summary["height"]) != (1280, 704):
        problems.append(f"raw size={summary['width']}x{summary['height']}")
    if summary["r_frame_rate"] != "16/1" or summary["avg_frame_rate"] != "16/1":
        problems.append(
            f"raw fps r={summary['r_frame_rate']} avg={summary['avg_frame_rate']}"
        )
    if summary["nb_read_frames"] != 81:
        problems.append(f"raw nb_read_frames={summary['nb_read_frames']}")
    if summary["audio_streams"] != 0:
        problems.append(f"raw audio_streams={summary['audio_streams']}")
    if problems:
        raise HarnessError("media_normalization", "; ".join(problems))
    return summary


def _decode_check_raw(toolkit: FFmpegToolkit, raw: Path) -> None:
    try:
        toolkit.verify_video_decodable(raw)
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc


def _final_audio_evidence(summary: dict[str, Any]) -> int:
    problems: list[str] = []
    if summary.get("audio_streams") != 1:
        problems.append(f"audio_streams={summary.get('audio_streams')}")
    if summary.get("audio_time_base") != "1/48000":
        problems.append(f"audio_time_base={summary.get('audio_time_base')}")
    duration_ts = summary.get("audio_duration_ts")
    try:
        measured = int(duration_ts)
    except (TypeError, ValueError):
        measured = -1
        problems.append(f"audio_duration_ts={duration_ts!r}")
    if measured != TOTAL_AUDIO_SAMPLES:
        problems.append(
            f"audio_duration_ts={duration_ts!r} expected={TOTAL_AUDIO_SAMPLES}"
        )
    if problems:
        raise HarnessError(
            "renderer_interface", "final audio gate: " + "; ".join(problems)
        )
    return measured


def _timeline_dict(request: dict[str, Any], generated: Path) -> dict[str, Any]:
    emotion = (
        None if request.get("emotion") is None else Emotion(request["emotion"])
    )
    shot_scale = (
        None
        if request.get("shot_scale") is None
        else ShotScale(request["shot_scale"])
    )
    source_text = (
        f"Manual-keyframe request {request['request_id']}: "
        f"{request['subject_description']} | {request['scene_description']} | "
        f"action: {request['action']} | {request['start_state']} -> "
        f"{request['end_state']}"
    )
    requirement = ShotRequirement(
        id="g1mk1-shot",
        order=1,
        source_text=source_text,
        characters=[],
        location_id=None,
        location_name=None,
        action=request["action"],
        target_frames=TARGET_FRAMES,
        dialogue=None,
        emotion=emotion,
        shot_scale=shot_scale,
    )
    item = TimelineItem(
        shot_id="g1mk1-shot",
        order=1,
        requirement=requirement,
        strategy=TimelineStrategy.CLIP,
        source_asset_id=request["request_id"],
        source_path="generated_clip.mp4",
        source_size_bytes=generated.stat().st_size,
        source_sha256=sha256_file(generated),
        source_in_frame=0,
        source_frame_count=TARGET_FRAMES,
        target_frames=TARGET_FRAMES,
        score=None,
        reason_code="exact_length",
        reason="generated manual-keyframe clip with exact target length",
    )
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )
    return json.loads(timeline.model_dump_json())


def generated_clip_description(request: dict[str, Any]) -> str:
    """Deterministic clips.json description built only from request fields."""

    return (
        f"Generated manual-keyframe clip: {request['subject_description']} | "
        f"{request['scene_description']} | action: {request['action']} | "
        f"{request['start_state']} -> {request['end_state']}"
    )


def _clips_document_dict(request: dict[str, Any]) -> dict[str, Any]:
    """Build the strict existing-schema one-asset ClipsDocument (schema 1.9).

    The asset ID is the already-validated ``request_id`` (the existing
    request regex is identical to the domain ``ID_PATTERN``), the path is the
    exact normalized ``generated_clip.mp4``, characters stay empty, location
    fields stay null, and action/emotion/shot_scale/description map only from
    request data without inventing identity or location metadata.
    """

    emotion = (
        None if request.get("emotion") is None else Emotion(request["emotion"])
    )
    shot_scale = (
        None
        if request.get("shot_scale") is None
        else ShotScale(request["shot_scale"])
    )
    asset = ClipAsset(
        id=request["request_id"],
        path="generated_clip.mp4",
        characters=[],
        location_id=None,
        location_name=None,
        action=request["action"],
        description=generated_clip_description(request),
        emotion=emotion,
        shot_scale=shot_scale,
    )
    document = ClipsDocument(schema_version="1.9", clips=[asset])
    return json.loads(document.model_dump_json())


def _map_anime_error(layer: str, exc: AnimeRemixError) -> HarnessError:
    return HarnessError(layer, str(exc))


def cmd_finalize(
    package: Path,
    raw: Path,
    remote_receipt: Path,
    output: Path,
) -> Path:
    """Verify package/receipt/raw, normalize, render and publish a run dir."""

    package = Path(package)
    raw = Path(raw)
    remote_receipt = Path(remote_receipt)
    output = Path(output)
    if output.exists():
        raise HarnessError("input_contract", f"run output already exists: {output}")

    _reject_link_or_reparse(package, "package root")
    package_root = package.resolve()
    manifest_bytes = _capture_bytes(
        package_root / "package_manifest.json", "evidence_incomplete"
    )
    manifest = _validate_package_manifest(
        _load_json_bytes(
            manifest_bytes, "package_manifest", "evidence_incomplete"
        ),
        package_root,
    )
    package_manifest_sha256 = _sha256_bytes(manifest_bytes)

    members = _capture_package_members(package_root)
    _validate_package_manifest_bindings(manifest, members)

    request_bytes = members["request.json"]
    info = _validate_request_bytes(
        request_bytes, package_root / "request.json"
    )
    if info["request_id"] != manifest["request_id"]:
        raise HarnessError(
            "evidence_incomplete",
            "package_manifest request_id does not match packaged request",
        )
    if info["request_sha256"] != manifest["request_sha256"]:
        raise HarnessError(
            "evidence_incomplete",
            "package_manifest request_sha256 does not match packaged request",
        )
    _verify_guide_snapshot_hashes(
        info, members["inputs/k0.png"], members["inputs/k_end.png"]
    )
    if (
        info["start_sha256"] != manifest["start_sha256"]
        or info["end_sha256"] != manifest["end_sha256"]
    ):
        raise HarnessError(
            "evidence_incomplete",
            "package_manifest guide hashes do not match packaged request",
        )

    inspection_bytes = members["inspection.json"]
    inspection_data = _load_json_bytes(
        inspection_bytes, "inspection", "approval_blocked"
    )
    _validate_inspection(
        inspection_data,
        info,
        members["inputs/k0.png"],
        members["inputs/k_end.png"],
        "request.json",
    )
    gate_active = inspection_data["first_formal_gate"]["active"]
    gate_problems = inspection_data["first_formal_gate"]["problems"]

    approval_bytes = members["approval.json"]
    _validate_approval(
        _load_json_bytes(approval_bytes, "approval", "approval_blocked"),
        info["request_id"],
        info["request_sha256"],
        info["start_sha256"],
        info["end_sha256"],
    )

    contract_bytes = members["sampling_contract.json"]
    contract = _validate_sampling_contract(
        _load_json_bytes(
            contract_bytes, "sampling_contract", "evidence_incomplete"
        ),
        info,
        gate_active,
        gate_problems,
    )
    sampling_contract_sha256 = _sha256_bytes(contract_bytes)
    _validate_anisora_input_bytes(contract, members["anisora_input.txt"])

    receipt_bytes = _capture_bytes(remote_receipt, "evidence_incomplete")
    receipt = _validate_receipt(
        _load_json_bytes(receipt_bytes, "remote receipt", "evidence_incomplete"),
        info["request_id"],
        info["request_sha256"],
        package_manifest_sha256,
        sampling_contract_sha256,
        info["start_sha256"],
        info["end_sha256"],
        contract["frozen_parameters"]["sample_steps"],
    )

    raw_bytes = _capture_bytes(raw, "media_normalization")
    if _sha256_bytes(raw_bytes) != receipt["raw_sha256"]:
        raise HarnessError(
            "media_normalization",
            "raw SHA256 does not match the remote sampling receipt",
        )

    toolkit = FFmpegToolkit()
    try:
        toolkit.check_capabilities()
    except AnimeRemixError as exc:
        raise _map_anime_error("media_normalization", exc) from exc

    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        fd, probe_name = tempfile.mkstemp(
            prefix=".g1mk1-raw-", suffix=".mp4", dir=str(parent)
        )
        probe_path = Path(probe_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        raw_summary = _preflight_raw(toolkit, probe_path, receipt["raw_sha256"])
        _decode_check_raw(toolkit, probe_path)
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except OSError:
                pass

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent))
    )
    try:
        _write_staged_bytes(staging / "raw_shot.mp4", raw_bytes)
        if sha256_file(staging / "raw_shot.mp4") != receipt["raw_sha256"]:
            raise HarnessError(
                "evidence_incomplete",
                "TOCTOU: raw changed while copying into the run directory",
            )
        generated = staging / "generated_clip.mp4"
        try:
            toolkit.normalize_generated_source(
                staging / "raw_shot.mp4",
                generated,
                target_frames=TARGET_FRAMES,
            )
        except AnimeRemixError as exc:
            raise _map_anime_error("media_normalization", exc) from exc
        normalized_summary = _probe_summary(toolkit, generated)

        timeline_path = staging / "timeline.json"
        dump_json_atomic(timeline_path, _timeline_dict(info["data"], generated))
        clips_path = staging / "clips.json"
        dump_json_atomic(clips_path, _clips_document_dict(info["data"]))
        (staging / ".anime-remix-run").write_bytes(b"{}\n")
        try:
            render_timeline(
                timeline_path,
                staging / "output.mp4",
                allow_managed_output=True,
                log_path=staging / "render.log",
                toolkit=toolkit,
            )
        except AnimeRemixError as exc:
            raise _map_anime_error("renderer_interface", exc) from exc

        staged_text: list[tuple[str, bytes]] = [
            ("request.json", request_bytes),
            ("approval.json", approval_bytes),
            ("package_manifest.json", manifest_bytes),
            ("remote_receipt.json", receipt_bytes),
        ]
        for name, data_bytes in staged_text:
            _write_staged_bytes(staging / name, data_bytes)
        for name, data_bytes in staged_text:
            if sha256_file(staging / name) != _sha256_bytes(data_bytes):
                raise HarnessError(
                    "evidence_incomplete",
                    f"TOCTOU: {name} changed while copying into the run directory",
                )

        final_summary = _probe_summary(toolkit, staging / "output.mp4")
        measured_audio_samples = _final_audio_evidence(final_summary)
        final_summary["total_audio_samples"] = measured_audio_samples
        generation_manifest = {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "request_id": info["request_id"],
            "request_sha256": info["request_sha256"],
            "approval_sha256": _sha256_bytes(approval_bytes),
            "package": {
                "path": "package_manifest.json",
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "remote_receipt": {
                "path": "remote_receipt.json",
                "sha256": _sha256_bytes(receipt_bytes),
            },
            "raw": {
                "path": "raw_shot.mp4",
                "sha256": sha256_file(staging / "raw_shot.mp4"),
                "probe": raw_summary,
            },
            "normalized": {
                "path": "generated_clip.mp4",
                "sha256": sha256_file(staging / "generated_clip.mp4"),
                "probe": normalized_summary,
            },
            "timeline": {
                "path": "timeline.json",
                "sha256": sha256_file(staging / "timeline.json"),
            },
            "clips": {
                "path": "clips.json",
                "sha256": sha256_file(staging / "clips.json"),
            },
            "render_log": {
                "path": "render.log",
                "sha256": sha256_file(staging / "render.log"),
            },
            "final": {
                "path": "output.mp4",
                "sha256": sha256_file(staging / "output.mp4"),
                "probe": final_summary,
            },
            "model_params": dict(
                FROZEN_SAMPLING,
                sample_steps=contract["frozen_parameters"]["sample_steps"],
            ),
            "total_audio_samples": measured_audio_samples,
            "failure_layer": None,
            "created_at": _utc_now(),
        }
        dump_json_atomic(
            staging / "generation_manifest.json", generation_manifest
        )
        _publish_dir(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manual_keyframe_mvp",
        description="G1-MK1-L manual keyframe MVP local harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate request/keyframes/provenance and emit pending approval",
    )
    inspect_parser.add_argument("--request", required=True, type=Path)
    inspect_parser.add_argument("--output", required=True, type=Path)

    package_parser = subparsers.add_parser(
        "package", help="re-verify approval chain and publish a remote package"
    )
    package_parser.add_argument("--request", required=True, type=Path)
    package_parser.add_argument("--inspection", required=True, type=Path)
    package_parser.add_argument("--approval", required=True, type=Path)
    package_parser.add_argument("--output", required=True, type=Path)
    package_parser.add_argument(
        "--sample-steps",
        type=int,
        default=SAMPLE_STEPS_DEFAULT,
        metavar="INTEGER",
        help=(
            f"sampling steps recorded in the packaged contract "
            f"(default {SAMPLE_STEPS_DEFAULT}; integer 1..100)"
        ),
    )

    finalize_parser = subparsers.add_parser(
        "finalize", help="normalize raw, build Timeline 1.9 and render"
    )
    finalize_parser.add_argument("--package", required=True, type=Path)
    finalize_parser.add_argument("--raw", required=True, type=Path)
    finalize_parser.add_argument("--remote-receipt", required=True, type=Path)
    finalize_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            cmd_inspect(args.request, args.output)
        elif args.command == "package":
            cmd_package(
                args.request,
                args.inspection,
                args.approval,
                args.output,
                args.sample_steps,
            )
        elif args.command == "finalize":
            cmd_finalize(args.package, args.raw, args.remote_receipt, args.output)
        else:
            raise HarnessError("input_contract", f"unknown command: {args.command}")
    except HarnessError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
