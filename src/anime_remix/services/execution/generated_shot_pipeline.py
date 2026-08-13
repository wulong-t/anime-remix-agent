"""Recoverable local assembly of one Image-First ``GeneratedShot``.

Qwen and Vidu remain explicit upstream steps with separate approval and
billing boundaries.  This module imports only approved artifacts, snapshots
them by content hash, normalizes each segment and assembles one standard shot.
Missing later anchors or videos pause the run without invalidating prior work.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.models import RenderProfile
from anime_remix.errors import (
    InputValidationError,
    OutputValidationError,
    SourceDriftError,
)
from anime_remix.json_io import dump_json_atomic, load_json_object, sha256_file
from anime_remix.services.execution.artifact_store import canonical_json_bytes
from anime_remix.services.script.generation_segment_plan import (
    GenerationSegmentPlanDocument,
    parse_generation_segment_plan,
)

_SCHEMA_VERSION = "generated-shot-manifest-v1"
_INPUTS_SCHEMA_VERSION = "generated-shot-inputs-v1"
_HANDOFF_SCHEMA_VERSION = "handoff-frame-compose-run-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
_FPS = 24
_MAX_SEGMENT_DURATION_SECONDS = 2.0


class SegmentVideoInput(BaseModel):
    """One approved raw video and its provider-side evidence manifest."""

    model_config = _STRICT_CONFIG
    segment_id: str
    raw_video_path: str
    provider_manifest_path: str
    human_review: Literal["approved"] = "approved"

    @field_validator("segment_id", mode="before")
    @classmethod
    def _segment_id(cls, value: object) -> object:
        return _clean_id(value, "segment_id")

    @field_validator("raw_video_path", "provider_manifest_path", mode="before")
    @classmethod
    def _path(cls, value: object, info) -> object:
        return _clean_path_text(value, info.field_name)


class GeneratedShotInputsDocument(BaseModel):
    """Progressive registration document for approved segment videos."""

    model_config = _STRICT_CONFIG
    schema_version: Literal["generated-shot-inputs-v1"] = _INPUTS_SCHEMA_VERSION
    shot_id: str
    segments: list[SegmentVideoInput]

    @field_validator("shot_id", mode="before")
    @classmethod
    def _shot_id(cls, value: object) -> object:
        return _clean_id(value, "shot_id")

    @model_validator(mode="after")
    def _unique_segments(self) -> GeneratedShotInputsDocument:
        ids = [item.segment_id for item in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment video input ids must be unique")
        return self


def parse_generated_shot_inputs(data: object) -> GeneratedShotInputsDocument:
    """Validate the strict progressive input document."""

    try:
        return TypeAdapter(GeneratedShotInputsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            "invalid generated-shot-inputs-v1 document",
            actual=str(exc),
        ) from exc


class GeneratedVideoToolkit(Protocol):
    """Narrow media interface used by the recoverable pipeline."""

    def normalize_generated_source(
        self,
        source: Path,
        output: Path,
        *,
        target_frames: int,
        profile: RenderProfile | None = None,
    ) -> None: ...

    def validate_segment(
        self,
        path: Path,
        *,
        target_frames: int,
        shot_id: str,
    ) -> dict[str, Any]: ...

    def concat_signatures_equal(
        self,
        signatures: list[dict[str, Any]],
    ) -> None: ...

    def concat_video(
        self,
        segments: list[Path],
        output: Path,
        *,
        durations: list[Decimal] | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class GeneratedShotRunResult:
    """Current durable state of one GeneratedShot pipeline run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    status: str
    next_segment_id: str | None
    completed_segment_ids: tuple[str, ...]
    shot_video_path: Path | None


def _clean_id(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not _ID_PATTERN.fullmatch(cleaned):
        raise ValueError(f"invalid {field} {value!r}")
    return cleaned


def _clean_path_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError(f"{field} must be a non-empty filesystem path")
    return cleaned


def _contract_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _target_frames(duration_seconds: float) -> int:
    exact = duration_seconds * _FPS
    frames = round(exact)
    if frames <= 0 or not math.isclose(exact, frames, abs_tol=1e-6):
        raise InputValidationError(
            "segment duration must resolve to an exact 24 fps frame count",
            actual=duration_seconds,
        )
    return frames


def _regular_file(path: Path, *, label: str, suffix: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise InputValidationError(
            f"{label} must be a regular non-symlink file",
            actual=str(candidate),
        )
    if suffix is not None and candidate.suffix.casefold() != suffix:
        raise InputValidationError(
            f"{label} must use the {suffix} extension",
            actual=candidate.suffix,
        )
    return candidate.resolve()


def _declared_path(value: str, *, base_dir: Path, label: str, suffix: str) -> Path:
    declared = Path(value)
    candidate = declared if declared.is_absolute() else base_dir / declared
    return _regular_file(candidate, label=label, suffix=suffix)


def _manifest_artifact_path(
    value: object,
    *,
    manifest_dir: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{label} has no relative artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise InputValidationError(
            f"{label} artifact path must remain inside its run directory",
            actual=value,
        )
    resolved = _regular_file(manifest_dir / relative, label=label)
    try:
        resolved.relative_to(manifest_dir.resolve())
    except ValueError as exc:
        raise InputValidationError(
            f"{label} artifact escapes its run directory",
            actual=value,
        ) from exc
    return resolved


def _fingerprint(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_expected_fingerprint(
    path: Path,
    expected: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    actual = _fingerprint(path)
    if actual["sha256"] != expected.get("sha256"):
        raise SourceDriftError(
            f"{label} SHA256 changed",
            actual=actual["sha256"],
        )
    expected_bytes = expected.get("bytes")
    if expected_bytes is not None and actual["bytes"] != expected_bytes:
        raise SourceDriftError(
            f"{label} byte size changed",
            actual=actual["bytes"],
        )
    return actual


def _run_artifact_path(root: Path, record: Mapping[str, object], *, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise OutputValidationError(f"{label} has no run-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise OutputValidationError(
            f"{label} path is not run-relative",
            actual=value,
        )
    resolved = _regular_file(root / relative, label=label)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise OutputValidationError(
            f"{label} escapes the GeneratedShot run directory",
            actual=value,
        ) from exc
    _verify_expected_fingerprint(resolved, record, label=label)
    return resolved


def _copy_snapshot(
    source: Path,
    target: Path,
    *,
    relative_path: str,
    expected: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source_fingerprint = _fingerprint(source)
    if expected is not None:
        if source_fingerprint["sha256"] != expected.get("sha256"):
            raise SourceDriftError(
                "source no longer matches its recorded SHA256",
                actual=source_fingerprint["sha256"],
            )
        expected_bytes = expected.get("bytes")
        if expected_bytes is not None and source_fingerprint["bytes"] != expected_bytes:
            raise SourceDriftError(
                "source no longer matches its recorded byte size",
                actual=source_fingerprint["bytes"],
            )
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise OutputValidationError(
                "registered snapshot target is not a regular file",
                actual=str(target),
            )
        if _fingerprint(target) != source_fingerprint:
            raise SourceDriftError(
                "registered snapshot differs from the supplied source",
                actual=str(target),
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.unlink(missing_ok=True)
        shutil.copyfile(source, temporary)
        if _fingerprint(temporary) != source_fingerprint:
            temporary.unlink(missing_ok=True)
            raise OutputValidationError("snapshot copy failed its hash check")
        os.replace(temporary, target)
    return {"path": relative_path, **source_fingerprint}


def _provider_output_sha256(document: Mapping[str, object]) -> str | None:
    execution = document.get("execution")
    if isinstance(execution, dict):
        output = execution.get("output")
        if isinstance(output, dict) and isinstance(output.get("sha256"), str):
            return output["sha256"]
    result = document.get("execution_result")
    if isinstance(result, dict) and isinstance(result.get("output_sha256"), str):
        return result["output_sha256"]
    return None


def _provider_summary(path: Path, *, raw_sha256: str) -> dict[str, object]:
    document = load_json_object(path)
    recorded_sha256 = _provider_output_sha256(document)
    if recorded_sha256 is None:
        raise InputValidationError("provider manifest must bind the raw output SHA256")
    if recorded_sha256 != raw_sha256:
        raise SourceDriftError(
            "provider manifest output SHA256 does not match the raw video",
            actual=recorded_sha256,
        )
    review = document.get("human_review")
    review_status = review.get("status") if isinstance(review, dict) else None
    if review_status not in {"accepted", "approved"}:
        raise InputValidationError(
            "provider manifest must record an accepted human review",
            actual=review_status,
        )
    return {
        "schema_version": document.get("schema_version"),
        "run_id": document.get("run_id"),
        "provider": document.get("provider"),
        "model": document.get("model"),
        "manifest_sha256": sha256_file(path),
        "output_sha256": recorded_sha256,
        "human_review": review_status,
    }


def _load_anchor_sources(
    manifest_path: Path,
    *,
    plan: GenerationSegmentPlanDocument,
) -> tuple[dict[str, object], dict[str, tuple[Path, dict[str, object]]]]:
    path = _regular_file(manifest_path, label="handoff manifest", suffix=".json")
    document = load_json_object(path)
    if document.get("schema_version") != _HANDOFF_SCHEMA_VERSION:
        raise InputValidationError("unsupported handoff-frame manifest")
    if (
        document.get("plan_id") != plan.plan_id
        or document.get("shot_id") != plan.shot_id
    ):
        raise InputValidationError(
            "handoff-frame manifest does not match the generation segment plan"
        )
    records = document.get("anchors")
    if not isinstance(records, list):
        raise InputValidationError("handoff-frame manifest has no anchor list")
    by_id = {
        item.get("anchor_id"): item
        for item in records
        if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
    }
    sources: dict[str, tuple[Path, dict[str, object]]] = {}
    for anchor in plan.anchors:
        record = by_id.get(anchor.anchor_id)
        if not isinstance(record, dict):
            raise InputValidationError(
                f"handoff-frame manifest is missing anchor {anchor.anchor_id!r}"
            )
        if record.get("status") != "completed" or record.get("approved") is not True:
            continue
        output = record.get("output")
        if not isinstance(output, dict):
            raise InputValidationError(
                f"approved anchor {anchor.anchor_id!r} has no output record"
            )
        source = _manifest_artifact_path(
            output.get("path"),
            manifest_dir=path.parent,
            label=f"anchor {anchor.anchor_id}",
        )
        actual = _verify_expected_fingerprint(
            source,
            output,
            label=f"anchor {anchor.anchor_id}",
        )
        sources[anchor.anchor_id] = (source, actual)
    return document, sources


def _new_manifest(
    *,
    run_id: str,
    plan: GenerationSegmentPlanDocument,
    contract_sha256: str,
    anchor_manifest: Mapping[str, object],
    profile: RenderProfile,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "shot_id": plan.shot_id,
        "plan_id": plan.plan_id,
        "status": "ready",
        "contract_sha256": contract_sha256,
        "policy": {
            "fps": _FPS,
            "max_segment_duration_seconds": _MAX_SEGMENT_DURATION_SECONDS,
            "video_model_calls": 0,
            "automatic_model_retries": 0,
            "progressive_registration": True,
            "render_profile": profile.model_dump(mode="json"),
        },
        "anchor_source": {
            "schema_version": anchor_manifest.get("schema_version"),
            "run_id": anchor_manifest.get("run_id"),
            "plan_id": anchor_manifest.get("plan_id"),
        },
        "anchors": [
            {
                "anchor_id": item.anchor_id,
                "order": item.order,
                "status": "awaiting_approval",
                "frame": None,
            }
            for item in plan.anchors
        ],
        "segments": [
            {
                "segment_id": item.segment_id,
                "order": item.order,
                "start_anchor_id": item.start_anchor_id,
                "end_anchor_id": item.end_anchor_id,
                "duration_seconds": item.duration_seconds,
                "target_frames": _target_frames(item.duration_seconds),
                "process_description": item.process_description,
                "dominant_motion": item.dominant_motion,
                "camera_motion": item.camera_motion,
                "status": "awaiting_anchors",
                "start_anchor": None,
                "end_anchor": None,
                "raw_video": None,
                "provider_manifest": None,
                "normalized_video": None,
                "normalization_attempts": 0,
                "error": None,
            }
            for item in plan.segments
        ],
        "summary": {
            "approved_anchor_count": 0,
            "registered_video_count": 0,
            "completed_segment_count": 0,
            "normalization_attempt_count": 0,
        },
        "next_segment_id": plan.segments[0].segment_id,
        "generated_shot": None,
        "assembly_error": None,
    }


def _anchor_reference(record: Mapping[str, object]) -> dict[str, object] | None:
    frame = record.get("frame")
    if not isinstance(frame, dict):
        return None
    return {
        "anchor_id": record["anchor_id"],
        "path": frame["path"],
        "bytes": frame["bytes"],
        "sha256": frame["sha256"],
    }


def _sync_segment_states(manifest: dict[str, object]) -> None:
    anchors = {
        item["anchor_id"]: item
        for item in manifest["anchors"]
        if isinstance(item, dict)
    }
    for segment in manifest["segments"]:
        if not isinstance(segment, dict):
            continue
        segment["start_anchor"] = _anchor_reference(anchors[segment["start_anchor_id"]])
        segment["end_anchor"] = _anchor_reference(anchors[segment["end_anchor_id"]])
        if isinstance(segment.get("normalized_video"), dict):
            segment["status"] = "completed"
        elif segment.get("error") is not None:
            segment["status"] = "failed"
        elif segment.get("status") == "normalizing":
            continue
        elif segment["start_anchor"] is None or segment["end_anchor"] is None:
            segment["status"] = "awaiting_anchors"
        elif not isinstance(segment.get("raw_video"), dict):
            segment["status"] = "awaiting_video"
        else:
            segment["status"] = "ready_to_normalize"


def _refresh_manifest(
    manifest: dict[str, object],
    *,
    paused_for_limit: bool = False,
) -> None:
    _sync_segment_states(manifest)
    anchors = [item for item in manifest["anchors"] if isinstance(item, dict)]
    segments = [item for item in manifest["segments"] if isinstance(item, dict)]
    summary = manifest["summary"]
    summary["approved_anchor_count"] = sum(
        isinstance(item.get("frame"), dict) for item in anchors
    )
    summary["registered_video_count"] = sum(
        isinstance(item.get("raw_video"), dict) for item in segments
    )
    summary["completed_segment_count"] = sum(
        item.get("status") == "completed" for item in segments
    )
    summary["normalization_attempt_count"] = sum(
        int(item.get("normalization_attempts", 0)) for item in segments
    )
    next_segment = next(
        (item for item in segments if item.get("status") != "completed"),
        None,
    )
    manifest["next_segment_id"] = (
        next_segment["segment_id"] if next_segment is not None else None
    )
    statuses = {item.get("status") for item in segments}
    if isinstance(manifest.get("generated_shot"), dict):
        manifest["status"] = "completed"
    elif "failed" in statuses:
        manifest["status"] = "failed"
    elif "awaiting_anchors" in statuses:
        manifest["status"] = "awaiting_anchors"
    elif "awaiting_video" in statuses:
        manifest["status"] = "awaiting_video"
    elif "ready_to_normalize" in statuses:
        manifest["status"] = "paused_limit" if paused_for_limit else "ready"
    elif "normalizing" in statuses:
        manifest["status"] = "running"
    else:
        manifest["status"] = "ready"


def _write_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    paused_for_limit: bool = False,
) -> None:
    _refresh_manifest(manifest, paused_for_limit=paused_for_limit)
    dump_json_atomic(path, manifest, sort_keys=True)


def _result(root: Path, manifest: Mapping[str, object]) -> GeneratedShotRunResult:
    generated = manifest.get("generated_shot")
    shot_path = (
        root / generated["path"]
        if isinstance(generated, dict) and isinstance(generated.get("path"), str)
        else None
    )
    return GeneratedShotRunResult(
        run_id=str(manifest["run_id"]),
        run_dir=root,
        manifest_path=root / "generated_shot_manifest.json",
        status=str(manifest["status"]),
        next_segment_id=(
            str(manifest["next_segment_id"])
            if manifest.get("next_segment_id") is not None
            else None
        ),
        completed_segment_ids=tuple(
            str(item["segment_id"])
            for item in manifest["segments"]
            if isinstance(item, dict) and item.get("status") == "completed"
        ),
        shot_video_path=shot_path,
    )


def _validate_run_artifacts(
    root: Path,
    manifest: dict[str, object],
    *,
    toolkit: GeneratedVideoToolkit,
) -> None:
    for anchor in manifest["anchors"]:
        if isinstance(anchor, dict) and isinstance(anchor.get("frame"), dict):
            _run_artifact_path(
                root,
                anchor["frame"],
                label=f"anchor {anchor['anchor_id']}",
            )
    for segment in manifest["segments"]:
        if not isinstance(segment, dict):
            continue
        raw = segment.get("raw_video")
        if isinstance(raw, dict):
            _run_artifact_path(
                root,
                raw,
                label=f"segment {segment['segment_id']} raw video",
            )
        normalized = segment.get("normalized_video")
        if isinstance(normalized, dict):
            path = _run_artifact_path(
                root,
                normalized,
                label=f"segment {segment['segment_id']} normalized video",
            )
            toolkit.validate_segment(
                path,
                target_frames=int(segment["target_frames"]),
                shot_id=str(segment["segment_id"]),
            )
    generated = manifest.get("generated_shot")
    if isinstance(generated, dict):
        path = _run_artifact_path(root, generated, label="generated shot")
        toolkit.validate_segment(
            path,
            target_frames=int(generated["target_frames"]),
            shot_id=str(manifest["shot_id"]),
        )


def run_generated_shot_pipeline(
    *,
    run_dir: str | Path,
    run_id: str,
    plan: GenerationSegmentPlanDocument | dict[str, object],
    anchor_manifest_path: str | Path,
    inputs: GeneratedShotInputsDocument | dict[str, object],
    inputs_base_dir: str | Path = ".",
    toolkit: GeneratedVideoToolkit | None = None,
    max_new_normalizations: int = 1,
    retry_failed_segment_ids: Collection[str] = (),
) -> GeneratedShotRunResult:
    """Import, normalize and assemble one recoverable GeneratedShot.

    No image or video model is invoked.  A resume may add approved videos, but
    previously registered anchors, raw videos and outputs are SHA256-immutable.
    """

    try:
        cleaned_run_id = _clean_id(run_id, "run_id")
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "invalid GeneratedShot run_id",
            actual=str(exc),
        ) from exc
    if (
        isinstance(max_new_normalizations, bool)
        or not isinstance(max_new_normalizations, int)
        or max_new_normalizations < 0
    ):
        raise InputValidationError("max_new_normalizations must be an integer >= 0")
    plan_doc = (
        plan
        if isinstance(plan, GenerationSegmentPlanDocument)
        else parse_generation_segment_plan(plan)
    )
    if plan_doc.review_status != "approved":
        raise InputValidationError(
            "generation segment plan must be explicitly approved before execution"
        )
    for segment in plan_doc.segments:
        if segment.duration_seconds > _MAX_SEGMENT_DURATION_SECONDS + 1e-6:
            raise InputValidationError(
                "segment exceeds the verified 2 second Vidu boundary",
                shot_id=plan_doc.shot_id,
                actual={
                    "segment_id": segment.segment_id,
                    "duration_seconds": segment.duration_seconds,
                },
            )
        _target_frames(segment.duration_seconds)
    input_doc = (
        inputs
        if isinstance(inputs, GeneratedShotInputsDocument)
        else parse_generated_shot_inputs(inputs)
    )
    if input_doc.shot_id != plan_doc.shot_id:
        raise InputValidationError("generated-shot inputs shot_id does not match plan")
    known_segment_ids = {item.segment_id for item in plan_doc.segments}
    supplied_ids = {item.segment_id for item in input_doc.segments}
    unknown_inputs = sorted(supplied_ids - known_segment_ids)
    if unknown_inputs:
        raise InputValidationError(
            "generated-shot inputs contain unknown segment ids",
            actual=unknown_inputs,
        )
    retries = set(retry_failed_segment_ids)
    unknown_retries = sorted(retries - known_segment_ids)
    if unknown_retries:
        raise InputValidationError(
            "retry_failed_segment_ids contains unknown ids",
            actual=unknown_retries,
        )

    requested_root = Path(run_dir)
    if requested_root.is_symlink() or (
        requested_root.exists() and not requested_root.is_dir()
    ):
        raise InputValidationError(
            "GeneratedShot run_dir must be a real directory",
            actual=str(requested_root),
        )
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    media = toolkit or FFmpegToolkit()
    profile = RenderProfile()
    anchor_document, anchor_sources = _load_anchor_sources(
        Path(anchor_manifest_path),
        plan=plan_doc,
    )
    contract = {
        "plan": plan_doc.model_dump(mode="json"),
        "fps": _FPS,
        "max_segment_duration_seconds": _MAX_SEGMENT_DURATION_SECONDS,
        "render_profile": profile.model_dump(mode="json"),
    }
    contract_sha256 = _contract_sha256(contract)
    manifest_path = root / "generated_shot_manifest.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise InputValidationError("unsupported GeneratedShot run manifest")
        if manifest.get("run_id") != cleaned_run_id:
            raise InputValidationError(
                "cannot change run_id while resuming GeneratedShot"
            )
        if manifest.get("contract_sha256") != contract_sha256:
            raise InputValidationError(
                "cannot resume: GeneratedShot plan or normalization contract changed"
            )
        for segment in manifest.get("segments", []):
            if isinstance(segment, dict) and segment.get("status") == "normalizing":
                segment["status"] = "failed"
                segment["error"] = {
                    "type": "InterruptedExecution",
                    "message": (
                        "previous local normalization was interrupted; explicit "
                        "segment retry required"
                    ),
                }
        _validate_run_artifacts(root, manifest, toolkit=media)
    else:
        manifest = _new_manifest(
            run_id=cleaned_run_id,
            plan=plan_doc,
            contract_sha256=contract_sha256,
            anchor_manifest=anchor_document,
            profile=profile,
        )

    anchor_records = {
        item["anchor_id"]: item
        for item in manifest["anchors"]
        if isinstance(item, dict)
    }
    for anchor in plan_doc.anchors:
        record = anchor_records[anchor.anchor_id]
        available = anchor_sources.get(anchor.anchor_id)
        if available is None:
            if isinstance(record.get("frame"), dict):
                raise SourceDriftError(
                    f"previously imported anchor {anchor.anchor_id!r} "
                    "is no longer approved"
                )
            continue
        source, source_fingerprint = available
        relative = f"anchors/{anchor.order:03d}_{anchor.anchor_id}/frame.png"
        frame = _copy_snapshot(
            source,
            root / Path(relative),
            relative_path=relative,
            expected=source_fingerprint,
        )
        existing = record.get("frame")
        if isinstance(existing, dict) and existing != frame:
            raise SourceDriftError(
                f"approved anchor {anchor.anchor_id!r} changed after registration"
            )
        record["status"] = "approved"
        record["frame"] = frame

    _sync_segment_states(manifest)
    input_by_id = {item.segment_id: item for item in input_doc.segments}
    segment_records = {
        item["segment_id"]: item
        for item in manifest["segments"]
        if isinstance(item, dict)
    }
    base_dir = Path(inputs_base_dir).resolve()
    for segment_id, item in input_by_id.items():
        record = segment_records[segment_id]
        if record.get("start_anchor") is None or record.get("end_anchor") is None:
            raise InputValidationError(
                f"cannot register video for {segment_id!r} before both anchors "
                "are approved"
            )
        source = _declared_path(
            item.raw_video_path,
            base_dir=base_dir,
            label=f"segment {segment_id} raw video",
            suffix=".mp4",
        )
        source_fingerprint = _fingerprint(source)
        provider_path = _declared_path(
            item.provider_manifest_path,
            base_dir=base_dir,
            label=f"segment {segment_id} provider manifest",
            suffix=".json",
        )
        provider = _provider_summary(
            provider_path,
            raw_sha256=str(source_fingerprint["sha256"]),
        )
        relative = f"segments/{int(record['order']):03d}_{segment_id}/raw.mp4"
        raw_record = _copy_snapshot(
            source,
            root / Path(relative),
            relative_path=relative,
            expected=source_fingerprint,
        )
        existing_raw = record.get("raw_video")
        if isinstance(existing_raw, dict) and existing_raw != raw_record:
            raise SourceDriftError(
                f"raw video for {segment_id!r} changed after registration"
            )
        existing_provider = record.get("provider_manifest")
        if isinstance(existing_provider, dict) and existing_provider != provider:
            raise SourceDriftError(
                f"provider manifest for {segment_id!r} changed after registration"
            )
        record["raw_video"] = raw_record
        record["provider_manifest"] = provider

    for segment_id in retries:
        record = segment_records[segment_id]
        if record.get("status") != "failed" and record.get("error") is None:
            raise InputValidationError(
                f"segment {segment_id!r} is not in a failed state"
            )
        record["error"] = None

    _write_manifest(manifest_path, manifest)
    new_normalizations = 0
    for segment in manifest["segments"]:
        if not isinstance(segment, dict):
            continue
        _sync_segment_states(manifest)
        if segment.get("status") != "ready_to_normalize":
            continue
        if new_normalizations >= max_new_normalizations:
            break
        raw = segment["raw_video"]
        source = _run_artifact_path(
            root,
            raw,
            label=f"segment {segment['segment_id']} raw video",
        )
        relative = (
            f"segments/{int(segment['order']):03d}_{segment['segment_id']}"
            "/normalized.mp4"
        )
        output = root / Path(relative)
        if output.exists():
            raise OutputValidationError(
                "unregistered normalization output already exists",
                actual=relative,
            )
        temporary = output.with_name(".normalized.mp4.tmp")
        temporary.unlink(missing_ok=True)
        segment["status"] = "normalizing"
        segment["normalization_attempts"] = (
            int(segment.get("normalization_attempts", 0)) + 1
        )
        _write_manifest(manifest_path, manifest)
        try:
            media.normalize_generated_source(
                source,
                temporary,
                target_frames=int(segment["target_frames"]),
                profile=profile,
            )
            media.validate_segment(
                temporary,
                target_frames=int(segment["target_frames"]),
                shot_id=str(segment["segment_id"]),
            )
            os.replace(temporary, output)
            segment["normalized_video"] = {
                "path": relative,
                **_fingerprint(output),
                "target_frames": segment["target_frames"],
            }
            segment["error"] = None
            new_normalizations += 1
            _write_manifest(manifest_path, manifest)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            segment["error"] = {
                "type": type(exc).__name__,
                "message": (
                    "local normalization failed; explicit segment retry required"
                ),
            }
            _write_manifest(manifest_path, manifest)
            raise

    _sync_segment_states(manifest)
    completed = all(
        isinstance(item, dict) and item.get("status") == "completed"
        for item in manifest["segments"]
    )
    if completed and not isinstance(manifest.get("generated_shot"), dict):
        normalized_paths: list[Path] = []
        signatures: list[dict[str, Any]] = []
        durations: list[Decimal] = []
        total_frames = 0
        for segment in manifest["segments"]:
            normalized = segment["normalized_video"]
            path = _run_artifact_path(
                root,
                normalized,
                label=f"segment {segment['segment_id']} normalized video",
            )
            frames = int(segment["target_frames"])
            normalized_paths.append(path)
            signatures.append(
                media.validate_segment(
                    path,
                    target_frames=frames,
                    shot_id=str(segment["segment_id"]),
                )
            )
            durations.append(Decimal(frames) / Decimal(_FPS))
            total_frames += frames
        media.concat_signatures_equal(signatures)
        output = root / "generated_shot.mp4"
        if output.exists():
            raise OutputValidationError(
                "unregistered generated_shot.mp4 already exists"
            )
        temporary = root / ".generated_shot.mp4.tmp"
        temporary.unlink(missing_ok=True)
        try:
            if len(normalized_paths) == 1:
                shutil.copyfile(normalized_paths[0], temporary)
            else:
                media.concat_video(
                    normalized_paths,
                    temporary,
                    durations=durations,
                )
            media.validate_segment(
                temporary,
                target_frames=total_frames,
                shot_id=plan_doc.shot_id,
            )
            os.replace(temporary, output)
            manifest["generated_shot"] = {
                "path": "generated_shot.mp4",
                **_fingerprint(output),
                "target_frames": total_frames,
                "duration_seconds": total_frames / _FPS,
                "segment_count": len(normalized_paths),
            }
            manifest["assembly_error"] = None
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            manifest["assembly_error"] = {
                "type": type(exc).__name__,
                "message": "local GeneratedShot assembly failed",
            }
            _write_manifest(manifest_path, manifest)
            raise

    _sync_segment_states(manifest)
    has_ready = any(
        isinstance(item, dict) and item.get("status") == "ready_to_normalize"
        for item in manifest["segments"]
    )
    paused = has_ready and new_normalizations >= max_new_normalizations
    _write_manifest(manifest_path, manifest, paused_for_limit=paused)
    _validate_run_artifacts(root, manifest, toolkit=media)
    return _result(root, manifest)
