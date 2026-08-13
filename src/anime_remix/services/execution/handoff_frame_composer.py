"""Recoverable generation of shared handoff anchors inside one shot.

The approved canonical first-frame image is adopted as anchor 1.  Every later
model-generated anchor edits the immediately previous approved anchor and may
use one additional scoped visual reference.  This gives each model request at
most two images and ensures adjacent video segments receive the exact same
boundary bytes.

There are no automatic retries.  Model results pause for human approval by
default, and one invocation creates at most one new paid-capable frame unless
the caller explicitly raises the limit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, UnidentifiedImageError

from anime_remix.errors import InputValidationError, OutputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object, sha256_file
from anime_remix.services.execution.adapter import Adapter, Executor
from anime_remix.services.execution.artifact_store import canonical_json_bytes
from anime_remix.services.script.generation_segment_plan import (
    GenerationSegmentPlanDocument,
    parse_generation_segment_plan,
)

_SCHEMA_VERSION = "handoff-frame-compose-run-v1"
_VISIBLE_DELTA_THRESHOLD = 20
_MIN_VISIBLE_DELTA_FRACTION = 0.001


@dataclass(frozen=True)
class HandoffFrameComposeResult:
    """Current durable state of a shared-anchor composition run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    status: str
    next_anchor_id: str | None
    completed_anchor_ids: tuple[str, ...]
    final_frame_path: Path | None


def _adapter_contract(adapter: Adapter) -> dict:
    state: dict[str, object] = {}
    for key, value in vars(adapter).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            state[key] = value
    return {
        "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "adapter_id": adapter.adapter_id,
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "state": state,
    }


def _normalize_png(data: bytes, *, label: str) -> tuple[bytes, int, int]:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.width <= 0 or image.height <= 0:
                raise OutputValidationError(f"{label} has empty dimensions")
            normalized = image.convert("RGB")
            width, height = normalized.size
            output = BytesIO()
            normalized.save(output, format="PNG")
    except (OSError, UnidentifiedImageError) as exc:
        raise OutputValidationError(
            f"cannot decode {label} as PNG/JPEG", actual=str(exc)
        ) from exc
    return output.getvalue(), width, height


def _fingerprint(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise InputValidationError(f"{label} is not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputValidationError(f"cannot read {label}", actual=str(exc)) from exc
    _, width, height = _normalize_png(data, label=label)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "width": width,
        "height": height,
    }


def _write_frame(
    *,
    root: Path,
    anchor_id: str,
    order: int,
    attempt: int,
    data: bytes,
) -> dict:
    normalized, width, height = _normalize_png(data, label=f"anchor {anchor_id} output")
    frame_dir = root / "anchors" / f"{order:03d}_{anchor_id}"
    attempt_dir = frame_dir / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempt_dir / "output.png"
    named_path = frame_dir / "frame.png"
    attempt_tmp = attempt_dir / ".output.png.tmp"
    named_tmp = frame_dir / ".frame.png.tmp"
    attempt_tmp.write_bytes(normalized)
    attempt_tmp.replace(attempt_path)
    named_tmp.write_bytes(normalized)
    named_tmp.replace(named_path)
    return {
        "path": named_path.relative_to(root).as_posix(),
        "attempt_path": attempt_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(normalized).hexdigest(),
        "bytes": len(normalized),
        "width": width,
        "height": height,
    }


def _validate_output(root: Path, frame: dict) -> None:
    output = frame.get("output")
    if not isinstance(output, dict):
        raise InputValidationError(
            f"completed anchor {frame['anchor_id']} has no output record"
        )
    path = root / output["path"]
    if not path.is_file():
        raise InputValidationError(f"completed anchor output is missing: {path}")
    actual = sha256_file(path)
    if actual != output["sha256"]:
        raise InputValidationError(
            f"completed anchor output drifted: {frame['anchor_id']}",
            actual=actual,
        )
    _normalize_png(path.read_bytes(), label=f"anchor {frame['anchor_id']} output")


def _safe_provider_metadata(executor: Executor) -> dict | None:
    value = getattr(executor, "last_metadata", None)
    if not isinstance(value, dict):
        return None
    allowed = {
        "provider",
        "model",
        "request_id",
        "status",
        "duration_ms",
        "usage",
        "http_status",
        "provider_code",
        "provider_message",
    }
    return {key: item for key, item in value.items() if key in allowed}


def _visible_delta_qa(base: bytes, output: bytes) -> dict:
    """Flag model outputs that made no material visible change to the anchor."""

    base_png, base_width, base_height = _normalize_png(base, label="anchor base")
    output_png, output_width, output_height = _normalize_png(
        output, label="anchor output"
    )
    if (base_width, base_height) != (output_width, output_height):
        return {
            "metric": "luma_changed_pixel_fraction",
            "threshold": _VISIBLE_DELTA_THRESHOLD,
            "changed_fraction": 1.0,
            "minimum_fraction": _MIN_VISIBLE_DELTA_FRACTION,
            "semantic_change_suspect": False,
            "note": "dimensions changed, so the anchor requires visual review",
        }
    with (
        Image.open(BytesIO(base_png)) as base_image,
        Image.open(BytesIO(output_png)) as output_image,
    ):
        histogram = (
            ImageChops.difference(base_image, output_image).convert("L").histogram()
        )
    pixels = base_width * base_height
    changed = sum(histogram[_VISIBLE_DELTA_THRESHOLD + 1 :])
    fraction = changed / pixels
    suspect = fraction < _MIN_VISIBLE_DELTA_FRACTION
    return {
        "metric": "luma_changed_pixel_fraction",
        "threshold": _VISIBLE_DELTA_THRESHOLD,
        "changed_fraction": round(fraction, 8),
        "minimum_fraction": _MIN_VISIBLE_DELTA_FRACTION,
        "semantic_change_suspect": suspect,
        "note": (
            "model output may have ignored the requested anchor change; "
            "explicit visual approval is required"
            if suspect
            else "output contains a material visible delta"
        ),
    }


def _new_manifest(
    *,
    run_id: str,
    plan: GenerationSegmentPlanDocument,
    contract_sha256: str,
    adapter: Adapter,
    auto_approve: bool,
    first_output: dict,
) -> dict:
    frames: list[dict] = []
    for anchor in plan.anchors:
        first = anchor.order == 1
        model_call = anchor.generation_method == "edit_previous"
        frames.append(
            {
                "anchor_id": anchor.anchor_id,
                "order": anchor.order,
                "time_seconds": anchor.time_seconds,
                "position": anchor.position,
                "generation_method": anchor.generation_method,
                "base_anchor_id": anchor.base_anchor_id,
                "reference_asset_id": anchor.reference_asset_id,
                "information_added": list(anchor.information_added),
                "generation_risk": anchor.generation_risk,
                "risk_factors": list(anchor.risk_factors),
                "model_call": model_call,
                "requires_review": (
                    anchor.generation_risk == "high"
                    or "camera_or_composition_reset" in anchor.control_reasons
                ),
                "status": "completed" if first else "pending",
                "approved": first,
                "attempts": 0,
                "request": None,
                "output": first_output if first else None,
                "error": None,
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "shot_id": plan.shot_id,
        "status": "ready",
        "contract_sha256": contract_sha256,
        "execution": {
            "adapter_id": adapter.adapter_id,
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "auto_approve": auto_approve,
            "automatic_retries": 0,
            "continuity": "immediately_previous_approved_anchor",
            "per_invocation_new_model_frame_limit_default": 1,
            "max_primary_visual_references_per_model_call": 2,
        },
        "completed_anchor_count": 1,
        "approved_anchor_count": 1,
        "model_call_count": 0,
        "next_anchor_id": plan.anchors[1].anchor_id,
        "final_frame": None,
        "anchors": frames,
        "segments": [item.model_dump(mode="json") for item in plan.segments],
    }


def _refresh_manifest(manifest: dict) -> None:
    frames = manifest["anchors"]
    manifest["completed_anchor_count"] = sum(
        item["status"] == "completed" for item in frames
    )
    manifest["approved_anchor_count"] = sum(
        item["status"] == "completed" and item["approved"] for item in frames
    )
    manifest["model_call_count"] = sum(
        item["status"] == "completed" and item["model_call"] for item in frames
    )
    next_frame = next(
        (
            item
            for item in frames
            if item["status"] != "completed" or not item["approved"]
        ),
        None,
    )
    manifest["next_anchor_id"] = (
        next_frame["anchor_id"] if next_frame is not None else None
    )
    manifest["final_frame"] = None if next_frame else frames[-1]["output"]


def _write_manifest(path: Path, manifest: dict) -> None:
    _refresh_manifest(manifest)
    dump_json_atomic(path, manifest, sort_keys=True)


def _result(root: Path, manifest: dict) -> HandoffFrameComposeResult:
    final = manifest.get("final_frame")
    return HandoffFrameComposeResult(
        run_id=manifest["run_id"],
        run_dir=root,
        manifest_path=root / "handoff-frame-compose-run.json",
        status=manifest["status"],
        next_anchor_id=manifest["next_anchor_id"],
        completed_anchor_ids=tuple(
            item["anchor_id"]
            for item in manifest["anchors"]
            if item["status"] == "completed"
        ),
        final_frame_path=(root / final["path"] if isinstance(final, dict) else None),
    )


def _condition(
    *,
    condition_id: str,
    role: str,
    payload_ref: str,
    source_asset_id: str | None,
    derived_from: str | None,
    constraint: str,
) -> dict:
    return {
        "condition_id": condition_id,
        "role": role,
        "kind": "image",
        "payload_ref": payload_ref,
        "satisfied_constraints": [constraint],
        "scores": {"selected": 1.0},
        "provenance": {
            "source_asset_id": source_asset_id,
            "derived_from": derived_from,
        },
    }


def run_handoff_frame_composition(
    *,
    run_dir: str | Path,
    run_id: str,
    plan: GenerationSegmentPlanDocument | dict,
    first_frame_path: str | Path,
    asset_map: Mapping[str, str | Path],
    adapter: Adapter,
    executor: Executor,
    approved_anchor_ids: Collection[str] = (),
    auto_approve: bool = False,
    max_new_model_frames: int = 1,
) -> HandoffFrameComposeResult:
    """Execute or resume the ordered shared-anchor frame plan."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise InputValidationError("run_id must be a non-empty string")
    if not isinstance(auto_approve, bool):
        raise InputValidationError("auto_approve must be a boolean")
    if (
        isinstance(max_new_model_frames, bool)
        or not isinstance(max_new_model_frames, int)
        or max_new_model_frames < 0
    ):
        raise InputValidationError("max_new_model_frames must be an integer >= 0")
    document = (
        plan
        if isinstance(plan, GenerationSegmentPlanDocument)
        else parse_generation_segment_plan(plan)
    )
    if document.review_status != "approved":
        raise InputValidationError(
            "generation segment plan must be explicitly approved before execution"
        )
    if document.decision == "blocked":
        raise InputValidationError("blocked generation segment plan cannot execute")
    if not getattr(adapter, "supports_first_frame_fusion", False):
        raise InputValidationError(
            f"adapter {adapter.adapter_id!r} does not support frame fusion"
        )

    first_path = Path(first_frame_path)
    first_fingerprint = _fingerprint(first_path, label="approved first frame")
    asset_paths = {asset_id: Path(path) for asset_id, path in asset_map.items()}
    required_assets = {
        anchor.reference_asset_id
        for anchor in document.anchors[1:]
        if anchor.reference_asset_id is not None
    }
    missing = sorted(required_assets - set(asset_paths))
    if missing:
        raise InputValidationError(
            f"asset_map is missing handoff-frame references: {missing}"
        )
    asset_fingerprints = {
        asset_id: _fingerprint(
            asset_paths[asset_id], label=f"reference asset {asset_id}"
        )
        for asset_id in sorted(required_assets)
    }
    contract = {
        "plan": document.model_dump(mode="json"),
        "first_frame": first_fingerprint,
        "reference_assets": asset_fingerprints,
        "adapter": _adapter_contract(adapter),
    }
    digest = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "handoff-frame-compose-run.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise InputValidationError("unsupported handoff-frame run manifest")
        if manifest.get("run_id") != run_id:
            raise InputValidationError("run_id does not match existing manifest")
        if manifest.get("contract_sha256") != digest:
            raise InputValidationError(
                "cannot resume: handoff-frame composition contract changed"
            )
        if manifest["execution"]["auto_approve"] is not auto_approve:
            raise InputValidationError(
                "cannot change auto_approve while resuming handoff-frame run"
            )
        for frame in manifest["anchors"]:
            if frame["status"] == "running":
                frame["status"] = "failed"
                frame["error"] = {
                    "type": "InterruptedRun",
                    "message": "previous frame attempt ended before completion",
                }
            if frame["status"] == "completed":
                _validate_output(root, frame)
    else:
        first_output = _write_frame(
            root=root,
            anchor_id=document.anchors[0].anchor_id,
            order=1,
            attempt=0,
            data=first_path.read_bytes(),
        )
        manifest = _new_manifest(
            run_id=run_id,
            plan=document,
            contract_sha256=digest,
            adapter=adapter,
            auto_approve=auto_approve,
            first_output=first_output,
        )
        _write_manifest(manifest_path, manifest)

    anchor_ids = {item.anchor_id for item in document.anchors}
    first_anchor_id = document.anchors[0].anchor_id
    approvals = set(approved_anchor_ids)
    unknown = sorted(approvals - anchor_ids)
    if unknown:
        raise InputValidationError(
            f"approved_anchor_ids contains unknown ids: {unknown}"
        )
    completed = {
        item["anchor_id"]
        for item in manifest["anchors"]
        if item["status"] == "completed"
    }
    premature = sorted(approvals - completed)
    if premature and not auto_approve:
        raise InputValidationError(
            f"cannot approve anchors before outputs exist: {premature}"
        )
    for frame in manifest["anchors"]:
        if frame["status"] == "completed" and (
            frame["anchor_id"] == first_anchor_id
            or frame["anchor_id"] in approvals
            or not frame["model_call"]
            or (auto_approve and not frame.get("requires_review", False))
        ):
            frame["approved"] = True

    plan_by_id = {item.anchor_id: item for item in document.anchors}
    manifest_by_id = {item["anchor_id"]: item for item in manifest["anchors"]}
    new_model_frames = 0
    for frame in manifest["anchors"]:
        if frame["status"] == "completed":
            if not frame["approved"]:
                manifest["status"] = "awaiting_review"
                _write_manifest(manifest_path, manifest)
                return _result(root, manifest)
            continue
        anchor = plan_by_id[frame["anchor_id"]]
        if frame["model_call"] and new_model_frames >= max_new_model_frames:
            manifest["status"] = "paused_limit"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)
        previous = manifest_by_id[anchor.base_anchor_id]
        if previous["status"] != "completed" or not previous["approved"]:
            raise InputValidationError(
                f"base anchor {anchor.base_anchor_id!r} is not approved"
            )
        frame["attempts"] += 1
        attempt = frame["attempts"]
        attempt_dir = (
            root
            / "anchors"
            / f"{anchor.order:03d}_{anchor.anchor_id}"
            / f"attempt_{attempt:03d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        frame["status"] = "running"
        frame["error"] = None
        manifest["status"] = "running"
        _write_manifest(manifest_path, manifest)
        request_record = None
        try:
            if anchor.generation_method == "reuse_existing_asset":
                output_data = asset_paths[anchor.reference_asset_id].read_bytes()
                request_record = None
            else:
                previous_path = root / previous["output"]["path"]
                base_id = f"cond_{anchor.anchor_id}_base"
                conditions = [
                    _condition(
                        condition_id=base_id,
                        role="source_frame",
                        payload_ref=f"artifact://handoff/{anchor.base_anchor_id}",
                        source_asset_id=None,
                        derived_from=f"anchor://{anchor.base_anchor_id}",
                        constraint="preserve the previous approved anchor",
                    )
                ]
                inputs = {base_id: previous_path.read_bytes()}
                if anchor.reference_asset_id is not None:
                    reference_id = f"cond_{anchor.anchor_id}_reference"
                    conditions.append(
                        _condition(
                            condition_id=reference_id,
                            role=anchor.reference_role,
                            payload_ref=(
                                "asset://anime-remix/catalog/"
                                f"{anchor.reference_asset_id}@v1"
                            ),
                            source_asset_id=anchor.reference_asset_id,
                            derived_from=None,
                            constraint="supply only the scoped new visual fact",
                        )
                    )
                    inputs[reference_id] = asset_paths[
                        anchor.reference_asset_id
                    ].read_bytes()
                compiled = adapter.compile(
                    operation="first_frame_fusion",
                    intent={
                        "stage_id": anchor.anchor_id,
                        "stage_operation": (
                            "fuse_component"
                            if anchor.reference_asset_id is not None
                            else "apply_text_delta"
                        ),
                        "component_ids": list(anchor.information_added)
                        or [anchor.anchor_id],
                        "instruction": anchor.delta_instruction,
                        "reference_attributes": list(anchor.reference_attributes),
                        "text_fallbacks": {},
                    },
                    keyframe_state={},
                    scene_description=anchor.composition,
                    conditions=conditions,
                )
                request_path = attempt_dir / "request.json"
                dump_json_atomic(request_path, compiled, sort_keys=True)
                request_record = {
                    "path": request_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(request_path),
                    "adapter_id": compiled["adapter_id"],
                    "condition_count": len(compiled["conditions"]),
                    "prompt_sha256": hashlib.sha256(
                        compiled["prompt"].encode("utf-8")
                    ).hexdigest(),
                }
                frame["request"] = request_record
                _write_manifest(manifest_path, manifest)
                selected_inputs = {
                    slot["condition_ref"]: inputs[slot["condition_ref"]]
                    for slot in compiled["conditions"]
                }
                output_data = executor.execute(
                    request_payload=compiled,
                    operation="first_frame_fusion",
                    inputs=selected_inputs,
                )
                if anchor.generation_method == "edit_previous":
                    frame["qa"] = _visible_delta_qa(
                        selected_inputs[base_id],
                        output_data,
                    )
                    if frame["qa"]["semantic_change_suspect"]:
                        frame["requires_review"] = True
            frame["request"] = request_record
            frame["output"] = _write_frame(
                root=root,
                anchor_id=anchor.anchor_id,
                order=anchor.order,
                attempt=attempt,
                data=output_data,
            )
            metadata = _safe_provider_metadata(executor)
            if metadata is not None and frame["request"] is not None:
                frame["request"]["provider"] = metadata
        except Exception as exc:
            metadata = _safe_provider_metadata(executor)
            if metadata is not None and frame["request"] is not None:
                frame["request"]["provider"] = metadata
            frame["status"] = "failed"
            frame["error"] = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise
        frame["status"] = "completed"
        frame["approved"] = not frame["model_call"] or (
            auto_approve and not frame.get("requires_review", False)
        )
        if frame["model_call"]:
            new_model_frames += 1
        _write_manifest(manifest_path, manifest)
        if not frame["approved"]:
            manifest["status"] = "awaiting_review"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)

    manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    return _result(root, manifest)
