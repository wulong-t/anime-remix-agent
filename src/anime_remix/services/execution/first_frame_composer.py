"""Recoverable staged construction of a shot's canonical first frame.

The composer consumes an approved ``first-frame-plan-v1``.  It resolves only
the exact selected assets, executes the ordered plan one stage at a time and
persists a restart-safe manifest.  Deterministic anchor adoption is approved
automatically unless a quality gate is attached.  Full-frame RGBA foreground
overlays can be alpha-composited without a model call.  Model stages pause for
human approval by default.  There are no automatic retries and at most one
new model stage is executed per call unless the caller explicitly raises the
limit.
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
from anime_remix.services.script.first_frame_plan import (
    FirstFramePlanDocument,
    parse_first_frame_plan,
)

_SCHEMA_VERSION = "first-frame-compose-run-v1"
_VISIBLE_DELTA_THRESHOLD = 20
_MIN_VISIBLE_DELTA_FRACTION = 0.001


@dataclass(frozen=True)
class FirstFrameComposeResult:
    """Current durable state of one staged first-frame run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    status: str
    next_stage_id: str | None
    completed_stage_ids: tuple[str, ...]
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


def _contract_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


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


def _composite_overlay(base: bytes, overlay: bytes) -> bytes:
    """Apply one same-size RGBA overlay without regenerating existing pixels."""

    try:
        with (
            Image.open(BytesIO(base)) as base_image,
            Image.open(BytesIO(overlay)) as overlay_image,
        ):
            if base_image.size != overlay_image.size:
                raise InputValidationError(
                    "deterministic overlay dimensions must match the canvas",
                    expected=f"{base_image.width}x{base_image.height}",
                    actual=f"{overlay_image.width}x{overlay_image.height}",
                )
            if "A" not in overlay_image.getbands():
                raise InputValidationError(
                    "deterministic overlay must contain an alpha channel"
                )
            rgba_overlay = overlay_image.convert("RGBA")
            alpha_extrema = rgba_overlay.getchannel("A").getextrema()
            if alpha_extrema == (0, 0):
                raise InputValidationError(
                    "deterministic overlay must contain visible pixels"
                )
            if alpha_extrema == (255, 255):
                raise InputValidationError(
                    "deterministic overlay cannot be fully opaque"
                )
            composed = Image.alpha_composite(
                base_image.convert("RGBA"),
                rgba_overlay,
            ).convert("RGB")
            output = BytesIO()
            composed.save(output, format="PNG")
            return output.getvalue()
    except InputValidationError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise OutputValidationError(
            "cannot decode deterministic overlay inputs", actual=str(exc)
        ) from exc


def _normalize_assets(
    asset_map: Mapping[str, str | Path],
    *,
    plan: FirstFramePlanDocument,
) -> tuple[dict[str, Path], dict[str, dict]]:
    paths = {asset_id: Path(path) for asset_id, path in asset_map.items()}
    selected = list(plan.selected_asset_ids)
    missing = sorted(set(selected) - set(paths))
    if missing:
        raise InputValidationError(
            f"asset_map is missing selected first-frame assets: {missing}"
        )
    fingerprints: dict[str, dict] = {}
    for asset_id in selected:
        path = paths[asset_id]
        if not path.is_file():
            raise InputValidationError(
                f"selected first-frame asset is not a file: {path}",
                asset_id=asset_id,
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise InputValidationError(
                f"cannot read selected first-frame asset {asset_id!r}",
                actual=str(exc),
            ) from exc
        _, width, height = _normalize_png(data, label=f"asset {asset_id}")
        fingerprints[asset_id] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "width": width,
            "height": height,
        }
    return paths, fingerprints


def _new_manifest(
    *,
    run_id: str,
    plan: FirstFramePlanDocument,
    contract_sha256: str,
    adapter: Adapter,
    auto_approve: bool,
) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "shot_id": plan.shot_id,
        "keyframe_id": plan.keyframe_id,
        "content_plan_id": plan.content_plan_id,
        "prepared_component_plan_id": plan.prepared_component_plan_id,
        "prepared_component_asset_ids": list(plan.prepared_component_asset_ids),
        "status": "ready",
        "contract_sha256": contract_sha256,
        "execution": {
            "adapter_id": adapter.adapter_id,
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "auto_approve": auto_approve,
            "automatic_retries": 0,
            "per_invocation_new_model_stage_limit_default": 1,
            "max_primary_visual_references_per_model_call": (
                plan.max_primary_visual_references_per_model_call
            ),
        },
        "information_coverage": [
            item.model_dump(mode="json") for item in plan.information_coverage
        ],
        "reference_admissions": [
            item.model_dump(mode="json") for item in plan.reference_admissions
        ],
        "interaction_units": [
            item.model_dump(mode="json") for item in plan.interaction_units
        ],
        "quality_gates": [item.model_dump(mode="json") for item in plan.quality_gates],
        "completed_stage_count": 0,
        "approved_stage_count": 0,
        "model_call_count": 0,
        "next_stage_id": plan.stages[0].stage_id,
        "final_frame": None,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "order": stage.order,
                "operation": stage.operation,
                "component_ids": list(stage.component_ids),
                "quality_gate_ids": list(stage.quality_gate_ids),
                "model_call": stage.operation
                not in {"adopt_anchor", "composite_overlay"},
                "status": "pending",
                "approved": False,
                "attempts": 0,
                "request": None,
                "output": None,
                "qa": None,
                "requires_review": False,
                "error": None,
            }
            for stage in plan.stages
        ],
    }


def _refresh_manifest(manifest: dict) -> None:
    stages = manifest["stages"]
    manifest["completed_stage_count"] = sum(
        item["status"] == "completed" for item in stages
    )
    manifest["approved_stage_count"] = sum(
        item["status"] == "completed" and item["approved"] for item in stages
    )
    manifest["model_call_count"] = sum(
        item["status"] == "completed" and item["model_call"] for item in stages
    )
    next_stage = next(
        (
            item
            for item in stages
            if item["status"] != "completed" or not item["approved"]
        ),
        None,
    )
    manifest["next_stage_id"] = (
        next_stage["stage_id"] if next_stage is not None else None
    )
    if next_stage is None and stages:
        manifest["final_frame"] = stages[-1]["output"]


def _write_manifest(path: Path, manifest: dict) -> None:
    _refresh_manifest(manifest)
    dump_json_atomic(path, manifest, sort_keys=True)


def _validate_output(root: Path, stage: dict) -> None:
    output = stage.get("output")
    if not isinstance(output, dict):
        raise InputValidationError(
            f"completed stage {stage['stage_id']} has no output record"
        )
    path = root / output["path"]
    if not path.is_file():
        raise InputValidationError(f"completed stage output is missing: {path}")
    actual = sha256_file(path)
    if actual != output["sha256"]:
        raise InputValidationError(
            f"completed stage output drifted: {stage['stage_id']}",
            actual=actual,
        )
    _normalize_png(path.read_bytes(), label=f"stage {stage['stage_id']} output")


def _visible_delta_qa(base: bytes, output: bytes) -> dict:
    base_png, base_width, base_height = _normalize_png(base, label="stage base")
    output_png, output_width, output_height = _normalize_png(
        output, label="stage output"
    )
    if (base_width, base_height) != (output_width, output_height):
        return {
            "metric": "luma_changed_pixel_fraction",
            "threshold": _VISIBLE_DELTA_THRESHOLD,
            "changed_fraction": 1.0,
            "minimum_fraction": _MIN_VISIBLE_DELTA_FRACTION,
            "semantic_change_suspect": False,
            "note": "dimensions changed, so the frame requires visual review",
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
            "model output may have ignored the requested component; explicit "
            "visual approval is required"
            if suspect
            else "output contains a material visible delta"
        ),
    }


def _stage_output(
    *,
    root: Path,
    stage_id: str,
    attempt: int,
    data: bytes,
) -> dict:
    normalized, width, height = _normalize_png(data, label=f"stage {stage_id} output")
    stage_dir = root / "stages" / stage_id
    attempt_dir = stage_dir / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempt_dir / "output.png"
    named_path = stage_dir / "frame.png"
    attempt_tmp = attempt_dir / ".output.png.tmp"
    named_tmp = stage_dir / ".frame.png.tmp"
    attempt_tmp.write_bytes(normalized)
    attempt_tmp.replace(attempt_path)
    named_tmp.write_bytes(normalized)
    named_tmp.replace(named_path)
    digest = hashlib.sha256(normalized).hexdigest()
    return {
        "path": named_path.relative_to(root).as_posix(),
        "attempt_path": attempt_path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": len(normalized),
        "width": width,
        "height": height,
    }


def _condition_role(role: str) -> str:
    return {
        "base": "source_frame",
        "source_frame": "source_frame",
        "outfit": "costume",
        "foreground": "scene",
    }.get(role, role)


def _resolve_stage_inputs(
    *,
    root: Path,
    stage_plan,
    manifest_by_id: Mapping[str, dict],
    asset_paths: Mapping[str, Path],
) -> tuple[list[dict], dict[str, bytes]]:
    conditions: list[dict] = []
    data_by_condition: dict[str, bytes] = {}
    for item in stage_plan.inputs:
        condition_id = f"cond_{stage_plan.stage_id}_{item.slot}"
        if item.source_type == "asset":
            path = asset_paths[item.source_id]
            source_asset_id: str | None = item.source_id
            derived_from: str | None = None
        else:
            prior = manifest_by_id[item.source_id]
            if prior["status"] != "completed" or not prior["approved"]:
                raise InputValidationError(
                    f"prior stage {item.source_id!r} is not approved"
                )
            path = root / prior["output"]["path"]
            source_asset_id = None
            derived_from = f"stage://{item.source_id}"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise InputValidationError(
                f"cannot read stage input {item.source_id!r}", actual=str(exc)
            ) from exc
        data_by_condition[condition_id] = data
        conditions.append(
            {
                "condition_id": condition_id,
                "role": _condition_role(item.role),
                "kind": "image",
                "payload_ref": (
                    f"asset://anime-remix/catalog/{item.source_id}@v1"
                    if item.source_type == "asset"
                    else f"artifact://first-frame/{item.source_id}"
                ),
                "satisfied_constraints": [
                    f"first_frame.stage.{stage_plan.stage_id}.slot_{item.slot}"
                ],
                "scores": {"selected": 1.0},
                "provenance": {
                    "source_asset_id": source_asset_id,
                    "derived_from": derived_from,
                },
            }
        )
    return conditions, data_by_condition


def _result(root: Path, manifest: dict) -> FirstFrameComposeResult:
    final = manifest.get("final_frame")
    final_path = root / final["path"] if isinstance(final, dict) else None
    return FirstFrameComposeResult(
        run_id=manifest["run_id"],
        run_dir=root,
        manifest_path=root / "first-frame-compose-run.json",
        status=manifest["status"],
        next_stage_id=manifest["next_stage_id"],
        completed_stage_ids=tuple(
            item["stage_id"]
            for item in manifest["stages"]
            if item["status"] == "completed"
        ),
        final_frame_path=final_path,
    )


def run_first_frame_composition(
    *,
    run_dir: str | Path,
    run_id: str,
    plan: FirstFramePlanDocument | dict,
    asset_map: Mapping[str, str | Path],
    adapter: Adapter,
    executor: Executor,
    approved_stage_ids: Collection[str] = (),
    auto_approve: bool = False,
    max_new_model_stages: int = 1,
) -> FirstFrameComposeResult:
    """Execute or resume an approved staged first-frame plan.

    A model failure is recorded and reattempted only on a later explicit
    invocation.  Manual mode pauses after every new model stage.  Adoption of
    an exact full-frame anchor is deterministic and does not consume the paid
    model-stage limit.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise InputValidationError("run_id must be a non-empty string")
    if not isinstance(auto_approve, bool):
        raise InputValidationError("auto_approve must be a boolean")
    if (
        isinstance(max_new_model_stages, bool)
        or not isinstance(max_new_model_stages, int)
        or max_new_model_stages < 1
    ):
        raise InputValidationError("max_new_model_stages must be a positive integer")
    plan_doc = (
        plan
        if isinstance(plan, FirstFramePlanDocument)
        else parse_first_frame_plan(plan)
    )
    if plan_doc.review_status != "approved":
        raise InputValidationError(
            "first-frame plan must be explicitly approved before execution"
        )
    if plan_doc.decision == "blocked":
        raise InputValidationError("blocked first-frame plan cannot execute")
    if not getattr(adapter, "supports_first_frame_fusion", False):
        raise InputValidationError(
            f"adapter {adapter.adapter_id!r} does not support staged first-frame fusion"
        )

    asset_paths, fingerprints = _normalize_assets(asset_map, plan=plan_doc)
    normalized_plan = plan_doc.model_dump(mode="json")
    contract = {
        "plan": normalized_plan,
        "asset_fingerprints": fingerprints,
        "adapter": _adapter_contract(adapter),
    }
    digest = _contract_sha256(contract)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "first-frame-compose-run.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise InputValidationError("unsupported first-frame run manifest")
        if manifest.get("run_id") != run_id:
            raise InputValidationError("run_id does not match existing manifest")
        if manifest.get("contract_sha256") != digest:
            raise InputValidationError(
                "cannot resume: first-frame composition contract changed"
            )
        if manifest["execution"]["auto_approve"] is not auto_approve:
            raise InputValidationError(
                "cannot change auto_approve while resuming first-frame run"
            )
        for stage in manifest["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
                stage["error"] = {
                    "type": "InterruptedRun",
                    "message": "previous stage attempt ended before completion",
                }
            if stage["status"] == "completed":
                _validate_output(root, stage)
    else:
        manifest = _new_manifest(
            run_id=run_id,
            plan=plan_doc,
            contract_sha256=digest,
            adapter=adapter,
            auto_approve=auto_approve,
        )
        _write_manifest(manifest_path, manifest)

    stage_ids = {item.stage_id for item in plan_doc.stages}
    approved = set(approved_stage_ids)
    unknown = sorted(approved - stage_ids)
    if unknown:
        raise InputValidationError(
            f"approved_stage_ids contains unknown ids: {unknown}"
        )
    completed_ids = {
        item["stage_id"] for item in manifest["stages"] if item["status"] == "completed"
    }
    premature = sorted(approved - completed_ids)
    if premature and not auto_approve:
        raise InputValidationError(
            f"cannot approve stages before their outputs exist: {premature}"
        )
    for item in manifest["stages"]:
        if item["status"] == "completed" and (
            (not item["model_call"] and not item.get("requires_review", False))
            or item["stage_id"] in approved
            or (auto_approve and not item.get("requires_review", False))
        ):
            item["approved"] = True

    manifest_by_id = {item["stage_id"]: item for item in manifest["stages"]}
    plan_by_id = {item.stage_id: item for item in plan_doc.stages}
    new_model_stages = 0
    for stage in manifest["stages"]:
        if stage["status"] == "completed":
            if not stage["approved"]:
                manifest["status"] = "awaiting_review"
                _write_manifest(manifest_path, manifest)
                return _result(root, manifest)
            continue
        if stage["model_call"] and new_model_stages >= max_new_model_stages:
            manifest["status"] = "paused_limit"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)

        stage_plan = plan_by_id[stage["stage_id"]]
        stage["attempts"] += 1
        attempt = stage["attempts"]
        attempt_dir = root / "stages" / stage["stage_id"] / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stage["status"] = "running"
        stage["error"] = None
        stage["requires_review"] = bool(stage_plan.quality_gate_ids)
        manifest["status"] = "running"
        _write_manifest(manifest_path, manifest)
        request_record = None
        try:
            conditions, inputs = _resolve_stage_inputs(
                root=root,
                stage_plan=stage_plan,
                manifest_by_id=manifest_by_id,
                asset_paths=asset_paths,
            )
            if stage_plan.operation == "adopt_anchor":
                output_data = next(iter(inputs.values()))
                request_record = None
            elif stage_plan.operation == "composite_overlay":
                base_id = conditions[0]["condition_id"]
                overlay_id = conditions[1]["condition_id"]
                output_data = _composite_overlay(
                    inputs[base_id],
                    inputs[overlay_id],
                )
                stage["qa"] = {
                    "deterministic_alpha_composite": True,
                    "manual_quality_gate_ids": list(stage_plan.quality_gate_ids),
                }
            else:
                compiled = adapter.compile(
                    operation="first_frame_fusion",
                    intent={
                        "stage_id": stage_plan.stage_id,
                        "stage_operation": stage_plan.operation,
                        "component_ids": list(stage_plan.component_ids),
                        "instruction": stage_plan.instruction,
                        "reference_attributes": list(stage_plan.reference_attributes),
                        "text_fallbacks": dict(stage_plan.text_fallbacks),
                    },
                    keyframe_state={},
                    scene_description=plan_doc.intent.setting,
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
                stage["request"] = request_record
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
                if stage_plan.operation == "fuse_component":
                    base_condition_id = compiled["conditions"][0]["condition_ref"]
                    stage["qa"] = _visible_delta_qa(
                        selected_inputs[base_condition_id], output_data
                    )
                    stage["requires_review"] = stage["qa"][
                        "semantic_change_suspect"
                    ] or bool(stage_plan.quality_gate_ids)
                if stage_plan.quality_gate_ids:
                    if stage["qa"] is None:
                        stage["qa"] = {}
                    stage["qa"]["manual_quality_gate_ids"] = list(
                        stage_plan.quality_gate_ids
                    )
            stage["request"] = request_record
            stage["output"] = _stage_output(
                root=root,
                stage_id=stage["stage_id"],
                attempt=attempt,
                data=output_data,
            )
            metadata = _safe_provider_metadata(executor)
            if metadata is not None and stage["request"] is not None:
                stage["request"]["provider"] = metadata
        except Exception as exc:
            metadata = _safe_provider_metadata(executor)
            if metadata is not None and stage["request"] is not None:
                stage["request"]["provider"] = metadata
            stage["status"] = "failed"
            stage["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise

        stage["status"] = "completed"
        stage["approved"] = (
            stage_plan.operation in {"adopt_anchor", "composite_overlay"}
            or auto_approve
        ) and not stage["requires_review"]
        if stage["model_call"]:
            new_model_stages += 1
        _write_manifest(manifest_path, manifest)
        if not stage["approved"]:
            manifest["status"] = "awaiting_review"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)

    manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    return _result(root, manifest)
