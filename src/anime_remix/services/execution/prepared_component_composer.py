"""Recoverable execution of approved WHO/HOW component preparation tasks.

The runner executes independent tasks from ``prepared-component-plan-v1``.
Only ``model_inputs`` are read and uploaded; structure-only control evidence
is retained in the manifest by id and never resolved to bytes.  Every output
is a generated candidate that requires separate human review, catalog
registration and promotion before the preparation plan can be completed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from anime_remix.errors import InputValidationError, OutputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object, sha256_file
from anime_remix.services.execution.adapter import Adapter, Executor
from anime_remix.services.execution.artifact_store import canonical_json_bytes
from anime_remix.services.script.prepared_component_plan import (
    PreparedComponentPlanDocument,
    PreparedComponentTask,
    parse_prepared_component_plan,
)

_SCHEMA_VERSION = "prepared-component-compose-run-v1"


@dataclass(frozen=True)
class PreparedComponentComposeResult:
    """Current durable state of one component-generation run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    status: str
    next_task_id: str | None
    completed_task_ids: tuple[str, ...]
    output_paths: tuple[tuple[str, Path], ...]


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


def _model_input_ids(plan: PreparedComponentPlanDocument) -> list[str]:
    return list(
        dict.fromkeys(
            item.asset_id for task in plan.tasks for item in task.model_inputs
        )
    )


def _normalize_assets(
    asset_map: Mapping[str, str | Path],
    *,
    plan: PreparedComponentPlanDocument,
) -> tuple[dict[str, Path], dict[str, dict]]:
    paths = {asset_id: Path(path) for asset_id, path in asset_map.items()}
    required = _model_input_ids(plan)
    missing = sorted(set(required) - set(paths))
    if missing:
        raise InputValidationError(
            f"asset_map is missing component model inputs: {missing}"
        )
    fingerprints: dict[str, dict] = {}
    for asset_id in required:
        path = paths[asset_id]
        if not path.is_file():
            raise InputValidationError(
                f"component model input is not a file: {path}", asset_id=asset_id
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise InputValidationError(
                f"cannot read component model input {asset_id!r}", actual=str(exc)
            ) from exc
        _, width, height = _normalize_png(data, label=f"asset {asset_id}")
        fingerprints[asset_id] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "width": width,
            "height": height,
        }
    return paths, fingerprints


def _condition_role(function: str) -> str:
    return {"who": "identity", "how": "pose", "prop_visual": "prop"}[function]


def _resolve_task_inputs(
    task: PreparedComponentTask,
    *,
    asset_paths: Mapping[str, Path],
) -> tuple[list[dict], dict[str, bytes]]:
    conditions: list[dict] = []
    inputs: dict[str, bytes] = {}
    for slot, item in enumerate(task.model_inputs, start=1):
        condition_id = f"cond_{task.task_id.replace('.', '_')}_{slot}"
        path = asset_paths[item.asset_id]
        try:
            inputs[condition_id] = path.read_bytes()
        except OSError as exc:
            raise InputValidationError(
                f"cannot read component task input {item.asset_id!r}",
                actual=str(exc),
            ) from exc
        conditions.append(
            {
                "condition_id": condition_id,
                "role": _condition_role(item.function),
                "kind": "image",
                "payload_ref": f"asset://anime-remix/catalog/{item.asset_id}@v1",
                "satisfied_constraints": [
                    f"prepared_component.task.{task.task_id}.{item.function}"
                ],
                "scores": {"selected": 1.0},
                "provenance": {
                    "source_asset_id": item.asset_id,
                    "derived_from": None,
                },
            }
        )
    return conditions, inputs


def _compile_task(
    task: PreparedComponentTask,
    *,
    adapter: Adapter,
    conditions: list[dict],
) -> dict:
    return adapter.compile(
        operation="first_frame_fusion",
        intent={
            "task_id": task.task_id,
            "stage_operation": "synthesize_component",
            "component_ids": list(task.component_ids),
            "instruction": (
                "Create only the planned component or interacting component group; "
                "keep all image-covered appearance facts unchanged."
            ),
            "reference_attributes": list(task.preserve_attributes),
            "text_fallbacks": dict(task.allowed_text_fallbacks),
        },
        keyframe_state={},
        scene_description="component preparation",
        conditions=conditions,
    )


def _write_output(
    *,
    root: Path,
    task: PreparedComponentTask,
    attempt: int,
    data: bytes,
) -> dict:
    normalized, width, height = _normalize_png(
        data, label=f"component task {task.task_id} output"
    )
    task_dir = root / "tasks" / task.task_id
    attempt_dir = task_dir / f"attempt_{attempt:03d}"
    candidate_dir = root / "candidates"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = attempt_dir / "output.png"
    candidate_path = candidate_dir / f"{task.output_asset_id}.png"
    attempt_tmp = attempt_dir / ".output.png.tmp"
    candidate_tmp = candidate_dir / f".{task.output_asset_id}.png.tmp"
    attempt_tmp.write_bytes(normalized)
    attempt_tmp.replace(attempt_path)
    candidate_tmp.write_bytes(normalized)
    candidate_tmp.replace(candidate_path)
    return {
        "asset_id": task.output_asset_id,
        "path": candidate_path.relative_to(root).as_posix(),
        "attempt_path": attempt_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(normalized).hexdigest(),
        "bytes": len(normalized),
        "width": width,
        "height": height,
        "catalog_status": "unregistered_generated_candidate",
    }


def _validate_output(root: Path, task: dict) -> None:
    output = task.get("output")
    if not isinstance(output, dict):
        raise InputValidationError(
            f"completed component task {task['task_id']} has no output record"
        )
    path = root / output["path"]
    if not path.is_file():
        raise InputValidationError(f"component output is missing: {path}")
    actual = sha256_file(path)
    if actual != output["sha256"]:
        raise InputValidationError(
            f"completed component output drifted: {task['task_id']}", actual=actual
        )
    _normalize_png(path.read_bytes(), label=f"task {task['task_id']} output")


def _new_manifest(
    *,
    run_id: str,
    plan: PreparedComponentPlanDocument,
    contract_sha256: str,
    adapter: Adapter,
) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "plan_id": plan.plan_id,
        "shot_id": plan.shot_id,
        "content_plan_id": plan.content_plan_id,
        "status": "ready",
        "contract_sha256": contract_sha256,
        "execution": {
            "adapter_id": adapter.adapter_id,
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "automatic_retries": 0,
            "per_invocation_new_task_limit_default": 1,
            "max_primary_visual_references_per_model_call": 2,
        },
        "control_evidence": {
            "asset_ids": sorted(
                {
                    item
                    for task in plan.tasks
                    for item in task.control_evidence_asset_ids
                }
            ),
            "resolved_as_model_input": False,
            "uploaded_to_model": False,
        },
        "completed_task_count": 0,
        "model_call_count": 0,
        "next_task_id": plan.tasks[0].task_id if plan.tasks else None,
        "tasks": [
            {
                "task_id": task.task_id,
                "kind": task.kind,
                "component_ids": list(task.component_ids),
                "subjects": list(task.subjects),
                "output_asset_id": task.output_asset_id,
                "model_input_asset_ids": [
                    item.asset_id for item in task.model_inputs
                ],
                "control_evidence_asset_ids": list(
                    task.control_evidence_asset_ids
                ),
                "status": "pending",
                "attempts": 0,
                "request": None,
                "output": None,
                "requires_manual_review": True,
                "review_criteria": {
                    "preserve_attributes": list(task.preserve_attributes),
                    "target_state": task.target_state,
                    "plate_scope": (
                        "This plate must show only the planned component group. "
                        "Any target-side visibility item in external_attachments "
                        "is final-frame-only: the target scene object and its "
                        "anchor must be absent from the plate."
                    ),
                    "prop_affordances": [
                        item.model_dump(mode="json")
                        for item in task.prop_affordances
                    ],
                    "external_attachments": [
                        item.model_dump(mode="json")
                        for item in task.external_attachments
                    ],
                    "structured_gates": [
                        item.model_dump(mode="json") for item in task.review_gates
                    ],
                    "reject": [
                        "wrong identity, clothing, prop appearance or native style",
                        "malformed anatomy, false contact or floating objects",
                        (
                            "reversed prop affordance, wrong action axis or source "
                            "anchor position, target scene object rendered in the "
                            "plate, missing source-side required elements"
                        ),
                        "storyboard-like or visibly low-fidelity geometry",
                        "copied background, label, watermark or reference layout",
                    ],
                },
                "error": None,
            }
            for task in plan.tasks
        ],
    }


def _refresh_manifest(manifest: dict) -> None:
    tasks = manifest["tasks"]
    completed = [item for item in tasks if item["status"] == "completed"]
    manifest["completed_task_count"] = len(completed)
    manifest["model_call_count"] = sum(
        item["attempts"] for item in tasks if item["status"] in {"completed", "failed"}
    )
    next_task = next((item for item in tasks if item["status"] != "completed"), None)
    manifest["next_task_id"] = (
        next_task["task_id"] if next_task is not None else None
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    _refresh_manifest(manifest)
    dump_json_atomic(path, manifest, sort_keys=True)


def _result(root: Path, manifest: dict) -> PreparedComponentComposeResult:
    return PreparedComponentComposeResult(
        run_id=manifest["run_id"],
        run_dir=root,
        manifest_path=root / "prepared-component-compose-run.json",
        status=manifest["status"],
        next_task_id=manifest["next_task_id"],
        completed_task_ids=tuple(
            item["task_id"]
            for item in manifest["tasks"]
            if item["status"] == "completed"
        ),
        output_paths=tuple(
            (item["output_asset_id"], root / item["output"]["path"])
            for item in manifest["tasks"]
            if item["status"] == "completed"
        ),
    )


def run_prepared_component_composition(
    *,
    run_dir: str | Path,
    run_id: str,
    plan: PreparedComponentPlanDocument | dict,
    asset_map: Mapping[str, str | Path],
    adapter: Adapter,
    executor: Executor,
    max_new_model_tasks: int = 1,
) -> PreparedComponentComposeResult:
    """Execute or resume approved component preparation without auto-acceptance."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise InputValidationError("run_id must be a non-empty string")
    if (
        isinstance(max_new_model_tasks, bool)
        or not isinstance(max_new_model_tasks, int)
        or max_new_model_tasks < 1
    ):
        raise InputValidationError("max_new_model_tasks must be a positive integer")
    plan_doc = (
        plan
        if isinstance(plan, PreparedComponentPlanDocument)
        else parse_prepared_component_plan(plan)
    )
    if plan_doc.review_status != "approved":
        raise InputValidationError(
            "prepared component plan must be approved before execution"
        )
    if plan_doc.completion_status != "pending":
        raise InputValidationError("completed component plan cannot execute again")
    if plan_doc.decision == "blocked":
        raise InputValidationError("blocked component plan cannot execute")
    if not plan_doc.tasks:
        raise InputValidationError("component plan has no generation tasks")
    if not getattr(adapter, "supports_first_frame_fusion", False):
        raise InputValidationError(
            f"adapter {adapter.adapter_id!r} does not support component synthesis"
        )

    asset_paths, fingerprints = _normalize_assets(asset_map, plan=plan_doc)
    contract = {
        "plan": plan_doc.model_dump(mode="json"),
        "asset_fingerprints": fingerprints,
        "adapter": _adapter_contract(adapter),
    }
    digest = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "prepared-component-compose-run.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise InputValidationError("unsupported component run manifest")
        if manifest.get("run_id") != run_id:
            raise InputValidationError("run_id does not match component manifest")
        if manifest.get("contract_sha256") != digest:
            raise InputValidationError(
                "cannot resume: component composition contract changed"
            )
        for task in manifest["tasks"]:
            if task["status"] == "running":
                task["status"] = "failed"
                task["error"] = {
                    "type": "InterruptedRun",
                    "message": "previous task attempt ended before completion",
                }
            if task["status"] == "completed":
                _validate_output(root, task)
    else:
        manifest = _new_manifest(
            run_id=run_id,
            plan=plan_doc,
            contract_sha256=digest,
            adapter=adapter,
        )
        _write_manifest(manifest_path, manifest)

    plan_by_id = {item.task_id: item for item in plan_doc.tasks}
    new_tasks = 0
    for task in manifest["tasks"]:
        if task["status"] == "completed":
            continue
        if new_tasks >= max_new_model_tasks:
            manifest["status"] = "paused_limit"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)
        task_plan = plan_by_id[task["task_id"]]
        task["attempts"] += 1
        attempt = task["attempts"]
        attempt_dir = root / "tasks" / task_plan.task_id / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        task["status"] = "running"
        task["error"] = None
        manifest["status"] = "running"
        _write_manifest(manifest_path, manifest)
        try:
            conditions, inputs = _resolve_task_inputs(
                task_plan, asset_paths=asset_paths
            )
            compiled = _compile_task(
                task_plan, adapter=adapter, conditions=conditions
            )
            request_path = attempt_dir / "request.json"
            dump_json_atomic(request_path, compiled, sort_keys=True)
            task["request"] = {
                "path": request_path.relative_to(root).as_posix(),
                "sha256": sha256_file(request_path),
                "adapter_id": compiled["adapter_id"],
                "condition_count": len(compiled["conditions"]),
                "input_asset_ids": [item.asset_id for item in task_plan.model_inputs],
                "control_evidence_uploaded": False,
                "prompt_sha256": hashlib.sha256(
                    compiled["prompt"].encode("utf-8")
                ).hexdigest(),
            }
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
            task["output"] = _write_output(
                root=root,
                task=task_plan,
                attempt=attempt,
                data=output_data,
            )
            metadata = _safe_provider_metadata(executor)
            if metadata is not None:
                task["request"]["provider"] = metadata
        except Exception as exc:
            metadata = _safe_provider_metadata(executor)
            if metadata is not None and task.get("request") is not None:
                task["request"]["provider"] = metadata
            task["status"] = "failed"
            task["error"] = {"type": type(exc).__name__, "message": str(exc)}
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise
        task["status"] = "completed"
        new_tasks += 1
        _write_manifest(manifest_path, manifest)

    manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    return _result(root, manifest)
