"""Recoverable execution of one shot's first and last keyframes.

This is intentionally a thin coordinator over ``run_compose_keyframe``.  It
does not plan frames, choose a model, retry automatically, or call AniSora.
The Keyframe Planner still supplies the complete shot state, but this runner
selects exactly its first and last entries and records a stable shot-level
manifest.  Interior planned entries never trigger a model request.

This runner retains the earlier single-call first endpoint for compatibility.
The preferred high-quality first-frame path is now ``FirstFrameComposer``,
which can stage multiple bound references before handing its approved output
to endpoint/video continuity.  In this compatibility path, manual approval is
the default and the first endpoint may combine a canonical WHO with one
full-frame scene/source/style reference.  The last endpoint is generated only
after the first output has been approved, because that exact output becomes
the ``source_frame`` continuity reference.  Prompts contain only action/state
changes and uncovered visual deltas.  ``auto_approve`` exists for offline
tests and explicitly authorized unattended runs.
``max_new_frames`` defaults to one so a caller cannot accidentally expand
paid model usage.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from anime_remix.errors import InputValidationError, OutputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object, sha256_file
from anime_remix.services.execution.adapter import Adapter, Executor
from anime_remix.services.execution.artifact_store import (
    ArtifactStore,
    canonical_json_bytes,
)
from anime_remix.services.execution.ledger_writer import read_complete_records
from anime_remix.services.execution.orchestrator import (
    PORT_CHARACTER_CANDIDATE,
    run_compose_keyframe,
)
from anime_remix.services.execution.reference_package import (
    parse_reference_package,
)
from anime_remix.services.execution.shot_spec import parse_shot_spec
from anime_remix.services.script.keyframe_plan import parse_keyframe_plan

_SCHEMA_VERSION = "shot-keyframe-run-v1"
_CONTINUITY_REQUIREMENT_ID = "continuity.previous_keyframe"
_CONTINUITY_CONDITION_ID = "cond_previous_keyframe"


@dataclass(frozen=True)
class ShotKeyframeRunResult:
    """Current durable state of one shot-level keyframe run."""

    run_id: str
    run_dir: Path
    manifest_path: Path
    status: str
    next_keyframe_id: str | None
    generated_keyframe_ids: tuple[str, ...]


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


def _normalize_reference_packages(
    value: Mapping[str, object],
    *,
    selected_keyframe_ids: list[str],
    planned_keyframe_ids: list[str],
    shot_id: str,
) -> dict[str, dict]:
    if value.get("schema_version") == "reference-package-v1":
        raw_by_keyframe = {
            keyframe_id: value for keyframe_id in selected_keyframe_ids
        }
    else:
        missing = [key for key in selected_keyframe_ids if key not in value]
        extra = [key for key in value if key not in set(planned_keyframe_ids)]
        if missing or extra:
            raise InputValidationError(
                "reference_packages must contain the selected first/last "
                "keyframe ids and no unknown ids "
                f"(missing={missing}, unknown={extra})"
            )
        raw_by_keyframe = {
            keyframe_id: value[keyframe_id]
            for keyframe_id in selected_keyframe_ids
        }

    normalized: dict[str, dict] = {}
    for keyframe_id, raw in raw_by_keyframe.items():
        package = parse_reference_package(raw)
        if package.shot_id != shot_id:
            raise InputValidationError(
                f"reference package for {keyframe_id} has shot_id "
                f"{package.shot_id!r}, expected {shot_id!r}"
            )
        if keyframe_id != selected_keyframe_ids[0] and any(
            condition.role == "source_frame" for condition in package.conditions
        ):
            raise InputValidationError(
                "a static source_frame is allowed only for the first endpoint; "
                "the last endpoint reserves source_frame for the approved "
                f"first frame ({keyframe_id})"
            )
        normalized[keyframe_id] = package.model_dump(mode="json")
    return normalized


def _normalize_assets(
    asset_map: Mapping[str, str | Path],
    *,
    packages: Mapping[str, dict],
    selected_keyframe_ids: list[str],
    adapter: Adapter,
) -> tuple[dict[str, Path], dict[str, dict]]:
    paths = {ref: Path(path) for ref, path in asset_map.items()}
    required_refs: set[str] = set()
    selector = getattr(adapter, "shot_runner_static_condition_ids", None)
    if not callable(selector):
        raise InputValidationError(
            f"adapter {adapter.adapter_id!r} does not declare exact static "
            "reference selection"
        )
    for index, keyframe_id in enumerate(selected_keyframe_ids):
        package = packages[keyframe_id]
        selected_condition_ids = set(
            selector(
                package["conditions"],
                endpoint_role="first" if index == 0 else "last",
            )
        )
        for condition in package["conditions"]:
            if (
                condition["kind"] == "image"
                and condition["condition_id"] in selected_condition_ids
            ):
                required_refs.add(condition["payload_ref"])

    missing = sorted(ref for ref in required_refs if ref not in paths)
    if missing:
        raise InputValidationError(
            f"asset_map is missing model inputs: {missing}"
        )
    fingerprints: dict[str, dict] = {}
    for ref in sorted(required_refs):
        path = paths[ref]
        if not path.is_file():
            raise InputValidationError(
                f"model input is not a file: {path}", actual=ref
            )
        fingerprints[ref] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return paths, fingerprints


def _contract_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _new_manifest(
    *,
    run_id: str,
    shot_id: str,
    contract_sha256: str,
    adapter: Adapter,
    keyframes: list[dict],
    source_keyframe_count: int,
    auto_approve: bool,
) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "shot_id": shot_id,
        "status": "ready",
        "contract_sha256": contract_sha256,
        "execution": {
            "adapter_id": adapter.adapter_id,
            "model_id": adapter.model_id,
            "revision": adapter.revision,
            "auto_approve": auto_approve,
            "automatic_retries": 0,
            "continuity": "previous_approved_keyframe",
            "selection": "first_and_last",
            "per_invocation_new_frame_limit_default": 1,
        },
        "source_keyframe_count": source_keyframe_count,
        "selected_keyframe_count": len(keyframes),
        "completed_keyframe_count": 0,
        "approved_keyframe_count": 0,
        "next_keyframe_id": keyframes[0]["keyframe_id"],
        "frames": [
            {
                "keyframe_id": frame["keyframe_id"],
                "order": frame["order"],
                "time_seconds": frame["time_seconds"],
                "position": frame["position"],
                "status": "pending",
                "approved": False,
                "attempts": 0,
                "previous_keyframe_id": None,
                "continuity_artifact_ref": None,
                "selected_reference_condition_ids": [],
                "selected_reference_roles": [],
                "text_fallback_fields": [],
                "output": None,
                "error": None,
            }
            for frame in keyframes
        ],
    }


def _refresh_counts(manifest: dict) -> None:
    frames = manifest["frames"]
    manifest["completed_keyframe_count"] = sum(
        frame["status"] == "completed" for frame in frames
    )
    manifest["approved_keyframe_count"] = sum(
        frame["status"] == "completed" and frame["approved"]
        for frame in frames
    )
    next_frame = next(
        (
            frame
            for frame in frames
            if frame["status"] != "completed" or not frame["approved"]
        ),
        None,
    )
    manifest["next_keyframe_id"] = (
        next_frame["keyframe_id"] if next_frame is not None else None
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    _refresh_counts(manifest)
    dump_json_atomic(path, manifest, sort_keys=True)


def _validate_existing_output(root: Path, frame: dict) -> None:
    output = frame.get("output")
    if not isinstance(output, dict):
        raise InputValidationError(
            f"completed frame {frame['keyframe_id']} has no output record"
        )
    named_path = root / output["named_path"]
    if not named_path.is_file():
        raise InputValidationError(
            f"completed frame output is missing: {named_path}"
        )
    actual = sha256_file(named_path)
    if actual != output["sha256"]:
        raise InputValidationError(
            f"completed frame output drifted: {frame['keyframe_id']}",
            actual=actual,
        )


def _continuity_package(
    base: dict,
    *,
    keyframe_id: str,
    previous_frame: dict,
) -> tuple[dict, str]:
    package = copy.deepcopy(base)
    requirement_ids = {
        requirement["requirement_id"] for requirement in package["requirements"]
    }
    condition_ids = {
        condition["condition_id"] for condition in package["conditions"]
    }
    if _CONTINUITY_REQUIREMENT_ID in requirement_ids:
        raise InputValidationError(
            f"reserved requirement id already exists: {_CONTINUITY_REQUIREMENT_ID}"
        )
    if _CONTINUITY_CONDITION_ID in condition_ids:
        raise InputValidationError(
            f"reserved condition id already exists: {_CONTINUITY_CONDITION_ID}"
        )
    artifact_ref = previous_frame["output"]["artifact_ref"]
    package["requirements"].append(
        {
            "requirement_id": _CONTINUITY_REQUIREMENT_ID,
            "constraint": "preserve continuity from the previous approved keyframe",
            "priority": "required",
        }
    )
    package["conditions"].append(
        {
            "condition_id": _CONTINUITY_CONDITION_ID,
            "role": "source_frame",
            "kind": "image",
            "payload_ref": artifact_ref,
            "satisfied_constraints": [_CONTINUITY_REQUIREMENT_ID],
            "scores": {"continuity": 1.0},
            "provenance": {"source_asset_id": None, "derived_from": artifact_ref},
        }
    )
    package["candidate_sets"].append(
        {
            "requirement_id": _CONTINUITY_REQUIREMENT_ID,
            "scope": "keyframe",
            "keyframe_id": keyframe_id,
            "candidates": [_CONTINUITY_CONDITION_ID],
        }
    )
    return parse_reference_package(package).model_dump(mode="json"), artifact_ref


def _artifact_output(
    *,
    root: Path,
    frame_dir: Path,
    attempt_dir: Path,
    artifact_ref: str,
) -> dict:
    artifact_id = artifact_ref.rsplit("/", 1)[-1]
    record = next(
        (
            item
            for item in read_complete_records(attempt_dir / "execution-ledger.jsonl")
            if item.record_type == "artifact_registered"
            and item.payload.artifact_id == artifact_id
        ),
        None,
    )
    if record is None:
        raise OutputValidationError(
            f"artifact record not found for {artifact_ref}"
        )
    store = ArtifactStore(attempt_dir)
    blob_path = store.blob_path(record.payload.blob_ref)
    data = store.read_bytes(record.payload.blob_ref)
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise OutputValidationError(
                    f"keyframe output is not a valid non-empty PNG: {artifact_ref}"
                )
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise OutputValidationError(
            f"cannot decode keyframe PNG: {artifact_ref}", actual=str(exc)
        ) from exc

    named_path = frame_dir / "keyframe.png"
    temporary_path = frame_dir / ".keyframe.png.tmp"
    frame_dir.mkdir(parents=True, exist_ok=True)
    temporary_path.write_bytes(data)
    temporary_path.replace(named_path)
    digest = sha256_file(named_path)
    expected = record.payload.blob_ref.removeprefix("blob://sha256:")
    if digest != expected:
        raise OutputValidationError(
            f"materialized keyframe hash mismatch: {artifact_ref}",
            actual=digest,
        )
    return {
        "artifact_ref": artifact_ref,
        "blob_ref": record.payload.blob_ref,
        "blob_path": blob_path.relative_to(root).as_posix(),
        "named_path": named_path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": len(data),
        "width": width,
        "height": height,
    }


def _result(root: Path, manifest: dict) -> ShotKeyframeRunResult:
    return ShotKeyframeRunResult(
        run_id=manifest["run_id"],
        run_dir=root,
        manifest_path=root / "shot-keyframe-run.json",
        status=manifest["status"],
        next_keyframe_id=manifest["next_keyframe_id"],
        generated_keyframe_ids=tuple(
            frame["keyframe_id"]
            for frame in manifest["frames"]
            if frame["status"] == "completed"
        ),
    )


def run_shot_keyframes(
    *,
    run_dir: str | Path,
    run_id: str,
    shot_spec: dict,
    keyframe_plan: dict,
    reference_packages: Mapping[str, object],
    scene_geometry: dict,
    scene_crop: dict,
    asset_map: Mapping[str, str | Path],
    adapter: Adapter,
    executor: Executor,
    scene_image_path: str | Path | None = None,
    canvas: tuple[int, int] = (1280, 720),
    approved_keyframe_ids: Collection[str] = (),
    auto_approve: bool = False,
    max_new_frames: int = 1,
) -> ShotKeyframeRunResult:
    """Execute and durably resume a shot's first and last keyframes.

    No automatic retry occurs.  A failed frame is recorded and retried only
    when the caller deliberately invokes this function again with the same
    frozen contract.  The default manual mode pauses after every new frame;
    the caller approves it by passing its id in ``approved_keyframe_ids`` on a
    later invocation.  Interior plan entries are never executed.
    ``max_new_frames`` is a hard per-invocation ceiling.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise InputValidationError("run_id must be a non-empty string")
    if not isinstance(auto_approve, bool):
        raise InputValidationError("auto_approve must be a boolean")
    if (
        isinstance(max_new_frames, bool)
        or not isinstance(max_new_frames, int)
        or max_new_frames < 1
    ):
        raise InputValidationError("max_new_frames must be a positive integer")

    spec = parse_shot_spec(shot_spec)
    plan = parse_keyframe_plan(keyframe_plan)
    if plan.shot_id != spec.shot_id:
        raise InputValidationError(
            f"keyframe plan shot_id {plan.shot_id!r} != {spec.shot_id!r}"
        )
    if plan.shot_duration_seconds != spec.duration_seconds:
        raise InputValidationError(
            "keyframe plan duration does not match ShotSpec duration"
        )
    if len(plan.keyframes) > 1 and not getattr(
        adapter, "supports_previous_keyframe", False
    ):
        raise InputValidationError(
            f"adapter {adapter.adapter_id!r} does not support previous "
            "approved keyframe continuity"
        )
    if getattr(adapter, "uses_visual_how", False):
        raise InputValidationError(
            "first/last execution cannot use visual-HOW because the last "
            "frame reserves Image 2 for previous-frame continuity"
        )

    planned_keyframe_ids = [frame.keyframe_id for frame in plan.keyframes]
    selected_keyframes = [plan.keyframes[0], plan.keyframes[-1]]
    keyframe_ids = [frame.keyframe_id for frame in selected_keyframes]
    packages = _normalize_reference_packages(
        reference_packages,
        selected_keyframe_ids=keyframe_ids,
        planned_keyframe_ids=planned_keyframe_ids,
        shot_id=spec.shot_id,
    )
    paths, asset_fingerprints = _normalize_assets(
        asset_map,
        packages=packages,
        selected_keyframe_ids=keyframe_ids,
        adapter=adapter,
    )
    scene_path = (
        Path(scene_image_path)
        if scene_image_path is not None
        else next(iter(paths.values()))
    )
    if scene_image_path is not None and not scene_path.is_file():
        raise InputValidationError(
            f"scene_image_path is not a file: {scene_path}"
        )

    normalized_spec = spec.model_dump(mode="json")
    normalized_plan = plan.model_dump(mode="json")
    contract = {
        "shot_spec": normalized_spec,
        "keyframe_plan": normalized_plan,
        "reference_packages": packages,
        "asset_fingerprints": asset_fingerprints,
        "scene_geometry": scene_geometry,
        "scene_crop": scene_crop,
        "canvas": list(canvas),
        "adapter": _adapter_contract(adapter),
    }
    digest = _contract_sha256(contract)
    root = Path(run_dir)
    manifest_path = root / "shot-keyframe-run.json"
    if manifest_path.exists():
        manifest = load_json_object(manifest_path)
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise InputValidationError("unsupported shot keyframe run manifest")
        if manifest.get("run_id") != run_id:
            raise InputValidationError("run_id does not match existing manifest")
        if manifest.get("contract_sha256") != digest:
            raise InputValidationError(
                "cannot resume: shot keyframe execution contract changed"
            )
        if manifest["execution"]["auto_approve"] is not auto_approve:
            raise InputValidationError(
                "cannot change auto_approve while resuming a shot run"
            )
        for frame in manifest["frames"]:
            if frame["status"] == "running":
                frame["status"] = "failed"
                frame["error"] = {
                    "type": "InterruptedRun",
                    "message": "previous attempt ended before completion",
                }
            if frame["status"] == "completed":
                _validate_existing_output(root, frame)
    else:
        manifest = _new_manifest(
            run_id=run_id,
            shot_id=spec.shot_id,
            contract_sha256=digest,
            adapter=adapter,
            keyframes=[
                normalized_plan["keyframes"][0],
                normalized_plan["keyframes"][-1],
            ],
            source_keyframe_count=len(normalized_plan["keyframes"]),
            auto_approve=auto_approve,
        )
        _write_manifest(manifest_path, manifest)

    approved = set(approved_keyframe_ids)
    unknown_approvals = sorted(approved - set(keyframe_ids))
    if unknown_approvals:
        raise InputValidationError(
            f"approved_keyframe_ids contains unknown ids: {unknown_approvals}"
        )
    completed_ids = {
        frame["keyframe_id"]
        for frame in manifest["frames"]
        if frame["status"] == "completed"
    }
    premature = sorted(approved - completed_ids)
    if premature and not auto_approve:
        raise InputValidationError(
            "cannot approve keyframes before their outputs exist: "
            f"{premature}"
        )
    for frame in manifest["frames"]:
        if frame["status"] == "completed" and (
            auto_approve or frame["keyframe_id"] in approved
        ):
            frame["approved"] = True

    new_frames = 0
    previous_frame: dict | None = None
    for frame in manifest["frames"]:
        keyframe_id = frame["keyframe_id"]
        if frame["status"] == "completed":
            if not frame["approved"]:
                manifest["status"] = "awaiting_review"
                _write_manifest(manifest_path, manifest)
                return _result(root, manifest)
            previous_frame = frame
            continue
        if new_frames >= max_new_frames:
            manifest["status"] = "paused_limit"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)

        frame_dir = root / "frames" / f"{frame['order']:03d}_{keyframe_id}"
        frame["attempts"] += 1
        attempt_dir = frame_dir / f"attempt_{frame['attempts']:03d}"
        frame["status"] = "running"
        frame["error"] = None
        frame["previous_keyframe_id"] = (
            previous_frame["keyframe_id"] if previous_frame else None
        )
        package = packages[keyframe_id]
        frame_asset_map = dict(paths)
        prompt_fallback_fields: list[str]
        if previous_frame is not None:
            package, continuity_ref = _continuity_package(
                package,
                keyframe_id=keyframe_id,
                previous_frame=previous_frame,
            )
            previous_path = root / previous_frame["output"]["named_path"]
            frame_asset_map[continuity_ref] = previous_path
            frame["continuity_artifact_ref"] = continuity_ref
            first_state = normalized_plan["keyframes"][0]
            current_state = normalized_plan["keyframes"][-1]
            prompt_fallback_fields = [
                field
                for field in (
                    "composition",
                    "camera",
                    "background_state",
                    "foreground_state",
                    "prop_state",
                )
                if current_state[field] != first_state[field]
            ]
            selected_condition_ids = [
                *adapter.shot_runner_static_condition_ids(
                    package["conditions"], endpoint_role="last"
                ),
                _CONTINUITY_CONDITION_ID,
            ]
        else:
            selector = adapter.shot_runner_static_condition_ids
            selected_condition_ids = list(
                selector(package["conditions"], endpoint_role="first")
            )
            selected_ids = set(selected_condition_ids)
            selected_roles = {
                condition["role"]
                for condition in package["conditions"]
                if condition["condition_id"] in selected_ids
            }
            prompt_fallback_fields = (
                []
                if selected_roles & {"scene", "source_frame"}
                else [
                    "composition",
                    "camera",
                    "background_state",
                    "foreground_state",
                    "prop_state",
                ]
            )
        roles_by_id = {
            condition["condition_id"]: condition["role"]
            for condition in package["conditions"]
        }
        frame["selected_reference_condition_ids"] = selected_condition_ids
        frame["selected_reference_roles"] = [
            roles_by_id[condition_id]
            for condition_id in selected_condition_ids
        ]
        frame["text_fallback_fields"] = prompt_fallback_fields
        manifest["status"] = "running"
        _write_manifest(manifest_path, manifest)

        try:
            result = run_compose_keyframe(
                run_dir=attempt_dir,
                run_id=(
                    f"{run_id}-{keyframe_id}-attempt-{frame['attempts']:03d}"
                ),
                shot_spec=normalized_spec,
                keyframe_plan=normalized_plan,
                reference_package=package,
                scene_image_path=scene_path,
                scene_geometry=scene_geometry,
                scene_crop=scene_crop,
                asset_map=frame_asset_map,
                adapter=adapter,
                executor=executor,
                canvas=canvas,
                keyframe_id=keyframe_id,
                stop_after="character_synthesis",
                keyframe_prompt_policy={
                    "text_fallback_fields": prompt_fallback_fields,
                },
            )
            artifact_ref = result.ports[PORT_CHARACTER_CANDIDATE]
            frame["output"] = _artifact_output(
                root=root,
                frame_dir=frame_dir,
                attempt_dir=attempt_dir,
                artifact_ref=artifact_ref,
            )
        except Exception as exc:
            frame["status"] = "failed"
            frame["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise

        frame["status"] = "completed"
        frame["approved"] = auto_approve
        new_frames += 1
        previous_frame = frame
        if not frame["approved"]:
            manifest["status"] = "awaiting_review"
            _write_manifest(manifest_path, manifest)
            return _result(root, manifest)
        _write_manifest(manifest_path, manifest)

    manifest["status"] = "completed"
    _write_manifest(manifest_path, manifest)
    return _result(root, manifest)
