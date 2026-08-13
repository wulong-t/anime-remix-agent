"""Thin fixed-path orchestrator for one compose keyframe (Phase 3).

This is deliberately NOT a generic DAG Runner: it calls the frozen compose
stages in a fixed order and records every fact in the Execution Ledger.  Its
only job is to prove that the frozen contracts connect end-to-end.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.adapter import Adapter, Executor
from anime_remix.services.execution.artifact_store import (
    ArtifactStore,
    canonical_json_bytes,
    register_artifact,
)
from anime_remix.services.execution.execution_ledger import LedgerRecord
from anime_remix.services.execution.imaging import (
    image_to_png_bytes,
    load_rgb,
)
from anime_remix.services.execution.layout_plan import parse_layout_plan
from anime_remix.services.execution.ledger_writer import LedgerWriter
from anime_remix.services.execution.shot_spec import (
    ShotSpecDocument,
    parse_shot_spec,
)
from anime_remix.services.execution.stages.character_extract import (
    extract_character_layer,
)
from anime_remix.services.execution.stages.character_validate import (
    validate_character_layer,
)
from anime_remix.services.execution.stages.composite import (
    composite_keyframe,
)
from anime_remix.services.execution.stages.final_validate import (
    validate_final_keyframe,
)
from anime_remix.services.execution.stages.geometry_extract import (
    extract_character_geometry,
)
from anime_remix.services.execution.stages.layout import (
    derive_layout_intent,
    plan_layout,
)
from anime_remix.services.execution.stages.mask_generate import (
    generate_masks,
)

PORT_CHARACTER_CANDIDATE = "character_candidate"
PORT_CHARACTER_LAYER = "character_layer"
PORT_CHARACTER_GEOMETRY = "character_geometry"
PORT_LAYOUT_INTENT = "layout_intent"
PORT_LAYOUT_PLAN = "layout_plan"
PORT_COMPOSITE_MASK = "composite_mask"
PORT_INPAINT_MASK = "inpaint_mask"
PORT_COMPOSITE_IMAGE = "composite_image"
PORT_FINAL_KEYFRAME = "final_keyframe"

_TEMPLATE_REF = "compose-extract-alpha-v1"
_CONSTRAINT_SET_REF = "constraints_003"


@dataclass
class ComposeRunResult:
    run_id: str
    run_dir: Path
    ledger_path: Path
    final_keyframe_ref: str | None
    ports: dict[str, str]


@dataclass
class NodeOutcome:
    refs: list[str]
    datas: list[object]
    finished: LedgerRecord


def _port_ref(plan_id: str, port_name: str) -> str:
    return f"plan://{plan_id}/ports/{port_name}"


def _input_hash(compiled: dict) -> str:
    canonical = canonical_json_bytes(
        {
            "conditions": compiled["conditions"],
            "prompt": compiled["prompt"],
            "parameters": compiled["parameters"],
        }
    )
    return hashlib.sha256(canonical).hexdigest()


def _scene_description(shot_spec: ShotSpecDocument) -> str:
    parts = [
        (
            f"scene {shot_spec.scene_id} "
            f"({shot_spec.locks.scene.time_of_day})"
        ),
        shot_spec.narrative_purpose,
    ]
    if shot_spec.compose is not None:
        parts.extend(
            f"{r.subject} {r.relation} {r.object}"
            for r in shot_spec.compose.spatial_relations
        )
    return "; ".join(parts)


def _node_run_started(
    writer: LedgerWriter,
    *,
    plan_id: str,
    node_id: str,
    operation: str,
    instance_id: str,
    node_type: str,
    inputs: list[str],
) -> LedgerRecord:
    return writer.append(
        {
            "record_type": "node_run_started",
            "causal_refs": [],
            "payload": {
                "instance_id": instance_id,
                "plan_id": plan_id,
                "node_id": node_id,
                "operation": operation,
                "node_type": node_type,
                "inputs": inputs,
                "started_at": "2026-08-11T12:00:00+08:00",
            },
        }
    )


def _node_run_finished(
    writer: LedgerWriter,
    *,
    started: LedgerRecord,
    outputs: list[str],
    status: str = "success",
) -> LedgerRecord:
    return writer.append(
        {
            "record_type": "node_run_finished",
            "causal_refs": [],
            "payload": {
                "instance_id": started.payload.instance_id,
                "started_ref": (
                    f"ledger://{writer.run_ref}/{started.record_id}"
                ),
                "outputs": outputs,
                "status": status,
                "finished_at": "2026-08-11T12:00:01+08:00",
            },
        }
    )


def _bind_port(
    writer: LedgerWriter,
    *,
    plan_id: str,
    port_name: str,
    artifact_ref: str,
) -> None:
    writer.append(
        {
            "record_type": "port_bound",
            "causal_refs": [],
            "payload": {
                "binding_id": f"bind-{port_name}",
                "logical_port_ref": _port_ref(plan_id, port_name),
                "artifact_ref": artifact_ref,
            },
        }
    )


def _validation_gate(
    writer: LedgerWriter,
    *,
    node_id: str,
    derived_from: LedgerRecord,
    valid: bool,
    checks: list[dict],
) -> None:
    if not valid:
        raise InputValidationError(f"{node_id} validation failed: {checks}")
    writer.append(
        {
            "record_type": "validation_result",
            "causal_refs": [
                {
                    "record_ref": (
                        f"ledger://{writer.run_ref}/"
                        f"{derived_from.record_id}"
                    ),
                    "relation": "derived_from",
                }
            ],
            "payload": {
                "result_id": f"result-{node_id}",
                "node_id": node_id,
                "valid": True,
                "checks": checks,
                "failure_category": None,
            },
        }
    )


def _deterministic_node(
    writer: LedgerWriter,
    store: ArtifactStore,
    *,
    plan_id: str,
    node_id: str,
    operation: str,
    instance_id: str,
    inputs: list[str],
    productions: list[dict],
) -> NodeOutcome:
    started = _node_run_started(
        writer,
        plan_id=plan_id,
        node_id=node_id,
        operation=operation,
        instance_id=instance_id,
        node_type="deterministic",
        inputs=inputs,
    )
    refs: list[str] = []
    datas: list[object] = []
    for production in productions:
        artifact_ref, _ = register_artifact(
            store,
            writer,
            artifact_kind=production["artifact_kind"],
            schema_version=production["schema_version"],
            producer_started_ref=(
                f"ledger://{writer.run_ref}/{started.record_id}"
            ),
            data=production["data"],
            canonicalize=production.get("canonicalize", False),
        )
        refs.append(artifact_ref)
        datas.append(production["data"])
    finished = _node_run_finished(writer, started=started, outputs=refs)
    return NodeOutcome(refs=refs, datas=datas, finished=finished)


def _model_node(
    writer: LedgerWriter,
    store: ArtifactStore,
    *,
    plan_id: str,
    node_id: str,
    operation: str,
    instance_id: str,
    inputs: list[str],
    intent: dict,
    visual_intent: dict,
    plan_instantiated_ref: str,
    adapter: Adapter,
    executor: Executor,
    keyframe_state: dict,
    scene_description: str,
    conditions: list[dict],
    executor_inputs: dict[str, bytes | Path],
    artifact_kind: str,
    schema_version: str,
) -> NodeOutcome:
    started = _node_run_started(
        writer,
        plan_id=plan_id,
        node_id=node_id,
        operation=operation,
        instance_id=instance_id,
        node_type="model",
        inputs=inputs,
    )
    intent_record = writer.append(
        {
            "record_type": "render_intent_created",
            "causal_refs": [
                {
                    "record_ref": plan_instantiated_ref,
                    "relation": "triggered_by",
                }
            ],
            "payload": {
                "intent_id": intent["intent_id"],
                "operation": operation,
                "shot_id": intent["shot_id"],
                "keyframe_id": intent["keyframe_id"],
                "requirements": intent["requirements"],
                "reference_package_ref": intent["reference_package_ref"],
                "constraint_set_ref": _CONSTRAINT_SET_REF,
            },
        }
    )
    compiled = adapter.compile(
        operation=operation,
        intent=visual_intent,
        keyframe_state=keyframe_state,
        scene_description=scene_description,
        conditions=conditions,
    )
    request_record = writer.append(
        {
            "record_type": "model_render_request_created",
            "causal_refs": [
                {
                    "record_ref": (
                        f"ledger://{writer.run_ref}/"
                        f"{intent_record.record_id}"
                    ),
                    "relation": "input",
                }
            ],
            "payload": {
                "request_id": f"req-{node_id}",
                "intent_ref": (
                    f"ledger://{writer.run_ref}/{intent_record.record_id}"
                ),
                "adapter_id": compiled["adapter_id"],
                "model_id": compiled["model_id"],
                "revision": compiled["revision"],
                "conditions": compiled["conditions"],
                "prompt": compiled["prompt"],
                "parameters": compiled["parameters"],
                "input_hash": _input_hash(compiled),
            },
        }
    )
    attempt_started = writer.append(
        {
            "record_type": "render_attempt_started",
            "causal_refs": [
                {
                    "record_ref": (
                        f"ledger://{writer.run_ref}/"
                        f"{request_record.record_id}"
                    ),
                    "relation": "input",
                }
            ],
            "payload": {
                "attempt_id": f"attempt-{node_id}",
                "request_ref": (
                    f"ledger://{writer.run_ref}/{request_record.record_id}"
                ),
                "started_at": "2026-08-11T12:00:02+08:00",
            },
        }
    )
    selected_inputs: dict[str, bytes] = {}
    selected_names = (
        [slot["condition_ref"] for slot in compiled["conditions"]]
        if compiled["conditions"]
        else list(executor_inputs)
    )
    for condition_ref in selected_names:
        value = executor_inputs.get(condition_ref)
        if value is None:
            raise InputValidationError(
                f"missing selected model input {condition_ref!r}"
            )
        if isinstance(value, Path):
            try:
                value = value.read_bytes()
            except OSError as exc:
                raise InputValidationError(
                    f"cannot read selected model input {condition_ref!r}",
                    actual=str(exc),
                ) from exc
        selected_inputs[condition_ref] = value
    output_bytes = executor.execute(
        request_payload=compiled,
        operation=operation,
        inputs=selected_inputs,
    )
    artifact_ref, _ = register_artifact(
        store,
        writer,
        artifact_kind=artifact_kind,
        schema_version=schema_version,
        producer_started_ref=(
            f"ledger://{writer.run_ref}/{started.record_id}"
        ),
        data=output_bytes,
    )
    writer.append(
        {
            "record_type": "render_attempt_finished",
            "causal_refs": [],
            "payload": {
                "attempt_id": f"attempt-{node_id}",
                "started_ref": (
                    f"ledger://{writer.run_ref}/"
                    f"{attempt_started.record_id}"
                ),
                "request_ref": (
                    f"ledger://{writer.run_ref}/"
                    f"{request_record.record_id}"
                ),
                "status": "success",
                "output_artifact_ref": artifact_ref,
                "runtime": {
                    "device": getattr(executor, "provider", "stub"),
                    "duration_ms": 0,
                },
                "finished_at": "2026-08-11T12:00:03+08:00",
            },
        }
    )
    finished = _node_run_finished(writer, started=started, outputs=[artifact_ref])
    return NodeOutcome(
        refs=[artifact_ref], datas=[output_bytes], finished=finished
    )


def _image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGBA")


def run_compose_keyframe(
    *,
    run_dir: str | Path,
    run_id: str,
    shot_spec: ShotSpecDocument | dict,
    keyframe_plan: dict,
    reference_package: dict,
    scene_image_path: str | Path,
    scene_geometry: dict,
    scene_crop: dict,
    asset_map: dict[str, Path],
    adapter: Adapter,
    executor: Executor,
    canvas: tuple[int, int] = (1280, 720),
    keyframe_id: str | None = None,
    stop_after: str | None = None,
    keyframe_prompt_policy: dict | None = None,
) -> ComposeRunResult:
    """Execute one compose keyframe through the frozen stage chain.

    ``stop_after="character_synthesis"`` is an experiment affordance for the
    staged Phase 3 Real run (work package 4): it records the full synthesis
    lifecycle and binds the ``character_candidate`` port, then finishes the
    run without touching the deterministic/inpaint stages.  It is not part of
    the frozen compose contract.
    """

    spec = (
        parse_shot_spec(shot_spec)
        if isinstance(shot_spec, dict)
        else shot_spec
    )
    if spec.compose is None:
        raise InputValidationError(
            "run_compose_keyframe requires a compose ShotSpec"
        )
    if keyframe_id is None:
        if len(keyframe_plan["keyframes"]) != 1:
            raise InputValidationError(
                "Phase 3 requires exactly one keyframe or explicit "
                "keyframe_id"
            )
        keyframe_id = keyframe_plan["keyframes"][0]["keyframe_id"]
    keyframe = next(
        kf
        for kf in keyframe_plan["keyframes"]
        if kf["keyframe_id"] == keyframe_id
    )
    keyframe_state = {
        key: keyframe[key]
        for key in (
            "visual_description",
            "subject_pose",
            "expression",
            "gaze",
            "composition",
            "camera",
            "background_state",
            "foreground_state",
            "prop_state",
            "motion_from_previous",
            "required_assets",
        )
    }
    locked_attributes: list[str] = []
    for required_asset in keyframe["required_assets"]:
        for attribute in required_asset["locked_attributes"]:
            if attribute not in locked_attributes:
                locked_attributes.append(attribute)
    keyframe_state["locked_attributes"] = locked_attributes
    keyframe_state["character_locks"] = {
        "identity": spec.locks.character.identity,
        "hairstyle": spec.locks.character.hairstyle,
        "costume_variant": spec.locks.character.costume_variant,
    }
    if keyframe_prompt_policy is not None:
        if not isinstance(keyframe_prompt_policy, dict):
            raise InputValidationError(
                "keyframe_prompt_policy must be a dict when provided"
            )
        keyframe_state["prompt_policy"] = dict(keyframe_prompt_policy)
    layout_intent = derive_layout_intent(spec, keyframe_state)

    plan_id = f"{spec.shot_id}-{keyframe_id}-plan-v1"
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    writer = LedgerWriter(root / "execution-ledger.jsonl", run_id)
    store = ArtifactStore(root)

    shot_spec_ref = f"asset://anime-remix/spec/{spec.shot_id}@v1"
    keyframe_plan_ref = f"asset://anime-remix/spec/{plan_id}-keyframes@v1"
    reference_package_ref = (
        f"asset://anime-remix/refpkg/"
        f"{reference_package['package_id']}@v1"
    )
    scene_asset_ref = f"asset://anime-remix/scene/{spec.scene_id}@v1"
    scene_geometry_ref = (
        f"asset://anime-remix/geometry/{spec.scene_id}@v1"
    )

    writer.append(
        {
            "record_type": "run_started",
            "causal_refs": [],
            "payload": {
                "run_id": run_id,
                "execution_template_ref": f"template://{_TEMPLATE_REF}",
                "policy_refs": [
                    "policy://layout-v1",
                    "policy://inpaint-v1",
                ],
                "started_at": "2026-08-11T12:00:00+08:00",
            },
        }
    )
    plan_instantiated = writer.append(
        {
            "record_type": "plan_instantiated",
            "causal_refs": [],
            "payload": {
                "plan_id": plan_id,
                "template_ref": f"template://{_TEMPLATE_REF}",
                "shot_id": spec.shot_id,
                "keyframe_id": keyframe_id,
                "shot_spec_ref": shot_spec_ref,
                "keyframe_plan_ref": keyframe_plan_ref,
                "reference_package_ref": reference_package_ref,
                "policy_refs": [
                    "policy://layout-v1",
                    "policy://inpaint-v1",
                ],
            },
        }
    )
    plan_instantiated_ref = (
        f"ledger://{run_id}/{plan_instantiated.record_id}"
    )
    ports: dict[str, str] = {}

    conditions = reference_package["conditions"]
    synthesis_inputs: dict[str, bytes | Path] = {}
    for condition in conditions:
        if condition["role"] in {
            "identity",
            "pose",
            "expression",
            "source_frame",
            "scene",
            "style",
        }:
            path = asset_map.get(condition["payload_ref"])
            if path is not None:
                synthesis_inputs[condition["condition_id"]] = path

    candidate = _model_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="character_synthesis",
        operation="character_synthesis",
        instance_id="character-synthesis-run",
        inputs=[],
        intent={
            "intent_id": "intent-character-synthesis",
            "operation": "character_synthesis",
            "shot_id": spec.shot_id,
            "keyframe_id": keyframe_id,
            "requirements": [
                req["requirement_id"]
                for req in reference_package["requirements"]
            ],
            "reference_package_ref": reference_package_ref,
        },
        visual_intent=layout_intent,
        plan_instantiated_ref=plan_instantiated_ref,
        adapter=adapter,
        executor=executor,
        keyframe_state=keyframe_state,
        scene_description=_scene_description(spec),
        conditions=conditions,
        executor_inputs=synthesis_inputs,
        artifact_kind="character_candidate",
        schema_version="png",
    )
    candidate_ref = candidate.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_CHARACTER_CANDIDATE,
        artifact_ref=candidate_ref,
    )
    ports[PORT_CHARACTER_CANDIDATE] = candidate_ref
    if stop_after == "character_synthesis":
        writer.append(
            {
                "record_type": "run_finished",
                "causal_refs": [],
                "payload": {
                    "run_id": run_id,
                    "status": "completed",
                    "finished_at": "2026-08-11T12:00:04+08:00",
                },
            }
        )
        return ComposeRunResult(
            run_id=run_id,
            run_dir=root,
            ledger_path=root / "execution-ledger.jsonl",
            final_keyframe_ref=None,
            ports=ports,
        )
    if stop_after is not None:
        raise InputValidationError(
            f"unsupported stop_after {stop_after!r}"
        )

    layer_image = extract_character_layer(
        _image_from_bytes(candidate.datas[0])
    )
    layer = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="character_extract",
        operation="character_extract",
        instance_id="character-extract-run",
        inputs=[candidate_ref],
        productions=[
            {
                "data": image_to_png_bytes(layer_image),
                "canonicalize": False,
                "artifact_kind": "character_layer",
                "schema_version": "png",
            }
        ],
    )
    layer_ref = layer.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_CHARACTER_LAYER,
        artifact_ref=layer_ref,
    )
    ports[PORT_CHARACTER_LAYER] = layer_ref

    valid, checks = validate_character_layer(
        layer_image,
        canvas_w=layer_image.width,
        canvas_h=layer_image.height,
    )
    _validation_gate(
        writer,
        node_id="character_validate",
        derived_from=layer.finished,
        valid=valid,
        checks=checks,
    )

    geometry = extract_character_geometry(layer_image)
    geometry_node = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="geometry_extract",
        operation="geometry_extract",
        instance_id="geometry-extract-run",
        inputs=[layer_ref],
        productions=[
            {
                "data": geometry,
                "canonicalize": True,
                "artifact_kind": "character_geometry",
                "schema_version": "character-geometry-v1",
            }
        ],
    )
    geometry_ref = geometry_node.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_CHARACTER_GEOMETRY,
        artifact_ref=geometry_ref,
    )
    ports[PORT_CHARACTER_GEOMETRY] = geometry_ref

    intent_node = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="layout_intent_derive",
        operation="layout_intent_derive",
        instance_id="layout-intent-derive-run",
        inputs=[shot_spec_ref, keyframe_plan_ref],
        productions=[
            {
                "data": layout_intent,
                "canonicalize": True,
                "artifact_kind": "layout_intent",
                "schema_version": "layout-intent-v1",
            }
        ],
    )
    intent_ref = intent_node.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_LAYOUT_INTENT,
        artifact_ref=intent_ref,
    )
    ports[PORT_LAYOUT_INTENT] = intent_ref

    layout_plan = plan_layout(
        layout_intent=layout_intent,
        character_geometry=geometry,
        scene_geometry=scene_geometry,
        scene_crop=scene_crop,
        canvas={"width": canvas[0], "height": canvas[1]},
        plan_id=plan_id,
        shot_id=spec.shot_id,
        keyframe_id=keyframe_id,
        character_layer_ref=layer_ref,
        character_geometry_ref=geometry_ref,
        scene_asset_ref=scene_asset_ref,
        scene_geometry_ref=scene_geometry_ref,
        layout_intent_ref=intent_ref,
        keyframe_state_ref=keyframe_plan_ref,
    )
    parse_layout_plan(layout_plan)
    layout_node = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="layout",
        operation="layout",
        instance_id="layout-run",
        inputs=[geometry_ref, intent_ref],
        productions=[
            {
                "data": layout_plan,
                "canonicalize": True,
                "artifact_kind": "layout_plan",
                "schema_version": "layout-plan-v1",
            }
        ],
    )
    layout_ref = layout_node.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_LAYOUT_PLAN,
        artifact_ref=layout_ref,
    )
    ports[PORT_LAYOUT_PLAN] = layout_ref

    composite_mask_image, inpaint_mask_image = generate_masks(
        character_layer=layer_image,
        layout_plan=layout_plan,
        canvas_w=canvas[0],
        canvas_h=canvas[1],
    )
    mask_node = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="mask_generate",
        operation="mask_generate",
        instance_id="mask-generate-run",
        inputs=[layer_ref, layout_ref],
        productions=[
            {
                "data": image_to_png_bytes(composite_mask_image),
                "canonicalize": False,
                "artifact_kind": "composite_mask",
                "schema_version": "png",
            },
            {
                "data": image_to_png_bytes(inpaint_mask_image),
                "canonicalize": False,
                "artifact_kind": "inpaint_mask",
                "schema_version": "png",
            },
        ],
    )
    composite_mask_ref, inpaint_mask_ref = mask_node.refs
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_COMPOSITE_MASK,
        artifact_ref=composite_mask_ref,
    )
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_INPAINT_MASK,
        artifact_ref=inpaint_mask_ref,
    )
    ports[PORT_COMPOSITE_MASK] = composite_mask_ref
    ports[PORT_INPAINT_MASK] = inpaint_mask_ref

    scene_image = load_rgb(asset_map[scene_asset_ref])
    composite_image = composite_keyframe(
        scene_image=scene_image,
        character_layer=layer_image,
        layout_plan=layout_plan,
    )
    composite_node = _deterministic_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="composite",
        operation="composite",
        instance_id="composite-run",
        inputs=[layout_ref, layer_ref],
        productions=[
            {
                "data": image_to_png_bytes(composite_image),
                "canonicalize": False,
                "artifact_kind": "composite_image",
                "schema_version": "png",
            }
        ],
    )
    composite_ref = composite_node.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_COMPOSITE_IMAGE,
        artifact_ref=composite_ref,
    )
    ports[PORT_COMPOSITE_IMAGE] = composite_ref

    final = _model_node(
        writer,
        store,
        plan_id=plan_id,
        node_id="local_inpaint",
        operation="local_inpaint",
        instance_id="local-inpaint-run",
        inputs=[composite_ref, inpaint_mask_ref],
        intent={
            "intent_id": "intent-local-inpaint",
            "operation": "local_inpaint",
            "shot_id": spec.shot_id,
            "keyframe_id": keyframe_id,
            "requirements": [],
            "reference_package_ref": reference_package_ref,
        },
        visual_intent=layout_intent,
        plan_instantiated_ref=plan_instantiated_ref,
        adapter=adapter,
        executor=executor,
        keyframe_state=keyframe_state,
        scene_description=_scene_description(spec),
        conditions=conditions,
        executor_inputs={
            "composite_image": composite_node.datas[0],
            "inpaint_mask": mask_node.datas[1],
        },
        artifact_kind="final_keyframe",
        schema_version="png",
    )
    final_ref = final.refs[0]
    _bind_port(
        writer,
        plan_id=plan_id,
        port_name=PORT_FINAL_KEYFRAME,
        artifact_ref=final_ref,
    )
    ports[PORT_FINAL_KEYFRAME] = final_ref

    final_image = _image_from_bytes(final.datas[0]).convert("RGB")
    valid, checks = validate_final_keyframe(
        final_image, canvas_w=canvas[0], canvas_h=canvas[1]
    )
    _validation_gate(
        writer,
        node_id="final_validate",
        derived_from=final.finished,
        valid=valid,
        checks=checks,
    )

    writer.append(
        {
            "record_type": "run_finished",
            "causal_refs": [],
            "payload": {
                "run_id": run_id,
                "status": "completed",
                "finished_at": "2026-08-11T12:00:04+08:00",
            },
        }
    )
    return ComposeRunResult(
        run_id=run_id,
        run_dir=root,
        ledger_path=root / "execution-ledger.jsonl",
        final_keyframe_ref=final_ref,
        ports=ports,
    )
