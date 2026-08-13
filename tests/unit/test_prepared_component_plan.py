"""Tests for WHO/HOW component preparation before first-frame assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_content_plan import (
    approve_first_frame_content_plan,
    build_first_frame_content_plan,
)
from anime_remix.services.script.first_frame_plan import build_first_frame_plan
from anime_remix.services.script.prepared_component_plan import (
    approve_prepared_component_plan,
    build_prepared_component_plan,
    complete_prepared_component_plan,
    parse_prepared_component_plan,
)


def _record(
    asset_id: str,
    asset_type: str,
    *,
    subject: str,
    roles: tuple[str, ...] = (),
    source_tier: str = "canonical",
    analyzed: bool = True,
) -> ImageAssetRecord:
    return ImageAssetRecord(
        asset_id=asset_id,
        asset_type=asset_type,
        path=f"images/{asset_id}.png",
        rights_status="user-owned",
        resolved_path=Path(f"C:/{asset_id}.png"),
        format="png",
        width=1280,
        height=720,
        subject_or_scene_id=subject,
        source_tier=source_tier,
        reference_roles=roles,
        analysis_status="analyzed" if analyzed else "pending",
    )


def _catalog(*records: ImageAssetRecord) -> ImageAssetCatalog:
    return ImageAssetCatalog.build(list(records))


def _shot() -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "Anchor Mira's grip before movement.",
        "duration_seconds": 4.0,
        "shot_scale": "medium",
        "composition": "Mira left, reactor behind, key at waist height",
        "camera_position": "front three-quarter",
        "camera_motion": "slow push in",
        "subjects": ["Mira"],
        "setting": "observatory",
        "props": ["red key"],
        "start_state": "Mira kneels and grips the red key",
        "action_beats": [
            {"time_seconds": 0.0, "description": "Mira holds the key"}
        ],
        "end_state": "Mira turns the key",
        "emotion_arc": "focused",
    }


def _policy() -> dict:
    return {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "mira_who",
                "authority": "identity_only",
                "reason": "canonical identity",
            },
            {
                "asset_id": "key_visual",
                "authority": "final_visual",
                "reason": "canonical prop",
            },
            {
                "asset_id": "layout_only",
                "authority": "structure_only",
                "reason": "contact geometry only",
            },
        ],
        "interactions": [
            {
                "interaction_id": "mira_grips_key",
                "actor": "Mira",
                "target": "red key",
                "relation": "grips",
                "required_state": "Mira's fingers visibly wrap around the red key",
                "evidence_asset_ids": ["layout_only"],
                "hard_gate": True,
            }
        ],
        "require_production_quality_review": True,
    }


def _bundle(*records: ImageAssetRecord) -> dict:
    return {
        "schema_version": "reference-bundle-v1",
        "shot_id": "shot_001",
        "references": [
            {
                "asset_id": item.asset_id,
                "asset_type": item.asset_type,
                "path": item.path,
                "note": "",
            }
            for item in records
        ],
    }


def _content() -> object:
    return approve_first_frame_content_plan(
        build_first_frame_content_plan(_shot(), assembly_policy=_policy())
    )


def _content_with_external_attachment() -> object:
    payload = build_first_frame_content_plan(
        _shot(), assembly_policy=_policy()
    ).model_dump(mode="json")
    payload["prop_states"][0]["functional_affordance"] = {
        "grip_zone": "circular bow",
        "active_end": "shaft tip",
        "native_action_axis": "bow-to-shaft",
    }
    payload["character_states"][0]["facing_direction"] = "faces left"
    payload["attachment_graph"] = [
        {
            "attachment_id": "key_approaches_socket",
            "source_component_id": "prop_001",
            "source_anchor": "shaft tip",
            "source_anchor_position": {"x": 0.46, "y": 0.56},
            "target_component_id": "scene",
            "target_anchor": "empty socket",
            "target_anchor_position": {"x": 0.42, "y": 0.56},
            "relation": "approaches",
            "action_axis": "right-to-left",
            "initial_gap": "four percent of frame width",
            "required_visible_state": (
                "the shaft tip points toward the empty socket without entering it"
            ),
            "must_remain_visible": [
                "Mira's face",
                "right hand",
                "shaft tip",
                "empty socket",
                "insertion gap",
            ],
            "source_must_remain_visible": [
                "Mira's face",
                "right hand",
                "shaft tip",
            ],
            "target_must_remain_visible": ["empty socket", "insertion gap"],
            "hard_gate": True,
        }
    ]
    return approve_first_frame_content_plan(payload)


def test_interaction_is_prepared_as_one_atomic_two_visual_plate() -> None:
    who = _record(
        "mira_who", "character", subject="Mira", roles=("identity_reference",)
    )
    prop = _record(
        "key_visual", "prop", subject="red key", roles=("prop_reference",)
    )
    layout = _record("layout_only", "character", subject="Mira red key")
    plan = build_prepared_component_plan(
        _content(),
        reference_bundle=_bundle(who, prop, layout),
        catalog=_catalog(who, prop, layout),
        assembly_policy=_policy(),
    )

    assert plan.decision == "ready"
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.kind == "interaction_plate"
    assert task.component_ids == ["character_001", "prop_001"]
    assert [item.function for item in task.model_inputs] == ["who", "prop_visual"]
    assert task.control_evidence_asset_ids == ["layout_only"]
    assert "layout_only" not in [item.asset_id for item in task.model_inputs]
    assert set(task.allowed_text_fallbacks) == {
        "contact_relation",
        "spatial_placement",
    }


def test_external_attachment_becomes_component_geometry_and_review_gates() -> None:
    who = _record(
        "mira_who", "character", subject="Mira", roles=("identity_reference",)
    )
    prop = _record(
        "key_visual", "prop", subject="red key", roles=("prop_reference",)
    )
    layout = _record("layout_only", "character", subject="Mira red key")
    plan = build_prepared_component_plan(
        _content_with_external_attachment(),
        reference_bundle=_bundle(who, prop, layout),
        catalog=_catalog(who, prop, layout),
        assembly_policy=_policy(),
    )

    assert plan.decision == "ready"
    task = plan.tasks[0]
    assert task.prop_affordances[0].active_end == "shaft tip"
    assert task.external_attachments[0].target_anchor_x == 0.42
    assert task.external_attachments[0].action_axis == "right-to-left"
    assert task.external_attachments[0].source_must_remain_visible == [
        "Mira's face",
        "right hand",
        "shaft tip",
    ]
    assert {
        "prop_affordance",
        "facing_direction",
        "action_axis",
        "external_target",
        "required_visibility",
    } <= set(task.allowed_text_fallbacks)
    assert task.allowed_text_fallbacks["facing_direction"] == "Mira: faces left"
    assert "socket" not in task.allowed_text_fallbacks["spatial_placement"]
    assert "(0.460, 0.560)" in task.allowed_text_fallbacks["spatial_placement"]
    external_target = task.allowed_text_fallbacks["external_target"]
    assert "(0.420, 0.560)" in external_target
    assert "socket" not in external_target
    assert "absent from the plate" in external_target
    attachment_gate = next(
        item
        for item in task.review_gates
        if item.gate_id == "gate.attachment.key_approaches_socket"
    )
    assert "socket" not in attachment_gate.criterion
    assert "(0.420, 0.560)" in attachment_gate.criterion
    assert "absent from the plate" in attachment_gate.criterion
    assert "Plate-scope state" in task.external_attachments[0].required_visible_state
    assert {item.gate_id for item in task.review_gates} == {
        "gate.contact.mira_grips_key",
        "gate.affordance.prop_001",
        "gate.attachment.key_approaches_socket",
    }


def test_scene_attachment_target_description_does_not_leak_into_component_prompt_or_gates() -> None:
    payload = _content_with_external_attachment().model_dump(mode="json")
    payload["layers"][0]["subject"] = (
        "abandoned astronomical observatory with sealed circular mechanism"
    )
    content = approve_first_frame_content_plan(payload)
    who = _record(
        "mira_who", "character", subject="Mira", roles=("identity_reference",)
    )
    prop = _record(
        "key_visual", "prop", subject="red key", roles=("prop_reference",)
    )
    layout = _record("layout_only", "character", subject="Mira red key")
    plan = build_prepared_component_plan(
        content,
        reference_bundle=_bundle(who, prop, layout),
        catalog=_catalog(who, prop, layout),
        assembly_policy=_policy(),
    )

    task = plan.tasks[0]
    fallback_text = " ".join(task.allowed_text_fallbacks.values())
    gate_text = " ".join(item.criterion for item in task.review_gates)
    for needle in (
        "abandoned astronomical observatory",
        "sealed circular mechanism",
        "empty socket",
    ):
        assert needle not in fallback_text
        assert needle not in gate_text
    assert "layout-only target coordinate (0.420, 0.560)" in fallback_text
    assert "absent from the plate" in gate_text
    assert (
        task.external_attachments[0].target_subject
        == "abandoned astronomical observatory with sealed circular mechanism"
    )


def test_attachment_component_cannot_be_approved_until_every_gate_passes() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    draft = build_prepared_component_plan(
        _content_with_external_attachment(),
        reference_bundle=_bundle(who, prop),
        catalog=_catalog(who, prop),
    )
    payload = approve_prepared_component_plan(draft).model_dump(mode="json")
    payload["tasks"][0]["result"] = "approved"
    payload["tasks"][0]["result_review_notes"] = "visual review completed"

    with pytest.raises(InputValidationError, match="every review gate"):
        parse_prepared_component_plan(payload)

    payload["tasks"][0]["gate_results"] = [
        {
            "gate_id": item["gate_id"],
            "result": "pass",
            "note": "manually verified on the generated component plate",
        }
        for item in payload["tasks"][0]["review_gates"]
    ]
    output_id = payload["tasks"][0]["output_asset_id"]
    output = _record(
        output_id,
        "character",
        subject="Mira red key",
        source_tier="approved_generated",
    )
    completed = complete_prepared_component_plan(
        payload,
        catalog=_catalog(who, prop, output),
    )
    assert completed.completion_status == "completed"


def test_component_contract_rejects_appearance_text_fallback() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    layout = _record("layout_only", "character", subject="Mira red key")
    payload = build_prepared_component_plan(
        _content(),
        reference_bundle=_bundle(who, prop, layout),
        catalog=_catalog(who, prop, layout),
        assembly_policy=_policy(),
    ).model_dump(mode="json")
    payload["tasks"][0]["allowed_text_fallbacks"]["appearance"] = "blue hair"

    with pytest.raises(InputValidationError, match="only action/contact"):
        parse_prepared_component_plan(payload)


def test_overlapping_hard_contacts_block_ambiguous_component_preparation() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    reactor = _record("reactor_visual", "prop", subject="reactor")
    content = _content().model_dump(mode="json")
    content["layers"].append(
        {
            "layer_id": "layer.prop_002",
            "order": 3,
            "component_id": "prop_002",
            "kind": "prop",
            "subject": "reactor",
            "target_state": "sealed",
            "spatial_relationship": "behind Mira",
        }
    )
    content["prop_states"].append(
        {
            "component_id": "prop_002",
            "subject": "reactor",
            "state": "sealed",
            "screen_position": "behind Mira",
        }
    )
    content["contact_graph"].append(
        {
            "interaction_id": "mira_touches_reactor",
            "actor_component_id": "character_001",
            "target_component_id": "prop_002",
            "relation": "touches",
            "required_visible_state": "Mira's left palm touches the sealed reactor",
            "hard_gate": True,
        }
    )
    content["information"].append(
        {
            "fact_id": "prop_002.state",
            "component_id": "prop_002",
            "state": "visible",
            "fact": "reactor is sealed",
            "reason": "K0 must not reveal the interior",
        }
    )

    plan = build_prepared_component_plan(
        content,
        reference_bundle=_bundle(who, prop, reactor),
        catalog=_catalog(who, prop, reactor),
    )

    assert plan.decision == "blocked"
    assert any("overlapping hard interactions" in item for item in plan.warnings)
    with pytest.raises(InputValidationError, match="blocked"):
        approve_prepared_component_plan(plan)


def test_completion_requires_registered_analyzed_approved_generated_output() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    layout = _record("layout_only", "character", subject="Mira red key")
    draft = build_prepared_component_plan(
        _content(),
        reference_bundle=_bundle(who, prop, layout),
        catalog=_catalog(who, prop, layout),
        assembly_policy=_policy(),
    )
    payload = approve_prepared_component_plan(draft).model_dump(mode="json")
    payload["tasks"][0]["result"] = "approved"
    payload["tasks"][0]["result_review_notes"] = (
        "identity, anatomy and finger-key contact passed"
    )
    output_id = payload["tasks"][0]["output_asset_id"]

    with pytest.raises(InputValidationError, match="not registered"):
        complete_prepared_component_plan(payload, catalog=_catalog(who, prop, layout))

    output = _record(
        output_id,
        "character",
        subject="Mira",
        source_tier="approved_generated",
    )
    completed = complete_prepared_component_plan(
        payload, catalog=_catalog(who, prop, layout, output)
    )
    assert completed.completion_status == "completed"
    assert completed.approved_output_asset_ids == [output_id]


def test_final_assembly_consumes_component_output_not_raw_who_how_sources() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    layout = _record("layout_only", "character", subject="Mira red key")
    background = _record("observatory_bg", "background", subject="observatory")
    source_catalog = _catalog(who, prop, layout)
    draft = build_prepared_component_plan(
        _content(),
        reference_bundle=_bundle(who, prop, layout),
        catalog=source_catalog,
        assembly_policy=_policy(),
    )
    payload = approve_prepared_component_plan(draft).model_dump(mode="json")
    payload["tasks"][0]["result"] = "approved"
    payload["tasks"][0]["result_review_notes"] = "all reject-tier gates passed"
    output_id = payload["tasks"][0]["output_asset_id"]
    output = _record(
        output_id,
        "character",
        subject="Mira",
        source_tier="approved_generated",
        roles=("identity_reference", "pose_reference"),
    )
    final_catalog = _catalog(who, prop, layout, background, output)
    completed = complete_prepared_component_plan(payload, catalog=final_catalog)
    final_bundle = _bundle(background, output)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=final_bundle,
        catalog=final_catalog,
        content_plan=_content(),
        prepared_component_plan=completed,
    )

    assert plan.prepared_component_plan_id == completed.plan_id
    assert plan.prepared_component_asset_ids == [output_id]
    assert output_id in plan.selected_asset_ids
    assert not set(completed.source_asset_ids) & set(plan.bound_asset_ids)
    component_stage = next(
        stage
        for stage in plan.stages
        if output_id in [item.source_id for item in stage.inputs]
    )
    assert component_stage.component_ids == ["character_001", "prop_001"]
    assert "prop identity" in component_stage.reference_attributes
    assert "prop appearance" in component_stage.reference_attributes
    assert "interaction/contact geometry" in component_stage.reference_attributes
    assert "relative participant placement" in component_stage.reference_attributes
    interaction = next(
        item for item in plan.interaction_units if item.interaction_id == "mira_grips_key"
    )
    assert interaction.grounding == "visually_grounded"


def test_final_plan_preserves_and_gates_external_attachment_geometry() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    background = _record("observatory_bg", "background", subject="observatory")
    content = _content_with_external_attachment()
    draft = build_prepared_component_plan(
        content,
        reference_bundle=_bundle(who, prop),
        catalog=_catalog(who, prop),
    )
    payload = approve_prepared_component_plan(draft).model_dump(mode="json")
    payload["tasks"][0]["result"] = "approved"
    payload["tasks"][0]["result_review_notes"] = "all structured gates passed"
    payload["tasks"][0]["gate_results"] = [
        {
            "gate_id": item["gate_id"],
            "result": "pass",
            "note": "verified",
        }
        for item in payload["tasks"][0]["review_gates"]
    ]
    output_id = payload["tasks"][0]["output_asset_id"]
    output = _record(
        output_id,
        "character",
        subject="Mira red key",
        source_tier="approved_generated",
        roles=("identity_reference", "pose_reference"),
    )
    catalog = _catalog(who, prop, background, output)
    completed = complete_prepared_component_plan(payload, catalog=catalog)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(background, output),
        catalog=catalog,
        full_frame_anchor_asset_id="observatory_bg",
        content_plan=content,
        prepared_component_plan=completed,
    )

    attachment = plan.attachment_units[0]
    assert attachment.attachment_id == "key_approaches_socket"
    assert attachment.grounding == "visually_grounded"
    assert attachment.evidence_asset_ids == [output_id]
    gate = next(
        item
        for item in plan.quality_gates
        if item.gate_id == "gate.attachment.key_approaches_socket"
    )
    assert gate.kind == "attachment"
    assert gate.attachment_id == attachment.attachment_id
    component_stage = next(
        stage
        for stage in plan.stages
        if output_id in [item.source_id for item in stage.inputs]
    )
    assert "prop functional topology" in component_stage.reference_attributes
    assert "external attachment geometry" in component_stage.reference_attributes
    assert "final-frame canvas placement" in component_stage.reference_attributes
    assert "(0.460, 0.560)" in component_stage.instruction
    assert "empty socket" in component_stage.instruction


def test_transparent_attachment_component_is_overlaid_before_model_refinement() -> None:
    who = _record("mira_who", "character", subject="Mira")
    prop = _record("key_visual", "prop", subject="red key")
    background = _record("observatory_bg", "background", subject="observatory")
    content = _content_with_external_attachment()
    draft = build_prepared_component_plan(
        content,
        reference_bundle=_bundle(who, prop),
        catalog=_catalog(who, prop),
    )
    payload = approve_prepared_component_plan(draft).model_dump(mode="json")
    payload["tasks"][0]["result"] = "approved"
    payload["tasks"][0]["result_review_notes"] = "transparent layout plate passed"
    payload["tasks"][0]["gate_results"] = [
        {"gate_id": item["gate_id"], "result": "pass", "note": "verified"}
        for item in payload["tasks"][0]["review_gates"]
    ]
    output_id = payload["tasks"][0]["output_asset_id"]
    transparent_output = _record(
        output_id,
        "foreground",
        subject="Mira red key",
        source_tier="approved_generated",
    )
    catalog = _catalog(who, prop, background, transparent_output)
    completed = complete_prepared_component_plan(payload, catalog=catalog)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(background, transparent_output),
        catalog=catalog,
        full_frame_anchor_asset_id="observatory_bg",
        content_plan=content,
        prepared_component_plan=completed,
    )

    assert [item.operation for item in plan.stages] == [
        "adopt_anchor",
        "composite_overlay",
        "fuse_component",
    ]
    assert plan.stages[1].inputs[1].role == "foreground"
    assert "Harmonize only" in plan.stages[2].instruction
    assert "approved scale, screen position" in plan.stages[2].instruction
    assert sum(
        item.operation not in {"adopt_anchor", "composite_overlay"}
        for item in plan.stages
    ) == 1


def test_approved_content_layer_order_drives_final_assembly_order() -> None:
    background = _record("observatory_bg", "background", subject="observatory")
    who = _record(
        "mira_who",
        "character",
        subject="Mira",
        roles=("identity_reference", "pose_reference"),
    )
    prop = _record(
        "key_visual", "prop", subject="red key", roles=("prop_reference",)
    )
    records = (background, who, prop)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(*records),
        catalog=_catalog(*records),
        content_plan=_content(),
    )

    uploaded_assets = [
        next(
            (
                item.source_id
                for item in stage.inputs
                if item.source_type == "asset"
            ),
            None,
        )
        for stage in plan.stages
    ]
    assert uploaded_assets == ["observatory_bg", "key_visual", "mira_who"]
    assert any(
        gate.gate_id == "gate.content.hidden.scene.future_state"
        for gate in plan.quality_gates
    )
