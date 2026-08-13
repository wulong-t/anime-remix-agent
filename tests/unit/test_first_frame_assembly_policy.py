"""High-quality first-frame reference admission and interaction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_assembly_policy import (
    parse_first_frame_assembly_policy,
)
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
)


def _record(
    asset_id: str,
    asset_type: str,
    *,
    subject: str | None = None,
    roles: tuple[str, ...] = (),
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
        source_tier="canonical",
        reference_roles=roles,
        analysis_status="analyzed",
    )


def _shot(*, subjects: list[str] | None = None, props: list[str] | None = None) -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_001",
        "order": 1,
        "narrative_purpose": "Establish a reliable first-frame anchor.",
        "duration_seconds": 5.0,
        "shot_scale": "medium",
        "composition": "Mira kneels left of the sealed reactor",
        "camera_position": "low front-left three-quarter view",
        "camera_motion": "fixed",
        "subjects": ["Mira"] if subjects is None else subjects,
        "setting": "night observatory control room",
        "props": ["data key"] if props is None else props,
        "start_state": "Mira kneels with eyes closed and grips the data key",
        "action_beats": [
            {
                "time_seconds": 0.0,
                "description": "Mira holds the data key in her right hand",
            }
        ],
        "end_state": "Mira looks toward the reactor",
        "emotion_arc": "tense",
    }


def _bundle(*records: ImageAssetRecord) -> dict:
    return {
        "schema_version": "reference-bundle-v1",
        "shot_id": "shot_001",
        "references": [
            {
                "asset_id": record.asset_id,
                "asset_type": record.asset_type,
                "path": record.path,
                "note": "",
            }
            for record in records
        ],
    }


def test_policy_rejects_duplicate_reference_authority() -> None:
    with pytest.raises(InputValidationError, match="must be unique"):
        parse_first_frame_assembly_policy(
            {
                "schema_version": "first-frame-assembly-policy-v1",
                "shot_id": "shot_001",
                "reference_authorities": [
                    {
                        "asset_id": "layout",
                        "authority": "structure_only",
                        "reason": "schematic",
                    },
                    {
                        "asset_id": "layout",
                        "authority": "final_visual",
                        "reason": "conflicting rule",
                    },
                ],
            }
        )


def test_policy_isolates_layout_hidden_information_and_overlay() -> None:
    background = _record(
        "bg_observatory",
        "background",
        subject="night observatory control room",
        roles=("scene_reference",),
    )
    layout = _record("layout_board", "background", subject="shot layout")
    identity = _record(
        "mira_identity",
        "character",
        subject="Mira",
        roles=("identity_reference",),
    )
    action = _record(
        "mira_grip_how",
        "character",
        subject="Mira",
        roles=("pose_reference",),
    )
    data_key = _record(
        "data_key",
        "prop",
        subject="data key",
        roles=("prop_reference",),
    )
    steam = _record("steam", "foreground", subject="steam occlusion")
    reactor_inner = _record("reactor_inner", "prop", subject="reactor interior")
    records = (
        background,
        layout,
        identity,
        action,
        data_key,
        steam,
        reactor_inner,
    )
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "layout_board",
                "authority": "structure_only",
                "reason": "low-fidelity position evidence only",
            },
            {
                "asset_id": "steam",
                "authority": "deterministic_overlay",
                "reason": "preserve the approved alpha occlusion exactly",
            },
            {
                "asset_id": "reactor_inner",
                "authority": "hidden",
                "reason": "future information must not enter K0",
            },
        ],
        "interactions": [
            {
                "interaction_id": "mira_grips_key",
                "actor": "Mira.right_hand",
                "target": "data_key",
                "relation": "grasp",
                "required_state": "Mira's right hand visibly grips the data key",
                "evidence_asset_ids": ["mira_grip_how"],
                "hard_gate": True,
            }
        ],
    }

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(*records),
        catalog=ImageAssetCatalog.build(list(records)),
        assembly_policy=policy,
    )

    assert plan.control_asset_ids == ["layout_board"]
    assert plan.deferred_asset_ids == ["reactor_inner"]
    assert "layout_board" not in plan.selected_asset_ids
    assert "reactor_inner" not in plan.selected_asset_ids
    assert all(
        item.source_id not in {"layout_board", "reactor_inner"}
        for stage in plan.stages
        for item in stage.inputs
    )
    assert plan.interaction_units[0].grounding == "visually_grounded"
    assert plan.stages[-1].operation == "composite_overlay"
    assert {item.kind for item in plan.quality_gates} == {
        "interaction",
        "hidden_information",
        "production_quality",
    }
    assert set(plan.stages[-1].quality_gate_ids) == {
        item.gate_id for item in plan.quality_gates
    }
    assert plan.decision == "needs_review"
    assert approve_first_frame_plan(plan).review_status == "approved"


def test_hidden_character_reference_remains_future_eye_authority() -> None:
    identity = _record(
        "mira_identity",
        "character",
        subject="Mira",
        roles=("identity_reference",),
    )
    future_expression = _record(
        "mira_open_eyes",
        "character",
        subject="Mira",
        roles=("expression_reference",),
    )
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "mira_open_eyes",
                "authority": "hidden",
                "reason": "open-eye information is reserved for a later anchor",
            }
        ],
    }

    plan = build_first_frame_plan(
        _shot(props=[]),
        reference_bundle=_bundle(identity, future_expression),
        catalog=ImageAssetCatalog.build([identity, future_expression]),
        assembly_policy=policy,
    )

    eye_fact = next(
        item for item in plan.information_coverage if item.fact_id.endswith(".eyes")
    )
    assert plan.deferred_asset_ids == ["mira_open_eyes"]
    assert "mira_open_eyes" not in plan.selected_asset_ids
    assert all(
        item.source_id != "mira_open_eyes"
        for stage in plan.stages
        for item in stage.inputs
    )
    assert eye_fact.status == "occluded"
    assert eye_fact.source_asset_ids == ["mira_identity", "mira_open_eyes"]


def test_hidden_character_reference_remains_future_pose_authority() -> None:
    identity = _record(
        "mira_identity",
        "character",
        subject="Mira",
        roles=("identity_reference",),
    )
    future_pose = _record(
        "mira_side_profile",
        "character",
        subject="Mira",
        roles=("pose_reference",),
    )
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "mira_side_profile",
                "authority": "hidden",
                "reason": "side-profile pose is reserved for a later anchor",
            }
        ],
    }

    plan = build_first_frame_plan(
        _shot(props=[]),
        reference_bundle=_bundle(identity, future_pose),
        catalog=ImageAssetCatalog.build([identity, future_pose]),
        assembly_policy=policy,
    )

    pose_fact = next(
        item for item in plan.information_coverage if item.fact_id.endswith(".pose")
    )
    assert plan.deferred_asset_ids == ["mira_side_profile"]
    assert "mira_side_profile" not in plan.selected_asset_ids
    assert all(
        item.source_id != "mira_side_profile"
        for stage in plan.stages
        for item in stage.inputs
    )
    assert pose_fact.status == "known_from_reference"
    assert pose_fact.source_asset_ids == ["mira_side_profile"]


def test_unresolved_hard_interaction_blocks_approval() -> None:
    background = _record("bg", "background", subject="night observatory control room")
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "interactions": [
            {
                "interaction_id": "missing_contact",
                "actor": "missing_character.hand",
                "target": "missing_prop",
                "relation": "grasp",
                "required_state": "the missing hand grips the missing prop",
            }
        ],
    }
    plan = build_first_frame_plan(
        _shot(subjects=[], props=[]),
        reference_bundle=_bundle(background),
        catalog=ImageAssetCatalog.build([background]),
        full_frame_anchor_asset_id="bg",
        assembly_policy=policy,
    )

    assert plan.decision == "blocked"
    assert plan.interaction_units[0].grounding == "unresolved"
    with pytest.raises(InputValidationError, match="blocked"):
        approve_first_frame_plan(plan)


def test_structure_only_asset_cannot_be_full_frame_anchor() -> None:
    layout = _record("layout", "background", subject="shot layout")
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "layout",
                "authority": "structure_only",
                "reason": "schematic only",
            }
        ],
    }

    with pytest.raises(InputValidationError, match="renderable visual truth"):
        build_first_frame_plan(
            _shot(subjects=[], props=[]),
            reference_bundle=_bundle(layout),
            catalog=ImageAssetCatalog.build([layout]),
            full_frame_anchor_asset_id="layout",
            assembly_policy=policy,
        )
