"""Tests for minimal shared-boundary generation-segment planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
)
from anime_remix.services.script.generation_segment_plan import (
    approve_generation_segment_plan,
    build_generation_segment_plan,
    parse_generation_segment_plan,
)
from anime_remix.services.script.shot_plan import parse_shot_plan

runner = CliRunner()


def _shot_dict() -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_001",
        "order": 1,
        "narrative_purpose": "Reveal resolve without fragmenting the shot.",
        "duration_seconds": 6.0,
        "shot_scale": "medium",
        "composition": "Asuna at the desk, window on frame right",
        "camera_position": "front-left eye level",
        "camera_motion": "slow push in",
        "subjects": ["Asuna"],
        "setting": "afternoon classroom",
        "props": [],
        "start_state": "Asuna sits at the desk with both eyes closed",
        "action_beats": [
            {"time_seconds": 0.0, "description": "holds still, eyes closed"},
            {"time_seconds": 1.5, "description": "opens her eyes"},
            {"time_seconds": 3.0, "description": "turns toward the window"},
            {"time_seconds": 5.0, "description": "settles her gaze outside"},
        ],
        "end_state": "Asuna looks through the window with eyes open",
        "emotion_arc": "quiet to resolved",
        "dialogue": None,
        "continuity_in": None,
        "continuity_out": None,
    }


def _shot():
    return parse_shot_plan(
        {"schema_version": "shot-plan-v1", "shots": [_shot_dict()]}
    ).shots[0]


def _record(
    asset_id: str,
    asset_type: str,
    *,
    subject: str,
    roles: tuple[str, ...],
) -> ImageAssetRecord:
    return ImageAssetRecord(
        asset_id=asset_id,
        asset_type=asset_type,
        path=f"images/{asset_id}.png",
        rights_status="user-owned",
        resolved_path=Path(f"C:/{asset_id}.png"),
        format="png",
        width=640,
        height=360,
        subject_or_scene_id=subject,
        source_tier="canonical",
        reference_roles=roles,
        analysis_status="analyzed",
    )


def _approved_first_frame(*records: ImageAssetRecord):
    catalog = ImageAssetCatalog.build(list(records))
    bundle = {
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
    return approve_first_frame_plan(
        build_first_frame_plan(
            _shot(),
            reference_bundle=bundle,
            catalog=catalog,
        )
    )


def _identity() -> ImageAssetRecord:
    return _record(
        "char_asuna",
        "character",
        subject="Asuna",
        roles=("identity_reference", "expression_reference"),
    )


def _boundary_intents() -> dict:
    return {
        "schema_version": "segment-boundary-intents-v1",
        "shot_id": "shot_001",
        "boundaries": [
            {
                "anchor_id": "shot_001_eyes_open",
                "time_seconds": 2.0,
                "target_state": "Asuna sits with both eyes fully open",
                "composition": "same desk composition",
                "camera": "same front-left eye-level camera",
                "process_from_previous": "slowly opens both eyes",
                "dominant_motion": "eyelids open",
                "camera_motion": "slow push in continues",
                "delta_instruction": (
                    "Change only both eyelids to the naturally open state; "
                    "take eye shape and colour from the scoped reference."
                ),
                "control_reasons": ["information_reveal"],
                "reveal_fact_ids": ["character_001.eyes"],
                "generation_method": "edit_previous",
                "reference_asset_id": None,
                "reference_role": None,
                "reference_attributes": [],
                "locked_attributes": [
                    "identity",
                    "outfit",
                    "hair",
                    "style",
                    "scene_continuity",
                ],
            }
        ],
    }


def test_default_plan_does_not_turn_every_action_beat_into_a_cut() -> None:
    plan = build_generation_segment_plan(
        _shot(), first_frame_plan=_approved_first_frame(_identity())
    )

    assert len(_shot().action_beats) == 4
    assert len(plan.anchors) == 2
    assert len(plan.segments) == 1
    assert plan.anchors[0].roles == ["master_start"]
    assert plan.anchors[-1].roles == ["continuity", "final"]
    assert plan.segments[0].continuity_mode == "shared_boundary"


def test_reuse_previous_anchor_method_is_rejected_as_redundant() -> None:
    plan = approve_generation_segment_plan(
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=_approved_first_frame(_identity()),
            boundary_intents=_boundary_intents(),
        )
    )
    payload = plan.model_dump(mode="json")
    final = payload["anchors"][-1]
    final["generation_method"] = "reuse_previous"

    with pytest.raises(InputValidationError):
        parse_generation_segment_plan(payload)


@pytest.mark.parametrize("reason", ["motion_phase_change", "occlusion_change"])
def test_process_only_change_cannot_create_an_intermediate_anchor(reason: str) -> None:
    intents = _boundary_intents()
    boundary = intents["boundaries"][0]
    boundary["control_reasons"] = [reason]
    boundary["reveal_fact_ids"] = []
    boundary["reference_asset_id"] = None
    boundary["reference_role"] = None
    boundary["reference_attributes"] = []

    with pytest.raises(InputValidationError, match="no anchor-worthy"):
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=_approved_first_frame(_identity()),
            boundary_intents=intents,
        )


def test_information_boundary_is_shared_and_uses_canonical_authority() -> None:
    plan = build_generation_segment_plan(
        _shot(),
        first_frame_plan=_approved_first_frame(_identity()),
        boundary_intents=_boundary_intents(),
    )

    assert len(plan.anchors) == 3
    assert len(plan.segments) == 2
    boundary = plan.anchors[1]
    assert boundary.roles == ["information"]
    assert boundary.reference_asset_id == "char_asuna"
    assert boundary.reference_role == "expression"
    assert boundary.base_anchor_id == plan.anchors[0].anchor_id
    assert boundary.generation_risk == "medium"
    assert boundary.risk_factors == ["information_reveal"]
    assert plan.segments[0].end_anchor_id == boundary.anchor_id
    assert plan.segments[1].start_anchor_id == boundary.anchor_id
    eye = next(
        item for item in plan.information_ledger if item.fact_id == "character_001.eyes"
    )
    assert eye.status == "revealed"
    assert eye.first_visible_anchor_id == boundary.anchor_id


def test_information_boundary_can_use_deferred_expression_authority() -> None:
    identity = _identity()
    future_expression = _record(
        "char_asuna_open_eyes",
        "character",
        subject="Asuna",
        roles=("expression_reference",),
    )
    catalog = ImageAssetCatalog.build([identity, future_expression])
    bundle = {
        "schema_version": "reference-bundle-v1",
        "shot_id": "shot_001",
        "references": [
            {
                "asset_id": item.asset_id,
                "asset_type": item.asset_type,
                "path": item.path,
                "note": "",
            }
            for item in (identity, future_expression)
        ],
    }
    first = approve_first_frame_plan(
        build_first_frame_plan(
            _shot(),
            reference_bundle=bundle,
            catalog=catalog,
            assembly_policy={
                "schema_version": "first-frame-assembly-policy-v1",
                "shot_id": "shot_001",
                "reference_authorities": [
                    {
                        "asset_id": "char_asuna_open_eyes",
                        "authority": "hidden",
                        "reason": "reserved for the open-eye information anchor",
                    }
                ],
            },
        )
    )
    intents = _boundary_intents()
    intents["boundaries"][0].update(
        {
            "reference_asset_id": "char_asuna_open_eyes",
            "reference_role": "expression",
            "reference_attributes": ["eye_shape_and_colour"],
        }
    )

    plan = build_generation_segment_plan(
        _shot(), first_frame_plan=first, boundary_intents=intents
    )

    boundary = plan.anchors[1]
    assert boundary.reference_asset_id == "char_asuna_open_eyes"
    assert boundary.grounding == "previous_frame_and_visual_reference"
    eye = next(
        item for item in plan.information_ledger if item.fact_id == "character_001.eyes"
    )
    assert "char_asuna_open_eyes" in eye.authority_asset_ids


def test_information_boundary_can_use_deferred_pose_authority() -> None:
    identity = _identity()
    future_pose = _record(
        "char_asuna_side_profile",
        "character",
        subject="Asuna",
        roles=("pose_reference",),
    )
    catalog = ImageAssetCatalog.build([identity, future_pose])
    bundle = {
        "schema_version": "reference-bundle-v1",
        "shot_id": "shot_001",
        "references": [
            {
                "asset_id": item.asset_id,
                "asset_type": item.asset_type,
                "path": item.path,
                "note": "",
            }
            for item in (identity, future_pose)
        ],
    }
    first = approve_first_frame_plan(
        build_first_frame_plan(
            _shot(),
            reference_bundle=bundle,
            catalog=catalog,
            assembly_policy={
                "schema_version": "first-frame-assembly-policy-v1",
                "shot_id": "shot_001",
                "reference_authorities": [
                    {
                        "asset_id": "char_asuna_side_profile",
                        "authority": "hidden",
                        "reason": "reserved for the side-profile information anchor",
                    }
                ],
            },
        )
    )
    intents = _boundary_intents()
    intents["boundaries"][0].update(
        {
            "target_state": "Asuna turns only her head toward frame left",
            "process_from_previous": "turns her head toward frame left",
            "dominant_motion": "head turn",
            "delta_instruction": "Change only the head angle and gaze direction.",
            "reveal_fact_ids": ["character_001.pose"],
            "reference_asset_id": "char_asuna_side_profile",
            "reference_role": "pose",
            "reference_attributes": ["side_profile_head_orientation"],
        }
    )

    plan = build_generation_segment_plan(
        _shot(), first_frame_plan=first, boundary_intents=intents
    )

    boundary = plan.anchors[1]
    assert boundary.reference_asset_id == "char_asuna_side_profile"
    assert boundary.reference_role == "pose"
    assert boundary.grounding == "previous_frame_and_visual_reference"
    pose = next(
        item for item in plan.information_ledger if item.fact_id == "character_001.pose"
    )
    assert pose.status == "revealed"
    assert "char_asuna_side_profile" in pose.authority_asset_ids


def test_compound_information_and_topology_boundary_is_high_risk() -> None:
    intents = _boundary_intents()
    intents["boundaries"][0]["control_reasons"].append("contact_topology")

    plan = build_generation_segment_plan(
        _shot(),
        first_frame_plan=_approved_first_frame(_identity()),
        boundary_intents=intents,
    )

    boundary = plan.anchors[1]
    assert boundary.generation_risk == "high"
    assert boundary.risk_factors == ["information_reveal", "contact_topology"]
    assert plan.decision == "needs_review"
    assert "independent visual constraints" in plan.warnings[0]


def test_two_reveals_without_one_common_visual_authority_are_rejected() -> None:
    identity = _identity()
    scene = _record(
        "bg_classroom",
        "background",
        subject="afternoon classroom",
        roles=("scene_reference",),
    )
    first = _approved_first_frame(identity, scene).model_dump(mode="json")
    scene_fact = next(
        item
        for item in first["information_coverage"]
        if "bg_classroom" in item["source_asset_ids"]
    )
    scene_fact["status"] = "occluded"
    intents = _boundary_intents()
    intents["boundaries"][0]["reveal_fact_ids"].append(scene_fact["fact_id"])

    with pytest.raises(InputValidationError, match="do not share one"):
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=first,
            boundary_intents=intents,
        )


def test_intermediate_boundary_cannot_claim_shot_end() -> None:
    intents = _boundary_intents()
    intents["boundaries"][0]["control_reasons"] = ["shot_end"]
    intents["boundaries"][0]["reveal_fact_ids"] = []

    with pytest.raises(InputValidationError, match="cannot use shot_end"):
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=_approved_first_frame(_identity()),
            boundary_intents=intents,
        )


def test_graph_validator_rejects_non_shared_segment_boundary() -> None:
    plan = build_generation_segment_plan(
        _shot(),
        first_frame_plan=_approved_first_frame(_identity()),
        boundary_intents=_boundary_intents(),
    ).model_dump(mode="json")
    plan["segments"][1]["start_anchor_id"] = plan["anchors"][0]["anchor_id"]

    with pytest.raises(InputValidationError, match="adjacent anchor"):
        parse_generation_segment_plan(plan)


def test_plan_requires_approval_before_becoming_executable() -> None:
    draft = build_generation_segment_plan(
        _shot(), first_frame_plan=_approved_first_frame(_identity())
    )
    assert draft.review_status == "draft"
    approved = approve_generation_segment_plan(draft)
    assert approved.review_status == "approved"


def test_cli_builds_and_approves_segment_plan(tmp_path: Path) -> None:
    shot_path = tmp_path / "shot_plan.json"
    shot_path.write_text(
        json.dumps({"schema_version": "shot-plan-v1", "shots": [_shot_dict()]}),
        encoding="utf-8",
    )
    first_path = tmp_path / "first_frame_plan.json"
    first_path.write_text(
        _approved_first_frame(_identity()).model_dump_json(indent=2),
        encoding="utf-8",
    )
    boundaries_path = tmp_path / "boundaries.json"
    boundaries_path.write_text(json.dumps(_boundary_intents()), encoding="utf-8")
    draft_path = tmp_path / "generation_segment_plan.json"

    result = runner.invoke(
        app,
        [
            "director",
            "segment-plan",
            "--shot-plan",
            str(shot_path),
            "--shot-id",
            "shot_001",
            "--first-frame-plan",
            str(first_path),
            "--boundaries",
            str(boundaries_path),
            "--output",
            str(draft_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(draft_path.read_text(encoding="utf-8"))["review_status"] == (
        "draft"
    )

    approved_path = tmp_path / "generation_segment_plan.approved.json"
    result = runner.invoke(
        app,
        [
            "director",
            "segment-approve",
            "--plan",
            str(draft_path),
            "--output",
            str(approved_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        json.loads(approved_path.read_text(encoding="utf-8"))["review_status"]
        == "approved"
    )
