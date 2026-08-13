"""Tests for the model-independent first-frame content truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError
from anime_remix.services.script.first_frame_content_plan import (
    approve_first_frame_content_plan,
    build_first_frame_content_plan,
    parse_first_frame_content_plan,
)

runner = CliRunner()


def _shot() -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "Anchor Mira before the reactor opens.",
        "duration_seconds": 6.0,
        "shot_scale": "medium",
        "composition": "Mira left foreground, sealed reactor center right",
        "camera_position": "low three-quarter angle",
        "camera_motion": "slow push toward the reactor",
        "subjects": ["Mira"],
        "setting": "abandoned observatory",
        "props": ["red key", "sealed reactor"],
        "start_state": "Mira kneels with eyes closed and grips the red key",
        "action_beats": [
            {"time_seconds": 0.0, "description": "Mira holds still"},
            {"time_seconds": 2.0, "description": "Mira begins turning the key"},
        ],
        "end_state": "the reactor opens and Mira looks inside",
        "emotion_arc": "contained tension to alarm",
        "continuity_in": "steam drifts from frame left",
    }


def _policy() -> dict:
    return {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [],
        "interactions": [
            {
                "interaction_id": "mira_grips_key",
                "actor": "Mira",
                "target": "red key",
                "relation": "grips",
                "required_state": "Mira's fingers visibly wrap around the red key",
                "evidence_asset_ids": [],
                "hard_gate": True,
            }
        ],
        "require_production_quality_review": True,
    }


def test_content_plan_separates_k0_truth_from_future_information() -> None:
    plan = build_first_frame_content_plan(_shot(), assembly_policy=_policy())

    assert plan.review_status == "draft"
    assert [item.order for item in plan.layers] == list(range(len(plan.layers)))
    assert plan.layers[0].kind == "scene"
    assert plan.contact_graph[0].actor_component_id == "character_001"
    assert plan.contact_graph[0].target_component_id == "prop_001"
    deferred = {item.fact_id for item in plan.information if item.state == "deferred"}
    assert "character_001.eyes" in deferred
    assert "scene.future_state" in deferred
    assert "turning the key" in plan.motion_runway
    assert {item.tier for item in plan.acceptance_gates} == {
        "reject",
        "local_repair",
        "acceptable_variation",
    }
    assert approve_first_frame_content_plan(plan).review_status == "approved"


def test_content_plan_rejects_non_contiguous_back_to_front_layers() -> None:
    payload = build_first_frame_content_plan(_shot()).model_dump(mode="json")
    payload["layers"][1]["order"] = 8

    with pytest.raises(InputValidationError, match="contiguous"):
        parse_first_frame_content_plan(payload)


def test_content_plan_rejects_contact_to_unknown_component() -> None:
    payload = build_first_frame_content_plan(_shot()).model_dump(mode="json")
    payload["contact_graph"] = [
        {
            "interaction_id": "bad",
            "actor_component_id": "character_001",
            "target_component_id": "missing",
            "relation": "touches",
            "required_visible_state": "visible contact",
            "hard_gate": True,
        }
    ]

    with pytest.raises(InputValidationError, match="unknown component"):
        parse_first_frame_content_plan(payload)


def test_hard_external_attachment_requires_prop_affordance_before_approval() -> None:
    payload = build_first_frame_content_plan(
        _shot(), assembly_policy=_policy()
    ).model_dump(mode="json")
    payload["attachment_graph"] = [
        {
            "attachment_id": "key_approaches_socket",
            "source_component_id": "prop_001",
            "source_anchor": "key active tip",
            "source_anchor_position": {"x": 0.46, "y": 0.56},
            "target_component_id": "scene",
            "target_anchor": "reactor socket",
            "target_anchor_position": {"x": 0.42, "y": 0.56},
            "relation": "approaches",
            "action_axis": "right-to-left",
            "initial_gap": "four percent of frame width",
            "required_visible_state": (
                "the key tip points toward the empty socket without entering it"
            ),
            "must_remain_visible": ["key tip", "empty socket", "insertion gap"],
            "source_must_remain_visible": ["key tip"],
            "target_must_remain_visible": ["empty socket", "insertion gap"],
            "hard_gate": True,
        }
    ]

    draft = parse_first_frame_content_plan(payload)
    assert draft.attachment_graph[0].target_anchor_position.x == 0.42
    with pytest.raises(InputValidationError, match="functional affordance"):
        approve_first_frame_content_plan(draft)

    payload["prop_states"][0]["functional_affordance"] = {
        "grip_zone": "circular bow",
        "active_end": "shaft tip",
        "native_action_axis": "bow-to-shaft",
    }
    approved = approve_first_frame_content_plan(payload)
    assert approved.prop_states[0].functional_affordance is not None
    assert approved.attachment_graph[0].must_remain_visible == [
        "key tip",
        "empty socket",
        "insertion gap",
    ]
    assert approved.attachment_graph[0].source_must_remain_visible == ["key tip"]


def test_external_attachment_rejects_out_of_frame_anchor() -> None:
    payload = build_first_frame_content_plan(_shot()).model_dump(mode="json")
    payload["attachment_graph"] = [
        {
            "attachment_id": "bad_anchor",
            "source_component_id": "prop_001",
            "source_anchor": "key tip",
            "source_anchor_position": {"x": 1.2, "y": 0.5},
            "target_component_id": "scene",
            "target_anchor": "socket",
            "target_anchor_position": {"x": 0.4, "y": 0.5},
            "relation": "approaches",
            "action_axis": "right-to-left",
            "initial_gap": "visible",
            "required_visible_state": "key points toward socket",
            "must_remain_visible": ["key", "socket"],
            "hard_gate": True,
        }
    ]

    with pytest.raises(InputValidationError, match="within 0..1"):
        parse_first_frame_content_plan(payload)


def test_cli_builds_and_approves_editable_content_plan(tmp_path: Path) -> None:
    shot_path = tmp_path / "shot_plan.json"
    shot_path.write_text(
        json.dumps({"schema_version": "shot-plan-v1", "shots": [_shot()]}),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    draft_path = tmp_path / "first_frame_content.json"

    result = runner.invoke(
        app,
        [
            "director",
            "first-frame-content",
            "--shot-plan",
            str(shot_path),
            "--shot-id",
            "shot_001",
            "--assembly-policy",
            str(policy_path),
            "--output",
            str(draft_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(draft_path.read_text(encoding="utf-8"))["review_status"] == "draft"

    approved_path = tmp_path / "first_frame_content.approved.json"
    result = runner.invoke(
        app,
        [
            "director",
            "first-frame-content-approve",
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
