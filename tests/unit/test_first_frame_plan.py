"""Tests for deterministic reference-first first-frame planning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import (
    ImageAssetCatalog,
    ImageAssetRecord,
)
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
    parse_first_frame_plan,
)

runner = CliRunner()


def _record(
    asset_id: str,
    asset_type: str,
    *,
    subject: str | None = None,
    roles: tuple[str, ...] = (),
    pose: str | None = None,
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
        pose=pose,
        source_tier="canonical",
        reference_roles=roles,
        analysis_status="analyzed",
    )


def _catalog(*records: ImageAssetRecord) -> ImageAssetCatalog:
    return ImageAssetCatalog.build(list(records))


def _shot(*, subjects: list[str] | None = None, props: list[str] | None = None) -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "Anchor the scene and character.",
        "duration_seconds": 4.0,
        "shot_scale": "medium",
        "composition": "character on the right, desk in foreground",
        "camera_position": "front eye level",
        "camera_motion": "fixed",
        "subjects": ["Asuna"] if subjects is None else subjects,
        "setting": "afternoon classroom",
        "props": ["book"] if props is None else props,
        "start_state": "Asuna sits at the desk with both eyes closed",
        "action_beats": [
            {"time_seconds": 0.0, "description": "holds still with eyes closed"}
        ],
        "end_state": "Asuna opens her eyes",
        "emotion_arc": "calm",
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


def test_plan_consumes_relevant_assets_in_two_image_stages() -> None:
    background = _record(
        "bg_classroom",
        "background",
        subject="afternoon classroom",
        roles=("scene_reference",),
    )
    style = _record("style_master", "style", roles=("style_reference",))
    identity = _record(
        "char_asuna",
        "character",
        subject="Asuna",
        roles=("identity_reference", "outfit_reference"),
    )
    pose = _record(
        "pose_asuna_sitting",
        "character",
        subject="Asuna",
        roles=("pose_reference",),
        pose="sitting",
    )
    prop = _record(
        "prop_book",
        "prop",
        subject="book",
        roles=("prop_reference",),
    )
    records = (background, style, identity, pose, prop)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(*records),
        catalog=_catalog(*records),
    )

    assert plan.review_status == "draft"
    assert plan.selected_asset_ids == [
        "bg_classroom",
        "style_master",
        "char_asuna",
        "pose_asuna_sitting",
        "prop_book",
    ]
    assert plan.unused_bound_asset_ids == []
    assert [stage.operation for stage in plan.stages] == [
        "synthesize_base",
        "fuse_component",
        "fuse_component",
        "fuse_component",
    ]
    assert all(len(stage.inputs) <= 2 for stage in plan.stages)
    assert [item.source_id for item in plan.stages[0].inputs] == [
        "bg_classroom",
        "style_master",
    ]
    eye_fact = next(
        item for item in plan.information_coverage if item.fact_id.endswith(".eyes")
    )
    assert eye_fact.status == "occluded"
    assert "char_asuna" in eye_fact.source_asset_ids
    approved = approve_first_frame_plan(plan)
    assert approved.review_status == "approved"


def test_scene_only_shot_can_adopt_exact_full_frame_without_model_stage() -> None:
    background = _record(
        "bg_rooftop",
        "background",
        subject="afternoon classroom",
        roles=("scene_reference",),
    )

    plan = build_first_frame_plan(
        _shot(subjects=[], props=[]),
        reference_bundle=_bundle(background),
        catalog=_catalog(background),
        full_frame_anchor_asset_id="bg_rooftop",
    )

    assert plan.decision == "ready"
    assert [stage.operation for stage in plan.stages] == ["adopt_anchor"]
    assert plan.components[0].kind == "scene"
    assert not any(item.kind == "character" for item in plan.components)


def test_future_information_reference_is_deferred_from_first_frame_stages() -> None:
    background = _record("bg", "background", subject="afternoon classroom")
    exterior = _record("reactor_closed", "prop", subject="reactor")
    interior = _record("reactor_inner", "prop", subject="reactor")
    records = (background, exterior, interior)

    plan = build_first_frame_plan(
        _shot(subjects=[], props=["reactor"]),
        reference_bundle=_bundle(*records),
        catalog=_catalog(*records),
        full_frame_anchor_asset_id="bg",
        deferred_asset_ids={"reactor_inner"},
    )

    assert plan.deferred_asset_ids == ["reactor_inner"]
    assert "reactor_inner" not in plan.selected_asset_ids
    assert "reactor_inner" not in plan.unused_bound_asset_ids
    assert all(
        input_item.source_id != "reactor_inner"
        for stage in plan.stages
        for input_item in stage.inputs
    )


def test_deferred_first_frame_asset_must_be_bound() -> None:
    background = _record("bg", "background", subject="afternoon classroom")

    with pytest.raises(InputValidationError, match="not bound"):
        build_first_frame_plan(
            _shot(subjects=[], props=[]),
            reference_bundle=_bundle(background),
            catalog=_catalog(background),
            deferred_asset_ids={"future_core"},
        )


def test_no_scene_reference_creates_text_only_base_then_visual_character() -> None:
    identity = _record(
        "char_asuna",
        "character",
        subject="Asuna",
        roles=("identity_reference",),
    )

    plan = build_first_frame_plan(
        _shot(props=[]),
        reference_bundle=_bundle(identity),
        catalog=_catalog(identity),
    )

    assert plan.stages[0].operation == "synthesize_base"
    assert plan.stages[0].inputs == []
    assert plan.stages[0].text_fallbacks["setting"] == "afternoon classroom"
    assert plan.stages[1].inputs[1].source_id == "char_asuna"
    assert plan.decision == "needs_review"


def test_unassigned_character_reference_is_explicitly_unused() -> None:
    background = _record("bg", "background", subject="afternoon classroom")
    asuna = _record("asuna", "character", subject="Asuna")
    kirito = _record("kirito", "character", subject="Kirito")
    records = (background, asuna, kirito)

    plan = build_first_frame_plan(
        _shot(props=[]),
        reference_bundle=_bundle(*records),
        catalog=_catalog(*records),
    )

    assert plan.unused_bound_asset_ids == ["kirito"]
    assert any("unassigned character" in warning for warning in plan.warnings)
    assert plan.decision == "needs_review"


def test_approved_blocked_plan_is_rejected() -> None:
    background = _record("bg", "background", subject="afternoon classroom")
    plan = build_first_frame_plan(
        _shot(subjects=[], props=[]),
        reference_bundle=_bundle(background),
        catalog=_catalog(background),
        full_frame_anchor_asset_id="bg",
    ).model_dump(mode="json")
    plan["decision"] = "blocked"
    plan["review_status"] = "approved"

    with pytest.raises(InputValidationError, match="blocked"):
        parse_first_frame_plan(plan)


def test_cli_builds_then_explicitly_approves_first_frame_plan(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (64, 48), (20, 30, 40)).save(image_dir / "classroom.png")
    assets_path = tmp_path / "image_assets.json"
    assets_path.write_text(
        json.dumps(
            {
                "schema_version": "image-assets-v1",
                "assets": [
                    {
                        "asset_id": "bg_classroom",
                        "path": "images/classroom.png",
                        "asset_type": "background",
                        "rights_status": "user-owned",
                        "subject_or_scene_id": "afternoon classroom",
                        "reference_roles": ["scene_reference"],
                        "analysis_status": "analyzed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    shot_plan_path = tmp_path / "shot_plan.json"
    shot_plan_path.write_text(
        json.dumps(
            {"schema_version": "shot-plan-v1", "shots": [_shot(subjects=[], props=[])]}
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "reference_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "reference-bundle-v1",
                "shot_id": "shot_001",
                "references": [
                    {
                        "asset_id": "bg_classroom",
                        "asset_type": "background",
                        "path": "images/classroom.png",
                        "note": "complete target frame",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "first_frame_assembly_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "first-frame-assembly-policy-v1",
                "shot_id": "shot_001",
                "reference_authorities": [
                    {
                        "asset_id": "bg_classroom",
                        "authority": "final_visual",
                        "reason": "approved production-ready full frame",
                    }
                ],
                "require_production_quality_review": False,
            }
        ),
        encoding="utf-8",
    )
    draft = tmp_path / "first_frame_plan.json"

    result = runner.invoke(
        app,
        [
            "director",
            "first-frame-plan",
            "--shot-plan",
            str(shot_plan_path),
            "--shot-id",
            "shot_001",
            "--reference-bundle",
            str(bundle_path),
            "--image-assets",
            str(assets_path),
            "--full-frame-anchor",
            "bg_classroom",
            "--assembly-policy",
            str(policy_path),
            "--output",
            str(draft),
        ],
    )
    assert result.exit_code == 0, result.output
    draft_payload = json.loads(draft.read_text(encoding="utf-8"))
    assert draft_payload["review_status"] == "draft"
    assert draft_payload["reference_admissions"][0]["authority"] == "final_visual"

    approved = tmp_path / "first_frame_plan.approved.json"
    result = runner.invoke(
        app,
        [
            "director",
            "first-frame-approve",
            "--plan",
            str(draft),
            "--output",
            str(approved),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        json.loads(approved.read_text(encoding="utf-8"))["review_status"] == "approved"
    )


def test_prepared_placement_instruction_requires_unobstructed_target_region() -> None:
    from anime_remix.services.script.first_frame_plan import (
        _prepared_placement_instruction,
    )
    from anime_remix.services.script.prepared_component_plan import (
        PreparedComponentTask,
        PreparedExternalAttachment,
    )

    attachment = PreparedExternalAttachment(
        attachment_id="key_approaches_socket",
        source_component_id="prop_001",
        source_anchor="activation-key shaft tip",
        source_anchor_x=0.56,
        source_anchor_y=0.66,
        target_component_id="scene",
        target_subject="abandoned astronomical observatory",
        target_anchor="empty socket",
        target_anchor_x=0.5,
        target_anchor_y=0.66,
        relation="approaches",
        action_axis="right-to-left, horizontal",
        initial_gap="six percent of final-frame width, clearly visible",
        required_visible_state=(
            "the shaft tip points toward the empty socket without entering it"
        ),
        must_remain_visible=["shaft tip", "empty socket", "insertion gap"],
        source_must_remain_visible=["shaft tip"],
        target_must_remain_visible=["empty socket", "insertion gap"],
        hard_gate=True,
    )
    task = PreparedComponentTask(
        task_id="task.interaction.mira_grips_key",
        kind="interaction_plate",
        component_ids=["character_001", "prop_001"],
        subjects=["Mira", "red key"],
        target_state="Mira grips the red key",
        model_inputs=[{"asset_id": "mira_who", "function": "who"}],
        preserve_attributes=["identity"],
        allowed_text_fallbacks={"action": "grips the red key"},
        output_asset_id="prep_shot_comp_001_interaction_mira_grips_key",
        external_attachments=[attachment],
    )
    text = _prepared_placement_instruction(task)
    assert "remain fully visible and unobstructed" in text
    assert "(0.500, 0.660)" in text
    assert "character's body, arm, hand or the key" in text


def test_content_screen_position_flows_into_character_fusion_instruction() -> None:
    from anime_remix.services.script.first_frame_content_plan import (
        approve_first_frame_content_plan,
        build_first_frame_content_plan,
    )

    background = _record(
        "bg_classroom",
        "background",
        subject="afternoon classroom",
        roles=("scene_reference",),
    )
    identity = _record(
        "char_asuna",
        "character",
        subject="Asuna",
        roles=("identity_reference",),
    )
    payload = build_first_frame_content_plan(_shot()).model_dump(mode="json")
    payload["character_states"][0]["screen_position"] = (
        "right edge of the frame; Asuna and her arm must stay right of x = 0.60"
    )
    content = approve_first_frame_content_plan(payload)

    plan = build_first_frame_plan(
        _shot(),
        reference_bundle=_bundle(background, identity),
        catalog=_catalog(background, identity),
        full_frame_anchor_asset_id="bg_classroom",
        content_plan=content,
    )
    fuse = next(stage for stage in plan.stages if stage.operation == "fuse_component")
    assert "Placement constraint:" in fuse.instruction
    assert "right of x = 0.60" in fuse.instruction
