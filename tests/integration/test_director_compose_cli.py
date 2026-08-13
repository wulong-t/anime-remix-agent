"""CLI integration tests for recoverable first-frame and handoff composition."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.json_io import dump_json_atomic, load_json_object
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
)
from anime_remix.services.script.generation_segment_plan import (
    approve_generation_segment_plan,
    build_generation_segment_plan,
)
from anime_remix.services.script.shot_plan import parse_shot_plan

runner = CliRunner()


def _png(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (96, 64), colour).save(output, format="PNG")
    return output.getvalue()


def _record(
    asset_id: str,
    asset_type: str,
    path: Path,
    *,
    subject: str | None = None,
    roles: tuple[str, ...] = (),
) -> ImageAssetRecord:
    return ImageAssetRecord(
        asset_id=asset_id,
        asset_type=asset_type,
        path=path.name,
        rights_status="user-owned",
        resolved_path=path,
        format="png",
        width=96,
        height=64,
        subject_or_scene_id=subject,
        source_tier="canonical",
        reference_roles=roles,
        analysis_status="analyzed",
    )


def _write_catalog(records: list[ImageAssetRecord], root: Path) -> Path:
    catalog = root / "image_assets.json"
    dump_json_atomic(
        catalog,
        {
            "schema_version": "image-assets-v1",
            "assets": [
                {
                    "asset_id": item.asset_id,
                    "path": item.path,
                    "asset_type": item.asset_type,
                    "rights_status": item.rights_status,
                    "subject_or_scene_id": item.subject_or_scene_id,
                    "source_tier": item.source_tier,
                    "reference_roles": list(item.reference_roles),
                    "analysis_status": item.analysis_status,
                }
                for item in records
            ],
        },
        sort_keys=True,
    )
    return catalog


def _shot(*, props: list[str] | None = None) -> dict:
    return {
        "schema_version": "shot-plan-v1",
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_001",
                "order": 1,
                "narrative_purpose": "Create the canonical first frame.",
                "duration_seconds": 4.0,
                "shot_scale": "medium",
                "composition": "Asuna seated at the desk",
                "camera_position": "front-left eye level",
                "camera_motion": "fixed",
                "subjects": ["Asuna"],
                "setting": "afternoon classroom",
                "props": props or [],
                "start_state": "Asuna sits with both eyes closed",
                "action_beats": [
                    {"time_seconds": 0.0, "description": "eyes closed"},
                    {"time_seconds": 1.5, "description": "opens eyes"},
                ],
                "end_state": "Asuna looks toward the window, eyes open",
                "emotion_arc": "calm",
                "dialogue": None,
                "continuity_in": None,
                "continuity_out": None,
            }
        ],
    }


def _records(root: Path, *, with_prop: bool) -> list[ImageAssetRecord]:
    background_path = root / "background.png"
    background_path.write_bytes(_png((20, 30, 40)))
    identity_path = root / "identity.png"
    identity_path.write_bytes(_png((80, 30, 20)))
    records = [
        _record(
            "bg_classroom",
            "background",
            background_path,
            subject="afternoon classroom",
            roles=("scene_reference",),
        ),
        _record(
            "char_asuna",
            "character",
            identity_path,
            subject="Asuna",
            roles=("identity_reference",),
        ),
    ]
    if with_prop:
        prop_path = root / "prop_key.png"
        prop_path.write_bytes(_png((200, 180, 160)))
        records.append(
            _record(
                "prop_red_key",
                "prop",
                prop_path,
                subject="red key",
                roles=("prop_reference",),
            )
        )
    return records


def _approved_first_frame_plan(
    root: Path,
    *,
    with_prop: bool,
) -> tuple[Path, Path, list[ImageAssetRecord]]:
    records = _records(root, with_prop=with_prop)
    catalog = ImageAssetCatalog.build(records)
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
    shot = parse_shot_plan(_shot(props=["red key"] if with_prop else None)).shots[0]
    plan = build_first_frame_plan(
        shot,
        reference_bundle=bundle,
        catalog=catalog,
        full_frame_anchor_asset_id="bg_classroom",
    )
    approved = approve_first_frame_plan(plan)
    plan_path = root / "first_frame_plan.approved.json"
    dump_json_atomic(
        plan_path,
        approved.model_dump(mode="json"),
        sort_keys=True,
    )
    catalog_path = _write_catalog(records, root)
    return plan_path, catalog_path, records


def _manifest(run_dir: Path, name: str) -> dict:
    return load_json_object(run_dir / name)


def test_first_frame_compose_pauses_then_resumes_to_completed(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path, _ = _approved_first_frame_plan(
        tmp_path, with_prop=False
    )
    run_dir = tmp_path / "run"

    first = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-001",
            "--executor",
            "stub",
            "--max-new-stages",
            "1",
        ],
    )
    assert first.exit_code == 0, first.output
    assert "status=awaiting_review" in first.output
    assert "next=stage_002" in first.output
    manifest = _manifest(run_dir, "first-frame-compose-run.json")
    assert manifest["stages"][0]["approved"] is True
    assert manifest["stages"][1]["approved"] is False

    second = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-001",
            "--executor",
            "stub",
            "--approve-stage",
            "stage_002",
        ],
    )
    assert second.exit_code == 0, second.output
    assert "status=completed" in second.output
    assert "final frame:" in second.output
    manifest = _manifest(run_dir, "first-frame-compose-run.json")
    assert manifest["status"] == "completed"
    assert manifest["final_frame"] is not None
    final_path = run_dir / manifest["final_frame"]["path"]
    assert final_path.is_file()


def test_first_frame_compose_review_only_approves_without_model_call(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path, _ = _approved_first_frame_plan(
        tmp_path, with_prop=False
    )
    run_dir = tmp_path / "run"
    runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-review",
            "--executor",
            "stub",
        ],
    )
    before = _manifest(run_dir, "first-frame-compose-run.json")
    before_hash = before["stages"][1]["output"]["sha256"]

    review = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-review",
            "--executor",
            "review-only",
            "--approve-stage",
            "stage_002",
        ],
    )
    assert review.exit_code == 0, review.output
    assert "status=completed" in review.output
    after = _manifest(run_dir, "first-frame-compose-run.json")
    assert after["status"] == "completed"
    assert after["stages"][1]["output"]["sha256"] == before_hash


def test_first_frame_compose_review_only_blocks_unexpected_model_stage(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path, _ = _approved_first_frame_plan(tmp_path, with_prop=True)
    run_dir = tmp_path / "run"
    runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-guard",
            "--executor",
            "stub",
            "--max-new-stages",
            "1",
        ],
    )
    manifest = _manifest(run_dir, "first-frame-compose-run.json")
    assert manifest["status"] == "awaiting_review"
    assert len(manifest["stages"]) == 3

    review = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-guard",
            "--executor",
            "review-only",
            "--approve-stage",
            "stage_002",
        ],
    )
    assert review.exit_code != 0
    assert "unexpected model call" in review.output
    manifest = _manifest(run_dir, "first-frame-compose-run.json")
    assert manifest["stages"][2]["status"] == "failed"


def test_first_frame_compose_rejects_paid_flag_mismatches(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path, _ = _approved_first_frame_plan(
        tmp_path, with_prop=False
    )
    run_dir = tmp_path / "run"
    paid_without_confirm = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-paid",
            "--executor",
            "dashscope",
        ],
    )
    assert paid_without_confirm.exit_code != 0
    assert "paid-capable" in paid_without_confirm.output

    confirm_with_stub = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "first-frame-cli-paid",
            "--executor",
            "stub",
            "--confirm-paid",
        ],
    )
    assert confirm_with_stub.exit_code != 0
    assert "only valid with --executor dashscope" in confirm_with_stub.output


def test_first_frame_compose_rejects_unknown_stage_id(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path, _ = _approved_first_frame_plan(
        tmp_path, with_prop=False
    )
    result = runner.invoke(
        app,
        [
            "director",
            "first-frame-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--run-id",
            "first-frame-cli-unknown",
            "--executor",
            "stub",
            "--approve-stage",
            "stage_999",
        ],
    )
    assert result.exit_code != 0
    assert "unknown ids" in result.output


def _approved_handoff_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    plan_path, catalog_path, records = _approved_first_frame_plan(
        tmp_path, with_prop=False
    )
    first_frame_path = tmp_path / "first.png"
    first_frame_path.write_bytes(_png((10, 20, 30)))
    shot = parse_shot_plan(_shot()).shots[0]
    first_frame_doc = approve_first_frame_plan(
        build_first_frame_plan(
            shot,
            reference_bundle={
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
            },
            catalog=ImageAssetCatalog.build(records),
            full_frame_anchor_asset_id="bg_classroom",
        )
    )
    segment = approve_generation_segment_plan(
        build_generation_segment_plan(
            shot,
            first_frame_plan=first_frame_doc,
            boundary_intents={
                "schema_version": "segment-boundary-intents-v1",
                "shot_id": "shot_001",
                "boundaries": [
                    {
                        "anchor_id": "shot_001_eyes_open",
                        "time_seconds": 2.0,
                        "target_state": (
                            "Asuna remains seated with both eyes open"
                        ),
                        "composition": "same desk composition",
                        "camera": "same front-left camera",
                        "process_from_previous": "slowly opens both eyes",
                        "dominant_motion": "eyelids open",
                        "camera_motion": "fixed",
                        "delta_instruction": (
                            "Change only the eyelids and visible irises."
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
            },
        )
    )
    segment_path = tmp_path / "generation_segment_plan.approved.json"
    dump_json_atomic(
        segment_path,
        segment.model_dump(mode="json"),
        sort_keys=True,
    )
    return segment_path, first_frame_path, catalog_path, plan_path


def test_handoff_compose_pauses_then_resumes_to_completed(
    tmp_path: Path,
) -> None:
    segment_path, first_frame_path, catalog_path, _ = (
        _approved_handoff_inputs(tmp_path)
    )
    run_dir = tmp_path / "handoff-run"
    first = runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(first_frame_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "handoff-cli-001",
            "--executor",
            "stub",
        ],
    )
    assert first.exit_code == 0, first.output
    assert "status=awaiting_review" in first.output
    manifest = _manifest(run_dir, "handoff-frame-compose-run.json")
    assert manifest["anchors"][0]["approved"] is True
    assert manifest["anchors"][1]["approved"] is False

    second = runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(first_frame_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "handoff-cli-001",
            "--executor",
            "stub",
            "--approve-anchor",
            "shot_001_eyes_open",
        ],
    )
    assert second.exit_code == 0, second.output
    assert "status=awaiting_review" in second.output
    manifest = _manifest(run_dir, "handoff-frame-compose-run.json")
    assert manifest["anchors"][1]["approved"] is True
    assert manifest["anchors"][2]["approved"] is False

    third = runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(first_frame_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "handoff-cli-001",
            "--executor",
            "stub",
            "--approve-anchor",
            "shot_001_end",
        ],
    )
    assert third.exit_code == 0, third.output
    assert "status=completed" in third.output
    manifest = _manifest(run_dir, "handoff-frame-compose-run.json")
    assert manifest["status"] == "completed"
    assert manifest["final_frame"] is not None


def test_handoff_compose_review_only_blocks_next_model_anchor(
    tmp_path: Path,
) -> None:
    segment_path, first_frame_path, catalog_path, _ = (
        _approved_handoff_inputs(tmp_path)
    )
    run_dir = tmp_path / "handoff-run"
    runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(first_frame_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "handoff-cli-review",
            "--executor",
            "stub",
        ],
    )
    before = _manifest(run_dir, "handoff-frame-compose-run.json")
    before_hash = before["anchors"][1]["output"]["sha256"]

    review = runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(first_frame_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "handoff-cli-review",
            "--executor",
            "review-only",
            "--approve-anchor",
            "shot_001_eyes_open",
        ],
    )
    assert review.exit_code != 0
    assert "unexpected model call" in review.output
    after = _manifest(run_dir, "handoff-frame-compose-run.json")
    assert after["status"] == "failed"
    assert after["anchors"][1]["output"]["sha256"] == before_hash
    assert after["anchors"][2]["status"] == "failed"


def test_handoff_compose_rejects_missing_first_frame(
    tmp_path: Path,
) -> None:
    segment_path, _, catalog_path, _ = _approved_handoff_inputs(tmp_path)
    result = runner.invoke(
        app,
        [
            "director",
            "handoff-compose",
            "--plan",
            str(segment_path),
            "--first-frame",
            str(tmp_path / "missing.png"),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(tmp_path / "handoff-run"),
            "--run-id",
            "handoff-cli-missing",
            "--executor",
            "stub",
        ],
    )
    assert result.exit_code != 0
    assert "not a file" in result.output


def _component_plan_and_catalog(tmp_path: Path) -> tuple[Path, Path]:
    identity_path = tmp_path / "identity.png"
    identity_path.write_bytes(_png((120, 90, 70)))
    catalog = _write_catalog(
        [
            _record(
                "char_asuna",
                "character",
                identity_path,
                subject="Asuna",
                roles=("identity_reference",),
            )
        ],
        tmp_path,
    )
    plan = {
        "schema_version": "prepared-component-plan-v1",
        "plan_id": "shot_001-prepared-components",
        "shot_id": "shot_001",
        "content_plan_id": "shot_001-first-frame-content",
        "review_status": "approved",
        "completion_status": "pending",
        "decision": "ready",
        "max_primary_visual_references_per_model_call": 2,
        "source_asset_ids": ["char_asuna"],
        "tasks": [
            {
                "task_id": "task.interaction.mira_grips_key",
                "kind": "interaction_plate",
                "component_ids": ["character_001"],
                "subjects": ["Asuna"],
                "target_state": "Asuna grips the red key",
                "model_inputs": [{"asset_id": "char_asuna", "function": "who"}],
                "control_evidence_asset_ids": [],
                "preserve_attributes": ["identity"],
                "allowed_text_fallbacks": {"action": "grips the red key"},
                "prop_affordances": [],
                "external_attachments": [],
                "review_gates": [
                    {
                        "gate_id": "gate.contact.mira_grips_key",
                        "criterion": (
                            "Confirm the atomic contact is visually true: "
                            "Asuna's fingers wrap around the red key."
                        ),
                    }
                ],
                "output_asset_id": "prep_shot_001_interaction",
                "result": "pending",
                "result_review_notes": None,
                "gate_results": [],
            }
        ],
        "warnings": [],
    }
    plan_path = tmp_path / "prepared_component_plan.approved.json"
    dump_json_atomic(plan_path, plan, sort_keys=True)
    return plan_path, catalog


def test_component_compose_review_only_resume_makes_no_new_call(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path = _component_plan_and_catalog(tmp_path)
    run_dir = tmp_path / "component-run"
    first = runner.invoke(
        app,
        [
            "director",
            "component-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "component-cli-001",
            "--executor",
            "stub",
            "--max-new-tasks",
            "1",
        ],
    )
    assert first.exit_code == 0, first.output
    assert "status=completed" in first.output
    before = _manifest(run_dir, "prepared-component-compose-run.json")
    before_hash = before["tasks"][0]["output"]["sha256"]

    review = runner.invoke(
        app,
        [
            "director",
            "component-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "component-cli-001",
            "--executor",
            "review-only",
            "--max-new-tasks",
            "1",
        ],
    )
    assert review.exit_code == 0, review.output
    assert "status=completed" in review.output
    after = _manifest(run_dir, "prepared-component-compose-run.json")
    assert after["tasks"][0]["attempts"] == 1
    assert after["tasks"][0]["output"]["sha256"] == before_hash


def test_component_compose_review_only_blocks_pending_task(
    tmp_path: Path,
) -> None:
    plan_path, catalog_path = _component_plan_and_catalog(tmp_path)
    result = runner.invoke(
        app,
        [
            "director",
            "component-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(tmp_path / "component-run"),
            "--run-id",
            "component-cli-guard",
            "--executor",
            "review-only",
            "--max-new-tasks",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "unexpected model call" in result.output
    manifest = _manifest(tmp_path / "component-run", "prepared-component-compose-run.json")
    assert manifest["tasks"][0]["status"] == "failed"


def test_component_review_records_approved_result(tmp_path: Path) -> None:
    plan_path, _ = _component_plan_and_catalog(tmp_path)
    output_path = tmp_path / "prepared_component_plan.reviewed.json"
    result = runner.invoke(
        app,
        [
            "director",
            "component-review",
            "--plan",
            str(plan_path),
            "--result",
            "approved",
            "--gate-result",
            "gate.contact.mira_grips_key=pass",
            "--note",
            "visual review approved the generated component plate",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    reviewed = load_json_object(output_path)
    task = reviewed["tasks"][0]
    assert task["result"] == "approved"
    assert task["gate_results"] == [
        {
            "gate_id": "gate.contact.mira_grips_key",
            "result": "pass",
            "note": "visual review approved the generated component plate",
        }
    ]


def test_component_review_records_rejected_result(tmp_path: Path) -> None:
    plan_path, _ = _component_plan_and_catalog(tmp_path)
    output_path = tmp_path / "prepared_component_plan.reviewed.json"
    result = runner.invoke(
        app,
        [
            "director",
            "component-review",
            "--plan",
            str(plan_path),
            "--result",
            "rejected",
            "--gate-result",
            "gate.contact.mira_grips_key=fail",
            "--note",
            "target scene object was rendered in the plate",
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    task = load_json_object(output_path)["tasks"][0]
    assert task["result"] == "rejected"
    assert task["gate_results"][0]["result"] == "fail"


def test_component_review_rejects_incomplete_gates_for_approval(
    tmp_path: Path,
) -> None:
    plan_path, _ = _component_plan_and_catalog(tmp_path)
    result = runner.invoke(
        app,
        [
            "director",
            "component-review",
            "--plan",
            str(plan_path),
            "--result",
            "approved",
            "--note",
            "missing gate verdicts",
            "--output",
            str(tmp_path / "reviewed.json"),
        ],
    )
    assert result.exit_code != 0
    assert "requires a verdict for every review gate" in result.output


def test_component_review_rejects_unknown_gate(tmp_path: Path) -> None:
    plan_path, _ = _component_plan_and_catalog(tmp_path)
    result = runner.invoke(
        app,
        [
            "director",
            "component-review",
            "--plan",
            str(plan_path),
            "--result",
            "rejected",
            "--gate-result",
            "gate.does_not_exist=fail",
            "--note",
            "unknown gate",
            "--output",
            str(tmp_path / "reviewed.json"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown review gate" in result.output
