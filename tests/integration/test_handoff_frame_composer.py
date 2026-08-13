"""Offline integration tests for shared handoff-anchor generation."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from anime_remix.errors import InputValidationError, RenderError
from anime_remix.services.execution.adapter import QwenImage30Adapter
from anime_remix.services.execution.handoff_frame_composer import (
    run_handoff_frame_composition,
)
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


def _png(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (96, 64), colour).save(output, format="PNG")
    return output.getvalue()


class RecordingExecutor:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self.last_metadata: dict = {}
        self.fail_on_call = fail_on_call

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        assert operation == "first_frame_fusion"
        assert 1 <= len(inputs) <= 2
        call = {
            "request": request_payload,
            "input_hashes": {
                key: hashlib.sha256(value).hexdigest() for key, value in inputs.items()
            },
        }
        self.calls.append(call)
        self.last_metadata = {
            "provider": "offline-test",
            "model": "stub",
            "request_id": f"call-{len(self.calls)}",
            "status": "success",
        }
        if self.fail_on_call == len(self.calls):
            self.last_metadata.update(
                {
                    "status": "failed",
                    "http_status": 403,
                    "provider_code": "DataInspectionFailed",
                }
            )
            raise RenderError("injected handoff failure")
        return _png((40 * len(self.calls), 70, 100))


class NoChangeExecutor:
    """Returns the base anchor unchanged, simulating an ignored edit."""

    provider = "offline-no-change"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_metadata: dict = {}

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.calls.append(request_payload)
        self.last_metadata = {
            "provider": "offline-no-change",
            "model": "stub",
            "request_id": f"noop-{len(self.calls)}",
            "status": "succeeded",
        }
        return next(iter(inputs.values()))


def test_failed_handoff_persists_request_and_provider_metadata(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    run_dir = tmp_path / "run"

    with pytest.raises(RenderError, match="injected handoff"):
        run_handoff_frame_composition(
            run_dir=run_dir,
            run_id="handoff-failure",
            plan=_approved_segment_plan(),
            first_frame_path=first_path,
            asset_map={"char_asuna": identity_path},
            adapter=QwenImage30Adapter(),
            executor=RecordingExecutor(fail_on_call=1),
            auto_approve=True,
        )

    manifest = json.loads(
        (run_dir / "handoff-frame-compose-run.json").read_text(encoding="utf-8")
    )
    failed = manifest["anchors"][1]
    assert failed["request"]["provider"]["http_status"] == 403
    assert failed["request"]["provider"]["provider_code"] == ("DataInspectionFailed")


def _shot():
    return parse_shot_plan(
        {
            "schema_version": "shot-plan-v1",
            "shots": [
                {
                    "shot_id": "shot_001",
                    "scene_id": "scene_001",
                    "order": 1,
                    "narrative_purpose": "Reveal the eyes, then turn.",
                    "duration_seconds": 4.0,
                    "shot_scale": "medium",
                    "composition": "Asuna seated at the desk",
                    "camera_position": "front-left eye level",
                    "camera_motion": "fixed",
                    "subjects": ["Asuna"],
                    "setting": "afternoon classroom",
                    "props": [],
                    "start_state": "Asuna sits with both eyes closed",
                    "action_beats": [
                        {"time_seconds": 0.0, "description": "eyes closed"},
                        {"time_seconds": 1.5, "description": "opens eyes"},
                        {"time_seconds": 3.0, "description": "turns to window"},
                    ],
                    "end_state": "Asuna looks toward the window, eyes open",
                    "emotion_arc": "calm",
                    "dialogue": None,
                    "continuity_in": None,
                    "continuity_out": None,
                }
            ],
        }
    ).shots[0]


def _first_frame_plan():
    identity = ImageAssetRecord(
        asset_id="char_asuna",
        asset_type="character",
        path="images/char_asuna.png",
        rights_status="user-owned",
        resolved_path=Path("C:/char_asuna.png"),
        format="png",
        width=96,
        height=64,
        subject_or_scene_id="Asuna",
        source_tier="canonical",
        reference_roles=("identity_reference", "expression_reference"),
        analysis_status="analyzed",
    )
    catalog = ImageAssetCatalog.build([identity])
    bundle = {
        "schema_version": "reference-bundle-v1",
        "shot_id": "shot_001",
        "references": [
            {
                "asset_id": "char_asuna",
                "asset_type": "character",
                "path": "images/char_asuna.png",
                "note": "canonical identity and open-eye authority",
            }
        ],
    }
    return approve_first_frame_plan(
        build_first_frame_plan(_shot(), reference_bundle=bundle, catalog=catalog)
    )


def _boundary_intents() -> dict:
    return {
        "schema_version": "segment-boundary-intents-v1",
        "shot_id": "shot_001",
        "boundaries": [
            {
                "anchor_id": "shot_001_eyes_open",
                "time_seconds": 2.0,
                "target_state": "Asuna remains seated with both eyes open",
                "composition": "same desk composition",
                "camera": "same front-left camera",
                "process_from_previous": "slowly opens both eyes",
                "dominant_motion": "eyelids open",
                "camera_motion": "fixed",
                "delta_instruction": "Change only the eyelids and visible irises.",
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


def _approved_segment_plan():
    return approve_generation_segment_plan(
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=_first_frame_plan(),
            boundary_intents=_boundary_intents(),
        )
    )


def test_composer_generates_one_model_anchor_per_call_and_reuses_exact_bytes(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    plan = _approved_segment_plan()
    executor = RecordingExecutor()
    run_dir = tmp_path / "run"

    first_result = run_handoff_frame_composition(
        run_dir=run_dir,
        run_id="handoff-001",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
    )

    assert first_result.status == "awaiting_review"
    assert first_result.completed_anchor_ids == (
        "shot_001_first",
        "shot_001_eyes_open",
    )
    assert len(executor.calls) == 1
    assert len(executor.calls[0]["input_hashes"]) == 2
    middle_path = run_dir / "anchors/002_shot_001_eyes_open/frame.png"
    middle_hash = hashlib.sha256(middle_path.read_bytes()).hexdigest()

    second_result = run_handoff_frame_composition(
        run_dir=run_dir,
        run_id="handoff-001",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
        approved_anchor_ids={"shot_001_eyes_open"},
    )

    assert second_result.status == "awaiting_review"
    assert len(executor.calls) == 2
    assert len(executor.calls[1]["input_hashes"]) == 1
    assert middle_hash in executor.calls[1]["input_hashes"].values()

    completed = run_handoff_frame_composition(
        run_dir=run_dir,
        run_id="handoff-001",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
        approved_anchor_ids={"shot_001_eyes_open", "shot_001_end"},
    )
    assert completed.status == "completed"
    assert completed.final_frame_path is not None
    assert len(executor.calls) == 2


def test_zero_model_frame_budget_prepares_reusable_anchors_then_pauses(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    payload = _approved_segment_plan().model_dump(mode="json")
    payload["anchors"][1].update(
        {
            "generation_method": "reuse_existing_asset",
            "delta_instruction": None,
            "grounding": "exact_visual_asset",
        }
    )
    plan = parse_generation_segment_plan(payload)
    executor = RecordingExecutor()

    result = run_handoff_frame_composition(
        run_dir=tmp_path / "prepare-run",
        run_id="handoff-prepare-only",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
        max_new_model_frames=0,
    )

    assert result.status == "paused_limit"
    assert result.completed_anchor_ids == (
        "shot_001_first",
        "shot_001_eyes_open",
    )
    assert result.next_anchor_id == "shot_001_end"
    assert executor.calls == []
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_call_count"] == 0
    assert manifest["anchors"][1]["output"]["sha256"] == hashlib.sha256(
        identity_path.read_bytes()
    ).hexdigest()


def test_negative_model_frame_budget_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))

    with pytest.raises(InputValidationError, match="integer >= 0"):
        run_handoff_frame_composition(
            run_dir=tmp_path / "negative-budget-run",
            run_id="handoff-negative-budget",
            plan=_approved_segment_plan(),
            first_frame_path=first_path,
            asset_map={"char_asuna": identity_path},
            adapter=QwenImage30Adapter(),
            executor=RecordingExecutor(),
            max_new_model_frames=-1,
        )


def test_high_risk_compound_anchor_cannot_be_auto_approved(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    intents = _boundary_intents()
    intents["boundaries"][0]["control_reasons"].append("contact_topology")
    plan = approve_generation_segment_plan(
        build_generation_segment_plan(
            _shot(),
            first_frame_plan=_first_frame_plan(),
            boundary_intents=intents,
        )
    )

    result = run_handoff_frame_composition(
        run_dir=tmp_path / "risk-run",
        run_id="handoff-high-risk",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=RecordingExecutor(),
        auto_approve=True,
        max_new_model_frames=2,
    )

    assert result.status == "awaiting_review"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    boundary = manifest["anchors"][1]
    assert boundary["generation_risk"] == "high"
    assert boundary["requires_review"] is True
    assert boundary["approved"] is False


def test_exact_reference_anchor_is_adopted_without_model_call(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    exact_path = tmp_path / "exact-end.png"
    first_path.write_bytes(_png((1, 2, 3)))
    exact_path.write_bytes(_png((4, 5, 6)))
    first_plan = _first_frame_plan()
    intents = {
        "schema_version": "segment-boundary-intents-v1",
        "shot_id": "shot_001",
        "boundaries": [
            {
                "anchor_id": "shot_001_end",
                "time_seconds": 4.0,
                "target_state": "exact approved endpoint",
                "composition": "exact approved endpoint",
                "camera": "exact approved endpoint",
                "process_from_previous": "turns toward the window",
                "dominant_motion": "head turn",
                "camera_motion": "fixed",
                "delta_instruction": None,
                "control_reasons": ["shot_end"],
                "reveal_fact_ids": [],
                "generation_method": "reuse_existing_asset",
                "reference_asset_id": "char_asuna",
                "reference_role": "identity",
                "reference_attributes": ["complete target frame"],
                "locked_attributes": ["all visible facts"],
            }
        ],
    }
    plan = approve_generation_segment_plan(
        build_generation_segment_plan(
            _shot(), first_frame_plan=first_plan, boundary_intents=intents
        )
    )
    executor = RecordingExecutor()

    result = run_handoff_frame_composition(
        run_dir=tmp_path / "run",
        run_id="handoff-exact",
        plan=plan,
        first_frame_path=first_path,
        asset_map={"char_asuna": exact_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
    )

    assert result.status == "completed"
    assert executor.calls == []
    assert result.final_frame_path is not None
    with Image.open(result.final_frame_path) as image:
        assert image.getpixel((0, 0)) == (4, 5, 6)


def test_composer_rejects_draft_and_input_drift(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    draft = build_generation_segment_plan(
        _shot(),
        first_frame_plan=_first_frame_plan(),
        boundary_intents=_boundary_intents(),
    )
    executor = RecordingExecutor()

    with pytest.raises(InputValidationError, match="explicitly approved"):
        run_handoff_frame_composition(
            run_dir=tmp_path / "draft-run",
            run_id="draft",
            plan=draft,
            first_frame_path=first_path,
            asset_map={"char_asuna": identity_path},
            adapter=QwenImage30Adapter(),
            executor=executor,
        )

    approved = approve_generation_segment_plan(draft)
    run_handoff_frame_composition(
        run_dir=tmp_path / "drift-run",
        run_id="drift",
        plan=approved,
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=executor,
        auto_approve=True,
        max_new_model_frames=2,
    )
    first_path.write_bytes(_png((99, 88, 77)))

    with pytest.raises(InputValidationError, match="contract changed"):
        run_handoff_frame_composition(
            run_dir=tmp_path / "drift-run",
            run_id="drift",
            plan=approved,
            first_frame_path=first_path,
            asset_map={"char_asuna": identity_path},
            adapter=QwenImage30Adapter(),
            executor=executor,
            auto_approve=True,
            max_new_model_frames=2,
        )


def test_unchanged_handoff_edit_flags_semantic_change_suspect(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    identity_path = tmp_path / "identity.png"
    first_path.write_bytes(_png((10, 20, 30)))
    identity_path.write_bytes(_png((200, 180, 160)))
    run_dir = tmp_path / "run"

    result = run_handoff_frame_composition(
        run_dir=run_dir,
        run_id="handoff-noop",
        plan=_approved_segment_plan(),
        first_frame_path=first_path,
        asset_map={"char_asuna": identity_path},
        adapter=QwenImage30Adapter(),
        executor=NoChangeExecutor(),
        auto_approve=True,
    )

    assert result.status == "awaiting_review"
    manifest = json.loads(
        (run_dir / "handoff-frame-compose-run.json").read_text(encoding="utf-8")
    )
    middle = manifest["anchors"][1]
    assert middle["qa"]["semantic_change_suspect"] is True
    assert middle["qa"]["changed_fraction"] < 0.001
    assert middle["requires_review"] is True
    assert middle["approved"] is False
