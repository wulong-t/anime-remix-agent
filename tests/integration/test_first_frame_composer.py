"""Offline integration tests for recoverable staged first-frame fusion."""

from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from anime_remix.errors import InputValidationError, RenderError
from anime_remix.services.execution.adapter import QwenImage30Adapter
from anime_remix.services.execution.first_frame_composer import (
    run_first_frame_composition,
)
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
)


def _png(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 48), colour).save(output, format="PNG")
    return output.getvalue()


def _rgba_overlay(
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    colour: tuple[int, int, int, int],
) -> bytes:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            image.putpixel((x, y), colour)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class RecordingExecutor:
    provider = "offline-recording"

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        colour_offset: int = 0,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.colour_offset = colour_offset
        self.calls: list[dict] = []
        self.last_metadata: dict | None = None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.calls.append(
            {
                "request": copy.deepcopy(request_payload),
                "operation": operation,
                "inputs": dict(inputs),
            }
        )
        if self.fail_on_call == len(self.calls):
            self.last_metadata = {
                "provider": "offline-recording",
                "model": "stub",
                "request_id": f"failed-{len(self.calls)}",
                "status": "failed",
                "http_status": 403,
                "provider_code": "DataInspectionFailed",
            }
            raise RenderError("injected first-frame stage failure")
        self.last_metadata = {
            "provider": "offline-recording",
            "model": "stub",
            "request_id": f"success-{len(self.calls)}",
            "status": "succeeded",
        }
        return _png((40 * (len(self.calls) + self.colour_offset), 80, 120))


class NoOpComponentExecutor(RecordingExecutor):
    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.calls.append(
            {
                "request": copy.deepcopy(request_payload),
                "operation": operation,
                "inputs": dict(inputs),
            }
        )
        self.last_metadata = {
            "provider": "offline-recording",
            "model": "stub",
            "request_id": f"noop-{len(self.calls)}",
            "status": "succeeded",
        }
        if len(inputs) == 2:
            base_ref = request_payload["conditions"][0]["condition_ref"]
            return inputs[base_ref]
        return _png((40, 80, 120))


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
        width=64,
        height=48,
        subject_or_scene_id=subject,
        source_tier="canonical",
        reference_roles=roles,
        analysis_status="analyzed",
    )


def _shot(*, subjects: list[str] | None = None) -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "Create the canonical first frame.",
        "duration_seconds": 4.0,
        "shot_scale": "medium",
        "composition": "Asuna on the right",
        "camera_position": "front eye level",
        "camera_motion": "fixed",
        "subjects": ["Asuna"] if subjects is None else subjects,
        "setting": "afternoon classroom",
        "props": [],
        "start_state": "Asuna sits with both eyes closed",
        "action_beats": [{"time_seconds": 0.0, "description": "sits with eyes closed"}],
        "end_state": "Asuna opens her eyes",
        "emotion_arc": "calm",
    }


def _plan_and_assets(tmp_path: Path, *, scene_only: bool = False):
    background_path = tmp_path / "background.png"
    background_path.write_bytes(_png((20, 30, 40)))
    background = _record(
        "bg_classroom",
        "background",
        background_path,
        subject="afternoon classroom",
        roles=("scene_reference",),
    )
    records = [background]
    if not scene_only:
        identity_path = tmp_path / "identity.png"
        identity_path.write_bytes(_png((80, 30, 20)))
        records.append(
            _record(
                "char_asuna",
                "character",
                identity_path,
                subject="Asuna",
                roles=("identity_reference",),
            )
        )
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
    plan = build_first_frame_plan(
        _shot(subjects=[] if scene_only else None),
        reference_bundle=bundle,
        catalog=catalog,
        full_frame_anchor_asset_id=("bg_classroom" if scene_only else None),
    )
    approved = approve_first_frame_plan(plan)
    return approved, {item.asset_id: item.resolved_path for item in records}


def _manifest(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "run" / "first-frame-compose-run.json").read_text(encoding="utf-8")
    )


def _run(
    tmp_path: Path,
    executor: RecordingExecutor,
    *,
    plan=None,
    assets=None,
    approved: set[str] | None = None,
    auto_approve: bool = False,
    max_new_model_stages: int = 1,
):
    if plan is None or assets is None:
        plan, assets = _plan_and_assets(tmp_path)
    return run_first_frame_composition(
        run_dir=tmp_path / "run",
        run_id="shot-001-first-frame",
        plan=plan,
        asset_map=assets,
        adapter=QwenImage30Adapter(size="512*512"),
        executor=executor,
        approved_stage_ids=approved or set(),
        auto_approve=auto_approve,
        max_new_model_stages=max_new_model_stages,
    )


def test_manual_staged_flow_uses_canvas_then_component_reference(
    tmp_path: Path,
) -> None:
    plan, assets = _plan_and_assets(tmp_path)
    executor = RecordingExecutor()

    first = _run(tmp_path, executor, plan=plan, assets=assets)
    assert first.status == "awaiting_review"
    assert first.completed_stage_ids == ("stage_001",)
    assert len(executor.calls) == 1
    assert len(executor.calls[0]["request"]["conditions"]) == 1
    manifest = _manifest(tmp_path)
    assert manifest["content_plan_id"] is None
    assert manifest["prepared_component_plan_id"] is None
    assert manifest["prepared_component_asset_ids"] == []

    second = _run(
        tmp_path,
        executor,
        plan=plan,
        assets=assets,
        approved={"stage_001"},
    )
    assert second.status == "awaiting_review"
    assert second.completed_stage_ids == ("stage_001", "stage_002")
    assert len(executor.calls) == 2
    second_request = executor.calls[1]["request"]
    assert second_request["adapter_id"].endswith("fuse_component")
    assert len(second_request["conditions"]) == 2
    assert "Image 1 is the current approved frame canvas" in second_request["prompt"]
    assert "afternoon classroom" not in second_request["prompt"]

    final = _run(
        tmp_path,
        executor,
        plan=plan,
        assets=assets,
        approved={"stage_002"},
    )
    assert final.status == "completed"
    assert final.final_frame_path is not None
    assert final.final_frame_path.is_file()
    assert len(executor.calls) == 2


def test_exact_scene_anchor_completes_without_model_call(tmp_path: Path) -> None:
    plan, assets = _plan_and_assets(tmp_path, scene_only=True)
    executor = RecordingExecutor()

    result = _run(tmp_path, executor, plan=plan, assets=assets)

    assert result.status == "completed"
    assert result.final_frame_path is not None
    assert executor.calls == []
    manifest = _manifest(tmp_path)
    assert manifest["model_call_count"] == 0
    assert manifest["stages"][0]["approved"] is True


def test_failed_component_stage_resumes_without_rebuilding_base(tmp_path: Path) -> None:
    plan, assets = _plan_and_assets(tmp_path)
    failing = RecordingExecutor(fail_on_call=2)
    with pytest.raises(RenderError, match="injected"):
        _run(
            tmp_path,
            failing,
            plan=plan,
            assets=assets,
            auto_approve=True,
            max_new_model_stages=2,
        )
    failed = _manifest(tmp_path)
    assert [item["status"] for item in failed["stages"]] == [
        "completed",
        "failed",
    ]
    failed_request = failed["stages"][1]["request"]
    assert failed_request["provider"]["http_status"] == 403
    assert failed_request["provider"]["provider_code"] == "DataInspectionFailed"

    recovery = RecordingExecutor(colour_offset=2)
    result = _run(
        tmp_path,
        recovery,
        plan=plan,
        assets=assets,
        auto_approve=True,
        max_new_model_stages=2,
    )
    assert result.status == "completed"
    assert len(recovery.calls) == 1
    manifest = _manifest(tmp_path)
    assert manifest["stages"][0]["attempts"] == 1
    assert manifest["stages"][1]["attempts"] == 2


def test_noop_component_fusion_requires_explicit_visual_review(tmp_path: Path) -> None:
    plan, assets = _plan_and_assets(tmp_path)
    result = _run(
        tmp_path,
        NoOpComponentExecutor(),
        plan=plan,
        assets=assets,
        auto_approve=True,
        max_new_model_stages=2,
    )

    assert result.status == "awaiting_review"
    manifest = _manifest(tmp_path)
    suspect = manifest["stages"][1]
    assert suspect["status"] == "completed"
    assert suspect["approved"] is False
    assert suspect["requires_review"] is True
    assert suspect["qa"]["semantic_change_suspect"] is True
    assert suspect["qa"]["changed_fraction"] == 0.0


def test_resume_rejects_asset_drift_before_execution(tmp_path: Path) -> None:
    plan, assets = _plan_and_assets(tmp_path)
    executor = RecordingExecutor()
    _run(tmp_path, executor, plan=plan, assets=assets)
    Path(assets["char_asuna"]).write_bytes(_png((1, 2, 3)))

    with pytest.raises(InputValidationError, match="contract changed"):
        _run(
            tmp_path,
            executor,
            plan=plan,
            assets=assets,
            approved={"stage_001"},
        )
    assert len(executor.calls) == 1


def test_draft_plan_is_rejected_before_any_model_call(tmp_path: Path) -> None:
    approved, assets = _plan_and_assets(tmp_path)
    draft = approved.model_copy(update={"review_status": "draft"})
    executor = RecordingExecutor()

    with pytest.raises(InputValidationError, match="explicitly approved"):
        _run(tmp_path, executor, plan=draft, assets=assets)
    assert executor.calls == []


def test_deterministic_overlay_preserves_canvas_and_forces_quality_review(
    tmp_path: Path,
) -> None:
    background_path = tmp_path / "background.png"
    background_path.write_bytes(_png((20, 30, 40)))
    overlay_path = tmp_path / "steam.png"
    overlay_path.write_bytes(
        _rgba_overlay((64, 48), (16, 12, 48, 36), (220, 40, 60, 128))
    )
    background = _record(
        "bg",
        "background",
        background_path,
        subject="afternoon classroom",
        roles=("scene_reference",),
    )
    steam = _record(
        "steam",
        "foreground",
        overlay_path,
        subject="steam",
    )
    records = [background, steam]
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
    policy = {
        "schema_version": "first-frame-assembly-policy-v1",
        "shot_id": "shot_001",
        "reference_authorities": [
            {
                "asset_id": "steam",
                "authority": "deterministic_overlay",
                "reason": "apply the approved occlusion without regeneration",
            }
        ],
        "require_production_quality_review": True,
    }
    plan = approve_first_frame_plan(
        build_first_frame_plan(
            _shot(subjects=[]),
            reference_bundle=bundle,
            catalog=ImageAssetCatalog.build(records),
            full_frame_anchor_asset_id="bg",
            assembly_policy=policy,
        )
    )
    assert [item.operation for item in plan.stages] == [
        "adopt_anchor",
        "composite_overlay",
    ]

    executor = RecordingExecutor()
    assets = {item.asset_id: item.resolved_path for item in records}
    result = _run(
        tmp_path,
        executor,
        plan=plan,
        assets=assets,
        auto_approve=True,
    )

    assert result.status == "awaiting_review"
    assert result.next_stage_id == "stage_002"
    assert executor.calls == []
    manifest = _manifest(tmp_path)
    assert manifest["model_call_count"] == 0
    assert manifest["stages"][1]["requires_review"] is True
    assert manifest["stages"][1]["qa"]["deterministic_alpha_composite"] is True
    review_output = tmp_path / "run" / manifest["stages"][1]["output"]["path"]
    with Image.open(review_output) as output:
        assert output.getpixel((0, 0)) == (20, 30, 40)
        blended = output.getpixel((20, 20))
        assert blended[0] > 100
        assert blended[1] < 40

    completed = _run(
        tmp_path,
        executor,
        plan=plan,
        assets=assets,
        approved={"stage_002"},
        auto_approve=True,
    )
    assert completed.status == "completed"
    assert executor.calls == []
