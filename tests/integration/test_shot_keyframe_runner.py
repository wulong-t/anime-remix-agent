"""Shot-level first/last keyframe execution and recovery tests."""

from __future__ import annotations

import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from anime_remix.errors import InputValidationError, RenderError
from anime_remix.services.execution.adapter import QwenImage30Adapter
from anime_remix.services.execution.shot_keyframe_runner import (
    run_shot_keyframes,
)

IDENTITY_REF = "asset://anime-remix/character/asuna@v1"
POSE_REF = "asset://anime-remix/pose/asuna_sitting@v1"
SCENE_REF = "asset://anime-remix/scene/classroom@v1"


def _png_bytes(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 48), colour).save(output, format="PNG")
    return output.getvalue()


class RecordingExecutor:
    provider = "offline-recording"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[dict] = []

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
            raise RenderError("injected frame failure")
        level = 40 * len(self.calls)
        return _png_bytes((level, 80, 120))


def _shot_spec() -> dict:
    return {
        "schema_version": "shot-spec-v1",
        "shot_id": "shot_003",
        "scene_id": "classroom_01",
        "order": 1,
        "narrative_purpose": "Asuna changes from alert to thoughtful.",
        "duration_seconds": 4.0,
        "camera_motion": "static",
        "emotion_arc": "alert to sad",
        "start_state": "sitting upright with eyes open",
        "action_beats": [
            {"time_seconds": 0.0, "description": "looks forward"},
            {"time_seconds": 2.0, "description": "raises right hand"},
            {"time_seconds": 4.0, "description": "closes eyes"},
        ],
        "end_state": "eyes closed with right hand at temple",
        "generation_mode": "compose",
        "locks": {
            "character": {
                "identity": True,
                "hairstyle": True,
                "costume_variant": "school_uniform",
            },
            "scene": {
                "scene_id": "classroom_01",
                "time_of_day": "afternoon",
            },
            "style": {"visual_style_id": "source_anime"},
        },
        "compose": {
            "character": {
                "character_id": "asuna",
                "requirements": [
                    {
                        "requirement_id": "character.identity",
                        "constraint": "asuna",
                        "priority": "required",
                    }
                ],
            },
            "scene": {"scene_id": "classroom_01", "requirements": []},
            "composition": {
                "shot_scale": "medium",
                "camera_position": "front_left",
            },
            "spatial_relations": [],
        },
    }


def _frame(
    keyframe_id: str,
    order: int,
    time_seconds: float,
    position: float,
    state: str,
) -> dict:
    return {
        "keyframe_id": keyframe_id,
        "shot_id": "shot_003",
        "order": order,
        "time_seconds": time_seconds,
        "position": position,
        "visual_description": state,
        "subject_pose": state,
        "expression": "thoughtful",
        "gaze": "forward" if order == 1 else "eyes closed",
        "composition": "medium shot",
        "camera": "front_left",
        "background_state": "classroom, afternoon",
        "foreground_state": "desk",
        "prop_state": "desk",
        "required_assets": [
            {
                "asset_id": "asuna_001",
                "asset_type": "character",
                "locked_attributes": ["identity", "costume"],
            }
        ],
        "motion_from_previous": state,
    }


def _keyframe_plan() -> dict:
    return {
        "schema_version": "keyframe-plan-v1",
        "shot_id": "shot_003",
        "shot_duration_seconds": 4.0,
        "keyframes": [
            _frame("kf_001", 1, 0.0, 0.0, "sitting with eyes open"),
            _frame("kf_002", 2, 2.0, 0.5, "raising the right hand"),
            _frame(
                "kf_003",
                3,
                4.0,
                1.0,
                "eyes closed, right fingertips at right temple",
            ),
        ],
    }


def _reference_package(package_id: str = "refpkg_003") -> dict:
    return {
        "schema_version": "reference-package-v1",
        "package_id": package_id,
        "shot_id": "shot_003",
        "generation_mode": "compose",
        "requirements": [
            {
                "requirement_id": "character.identity",
                "constraint": "asuna",
                "priority": "required",
            },
            {
                "requirement_id": "pose.target",
                "constraint": "target pose",
                "priority": "preferred",
            },
        ],
        "conditions": [
            {
                "condition_id": "cond_identity",
                "role": "identity",
                "kind": "image",
                "payload_ref": IDENTITY_REF,
                "satisfied_constraints": ["character.identity"],
                "scores": {"identity": 1.0},
                "provenance": {"source_asset_id": "asuna_001"},
            },
            {
                "condition_id": "cond_pose",
                "role": "pose",
                "kind": "image",
                "payload_ref": POSE_REF,
                "satisfied_constraints": ["pose.target"],
                "scores": {"pose": 1.0},
                "provenance": {"source_asset_id": "pose_001"},
            },
        ],
        "candidate_sets": [
            {
                "requirement_id": "character.identity",
                "scope": "shot",
                "candidates": ["cond_identity"],
            },
            {
                "requirement_id": "pose.target",
                "scope": "shot",
                "candidates": ["cond_pose"],
            },
        ],
    }


def _inputs(tmp_path: Path) -> dict[str, Path]:
    identity = tmp_path / "identity.png"
    identity.write_bytes(_png_bytes((10, 20, 30)))
    return {IDENTITY_REF: identity}


def _run(
    tmp_path: Path,
    executor: RecordingExecutor,
    *,
    approved: set[str] | None = None,
    auto_approve: bool = False,
    max_new_frames: int = 1,
    plan: dict | None = None,
    packages: dict | None = None,
    extra_assets: dict[str, Path] | None = None,
):
    assets = _inputs(tmp_path)
    assets.update(extra_assets or {})
    return run_shot_keyframes(
        run_dir=tmp_path / "run",
        run_id="shot-003-endpoints",
        shot_spec=_shot_spec(),
        keyframe_plan=plan or _keyframe_plan(),
        reference_packages=packages or _reference_package(),
        scene_geometry={"anchors": {}, "regions": {}, "occluders": []},
        scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
        asset_map=assets,
        adapter=QwenImage30Adapter(),
        executor=executor,
        canvas=(64, 48),
        approved_keyframe_ids=approved or set(),
        auto_approve=auto_approve,
        max_new_frames=max_new_frames,
    )


def _manifest(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "run" / "shot-keyframe-run.json").read_text(
            encoding="utf-8"
        )
    )


def test_manual_flow_generates_only_first_and_last(tmp_path: Path) -> None:
    executor = RecordingExecutor()

    first = _run(tmp_path, executor)
    assert first.status == "awaiting_review"
    assert first.generated_keyframe_ids == ("kf_001",)
    assert first.next_keyframe_id == "kf_001"
    assert len(executor.calls) == 1

    second = _run(tmp_path, executor, approved={"kf_001"})
    assert second.status == "awaiting_review"
    assert second.generated_keyframe_ids == ("kf_001", "kf_003")
    assert second.next_keyframe_id == "kf_003"
    assert len(executor.calls) == 2

    final = _run(tmp_path, executor, approved={"kf_003"})
    assert final.status == "completed"
    assert final.next_keyframe_id is None
    assert len(executor.calls) == 2

    manifest = _manifest(tmp_path)
    assert manifest["source_keyframe_count"] == 3
    assert manifest["selected_keyframe_count"] == 2
    assert [frame["keyframe_id"] for frame in manifest["frames"]] == [
        "kf_001",
        "kf_003",
    ]
    assert not (tmp_path / "run" / "frames" / "002_kf_002").exists()
    assert all(frame["approved"] for frame in manifest["frames"])
    assert manifest["frames"][0]["selected_reference_roles"] == ["identity"]
    assert manifest["frames"][1]["selected_reference_roles"] == [
        "identity",
        "source_frame",
    ]

    first_request = executor.calls[0]["request"]
    assert [item["condition_ref"] for item in first_request["conditions"]] == [
        "cond_identity"
    ]
    assert set(executor.calls[0]["inputs"]) == {"cond_identity"}
    last_request = executor.calls[1]["request"]
    assert last_request["adapter_id"] == (
        "qwen-image-30-adapter-v6-continuity-action-delta"
    )
    assert [item["condition_ref"] for item in last_request["conditions"]] == [
        "cond_identity",
        "cond_previous_keyframe",
    ]
    assert set(executor.calls[1]["inputs"]) == {
        "cond_identity",
        "cond_previous_keyframe",
    }
    previous_bytes = executor.calls[1]["inputs"]["cond_previous_keyframe"]
    first_output = tmp_path / "run" / manifest["frames"][0]["output"]["named_path"]
    assert hashlib.sha256(previous_bytes).hexdigest() == hashlib.sha256(
        first_output.read_bytes()
    ).hexdigest()


def test_auto_approve_respects_default_one_frame_limit(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    first = _run(tmp_path, executor, auto_approve=True)
    assert first.status == "paused_limit"
    assert first.generated_keyframe_ids == ("kf_001",)
    assert first.next_keyframe_id == "kf_003"

    final = _run(tmp_path, executor, auto_approve=True)
    assert final.status == "completed"
    assert final.generated_keyframe_ids == ("kf_001", "kf_003")
    assert len(executor.calls) == 2


def test_failed_last_frame_resumes_only_that_frame(tmp_path: Path) -> None:
    failing = RecordingExecutor(fail_on_call=2)
    with pytest.raises(RenderError, match="injected frame failure"):
        _run(
            tmp_path,
            failing,
            auto_approve=True,
            max_new_frames=2,
        )
    failed_manifest = _manifest(tmp_path)
    assert [frame["status"] for frame in failed_manifest["frames"]] == [
        "completed",
        "failed",
    ]
    assert failed_manifest["frames"][1]["attempts"] == 1

    recovery = RecordingExecutor()
    result = _run(
        tmp_path,
        recovery,
        auto_approve=True,
        max_new_frames=2,
    )
    assert result.status == "completed"
    assert len(recovery.calls) == 1
    manifest = _manifest(tmp_path)
    assert manifest["frames"][0]["attempts"] == 1
    assert manifest["frames"][1]["attempts"] == 2
    assert (
        tmp_path
        / "run"
        / "frames"
        / "003_kf_003"
        / "attempt_002"
        / "execution-ledger.jsonl"
    ).exists()


def test_resume_rejects_contract_drift_before_execution(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    _run(tmp_path, executor)
    changed = _keyframe_plan()
    changed["keyframes"][-1]["visual_description"] = "different ending"

    with pytest.raises(InputValidationError, match="contract changed"):
        _run(tmp_path, executor, plan=changed, approved={"kf_001"})
    assert len(executor.calls) == 1


def test_mapping_may_include_unused_interior_package(tmp_path: Path) -> None:
    packages = {
        "kf_001": _reference_package("refpkg_first"),
        "kf_002": _reference_package("refpkg_middle"),
        "kf_003": _reference_package("refpkg_last"),
    }
    executor = RecordingExecutor()
    result = _run(tmp_path, executor, packages=packages)
    assert result.generated_keyframe_ids == ("kf_001",)


def test_manual_mode_rejects_approval_before_output_exists(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    with pytest.raises(InputValidationError, match="before their outputs exist"):
        _run(tmp_path, executor, approved={"kf_001"})
    assert executor.calls == []


def test_visual_how_is_rejected_before_any_frame_runs(tmp_path: Path) -> None:
    assets = _inputs(tmp_path)
    executor = RecordingExecutor()
    with pytest.raises(InputValidationError, match="reserves Image 2"):
        run_shot_keyframes(
            run_dir=tmp_path / "run",
            run_id="shot-003-endpoints",
            shot_spec=_shot_spec(),
            keyframe_plan=_keyframe_plan(),
            reference_packages=_reference_package(),
            scene_geometry={"anchors": {}, "regions": {}, "occluders": []},
            scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
            asset_map=assets,
            adapter=QwenImage30Adapter(visual_how=True),
            executor=executor,
            canvas=(64, 48),
        )
    assert executor.calls == []
    assert not (tmp_path / "run" / "shot-keyframe-run.json").exists()


def test_first_frame_uses_full_frame_reference_without_text_restatement(
    tmp_path: Path,
) -> None:
    package = _reference_package()
    package["requirements"].append(
        {
            "requirement_id": "scene.reference",
            "constraint": "use the approved classroom frame",
            "priority": "required",
        }
    )
    package["conditions"].append(
        {
            "condition_id": "cond_scene",
            "role": "scene",
            "kind": "image",
            "payload_ref": SCENE_REF,
            "satisfied_constraints": ["scene.reference"],
            "scores": {"scene": 1.0},
            "provenance": {"source_asset_id": "classroom_001"},
        }
    )
    package["candidate_sets"].append(
        {
            "requirement_id": "scene.reference",
            "scope": "shot",
            "candidates": ["cond_scene"],
        }
    )
    scene_reference = tmp_path / "frame-reference.png"
    scene_reference.write_bytes(_png_bytes((90, 60, 30)))
    executor = RecordingExecutor()

    result = _run(
        tmp_path,
        executor,
        packages=package,
        extra_assets={SCENE_REF: scene_reference},
    )
    assert result.generated_keyframe_ids == ("kf_001",)
    request = executor.calls[0]["request"]
    assert [item["condition_ref"] for item in request["conditions"]] == [
        "cond_identity",
        "cond_scene",
    ]
    assert set(executor.calls[0]["inputs"]) == {
        "cond_identity",
        "cond_scene",
    }
    assert "Image 2 is the full-frame visual reference" in request["prompt"]
    assert "No selected reference covers" not in request["prompt"]
    first_frame = _manifest(tmp_path)["frames"][0]
    assert first_frame["selected_reference_roles"] == ["identity", "scene"]
    assert first_frame["text_fallback_fields"] == []


def test_last_frame_text_only_fills_visual_fields_that_changed(
    tmp_path: Path,
) -> None:
    plan = _keyframe_plan()
    plan["keyframes"][-1]["background_state"] = "classroom after sunset"
    executor = RecordingExecutor()
    result = _run(
        tmp_path,
        executor,
        plan=plan,
        auto_approve=True,
        max_new_frames=2,
    )
    assert result.status == "completed"
    last_prompt = executor.calls[1]["request"]["prompt"]
    assert "Background: classroom after sunset" in last_prompt
    assert "Composition:" not in last_prompt
    assert "Camera:" not in last_prompt
    assert _manifest(tmp_path)["frames"][1]["text_fallback_fields"] == [
        "background_state"
    ]
