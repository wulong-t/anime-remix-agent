"""Phase 3 smoke test: one compose keyframe end-to-end (stub executor).

This test proves the frozen contracts connect without any real model:
every artifact, model request and PortBinding enters the Execution Ledger
and ``Resolver.current(final_keyframe)`` finds the adopted artifact from
the Ledger alone.  Real Qwen execution is exercised by the separate
experiments/phase3 script when the remote server is enabled.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from anime_remix.services.execution.adapter import (
    QwenImageEditAdapter,
    StubImageExecutor,
)
from anime_remix.services.execution.artifact_store import ArtifactStore
from anime_remix.services.execution.ledger_writer import (
    read_complete_records,
)
from anime_remix.services.execution.orchestrator import (
    PORT_CHARACTER_CANDIDATE,
    PORT_CHARACTER_GEOMETRY,
    PORT_CHARACTER_LAYER,
    PORT_COMPOSITE_IMAGE,
    PORT_COMPOSITE_MASK,
    PORT_FINAL_KEYFRAME,
    PORT_INPAINT_MASK,
    PORT_LAYOUT_INTENT,
    PORT_LAYOUT_PLAN,
    run_compose_keyframe,
)
from anime_remix.services.execution.resolver import Resolver

SCENE_REF = "asset://anime-remix/scene/classroom_01@v1"
IDENTITY_REF = "asset://anime-remix/character/asuna@v1"
POSE_REF = "asset://anime-remix/pose/asuna_sitting@v1"

ALL_PORTS = [
    PORT_CHARACTER_CANDIDATE,
    PORT_CHARACTER_LAYER,
    PORT_CHARACTER_GEOMETRY,
    PORT_LAYOUT_INTENT,
    PORT_LAYOUT_PLAN,
    PORT_COMPOSITE_MASK,
    PORT_INPAINT_MASK,
    PORT_COMPOSITE_IMAGE,
    PORT_FINAL_KEYFRAME,
]


def _shot_spec() -> dict:
    return {
        "schema_version": "shot-spec-v1",
        "shot_id": "shot_003",
        "scene_id": "classroom_01",
        "order": 1,
        "narrative_purpose": "Asuna sits down at the desk.",
        "duration_seconds": 4.0,
        "camera_motion": "static",
        "emotion_arc": "neutral to sad",
        "start_state": "standing beside desk",
        "action_beats": [
            {"time_seconds": 0.0, "description": "standing beside desk"},
            {"time_seconds": 2.0, "description": "sits down at desk"},
        ],
        "end_state": "sitting at desk, sad",
        "generation_mode": "compose",
        "locks": {
            "character": {
                "identity": True,
                "hairstyle": True,
                "costume_variant": "school_uniform",
            },
            "scene": {"scene_id": "classroom_01", "time_of_day": "afternoon"},
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
                    },
                    {
                        "requirement_id": "character.costume.school_uniform",
                        "constraint": "school uniform",
                        "priority": "required",
                    },
                    {
                        "requirement_id": "pose.sitting",
                        "constraint": "sitting",
                        "priority": "preferred",
                    },
                ],
            },
            "scene": {"scene_id": "classroom_01", "requirements": []},
            "composition": {
                "shot_scale": "medium",
                "camera_position": "front_left",
            },
            "spatial_relations": [
                {"subject": "asuna", "relation": "sitting_at", "object": "desk"},
                {
                    "subject": "desk",
                    "relation": "occludes",
                    "object": "asuna.lower_body",
                },
            ],
        },
    }


def _keyframe_plan() -> dict:
    return {
        "schema_version": "keyframe-plan-v1",
        "shot_id": "shot_003",
        "shot_duration_seconds": 4.0,
        "keyframes": [
            {
                "keyframe_id": "kf_002",
                "shot_id": "shot_003",
                "order": 1,
                "time_seconds": 2.0,
                "position": 0.5,
                "visual_description": (
                    "Asuna is seated at the desk with both eyes fully closed "
                    "and her right fingertips touching her right temple"
                ),
                "subject_pose": (
                    "seated at the desk with her right hand at her temple"
                ),
                "expression": "sad and thoughtful, both eyes fully closed",
                "gaze": "eyes fully closed",
                "composition": "medium shot, front-left",
                "camera": "front_left",
                "background_state": "classroom, afternoon",
                "foreground_state": "desk occludes lower body",
                "prop_state": "desk",
                "required_assets": [
                    {
                        "asset_id": "asuna_001",
                        "asset_type": "character",
                        "locked_attributes": ["identity", "costume"],
                    }
                ],
                "motion_from_previous": (
                    "has sat down and raised her right hand to her temple"
                ),
            }
        ],
    }


def _reference_package() -> dict:
    return {
        "schema_version": "reference-package-v1",
        "package_id": "refpkg_003",
        "shot_id": "shot_003",
        "generation_mode": "compose",
        "requirements": [
            {
                "requirement_id": "character.identity",
                "constraint": "asuna",
                "priority": "required",
            },
            {
                "requirement_id": "character.costume.school_uniform",
                "constraint": "school uniform",
                "priority": "required",
            },
            {
                "requirement_id": "pose.sitting",
                "constraint": "sitting",
                "priority": "preferred",
            },
        ],
        "conditions": [
            {
                "condition_id": "cond_001",
                "role": "identity",
                "kind": "image",
                "payload_ref": IDENTITY_REF,
                "satisfied_constraints": [
                    "character.identity",
                    "character.costume.school_uniform",
                ],
                "scores": {"identity": 0.96},
                "provenance": {"source_asset_id": "asuna_001"},
            },
            {
                "condition_id": "cond_002",
                "role": "pose",
                "kind": "image",
                "payload_ref": POSE_REF,
                "satisfied_constraints": ["pose.sitting"],
                "scores": {"pose": 0.83},
                "provenance": {"source_asset_id": "asuna_sitting_001"},
            },
        ],
        "candidate_sets": [
            {
                "requirement_id": "character.identity",
                "scope": "shot",
                "candidates": ["cond_001"],
            },
            {
                "requirement_id": "pose.sitting",
                "scope": "shot",
                "candidates": ["cond_002"],
            },
        ],
    }


def _make_images(tmp_path: Path) -> dict[str, Path]:
    scene = Image.new("RGB", (320, 180), (200, 190, 170))
    ImageDraw.Draw(scene).rectangle((115, 99, 291, 158), fill=(120, 100, 80))
    scene_path = tmp_path / "scene.png"
    scene.save(scene_path)

    identity = Image.new("RGB", (256, 256), (0, 0, 220))
    ImageDraw.Draw(identity).rounded_rectangle(
        (72, 48, 184, 208), radius=24, fill=(220, 60, 60)
    )
    identity_path = tmp_path / "identity.png"
    identity.save(identity_path)

    pose = Image.new("RGB", (256, 256), (0, 160, 0))
    ImageDraw.Draw(pose).rounded_rectangle(
        (80, 120, 176, 240), radius=16, fill=(240, 200, 40)
    )
    pose_path = tmp_path / "pose.png"
    pose.save(pose_path)
    return {
        SCENE_REF: scene_path,
        IDENTITY_REF: identity_path,
        POSE_REF: pose_path,
    }


def _scene_geometry() -> dict:
    return {
        "anchors": {"desk_left_seat": [0.48, 0.78]},
        "regions": {
            "desk_front": {
                "polygon": [
                    [0.40, 0.55],
                    [0.86, 0.55],
                    [0.91, 0.88],
                    [0.36, 0.88],
                ]
            }
        },
        "occluders": ["desk_front"],
    }


def test_compose_keyframe_full_chain(tmp_path) -> None:
    assets = _make_images(tmp_path)
    run_dir = tmp_path / "runs" / "run_phase3"
    result = run_compose_keyframe(
        run_dir=run_dir,
        run_id="run_phase3",
        shot_spec=_shot_spec(),
        keyframe_plan=_keyframe_plan(),
        reference_package=_reference_package(),
        scene_image_path=assets[SCENE_REF],
        scene_geometry=_scene_geometry(),
        scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
        asset_map=assets,
        adapter=QwenImageEditAdapter(seed=0),
        executor=StubImageExecutor(),
        canvas=(320, 180),
    )

    assert result.final_keyframe_ref.startswith("artifact://run_phase3/art_")
    resolver = Resolver(result.ledger_path)
    plan_id = "shot_003-kf_002-plan-v1"
    for port_name in ALL_PORTS:
        port_ref = f"plan://{plan_id}/ports/{port_name}"
        assert resolver.current(port_ref) == result.ports[port_name]
    assert resolver.current(f"plan://{plan_id}/ports/{PORT_FINAL_KEYFRAME}") == (
        result.final_keyframe_ref
    )

    records = read_complete_records(result.ledger_path)
    by_type: dict[str, int] = {}
    for record in records:
        by_type[record.record_type] = by_type.get(record.record_type, 0) + 1
    assert by_type["run_started"] == 1
    assert by_type["run_finished"] == 1
    assert by_type["plan_instantiated"] == 1
    assert by_type["node_run_started"] == 8
    assert by_type["node_run_finished"] == 8
    assert by_type["render_intent_created"] == 2
    assert by_type["model_render_request_created"] == 2
    assert by_type["render_attempt_started"] == 2
    assert by_type["render_attempt_finished"] == 2
    assert by_type["artifact_registered"] == 9
    assert by_type["port_bound"] == 9
    assert by_type["validation_result"] == 2

    final_artifact = next(
        r
        for r in records
        if r.record_type == "artifact_registered"
        and r.payload.artifact_kind == "final_keyframe"
    )
    store = ArtifactStore(run_dir)
    blob_path = store.blob_path(final_artifact.payload.blob_ref)
    assert blob_path.exists()
    final_image = Image.open(blob_path)
    assert final_image.size == (320, 180)

    local_inpaint_finished = [
        r
        for r in records
        if r.record_type == "render_attempt_finished"
        and r.payload.output_artifact_ref == result.final_keyframe_ref
    ]
    assert len(local_inpaint_finished) == 1
    requests = [
        r
        for r in records
        if r.record_type == "model_render_request_created"
    ]
    assert any("masked region" in r.payload.prompt for r in requests)
    assert any(
        "image 1 (WHO)" in r.payload.prompt
        and [(c.slot, c.condition_ref) for c in r.payload.conditions]
        == [
            (1, "cond_001"),
            (2, "cond_002"),
        ]
        for r in requests
    )
