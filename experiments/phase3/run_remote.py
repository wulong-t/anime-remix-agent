"""Phase 3 remote runner: one compose keyframe with the real Qwen model.

Local boundary: this script compiles the fixed ShotSpec / KeyframePlan /
ReferencePackage, runs the frozen compose chain with the Qwen adapter and
the RemoteQwenExecutor.  Until the GPU server is enabled, execution writes
the exact request manifests under ``<run_dir>/requests`` and stops with an
EnvironmentCapabilityError - that is the AGENTS.md "notify the user only
after local prep is ready" boundary.

Usage (after the user enables the server and provides assets)::

    python experiments/phase3/run_remote.py \
      --identity path/to/asuna_identity.png \
      --pose path/to/asuna_sitting.png \
      --scene path/to/classroom.png \
      --run-dir runs/phase3_real --seed 0
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from anime_remix.errors import EnvironmentCapabilityError
from anime_remix.services.execution.adapter import (
    QwenImageEditAdapter,
    RemoteQwenExecutor,
)
from anime_remix.services.execution.orchestrator import run_compose_keyframe


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
                    "Asuna is seated at the classroom desk with both eyes "
                    "fully closed and her right elbow bent; her right "
                    "fingertips touch her right temple while she looks sad "
                    "and thoughtful"
                ),
                "subject_pose": (
                    "seated upright at the desk, right elbow bent, right "
                    "hand raised with fingertips touching the right temple"
                ),
                "expression": "sad and thoughtful, with both eyes fully closed",
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


def _reference_package(identity_ref: str, pose_ref: str) -> dict:
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
                "payload_ref": identity_ref,
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
                "payload_ref": pose_ref,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--geometry-json", type=Path, default=None)
    parser.add_argument("--run-dir", default="runs/phase3_real", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    scene_ref = "asset://anime-remix/scene/classroom_01@v1"
    identity_ref = "asset://anime-remix/character/asuna@v1"
    pose_ref = "asset://anime-remix/pose/asuna_sitting@v1"
    scene_geometry = (
        json.loads(args.geometry_json.read_text(encoding="utf-8"))
        if args.geometry_json is not None
        else _scene_geometry()
    )
    run_id = f"phase3_real_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    result = run_compose_keyframe(
        run_dir=args.run_dir,
        run_id=run_id,
        shot_spec=_shot_spec(),
        keyframe_plan=_keyframe_plan(),
        reference_package=_reference_package(identity_ref, pose_ref),
        scene_image_path=args.scene,
        scene_geometry=scene_geometry,
        scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
        asset_map={
            scene_ref: args.scene,
            identity_ref: args.identity,
            pose_ref: args.pose,
        },
        adapter=QwenImageEditAdapter(seed=args.seed, steps=args.steps),
        executor=RemoteQwenExecutor(args.run_dir),
        canvas=(1280, 720),
    )
    print(result)


if __name__ == "__main__":
    try:
        main()
    except EnvironmentCapabilityError as exc:
        print(f"REMOTE NOT READY: {exc}")
        print(
            "Local request manifests are ready. Enable the GPU server, then "
            "re-run this script (execution continues where the manifests "
            "end)."
        )
