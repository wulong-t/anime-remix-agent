"""Unit tests for the shot-spec-v1 research contract."""

from __future__ import annotations

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.shot_spec import parse_shot_spec


def _base_compose() -> dict:
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


def _base_near_match() -> dict:
    document = _base_compose()
    document["generation_mode"] = "near_match"
    document.pop("compose")
    document["near_match"] = {
        "source_requirements": {
            "character_id": "asuna",
            "scene_id": "classroom_01",
            "shot_scale": "medium",
        },
        "preserve": ["composition", "camera", "pose"],
        "modify": {"expression": "sad"},
    }
    return document


def test_valid_compose_document_parses() -> None:
    document = parse_shot_spec(_base_compose())
    assert document.schema_version == "shot-spec-v1"
    assert document.generation_mode == "compose"
    assert document.compose is not None
    assert len(document.compose.character.requirements) == 3


def test_valid_near_match_document_parses() -> None:
    document = parse_shot_spec(_base_near_match())
    assert document.generation_mode == "near_match"
    assert document.near_match is not None
    assert document.near_match.modify == {"expression": "sad"}


def test_compose_mode_with_both_blocks_fails() -> None:
    document = _base_compose()
    document["near_match"] = {
        "source_requirements": {
            "character_id": "asuna",
            "scene_id": "classroom_01",
            "shot_scale": "medium",
        },
        "preserve": ["composition"],
        "modify": {"expression": "sad"},
    }
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_compose_mode_missing_compose_block_fails() -> None:
    document = _base_compose()
    document.pop("compose")
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_scene_id_mismatch_with_locks_fails() -> None:
    document = _base_compose()
    document["locks"]["scene"]["scene_id"] = "cafeteria_01"
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_empty_character_requirements_fail() -> None:
    document = _base_compose()
    document["compose"]["character"]["requirements"] = []
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_beat_outside_duration_fails() -> None:
    document = _base_compose()
    document["action_beats"][1]["time_seconds"] = 5.0
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_extra_field_rejected() -> None:
    document = _base_compose()
    document["unexpected"] = True
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)


def test_invalid_priority_rejected() -> None:
    document = _base_compose()
    document["compose"]["character"]["requirements"][0]["priority"] = "must"
    with pytest.raises(InputValidationError):
        parse_shot_spec(document)
