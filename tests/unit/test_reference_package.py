"""Unit tests for the reference-package-v1 research contract."""

from __future__ import annotations

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.reference_package import (
    parse_reference_package,
)


def _base_package() -> dict:
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
                "payload_ref": "asset://anime-remix/character/asuna@v1",
                "satisfied_constraints": ["character.identity"],
                "scores": {"identity": 0.96},
                "provenance": {"source_asset_id": "asuna_001"},
            },
            {
                "condition_id": "cond_002",
                "role": "pose",
                "kind": "image",
                "payload_ref": "asset://anime-remix/pose/asuna_sitting@v1",
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
                "scope": "keyframe",
                "keyframe_id": "kf_002",
                "candidates": ["cond_002"],
            },
        ],
    }


def test_valid_compose_package_parses() -> None:
    package = parse_reference_package(_base_package())
    assert package.schema_version == "reference-package-v1"
    assert len(package.conditions) == 2
    assert len(package.candidate_sets) == 2


def test_candidate_set_unknown_condition_fails() -> None:
    document = _base_package()
    document["candidate_sets"][0]["candidates"] = ["cond_999"]
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_candidate_set_unknown_requirement_fails() -> None:
    document = _base_package()
    document["candidate_sets"][0]["requirement_id"] = "pose.nonexistent"
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_score_out_of_range_fails() -> None:
    document = _base_package()
    document["conditions"][0]["scores"]["identity"] = 1.5
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_compose_without_identity_role_fails() -> None:
    document = _base_package()
    document["conditions"][0]["role"] = "pose"
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_near_match_without_source_frame_fails() -> None:
    document = _base_package()
    document["generation_mode"] = "near_match"
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_keyframe_scope_without_keyframe_id_fails() -> None:
    document = _base_package()
    document["candidate_sets"][1].pop("keyframe_id")
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_payload_ref_wrong_scope_fails() -> None:
    document = _base_package()
    document["conditions"][0]["payload_ref"] = "file://local/asuna.png"
    with pytest.raises(InputValidationError):
        parse_reference_package(document)


def test_duplicate_condition_ids_fail() -> None:
    document = _base_package()
    document["conditions"].append(dict(document["conditions"][0]))
    with pytest.raises(InputValidationError):
        parse_reference_package(document)
