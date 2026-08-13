"""Unit tests for the layout-plan-v1 research contract."""

from __future__ import annotations

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.layout_plan import parse_layout_plan


def _base_plan() -> dict:
    return {
        "schema_version": "layout-plan-v1",
        "plan_id": "layout_001",
        "shot_id": "shot_003",
        "keyframe_id": "kf_002",
        "refs": {
            "character_layer_ref": "artifact://run_001/art_000121",
            "character_geometry_ref": "artifact://run_001/art_000122",
            "scene_asset_ref": "asset://anime-remix/scene/classroom_01@v1",
            "scene_geometry_ref": "asset://anime-remix/geometry/classroom_01@v1",
            "layout_intent_ref": "artifact://run_001/art_000120",
            "keyframe_state_ref": "artifact://run_001/art_000119",
        },
        "canvas": {"width": 1280, "height": 720},
        "scene_crop": {"source_rect": [0.125, 0.0, 0.875, 1.0]},
        "layers": [
            {
                "layer_id": "scene",
                "source_ref": "asset://anime-remix/scene/classroom_01@v1",
                "transform": {
                    "scale": 1.0,
                    "translate": [0.0, 0.0],
                    "rotation_degrees": 0,
                },
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "z_index": 1,
            },
            {
                "layer_id": "asuna",
                "source_ref": "artifact://run_001/art_000121",
                "transform": {
                    "scale": 0.72,
                    "translate": [0.31, 0.08],
                    "rotation_degrees": 0,
                },
                "bbox": [0.31, 0.08, 0.63, 0.91],
                "z_index": 2,
            },
        ],
        "occlusions": [
            {
                "occluder_ref": "scene.desk_front",
                "target_layer": "asuna",
                "polygon": [
                    [0.40, 0.56],
                    [0.86, 0.56],
                    [0.91, 0.88],
                    [0.36, 0.88],
                ],
            }
        ],
        "contacts": [
            {
                "source_anchor": [0.52, 0.81],
                "target_anchor": [0.48, 0.71],
                "tolerance": 0.03,
            }
        ],
        "mask_generation": {
            "generator_version": "maskgen-v1",
            "parameters": {"edge_dilation_px": 12},
        },
        "policy_refs": {
            "layout_policy_ref": "policy://layout-v1",
            "inpaint_policy_ref": "policy://inpaint-v1",
        },
    }


def test_valid_plan_parses() -> None:
    plan = parse_layout_plan(_base_plan())
    assert plan.schema_version == "layout-plan-v1"
    assert plan.canvas.width == 1280
    assert len(plan.layers) == 2
    assert len(plan.occlusions) == 1


def test_bbox_outside_canvas_fails() -> None:
    document = _base_plan()
    document["layers"][1]["bbox"] = [0.31, 0.08, 1.4, 0.91]
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_degenerate_bbox_fails() -> None:
    document = _base_plan()
    document["layers"][1]["bbox"] = [0.63, 0.08, 0.31, 0.91]
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_occlusion_unknown_target_layer_fails() -> None:
    document = _base_plan()
    document["occlusions"][0]["target_layer"] = "nobody"
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_scene_crop_invalid_order_fails() -> None:
    document = _base_plan()
    document["scene_crop"]["source_rect"] = [0.875, 0.0, 0.125, 1.0]
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_rotation_non_zero_rejected_in_v1() -> None:
    document = _base_plan()
    document["layers"][1]["transform"]["rotation_degrees"] = 15.0
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_duplicate_z_index_fails() -> None:
    document = _base_plan()
    document["layers"][1]["z_index"] = 1
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_contact_anchor_out_of_range_fails() -> None:
    document = _base_plan()
    document["contacts"][0]["source_anchor"] = [1.2, 0.5]
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)


def test_empty_layers_fail() -> None:
    document = _base_plan()
    document["layers"] = []
    with pytest.raises(InputValidationError):
        parse_layout_plan(document)
