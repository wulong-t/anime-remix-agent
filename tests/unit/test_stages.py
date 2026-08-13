"""Unit tests for the deterministic Phase 3 stages."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.layout_plan import parse_layout_plan
from anime_remix.services.execution.stages.character_extract import (
    extract_character_layer,
)
from anime_remix.services.execution.stages.character_validate import (
    validate_character_layer,
)
from anime_remix.services.execution.stages.composite import (
    composite_keyframe,
)
from anime_remix.services.execution.stages.final_validate import (
    validate_final_keyframe,
)
from anime_remix.services.execution.stages.geometry_extract import (
    extract_character_geometry,
)
from anime_remix.services.execution.stages.layout import (
    map_through_crop,
    plan_layout,
)
from anime_remix.services.execution.stages.mask_generate import (
    generate_masks,
)


def _synthetic_candidate() -> Image.Image:
    image = Image.new("RGB", (256, 256), (0, 180, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((72, 48, 184, 208), fill=(220, 60, 60))
    return image


def _scene_image() -> Image.Image:
    image = Image.new("RGB", (320, 180), (200, 190, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((115, 99, 291, 158), fill=(120, 100, 80))
    return image


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


def _geometry() -> dict:
    layer = extract_character_layer(_synthetic_candidate())
    return extract_character_geometry(layer)


def _layout_plan() -> dict:
    return plan_layout(
        layout_intent={
            "subject_pose": "sitting",
            "camera_view": "front_left",
            "shot_scale": "medium",
            "relations": ["asuna sitting_at desk"],
        },
        character_geometry=_geometry(),
        scene_geometry=_scene_geometry(),
        scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
        canvas={"width": 320, "height": 180},
        plan_id="shot_003-kf_002-plan-v1",
        shot_id="shot_003",
        keyframe_id="kf_002",
        character_layer_ref="artifact://run_001/art_000001",
        character_geometry_ref="artifact://run_001/art_000002",
        scene_asset_ref="asset://anime-remix/scene/classroom_01@v1",
        scene_geometry_ref="asset://anime-remix/geometry/classroom_01@v1",
        layout_intent_ref="artifact://run_001/art_000003",
        keyframe_state_ref="asset://anime-remix/spec/plan-keyframes@v1",
    )


def test_extract_removes_background() -> None:
    layer = extract_character_layer(_synthetic_candidate())
    assert layer.mode == "RGBA"
    assert layer.getpixel((0, 0))[3] == 0
    assert layer.getpixel((128, 128))[3] == 255


def test_character_validate_ok() -> None:
    layer = extract_character_layer(_synthetic_candidate())
    valid, checks = validate_character_layer(
        layer, canvas_w=layer.width, canvas_h=layer.height
    )
    assert valid is True
    assert all(check["passed"] for check in checks)


def test_character_validate_empty_fails() -> None:
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    valid, _checks = validate_character_layer(
        empty, canvas_w=64, canvas_h=64
    )
    assert valid is False


def test_geometry_bbox_and_anchors() -> None:
    geometry = _geometry()
    assert geometry["source_size"] == [256, 256]
    assert geometry["alpha_bbox"] == pytest.approx(
        [72 / 256, 48 / 256, 185 / 256, 209 / 256]
    )
    assert geometry["anchors"]["bottom_center"] == pytest.approx(
        [128.5 / 256, 209 / 256]
    )


def test_plan_layout_is_deterministic_and_valid() -> None:
    plan = _layout_plan()
    parsed = parse_layout_plan(plan)
    assert parsed.canvas.width == 320
    character = next(
        layer for layer in plan["layers"] if layer["layer_id"] == "character"
    )
    assert character["transform"]["rotation_degrees"] == 0
    assert character["transform"]["translate"][0] >= 0
    assert character["transform"]["translate"][1] >= 0
    assert len(plan["occlusions"]) == 1
    assert plan["contacts"][0]["tolerance"] == 0.03


def test_map_through_crop() -> None:
    point = map_through_crop([0.45, 0.78], [0.125, 0.0, 0.875, 1.0])
    assert point[0] == pytest.approx((0.45 - 0.125) / 0.75)
    assert point[1] == pytest.approx(0.78)


def test_mask_generate_produces_bounded_masks() -> None:
    layer = extract_character_layer(_synthetic_candidate())
    plan = _layout_plan()
    composite_mask, inpaint_mask = generate_masks(
        character_layer=layer,
        layout_plan=plan,
        canvas_w=320,
        canvas_h=180,
    )
    assert composite_mask.size == (320, 180)
    assert inpaint_mask.size == (320, 180)
    area = inpaint_mask.point(lambda v: 1 if v > 0 else 0).histogram()[255]
    assert area / (320 * 180) <= 0.30


def test_mask_generate_rejects_oversized_mask() -> None:
    layer = Image.new("RGBA", (256, 256), (0, 0, 0, 255))
    plan = _layout_plan()
    with pytest.raises(InputValidationError):
        generate_masks(
            character_layer=layer,
            layout_plan=plan,
            canvas_w=320,
            canvas_h=180,
            max_mask_area_ratio=0.01,
        )


def test_composite_places_character_and_occludes() -> None:
    layer = extract_character_layer(_synthetic_candidate())
    plan = _layout_plan()
    scene = _scene_image()
    result = composite_keyframe(
        scene_image=scene, character_layer=layer, layout_plan=plan
    )
    assert result.size == (320, 180)
    polygon = plan["occlusions"][0]["polygon"]
    px, py = polygon[0]
    pixel = result.getpixel((int(px * 320), int(py * 180)))
    assert pixel == (120, 100, 80)


def test_final_validate() -> None:
    image = _scene_image()
    valid, _checks = validate_final_keyframe(
        image, canvas_w=320, canvas_h=180
    )
    assert valid is True
    invalid, _ = validate_final_keyframe(image, canvas_w=640, canvas_h=360)
    assert invalid is False
    blank = Image.new("RGB", (320, 180), (0, 0, 0))
    blank_valid, _ = validate_final_keyframe(blank, canvas_w=320, canvas_h=180)
    assert blank_valid is False
