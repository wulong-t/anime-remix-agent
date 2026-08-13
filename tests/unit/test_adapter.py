"""Unit tests for the Phase 3 Renderer Adapter boundary."""

from __future__ import annotations

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.adapter import (
    QwenImage30Adapter,
    QwenImageEditAdapter,
    StubImageExecutor,
)


def _conditions() -> list[dict]:
    return [
        {
            "condition_id": "cond_001",
            "role": "identity",
            "payload_ref": "asset://anime-remix/character/asuna@v1",
        },
        {
            "condition_id": "cond_002",
            "role": "pose",
            "payload_ref": "asset://anime-remix/pose/asuna_sitting@v1",
        },
    ]


def _intent() -> dict:
    return {
        "subject_pose": "sitting",
        "camera_view": "front_left",
        "shot_scale": "medium",
    }


def _keyframe_state() -> dict:
    return {
        "visual_description": (
            "Asuna is seated at the classroom desk with both eyes fully "
            "closed and her right fingertips touching her right temple"
        ),
        "subject_pose": (
            "seated at the desk, right elbow bent, right hand raised with "
            "fingertips touching the right temple"
        ),
        "expression": "sad and thoughtful, with both eyes fully closed",
        "gaze": "eyes fully closed",
        "composition": "medium shot, character center-left",
        "camera": "front-left",
        "background_state": "classroom in warm afternoon light",
        "foreground_state": "desk occludes the lower body",
        "prop_state": "wooden classroom desk",
        "motion_from_previous": "has sat down and raised her right hand",
        "locked_attributes": ["identity", "costume"],
        "character_locks": {
            "identity": True,
            "hairstyle": True,
            "costume_variant": "school_uniform",
        },
    }


def test_qwen_compile_character_synthesis_slots() -> None:
    adapter = QwenImageEditAdapter(seed=7, steps=30)
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions(),
    )
    assert compiled["adapter_id"] == "qwen-edit-2511-adapter-v1"
    assert [c["slot"] for c in compiled["conditions"]] == [1, 2]
    assert compiled["conditions"][0]["condition_ref"] == "cond_001"
    assert "Scene: scene classroom_01" in compiled["prompt"]
    assert compiled["parameters"]["seed"] == 7
    assert compiled["parameters"]["num_inference_steps"] == 30


def test_qwen_compile_requires_identity() -> None:
    adapter = QwenImageEditAdapter()
    conditions = [c for c in _conditions() if c["role"] != "identity"]
    with pytest.raises(InputValidationError):
        adapter.compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene",
            conditions=conditions,
        )


def test_qwen_compile_local_inpaint_has_no_slots() -> None:
    adapter = QwenImageEditAdapter()
    compiled = adapter.compile(
        operation="local_inpaint",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene",
        conditions=_conditions(),
    )
    assert compiled["conditions"] == []
    assert "masked region" in compiled["prompt"]


def test_stub_executor_is_deterministic() -> None:
    executor = StubImageExecutor()
    request = {
        "parameters": {"seed": 0},
        "prompt": "x",
        "conditions": [],
    }
    first = executor.execute(
        request_payload=request,
        operation="character_synthesis",
        inputs={},
    )
    second = executor.execute(
        request_payload=request,
        operation="character_synthesis",
        inputs={},
    )
    assert first == second


def _conditions_with_extra_slots() -> list[dict]:
    return [
        {
            "condition_id": "cond_001",
            "role": "identity",
            "payload_ref": "asset://anime-remix/character/asuna@v1",
        },
        {
            "condition_id": "cond_extra_identity",
            "role": "identity",
            "payload_ref": "asset://anime-remix/character/asuna@v2",
        },
        {
            "condition_id": "cond_002",
            "role": "pose",
            "payload_ref": "asset://anime-remix/pose/asuna_sitting@v1",
        },
        {
            "condition_id": "cond_003",
            "role": "expression",
            "payload_ref": "asset://anime-remix/expression/sad@v1",
        },
    ]


def test_qwen30_pro_compile_character_synthesis_frozen_params() -> None:
    adapter = QwenImage30Adapter(seed=9, size="1024*576")
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions(),
    )
    assert compiled["adapter_id"] == (
        "qwen-image-30-adapter-v6-reference-first"
    )
    assert compiled["model_id"] == "qwen-image-3.0"
    assert compiled["revision"] == "provider-managed-alias"
    assert [c["slot"] for c in compiled["conditions"]] == [1]
    assert compiled["conditions"][0]["condition_ref"] == "cond_001"
    parameters = compiled["parameters"]
    assert parameters["seed"] == 9
    assert parameters["n"] == 1
    assert parameters["size"] == "1024*576"
    assert parameters["prompt_extend"] is False
    assert parameters["prompt_extend_mode"] == "direct"
    assert parameters["watermark"] is False
    for keyword in ("text", "logo", "watermark", "deformed hands", "anatomy"):
        assert keyword in parameters["negative_prompt"]


def test_qwen30_pro_compile_default_parameters() -> None:
    adapter = QwenImage30Adapter()
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions(),
    )
    assert compiled["parameters"]["seed"] == 0
    assert compiled["parameters"]["size"] == "1280*720"


def test_qwen30_pro_text_how_sends_only_first_identity() -> None:
    adapter = QwenImage30Adapter()
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions_with_extra_slots(),
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"},
    ]


def test_qwen30_pro_compile_expression_before_pose_is_how_slot() -> None:
    adapter = QwenImage30Adapter(visual_how=True)
    conditions = [
        _conditions()[0],
        {
            "condition_id": "cond_003",
            "role": "expression",
            "payload_ref": "asset://anime-remix/expression/sad@v1",
        },
        _conditions()[1],
    ]
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=conditions,
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"},
        {"slot": 2, "condition_ref": "cond_003"},
    ]


def test_qwen30_pro_compile_prompt_keeps_complete_keyframe_semantics() -> None:
    adapter = QwenImage30Adapter()
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions(),
    )
    prompt = compiled["prompt"]
    assert "Image 1 is the only visual character reference" in prompt
    assert "Transform the same character from Image 1" in prompt
    assert "Do not replace or redesign that character's identity" in prompt
    assert "Image 2" not in prompt
    assert "both eyes fully closed" in prompt
    assert "fingertips touching the right temple" in prompt
    assert "Target gaze and eye state: eyes fully closed" in prompt
    assert "Composition: medium shot, character center-left" in prompt
    assert "Camera: front-left" in prompt
    assert "Background: classroom in warm afternoon light" in prompt
    assert "Foreground: desk occludes the lower body" in prompt
    assert "Props: wooden classroom desk" in prompt
    assert "No selected reference covers" in prompt
    assert "Visual state:" not in prompt
    assert (
        "Preserve from Image 1 exactly as shown: identity and facial "
        "features, hairstyle and hair color, clothing and accessories"
        in prompt
    )
    assert "Do not redesign, restyle" in prompt
    assert "scene classroom_01" not in prompt
    assert "sitting_at" not in prompt


def test_qwen30_pro_visual_how_uses_second_slot_and_leakage_guard() -> None:
    adapter = QwenImage30Adapter(visual_how=True)
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=_conditions(),
    )
    assert compiled["adapter_id"] == (
        "qwen-image-30-adapter-v4-visual-how"
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"},
        {"slot": 2, "condition_ref": "cond_002"},
    ]
    prompt = compiled["prompt"]
    assert "Image 2 is a pose-only HOW control reference" in prompt
    assert "Do not copy any identity" in prompt
    assert "clothing, texture, color, or background from Image 2" in prompt


def test_qwen30_pro_previous_keyframe_is_continuity_slot() -> None:
    conditions = [
        *_conditions(),
        {
            "condition_id": "cond_previous_keyframe",
            "role": "source_frame",
            "payload_ref": "artifact://previous-run/art_000007",
        },
    ]
    compiled = QwenImage30Adapter().compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=conditions,
    )
    assert compiled["adapter_id"] == (
        "qwen-image-30-adapter-v6-continuity-action-delta"
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"},
        {"slot": 2, "condition_ref": "cond_previous_keyframe"},
    ]
    prompt = compiled["prompt"]
    assert "previous approved keyframe from the same shot" in prompt
    assert "Image 1 is authoritative" in prompt
    assert "Required motion or state change" in prompt
    assert "No selected reference covers" not in prompt
    assert "Background:" not in prompt


def test_qwen30_pro_full_frame_reference_suppresses_visual_restatement() -> None:
    conditions = [
        *_conditions(),
        {
            "condition_id": "cond_scene",
            "role": "scene",
            "payload_ref": "asset://anime-remix/scene/classroom@v1",
        },
    ]
    compiled = QwenImage30Adapter().compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene classroom_01",
        conditions=conditions,
    )
    assert compiled["adapter_id"] == (
        "qwen-image-30-adapter-v6-reference-first-context"
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"},
        {"slot": 2, "condition_ref": "cond_scene"},
    ]
    prompt = compiled["prompt"]
    assert "Image 2 is the full-frame visual reference" in prompt
    assert "anime" not in prompt.lower()
    assert "Required motion or state change" in prompt
    assert "Target pose:" in prompt
    assert "No selected reference covers" not in prompt
    for label in ("Composition:", "Camera:", "Background:", "Foreground:", "Props:"):
        assert label not in prompt


def test_qwen30_pro_rejects_multiple_context_references() -> None:
    conditions = [
        *_conditions(),
        {
            "condition_id": "cond_scene",
            "role": "scene",
            "payload_ref": "asset://anime-remix/scene/classroom@v1",
        },
        {
            "condition_id": "cond_style",
            "role": "style",
            "payload_ref": "asset://anime-remix/style/anime@v1",
        },
    ]
    with pytest.raises(InputValidationError, match="fuse or stage"):
        QwenImage30Adapter().compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene classroom_01",
            conditions=conditions,
        )


def test_qwen30_pro_rejects_visual_how_with_continuity() -> None:
    conditions = [
        *_conditions(),
        {
            "condition_id": "cond_previous_keyframe",
            "role": "source_frame",
            "payload_ref": "artifact://previous-run/art_000007",
        },
    ]
    with pytest.raises(InputValidationError, match="cannot be combined"):
        QwenImage30Adapter(visual_how=True).compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene classroom_01",
            conditions=conditions,
        )


def test_qwen30_direct_scene_synthesis_three_slots_and_prompt_passthrough() -> None:
    conditions = [
        {
            "condition_id": "cond_identity",
            "role": "identity",
            "payload_ref": "asset://anime-remix/catalog/test-character@v1",
        },
        {
            "condition_id": "cond_desk",
            "role": "prop",
            "payload_ref": "asset://anime-remix/catalog/desk@v1",
        },
        {
            "condition_id": "cond_chair",
            "role": "prop",
            "payload_ref": "asset://anime-remix/catalog/chair@v1",
        },
    ]
    instruction = (
        "Image 1 is the character identity reference. Image 2 is the desk "
        "reference. Image 3 is the chair reference. Compose the final frame."
    )
    compiled = QwenImage30Adapter(seed=3, size="1280*720").compile(
        operation="direct_scene_synthesis",
        intent={"instruction": instruction},
        keyframe_state={},
        scene_description="",
        conditions=conditions,
    )
    assert (
        compiled["adapter_id"]
        == "qwen-image-30-adapter-v7-direct-frame-identity-prop-prop"
    )
    assert compiled["model_id"] == "qwen-image-3.0"
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_identity"},
        {"slot": 2, "condition_ref": "cond_desk"},
        {"slot": 3, "condition_ref": "cond_chair"},
    ]
    assert compiled["prompt"] == instruction
    assert compiled["parameters"]["seed"] == 3
    assert compiled["parameters"]["size"] == "1280*720"
    assert compiled["parameters"]["prompt_extend"] is False
    assert compiled["parameters"]["prompt_extend_mode"] == "direct"
    assert compiled["parameters"]["watermark"] is False


def test_qwen30_direct_scene_synthesis_requires_identity_reference() -> None:
    conditions = [
        {
            "condition_id": "cond_desk",
            "role": "prop",
            "payload_ref": "asset://anime-remix/catalog/desk@v1",
        },
    ]
    with pytest.raises(InputValidationError, match="exactly one identity"):
        QwenImage30Adapter().compile(
            operation="direct_scene_synthesis",
            intent={"instruction": "compose the frame"},
            keyframe_state={},
            scene_description="",
            conditions=conditions,
        )


def test_qwen30_direct_scene_synthesis_scene_only_allows_zero_identity() -> None:
    conditions = [
        {
            "condition_id": "cond_scene",
            "role": "scene",
            "payload_ref": "asset://anime-remix/catalog/office@v1",
        },
        {
            "condition_id": "cond_desk",
            "role": "prop",
            "payload_ref": "asset://anime-remix/catalog/desk@v1",
        },
        {
            "condition_id": "cond_chair",
            "role": "prop",
            "payload_ref": "asset://anime-remix/catalog/chair@v1",
        },
    ]
    compiled = QwenImage30Adapter().compile(
        operation="direct_scene_synthesis",
        intent={
            "instruction": "establish the scene only",
            "scene_only": True,
        },
        keyframe_state={},
        scene_description="",
        conditions=conditions,
    )
    assert (
        compiled["adapter_id"]
        == "qwen-image-30-adapter-v7-direct-frame-scene-prop-prop"
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_scene"},
        {"slot": 2, "condition_ref": "cond_desk"},
        {"slot": 3, "condition_ref": "cond_chair"},
    ]


def test_qwen30_direct_scene_synthesis_scene_only_rejects_identity() -> None:
    conditions = [
        {
            "condition_id": "cond_identity",
            "role": "identity",
            "payload_ref": "asset://anime-remix/catalog/test-character@v1",
        },
    ]
    with pytest.raises(InputValidationError, match="must not include"):
        QwenImage30Adapter().compile(
            operation="direct_scene_synthesis",
            intent={
                "instruction": "establish the scene only",
                "scene_only": True,
            },
            keyframe_state={},
            scene_description="",
            conditions=conditions,
        )


def test_qwen30_direct_scene_synthesis_rejects_more_than_three_references() -> None:
    conditions = [
        {
            "condition_id": f"cond_{index:02d}",
            "role": "prop" if index else "identity",
            "payload_ref": f"asset://anime-remix/catalog/item_{index}@v1",
        }
        for index in range(4)
    ]
    with pytest.raises(InputValidationError, match="1-3 reference"):
        QwenImage30Adapter().compile(
            operation="direct_scene_synthesis",
            intent={"instruction": "compose the frame"},
            keyframe_state={},
            scene_description="",
            conditions=conditions,
        )


def test_qwen30_first_frame_text_only_base_has_no_visual_restatement() -> None:
    request = QwenImage30Adapter(size="512*512").compile(
        operation="first_frame_fusion",
        intent={
            "stage_id": "stage_001",
            "stage_operation": "synthesize_base",
            "component_ids": ["scene"],
            "instruction": "Establish only the uncovered scene canvas.",
            "reference_attributes": [],
            "text_fallbacks": {
                "setting": "an empty classroom",
                "camera": "front eye level",
            },
        },
        keyframe_state={},
        scene_description="an empty classroom",
        conditions=[],
    )

    assert request["conditions"] == []
    assert request["adapter_id"].endswith("synthesize_base")
    assert "No visual reference exists" in request["prompt"]
    assert "an empty classroom" in request["prompt"]


def test_qwen30_first_frame_component_uses_canvas_then_scoped_reference() -> None:
    conditions = [
        {
            "condition_id": "cond_canvas",
            "role": "source_frame",
            "kind": "image",
        },
        {
            "condition_id": "cond_identity",
            "role": "identity",
            "kind": "image",
        },
    ]
    request = QwenImage30Adapter(size="512*512").compile(
        operation="first_frame_fusion",
        intent={
            "stage_id": "stage_002",
            "stage_operation": "fuse_component",
            "component_ids": ["character_001"],
            "instruction": "Place the character sitting with closed eyes.",
            "reference_attributes": ["identity", "face", "hair"],
            "text_fallbacks": {},
        },
        keyframe_state={},
        scene_description="classroom",
        conditions=conditions,
    )

    assert [item["condition_ref"] for item in request["conditions"]] == [
        "cond_canvas",
        "cond_identity",
    ]
    assert "authoritative only for this component's identity, face, hair" in (
        request["prompt"]
    )
    assert "classroom" not in request["prompt"]


def test_qwen30_first_frame_atomic_interaction_plate_keeps_prop_authority() -> None:
    request = QwenImage30Adapter(size="512*512").compile(
        operation="first_frame_fusion",
        intent={
            "stage_id": "stage_002",
            "stage_operation": "fuse_component",
            "component_ids": ["character_001", "prop_001"],
            "instruction": "Place the approved character-key group on the canvas.",
            "reference_attributes": [
                "identity",
                "pose",
                "prop identity",
                "prop appearance",
                "interaction/contact geometry",
            ],
            "text_fallbacks": {},
        },
        keyframe_state={},
        scene_description="observatory",
        conditions=[
            {"condition_id": "cond_canvas", "role": "source_frame", "kind": "image"},
            {"condition_id": "cond_group", "role": "identity", "kind": "image"},
        ],
    )

    assert "prop identity, prop appearance" in request["prompt"]
    assert "interaction/contact geometry" in request["prompt"]
    assert "clothing, or props" not in request["prompt"]


def test_qwen30_first_frame_locks_approved_external_attachment_geometry() -> None:
    request = QwenImage30Adapter(size="512*512").compile(
        operation="first_frame_fusion",
        intent={
            "stage_id": "stage_002",
            "stage_operation": "fuse_component",
            "component_ids": ["character_001", "prop_001"],
            "instruction": (
                "Preserve the key tip at (0.460, 0.560) pointing toward the "
                "empty socket at (0.420, 0.560)."
            ),
            "reference_attributes": [
                "identity",
                "prop functional topology",
                "external attachment geometry",
                "final-frame canvas placement",
            ],
            "text_fallbacks": {},
        },
        keyframe_state={},
        scene_description="observatory",
        conditions=[
            {"condition_id": "cond_canvas", "role": "source_frame", "kind": "image"},
            {"condition_id": "cond_group", "role": "identity", "kind": "image"},
        ],
    )

    assert "Do not mirror, rotate, translate, rescale or recompose" in request["prompt"]
    assert "integrate it at the recorded anchors" in request["prompt"]


def test_qwen30_pro_compile_requires_identity_and_visual_how_pose() -> None:
    adapter = QwenImage30Adapter()
    with pytest.raises(InputValidationError):
        adapter.compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene",
            conditions=[c for c in _conditions() if c["role"] != "identity"],
        )
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene",
        conditions=[c for c in _conditions() if c["role"] != "pose"],
    )
    assert compiled["conditions"] == [
        {"slot": 1, "condition_ref": "cond_001"}
    ]
    visual_adapter = QwenImage30Adapter(visual_how=True)
    with pytest.raises(InputValidationError, match="visual-HOW"):
        visual_adapter.compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene",
            conditions=[c for c in _conditions() if c["role"] != "pose"],
        )


def test_qwen30_pro_compile_local_inpaint_has_no_slots() -> None:
    adapter = QwenImage30Adapter()
    compiled = adapter.compile(
        operation="local_inpaint",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene",
        conditions=_conditions(),
    )
    assert compiled["conditions"] == []
    assert "masked region" in compiled["prompt"]


def test_qwen30_pro_rejects_invalid_seed() -> None:
    for seed in (-1, 2**31, 2**32, "7", True):
        with pytest.raises(InputValidationError):
            QwenImage30Adapter(seed=seed)


def test_qwen30_pro_rejects_non_boolean_visual_how() -> None:
    for visual_how in (0, 1, "false", None):
        with pytest.raises(InputValidationError, match="visual_how"):
            QwenImage30Adapter(visual_how=visual_how)


def test_qwen30_pro_accepts_max_seed() -> None:
    adapter = QwenImage30Adapter(seed=2**31 - 1)
    compiled = adapter.compile(
        operation="character_synthesis",
        intent=_intent(),
        keyframe_state=_keyframe_state(),
        scene_description="scene",
        conditions=_conditions(),
    )
    assert compiled["parameters"]["seed"] == 2**31 - 1


def test_qwen30_pro_accepts_size_boundaries() -> None:
    for size in ("512*512", "2048*2048", "512*4096", "4096*512", "1280*720"):
        adapter = QwenImage30Adapter(size=size)
        compiled = adapter.compile(
            operation="character_synthesis",
            intent=_intent(),
            keyframe_state=_keyframe_state(),
            scene_description="scene",
            conditions=_conditions(),
        )
        assert compiled["parameters"]["size"] == size


def test_qwen30_pro_rejects_invalid_size() -> None:
    for size in (
        "",
        "   ",
        "1280x720",
        "0*720",
        "1280*0",
        "01*01",
        "512*511",
        "2048*2049",
        "512*4097",
        "4097*512",
        -1,
        0,
        1024,
        True,
    ):
        with pytest.raises(InputValidationError):
            QwenImage30Adapter(size=size)
