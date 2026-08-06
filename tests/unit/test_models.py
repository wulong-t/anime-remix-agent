from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import (
    CharacterRef,
    ClipAsset,
    ClipsDocument,
    ProbedClip,
    RenderProfile,
    ScoreBreakdown,
    ShotRequirement,
    Timeline,
)


def _requirement(shot_id: str = "shot_001", order: int = 1) -> ShotRequirement:
    return ShotRequirement(
        id=shot_id,
        order=order,
        source_text="林夏站在天台。",
        action="独自站立",
        target_frames=72,
    )


def _clip_item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "shot_id": "shot_001",
        "order": 1,
        "requirement": _requirement().model_dump(mode="json"),
        "strategy": "clip",
        "source_asset_id": "clip_001",
        "source_path": "clips/clip_001.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "a" * 64,
        "source_in_frame": 0,
        "source_frame_count": 72,
        "target_frames": 72,
        "score": None,
        "reason_code": "exact_length",
        "reason": "exact_length",
    }
    base.update(overrides)
    return base


def _freeze_item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "shot_id": "shot_001",
        "order": 1,
        "requirement": _requirement().model_dump(mode="json"),
        "strategy": "freeze_frame",
        "source_asset_id": "clip_001",
        "source_path": "clips/clip_001.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "a" * 64,
        "source_in_frame": 0,
        "source_frame_count": 24,
        "target_frames": 72,
        "score": None,
        "reason_code": "short_source_freeze",
        "reason": "short_source_freeze",
    }
    base.update(overrides)
    return base


def _timeline_doc(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.9",
        "path_base": "timeline_dir",
        "render_profile": RenderProfile().model_dump(mode="json"),
        "items": items,
    }


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ClipAsset).validate_python(
            {
                "id": "clip_001",
                "path": "a.mp4",
                "action": "x",
                "description": "y",
                "surprise": 1,
            }
        )


def test_nan_and_infinity_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ClipsDocument).validate_python(
            {
                "schema_version": "1.9",
                "clips": [
                    {
                        "id": "clip_001",
                        "path": "a.mp4",
                        "characters": [{"id": "c", "name": float("nan")}],
                        "action": "x",
                        "description": "y",
                    }
                ],
            }
        )


def test_top_level_list_rejected_by_model() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ClipsDocument).validate_python([])


def test_character_ref_requires_id_or_name() -> None:
    with pytest.raises(ValidationError):
        CharacterRef(id="", name="")


def test_clip_id_pattern() -> None:
    with pytest.raises(ValidationError):
        ClipAsset(
            id="bad id!",
            path="a.mp4",
            action="x",
            description="y",
        )


def test_clips_document_max_and_duplicates() -> None:
    def clip(index: int) -> ClipAsset:
        return ClipAsset(
            id=f"clip_{index:03d}",
            path="a.mp4",
            action="x",
            description="y",
        )

    with pytest.raises(ValidationError):
        ClipsDocument(clips=[clip(1), clip(1)])
    with pytest.raises(ValidationError):
        ClipsDocument(clips=[clip(i) for i in range(51)])


def test_render_profile_literals() -> None:
    with pytest.raises(ValidationError):
        RenderProfile(width=1920)
    with pytest.raises(ValidationError):
        RenderProfile(fps=30)


def test_timeline_clip_requires_complete_source() -> None:
    doc = _timeline_doc([_clip_item(source_sha256=None)])
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(doc)


def test_timeline_placeholder_forbids_source_fields() -> None:
    item = _clip_item(
        strategy="placeholder",
        source_asset_id=None,
        source_path=None,
        source_size_bytes=None,
        source_sha256=None,
        source_in_frame=0,
        source_frame_count=0,
        reason_code="no_candidate",
        reason="no candidate",
    )
    item["source_asset_id"] = "clip_001"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(_timeline_doc([item]))


def test_timeline_orders_contiguous_and_unique() -> None:
    items = [
        _clip_item(shot_id="shot_001", order=2),
        _clip_item(shot_id="shot_002", order=1),
    ]
    items[1]["requirement"]["id"] = "shot_002"
    items[1]["requirement"]["order"] = 1
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(_timeline_doc(items))


def test_timeline_sha256_pattern() -> None:
    doc = _timeline_doc([_clip_item(source_sha256="z" * 64)])
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(doc)


def test_timeline_old_and_future_schema_rejected() -> None:
    valid = _timeline_doc([_clip_item()])
    for version in ("1.8", "2.0"):
        doc = dict(valid)
        doc["schema_version"] = version
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(doc)


def test_timeline_item_shot_id_must_match_requirement() -> None:
    doc = _timeline_doc([_clip_item(shot_id="shot_002")])
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(doc)


def test_strategy_enum_values() -> None:
    assert TimelineStrategy.CLIP.value == "clip"
    assert TimelineStrategy.FREEZE_FRAME.value == "freeze_frame"
    assert TimelineStrategy.PLACEHOLDER.value == "placeholder"


def test_path_rejects_int_bool_list_and_dict() -> None:
    for bad in (123, True, [1], {"x": 1}):
        with pytest.raises(ValidationError):
            TypeAdapter(ClipAsset).validate_python(
                {
                    "id": "clip_001",
                    "path": bad,
                    "action": "x",
                    "description": "y",
                }
            )


def test_strategy_rejects_int_bool_and_unknown_string() -> None:
    for bad in (1, False, "unfreeze", "CLIP"):
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(
                _timeline_doc([_clip_item(strategy=bad)])
            )


def test_decimal_rejects_bool_nan_and_infinity() -> None:
    base = {
        "character": None,
        "location": None,
        "duration": "1.0",
        "active_weights": {},
        "total": "1.0",
    }
    for bad in (True, False, "NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValidationError):
            TypeAdapter(ScoreBreakdown).validate_python(
                {**base, "action": bad}
            )
    with pytest.raises(ValidationError):
        TypeAdapter(ProbedClip).validate_python(
            {
                "asset": {
                    "id": "clip_001",
                    "path": "a.mp4",
                    "action": "x",
                    "description": "y",
                },
                "resolved_path": "a.mp4",
                "size_bytes": 1,
                "width": 1280,
                "height": 720,
                "fps_num": 24,
                "fps_den": 1,
                "nb_frames": 24,
                "duration_seconds": True,
            }
        )


def test_int_frame_fields_reject_booleans() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_clip_item(order=True)])
        )
    with pytest.raises(ValidationError):
        TypeAdapter(ShotRequirement).validate_python(
            {
                "id": "shot_001",
                "order": 1,
                "source_text": "x。",
                "action": "x",
                "target_frames": True,
            }
        )
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_clip_item(source_in_frame=True)])
        )
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_clip_item(source_frame_count=True)])
        )


def test_reason_code_is_closed_literal() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_clip_item(reason_code="bogus")])
        )
    TypeAdapter(Timeline).validate_python(
        _timeline_doc([_clip_item(reason_code="exact_length")])
    )
    TypeAdapter(Timeline).validate_python(
        _timeline_doc([_clip_item(reason_code="center_trim")])
    )
    TypeAdapter(Timeline).validate_python(
        _timeline_doc([_freeze_item(reason_code="short_source_freeze")])
    )
    placeholder = _freeze_item(
        strategy="placeholder",
        reason_code="no_candidate",
        source_asset_id=None,
        source_path=None,
        source_size_bytes=None,
        source_sha256=None,
        source_in_frame=0,
        source_frame_count=0,
    )
    TypeAdapter(Timeline).validate_python(_timeline_doc([placeholder]))


def test_freeze_frame_valid_model() -> None:
    TypeAdapter(Timeline).validate_python(_timeline_doc([_freeze_item()]))


def test_freeze_frame_requires_complete_source_fields() -> None:
    for field in (
        "source_asset_id",
        "source_path",
        "source_size_bytes",
        "source_sha256",
    ):
        item = _freeze_item(**{field: None})
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(_timeline_doc([item]))


def test_freeze_frame_reason_code_mapping() -> None:
    # freeze_frame must use short_source_freeze only.
    for bad in ("exact_length", "center_trim", "no_candidate"):
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(
                _timeline_doc([_freeze_item(reason_code=bad)])
            )
    # clip must not use short_source_freeze.
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_clip_item(reason_code="short_source_freeze")])
        )
    # placeholder must not use short_source_freeze.
    placeholder = _freeze_item(
        strategy="placeholder",
        reason_code="short_source_freeze",
        source_asset_id=None,
        source_path=None,
        source_size_bytes=None,
        source_sha256=None,
        source_in_frame=0,
        source_frame_count=0,
    )
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(_timeline_doc([placeholder]))


def test_freeze_frame_source_frame_count_bounds() -> None:
    for bad_count in (0, -1):
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(
                _timeline_doc([_freeze_item(source_frame_count=bad_count)])
            )
    for bad_count in (72, 100):
        with pytest.raises(ValidationError):
            TypeAdapter(Timeline).validate_python(
                _timeline_doc([_freeze_item(source_frame_count=bad_count)])
            )
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(
            _timeline_doc([_freeze_item(source_in_frame=-1)])
        )


def test_timeline_array_order_must_match_item_order() -> None:
    first = _clip_item(shot_id="shot_001", order=2)
    first["requirement"]["order"] = 2
    second = _clip_item(shot_id="shot_002", order=1)
    second["requirement"]["id"] = "shot_002"
    with pytest.raises(ValidationError):
        TypeAdapter(Timeline).validate_python(_timeline_doc([first, second]))


def test_timeline_array_order_valid_when_index_plus_one() -> None:
    first = _clip_item(shot_id="shot_001", order=1)
    second = _clip_item(
        shot_id="shot_002",
        order=2,
        source_asset_id="clip_002",
    )
    second["requirement"]["id"] = "shot_002"
    second["requirement"]["order"] = 2
    TypeAdapter(Timeline).validate_python(_timeline_doc([first, second]))
