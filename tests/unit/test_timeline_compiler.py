from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from anime_remix.domain.models import (
    CharacterRef,
    ClipAsset,
    ProbedClip,
    ShotRequirement,
)
from anime_remix.errors import TimelineValidationError
from anime_remix.services.clip_retriever import Selection
from anime_remix.services.timeline_compiler import compile_timeline


def _probed() -> ProbedClip:
    asset = ClipAsset(
        id="clip_001",
        path="clips/clip_001.mp4",
        characters=[CharacterRef(id="char_a", name="林夏")],
        location_id="loc_a",
        location_name="学校天台",
        action="独自站立",
        description="林夏站在学校天台。",
    )
    return ProbedClip(
        asset=asset,
        resolved_path=Path("synthetic-fixtures/clip_001.mp4").resolve(),
        size_bytes=1000,
        width=1280,
        height=720,
        fps_num=24,
        fps_den=1,
        nb_frames=96,
        duration_seconds=Decimal(4),
    )


def _requirement() -> ShotRequirement:
    return ShotRequirement(
        id="shot_001",
        order=1,
        source_text="林夏独自站在学校天台。",
        characters=[CharacterRef(id="char_a", name="林夏")],
        location_id="loc_a",
        location_name="学校天台",
        action="独自站立",
        target_frames=72,
    )


def test_compile_clip_timeline(tmp_path: Path) -> None:
    req = _requirement()
    probed = _probed()
    probed.resolved_path = (
        tmp_path / "synthetic-fixtures" / "clip_001.mp4"
    )
    selection = Selection(
        asset=probed,
        rank=1,
        reason_code="center_trim",
        source_in_frame=12,
        source_frame_count=72,
        score=None,
    )
    timeline = compile_timeline(
        [req],
        {"shot_001": selection},
        target_dir=tmp_path / "runs" / "demo-001",
        source_sha256={"clip_001": "a" * 64},
    )
    item = timeline.items[0]
    assert item.strategy.value == "clip"
    assert item.source_frame_count == 72
    assert item.source_path.endswith("synthetic-fixtures/clip_001.mp4")


def test_compile_placeholder_timeline() -> None:
    req = _requirement()
    selection = Selection(
        asset=None,
        rank=None,
        reason_code="no_candidate",
        source_in_frame=0,
        source_frame_count=0,
        score=None,
    )
    timeline = compile_timeline(
        [req],
        {"shot_001": selection},
        target_dir=Path("runs/demo-001"),
    )
    assert timeline.items[0].strategy.value == "placeholder"
    assert timeline.items[0].source_asset_id is None


def test_compile_freeze_timeline(tmp_path: Path) -> None:
    req = _requirement()
    probed = _probed()
    probed.nb_frames = 30
    probed.resolved_path = tmp_path / "demo" / "clips" / "clip_001.mp4"
    selection = Selection(
        asset=probed,
        rank=1,
        reason_code="short_source_freeze",
        source_in_frame=0,
        source_frame_count=30,
        score=None,
    )
    timeline = compile_timeline(
        [req],
        {"shot_001": selection},
        target_dir=tmp_path / "runs" / "demo-001",
        source_sha256={"clip_001": "a" * 64},
    )
    item = timeline.items[0]
    assert item.strategy.value == "freeze_frame"
    assert item.source_in_frame == 0
    assert item.source_frame_count == 30
    assert item.reason_code == "short_source_freeze"


def test_compile_freeze_rejects_planner_invariants(tmp_path: Path) -> None:
    req = _requirement()
    probed = _probed()
    probed.nb_frames = 30
    probed.resolved_path = tmp_path / "demo" / "clips" / "clip_001.mp4"

    bad_start = Selection(
        asset=probed,
        rank=1,
        reason_code="short_source_freeze",
        source_in_frame=1,
        source_frame_count=29,
        score=None,
    )
    with pytest.raises(TimelineValidationError):
        compile_timeline(
            [req],
            {"shot_001": bad_start},
            target_dir=tmp_path / "runs" / "demo-001",
            source_sha256={"clip_001": "a" * 64},
        )

    bad_count = Selection(
        asset=probed,
        rank=1,
        reason_code="short_source_freeze",
        source_in_frame=0,
        source_frame_count=20,
        score=None,
    )
    with pytest.raises(TimelineValidationError):
        compile_timeline(
            [req],
            {"shot_001": bad_count},
            target_dir=tmp_path / "runs" / "demo-001",
            source_sha256={"clip_001": "a" * 64},
        )


def test_missing_selection_raises() -> None:
    with pytest.raises(TimelineValidationError):
        compile_timeline(
            [_requirement()],
            {},
            target_dir=Path("runs/demo-001"),
        )


def test_missing_sha256_raises() -> None:
    selection = Selection(
        asset=_probed(),
        rank=1,
        reason_code="center_trim",
        source_in_frame=12,
        source_frame_count=72,
        score=None,
    )
    with pytest.raises(TimelineValidationError):
        compile_timeline(
            [_requirement()],
            {"shot_001": selection},
            target_dir=Path("runs/demo-001"),
        )


def test_cross_drive_relative_path_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = Selection(
        asset=_probed(),
        rank=1,
        reason_code="center_trim",
        source_in_frame=12,
        source_frame_count=72,
        score=None,
    )

    def _relpath_boom(source: object, start: object) -> str:
        raise ValueError("different drive")

    monkeypatch.setattr(os.path, "relpath", _relpath_boom)
    with pytest.raises(TimelineValidationError):
        compile_timeline(
            [_requirement()],
            {"shot_001": selection},
            target_dir=Path("runs/demo-001"),
            source_sha256={"clip_001": "a" * 64},
        )
