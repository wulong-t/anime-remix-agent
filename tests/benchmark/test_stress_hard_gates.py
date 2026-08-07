"""Phase B hard gates: 30x1000 scoring, determinism, shuffle, cache, SM3 (B2)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from dataset import (
    build_clips,
    build_requirements,
    scan_statistics,
    selection_summary,
)

from anime_remix.domain.models import (
    CharacterRef,
    ClipAsset,
    ProbedClip,
    ShotRequirement,
)
from anime_remix.services import clip_retriever as cr


def _small_probed(
    clip_id: str,
    frames: int,
    *,
    action: str,
    description: str,
) -> ProbedClip:
    asset = ClipAsset(
        id=clip_id,
        path=f"clips/{clip_id}.mp4",
        characters=[CharacterRef(id="char_lin_xia", name="林夏")],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action=action,
        description=description,
    )
    return ProbedClip(
        asset=asset,
        resolved_path=Path(f"clips/{clip_id}.mp4"),
        size_bytes=1000,
        width=1280,
        height=720,
        fps_num=24,
        fps_den=1,
        nb_frames=frames,
        duration_seconds=Decimal(frames) / Decimal(24),
    )


def _small_requirement(source_text: str, action: str) -> ShotRequirement:
    return ShotRequirement(
        id="shot_001",
        order=1,
        source_text=source_text,
        characters=[CharacterRef(id="char_lin_xia", name="林夏")],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action=action,
        target_frames=72,
    )


def _audit_json(audit: dict[str, Any]) -> str:
    return json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)


def test_all_30000_pairs_scored_and_top3_display_only(
    monkeypatch: Any,
) -> None:
    original = cr._score_requirement
    calls = 0

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "_score_requirement", counting)
    requirements = build_requirements()
    clips = build_clips()
    _selections, audit = cr.retrieve(requirements, clips)
    assert calls == 30_000
    assert len(audit["shots"]) == 30
    assert all(shot["total_candidates"] == 1000 for shot in audit["shots"])
    assert all(len(shot["top_3"]) == 3 for shot in audit["shots"])


def test_determinism_three_runs() -> None:
    requirements = build_requirements()
    clips = build_clips()
    summaries = []
    audit_dumps = []
    for _ in range(3):
        selections, audit = cr.retrieve(requirements, clips)
        summaries.append(selection_summary(selections))
        audit_dumps.append(_audit_json(audit))
    assert summaries[0] == summaries[1] == summaries[2]
    assert audit_dumps[0] == audit_dumps[1] == audit_dumps[2]


def test_shuffle_deterministic() -> None:
    requirements = build_requirements()
    clips = build_clips()
    shuffled = list(reversed(clips))
    original_selections, _ = cr.retrieve(requirements, clips)
    shuffled_selections, _ = cr.retrieve(requirements, shuffled)
    assert selection_summary(original_selections) == selection_summary(
        shuffled_selections
    )


def test_top3_boundary_rank_greater_than_three() -> None:
    freeze = [
        _small_probed(
            f"clip_f{i:02d}",
            70,
            action="独自站立",
            description="林夏独自站在学校天台。",
        )
        for i in range(1, 5)
    ]
    full = _small_probed(
        "clip_full",
        96,
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    requirement = _small_requirement(
        "林夏独自站在学校天台。",
        "独自站立",
    )
    selections, audit = cr.retrieve([requirement], [*freeze, full])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.asset.asset.id == "clip_full"
    assert selected.rank == 5
    shot = audit["shots"][0]
    assert len(shot["top_3"]) == 3
    assert len(shot["selection_trace"]["scanned_candidates"]) == 5


def test_clip_preprocessing_linear(monkeypatch: Any) -> None:
    clip_calls = 0
    req_calls = 0
    original_clip_index = cr._make_clip_index
    original_req_index = cr._make_requirement_index

    def counting_clip(clip: Any) -> Any:
        nonlocal clip_calls
        clip_calls += 1
        return original_clip_index(clip)

    def counting_req(req: Any) -> Any:
        nonlocal req_calls
        req_calls += 1
        return original_req_index(req)

    monkeypatch.setattr(cr, "_make_clip_index", counting_clip)
    monkeypatch.setattr(cr, "_make_requirement_index", counting_req)
    requirements = build_requirements()
    clips = build_clips()
    cr.retrieve(requirements, clips)
    # Static clip preprocessing is O(1000), requirement preprocessing O(30);
    # neither repeats per shot x clip pair.
    assert clip_calls == 1000
    assert req_calls == 30


def test_sequence_matcher_boundary(monkeypatch: Any) -> None:
    long_text = "啊" * 300
    long_req = _small_requirement(long_text, long_text)
    long_clips = [
        _small_probed("clip_long", 96, action=long_text, description=long_text)
    ]
    calls = 0
    original = cr.SequenceMatcher

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "SequenceMatcher", counting)
    cr.retrieve([long_req], long_clips)
    assert calls == 0

    short_req = _small_requirement("林夏独自站在学校天台。", "独自站立")
    short_clips = [
        _small_probed(
            "clip_short",
            96,
            action="独自站立",
            description="林夏独自站在学校天台。",
        )
    ]
    cr.retrieve([short_req], short_clips)
    assert calls > 0


def test_scanned_statistics_observed() -> None:
    requirements = build_requirements()
    clips = build_clips()
    _selections, audit = cr.retrieve(requirements, clips)
    average, maximum = scan_statistics(audit)
    assert 1 <= maximum <= 1000
    assert 0 < average <= 1000
