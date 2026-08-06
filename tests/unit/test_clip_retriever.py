from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from anime_remix.domain.models import (
    CharacterRef,
    ClipAsset,
    ProbedClip,
    ShotRequirement,
)
from anime_remix.services.clip_retriever import (
    _max_person_matches,
    _person_key,
    normalize_for_match,
    retrieve,
    text_similarity,
)


def _probed(clip_id: str, frames: int, **overrides: object) -> ProbedClip:
    asset = ClipAsset(
        id=clip_id,
        path=f"clips/{clip_id}.mp4",
        characters=overrides.pop(  # type: ignore[arg-type]
            "characters", [{"id": "char_lin_xia", "name": "林夏"}]
        ),
        location_id=overrides.pop("location_id", "loc_school_rooftop"),  # type: ignore[arg-type]
        location_name=overrides.pop("location_name", "学校天台"),  # type: ignore[arg-type]
        action=overrides.pop("action", "独自站立"),  # type: ignore[arg-type]
        description=overrides.pop(  # type: ignore[arg-type]
            "description", "林夏独自站在学校天台。"
        ),
    )
    return ProbedClip(
        asset=asset,
        resolved_path=Path(f"clips/{clip_id}.mp4").resolve(),
        size_bytes=1000,
        width=1280,
        height=720,
        fps_num=24,
        fps_den=1,
        nb_frames=frames,
        duration_seconds=Decimal(frames) / Decimal(24),
    )


def _requirement(**overrides: object) -> ShotRequirement:
    base: dict[str, object] = {
        "id": "shot_001",
        "order": 1,
        "source_text": "林夏独自站在学校天台。",
        "characters": [{"id": "char_lin_xia", "name": "林夏"}],
        "location_id": "loc_school_rooftop",
        "location_name": "学校天台",
        "action": "独自站立",
        "target_frames": 72,
    }
    base.update(overrides)
    return ShotRequirement(**base)


def test_normalize_for_match() -> None:
    assert normalize_for_match(" 林夏，STANDS ！") == "林夏stands"


def test_text_similarity_identical_and_empty() -> None:
    assert text_similarity("独自站立", "独自站立") == Decimal("1.000000")
    assert text_similarity("", "独自站立") == Decimal("0.000000")


def test_sequence_matcher_bound() -> None:
    long = "对" * 300
    assert text_similarity(long, long) == Decimal("1.000000")


def test_character_f2_quantized() -> None:
    probed = _probed("clip_001", 100, characters=[{"id": "char_lin_xia", "name": "林夏"}])
    selections, _audit = retrieve([_requirement()], [probed])
    score = selections["shot_001"].score
    assert score is not None
    assert score.character == Decimal("1.000000")

    # requirement has 2 people but clip has only 1 -> F2 = 5*1*0.5/(4*1+0.5)
    req = _requirement(
        characters=[
            {"id": "char_lin_xia", "name": "林夏"},
            {"id": "char_lu_chen", "name": "陆辰"},
        ]
    )
    selections2, _ = retrieve([req], [probed])
    score2 = selections2["shot_001"].score
    assert score2 is not None
    assert score2.character == Decimal("0.555556")


def test_active_weights_renormalized() -> None:
    req = _requirement(characters=[])
    probed = _probed("clip_001", 100, characters=[{"id": "char_lin_xia", "name": "林夏"}])
    selections, _ = retrieve([req], [probed])
    score = selections["shot_001"].score
    assert score is not None
    assert "character" not in score.active_weights
    total = sum(score.active_weights.values())
    assert abs(total - Decimal(1)) <= Decimal("0.000002")


def test_stable_sort_and_tie_break() -> None:
    first = _probed("clip_a", 100)
    second = _probed("clip_b", 100)
    _selections, audit = retrieve([_requirement()], [second, first])
    shot = audit["shots"][0]
    assert shot["top_3"][0]["asset_id"] == "clip_a"
    assert shot["selected"]["selected_asset_id"] == "clip_a"
    assert shot["selected"]["selected_strategy"] == "clip"


def test_gate_scan_continues_past_short_first_candidate() -> None:
    short = _probed(
        "clip_short",
        60,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏站在学校天台。",
    )
    good = _probed(
        "clip_good",
        96,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [short, good])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.asset.asset.id == "clip_good"
    assert selected.reason_code == "center_trim"
    assert selected.source_in_frame == 12
    shot = audit["shots"][0]
    assert shot["selected"]["selected_strategy"] == "clip"
    freeze_entries = [
        entry
        for entry in shot["checked_gates"]
        if entry["asset_id"] == "clip_short"
    ]
    assert freeze_entries
    assert freeze_entries[0]["frame_gate"] == "freeze_eligible"


def test_no_candidate_uses_placeholder() -> None:
    probed = _probed(
        "clip_unrelated",
        96,
        characters=[{"id": "other", "name": "路人"}],
        location_id="loc_street",
        location_name="街道",
        action="骑车",
        description="路人在街道上骑车。",
    )
    selections, audit = retrieve([_requirement()], [probed])
    selected = selections["shot_001"]
    assert selected.asset is None
    assert selected.reason_code == "no_candidate"
    shot = audit["shots"][0]
    assert shot["selected"]["selected_asset_id"] is None
    assert shot["selected"]["selected_strategy"] == "placeholder"
    assert shot["selected"]["reason_code"] == "no_candidate"


def test_exact_length() -> None:
    probed = _probed(
        "clip_exact",
        72,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, _ = retrieve([_requirement()], [probed])
    selected = selections["shot_001"]
    assert selected.reason_code == "exact_length"
    assert selected.source_in_frame == 0
    assert selected.source_frame_count == 72


def test_top_3_only_display() -> None:
    clips = [
        _probed(
            f"clip_{i:03d}",
            96,
            characters=[{"id": "char_lin_xia", "name": "林夏"}],
            location_id="loc_school_rooftop",
            location_name="学校天台",
            action="独自站立",
            description="林夏站在学校天台。",
        )
        for i in range(5)
    ]
    _, audit = retrieve([_requirement()], clips)
    assert len(audit["shots"][0]["top_3"]) == 3


def test_person_matching_deterministic() -> None:
    required = [
        _person_key(CharacterRef(id="a", name="林夏")),
        _person_key(CharacterRef(id="b", name="陆辰")),
    ]
    asset = [
        _person_key(CharacterRef(id="b", name="陆辰")),
        _person_key(CharacterRef(id="a", name="林夏")),
    ]
    assert _max_person_matches(required, asset) == 2


def test_person_matching_distinct_ids_never_merge_by_name() -> None:
    required = [
        _person_key(CharacterRef(id="char_a", name="林夏")),
        _person_key(CharacterRef(id="char_b", name="陆辰")),
    ]
    asset = [
        _person_key(CharacterRef(id="char_x", name="林夏")),
        _person_key(CharacterRef(id="char_y", name="陆辰")),
    ]
    # Same normalized names but every ID differs: no identity may match.
    assert _max_person_matches(required, asset) == 0


def test_person_matching_missing_id_allows_name() -> None:
    # One side without an ID matches by exact normalized name.
    assert (
        _max_person_matches(
            [_person_key(CharacterRef(name="林夏"))],
            [_person_key(CharacterRef(id="char_a", name="林夏"))],
        )
        == 1
    )
    assert (
        _max_person_matches(
            [_person_key(CharacterRef(id="char_a", name="林夏"))],
            [_person_key(CharacterRef(name="林夏"))],
        )
        == 1
    )
    assert (
        _max_person_matches(
            [_person_key(CharacterRef(name="林夏"))],
            [_person_key(CharacterRef(name="林夏"))],
        )
        == 1
    )


def test_frame_gate_too_short() -> None:
    probed = _probed(
        "clip_23f",
        23,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [probed])
    selected = selections["shot_001"]
    assert selected.asset is None
    assert selected.reason_code == "no_candidate"
    shot = audit["shots"][0]
    assert shot["checked_gates"][0]["frame_gate"] == "too_short"
    assert shot["unique_skip_reasons"] == ["too_short"]


def test_frame_gate_freeze_eligible_at_24_frames() -> None:
    probed = _probed(
        "clip_24f",
        24,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [probed])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.asset.asset.id == "clip_24f"
    assert selected.reason_code == "short_source_freeze"
    assert selected.source_in_frame == 0
    assert selected.source_frame_count == 24
    shot = audit["shots"][0]
    assert shot["checked_gates"][0]["frame_gate"] == "freeze_eligible"
    assert shot["selected"]["selected_strategy"] == "freeze_frame"
    assert shot["selected"]["reason_code"] == "short_source_freeze"


def test_frame_gate_clip_eligible_at_exact_target() -> None:
    probed = _probed(
        "clip_72f",
        72,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [probed])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.reason_code == "exact_length"
    shot = audit["shots"][0]
    assert shot["checked_gates"][0]["frame_gate"] == "clip_eligible"
    assert shot["selected"]["selected_strategy"] == "clip"


def test_high_rank_freeze_does_not_beat_later_full_clip() -> None:
    short = _probed(
        "clip_short",
        60,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    good = _probed(
        "clip_good",
        96,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [short, good])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.asset.asset.id == "clip_good"
    assert selected.reason_code == "center_trim"
    shot = audit["shots"][0]
    assert shot["selected"]["selected_strategy"] == "clip"
    assert shot["selected"]["selected_global_rank"] == 2
    freeze_entry = shot["checked_gates"][0]
    assert freeze_entry["frame_gate"] == "freeze_eligible"


def test_no_full_clip_selects_highest_rank_freeze() -> None:
    low = _probed(
        "clip_low",
        30,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    high = _probed(
        "clip_high",
        60,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    selections, audit = retrieve([_requirement()], [low, high])
    selected = selections["shot_001"]
    assert selected.asset is not None
    assert selected.asset.asset.id == "clip_high"
    assert selected.reason_code == "short_source_freeze"
    assert selected.source_in_frame == 0
    assert selected.source_frame_count == 60
    shot = audit["shots"][0]
    assert shot["selected"]["selected_strategy"] == "freeze_frame"
    assert shot["selected"]["selected_global_rank"] == 1


def test_content_gate_failing_short_source_is_not_fallback() -> None:
    unrelated = _probed(
        "clip_unrelated",
        30,
        characters=[{"id": "other", "name": "路人"}],
        location_id="loc_street",
        location_name="街道",
        action="骑车",
        description="路人在街道上骑车。",
    )
    selections, audit = retrieve([_requirement()], [unrelated])
    selected = selections["shot_001"]
    assert selected.asset is None
    assert selected.reason_code == "no_candidate"
    shot = audit["shots"][0]
    assert shot["checked_gates"][0]["frame_gate"] is None
    assert shot["selected"]["selected_strategy"] == "placeholder"


def test_retrieval_audit_records_frame_gates_and_strategy() -> None:
    short = _probed(
        "clip_short",
        60,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    good = _probed(
        "clip_good",
        96,
        characters=[{"id": "char_lin_xia", "name": "林夏"}],
        location_id="loc_school_rooftop",
        location_name="学校天台",
        action="独自站立",
        description="林夏独自站在学校天台。",
    )
    _selections, audit = retrieve([_requirement()], [short, good])
    shot = audit["shots"][0]
    frame_gates = {
        entry["asset_id"]: entry["frame_gate"]
        for entry in shot["checked_gates"]
    }
    assert frame_gates == {
        "clip_short": "freeze_eligible",
        "clip_good": "clip_eligible",
    }
    assert shot["selected"]["selected_strategy"] == "clip"
    assert shot["selected"]["reason_code"] == "center_trim"
