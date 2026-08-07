"""Unit tests for 3B-3 selection_trace (AGENTS.md v1.13 section 18.9)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from anime_remix.domain.models import (
    ClipAsset,
    FinalDecisionTrace,
    ProbedClip,
    ScannedCandidateTrace,
    SelectionTrace,
    ShotRequirement,
)
from anime_remix.services.clip_retriever import retrieve


def _probed(clip_id: str, frames: int, **overrides: object) -> ProbedClip:
    base: dict[str, object] = {
        "id": clip_id,
        "path": f"clips/{clip_id}.mp4",
        "characters": [{"id": "char_lin_xia", "name": "林夏"}],
        "location_id": "loc_school_rooftop",
        "location_name": "学校天台",
        "action": "独自站立",
        "description": "林夏独自站在学校天台。",
    }
    base.update(overrides)
    asset = ClipAsset(**base)
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


def _trace(audit: dict, shot_id: str = "shot_001") -> dict:
    return next(
        shot["selection_trace"]
        for shot in audit["shots"]
        if shot["shot_id"] == shot_id
    )


class TestTraceModels:
    def test_scanned_candidate_trace_accepts_valid_entry(self) -> None:
        entry = ScannedCandidateTrace(
            global_rank=1,
            asset_id="clip_001",
            total="0.770000",
            character="1.000000",
            location="1.000000",
            action="0.600000",
            duration="0.666667",
            emotion=None,
            shot_scale=None,
            content_gate="passed",
            frame_gate="clip_eligible",
            decision="selected_clip",
        )
        assert entry.total == Decimal("0.770000")

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("content_gate", "maybe"),
            ("frame_gate", "almost"),
            ("decision", "free text"),
        ],
    )
    def test_scanned_candidate_trace_rejects_unknown_values(
        self,
        field: str,
        bad: str,
    ) -> None:
        payload = {
            "global_rank": 1,
            "asset_id": "clip_001",
            "total": "1.0",
            "action": "1.0",
            "duration": "1.0",
            "content_gate": "passed",
            "frame_gate": "clip_eligible",
            "decision": "selected_clip",
        }
        payload[field] = bad
        with pytest.raises(ValidationError):
            TypeAdapter(ScannedCandidateTrace).validate_python(payload)

    def test_trace_rejects_unknown_keys(self) -> None:
        payload = {
            "global_rank": 1,
            "asset_id": "clip_001",
            "total": "1.0",
            "action": "1.0",
            "duration": "1.0",
            "content_gate": "passed",
            "frame_gate": "clip_eligible",
            "decision": "selected_clip",
            "source_text": "秘密",
        }
        with pytest.raises(ValidationError):
            TypeAdapter(ScannedCandidateTrace).validate_python(payload)

    def test_frame_gate_requires_passed_content_gate(self) -> None:
        payload = {
            "global_rank": 1,
            "asset_id": "clip_001",
            "total": "1.0",
            "action": "1.0",
            "duration": "1.0",
            "content_gate": "failed_character",
            "frame_gate": "clip_eligible",
            "decision": "skipped_character_gate",
        }
        with pytest.raises(ValidationError):
            TypeAdapter(ScannedCandidateTrace).validate_python(payload)

    def test_selection_trace_stop_reason_closed(self) -> None:
        final = FinalDecisionTrace(
            selected_asset_id=None,
            selected_global_rank=None,
            selected_strategy="placeholder",
            reason_code="no_candidate",
            source_in_frame=0,
            source_frame_count=0,
            target_frames=72,
        )
        with pytest.raises(ValidationError):
            SelectionTrace(
                scanned_candidates=[],
                stop_reason="because i felt like it",
                freeze_fallback_asset_id=None,
                final_decision=final,
            )

    def test_placeholder_final_decision_invariants(self) -> None:
        with pytest.raises(ValidationError):
            FinalDecisionTrace(
                selected_asset_id=None,
                selected_global_rank=None,
                selected_strategy="placeholder",
                reason_code="center_trim",
                source_in_frame=0,
                source_frame_count=0,
                target_frames=72,
            )

    def test_selection_trace_freeze_fallback_must_match_entry(self) -> None:
        entry = ScannedCandidateTrace(
            global_rank=1,
            asset_id="clip_001",
            total="1.0",
            action="1.0",
            duration="1.0",
            content_gate="passed",
            frame_gate="freeze_eligible",
            decision="saved_freeze_fallback",
        )
        final = FinalDecisionTrace(
            selected_asset_id=None,
            selected_global_rank=None,
            selected_strategy="placeholder",
            reason_code="no_candidate",
            source_in_frame=0,
            source_frame_count=0,
            target_frames=72,
        )
        with pytest.raises(ValidationError):
            SelectionTrace(
                scanned_candidates=[entry],
                stop_reason="exhausted_candidates",
                freeze_fallback_asset_id="clip_other",
                final_decision=final,
            )


class TestTraceScenarios:
    def test_selected_clip_trace(self) -> None:
        clip = _probed("clip_001", 96)
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        assert len(trace["scanned_candidates"]) == 1
        entry = trace["scanned_candidates"][0]
        assert entry["global_rank"] == 1
        assert entry["asset_id"] == "clip_001"
        assert entry["content_gate"] == "passed"
        assert entry["frame_gate"] == "clip_eligible"
        assert entry["decision"] == "selected_clip"
        assert trace["stop_reason"] == "selected_clip"
        assert trace["freeze_fallback_asset_id"] is None
        final = trace["final_decision"]
        assert final["selected_asset_id"] == "clip_001"
        assert final["selected_global_rank"] == 1
        assert final["selected_strategy"] == "clip"
        assert final["reason_code"] == "center_trim"
        assert final["source_in_frame"] == 12
        assert final["source_frame_count"] == 72
        assert final["target_frames"] == 72

    def test_freeze_fallback_saved_trace(self) -> None:
        clip = _probed("clip_short", 30)
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        assert trace["scanned_candidates"][0]["decision"] == (
            "saved_freeze_fallback"
        )
        assert trace["scanned_candidates"][0]["frame_gate"] == "freeze_eligible"
        assert trace["freeze_fallback_asset_id"] == "clip_short"
        assert trace["stop_reason"] == "exhausted_candidates"
        final = trace["final_decision"]
        assert final["selected_strategy"] == "freeze_frame"
        assert final["reason_code"] == "short_source_freeze"
        assert final["source_frame_count"] == 30

    def test_freeze_fallback_then_clip_trace(self) -> None:
        short = _probed(
            "clip_short",
            60,
            description="林夏站在学校天台。",
        )
        full = _probed(
            "clip_full",
            96,
            description="林夏独自站在学校天台。",
        )
        selections, audit = retrieve([_requirement()], [short, full])
        trace = _trace(audit)
        assert [entry["decision"] for entry in trace["scanned_candidates"]] == [
            "saved_freeze_fallback",
            "selected_clip",
        ]
        assert trace["freeze_fallback_asset_id"] == "clip_short"
        assert trace["stop_reason"] == "selected_clip"
        assert trace["final_decision"]["selected_asset_id"] == "clip_full"
        assert selections["shot_001"].asset.asset.id == "clip_full"

    def test_multiple_freeze_only_first_saved(self) -> None:
        low = _probed("clip_low", 30)
        high = _probed("clip_high", 60)
        _selections, audit = retrieve([_requirement()], [low, high])
        trace = _trace(audit)
        assert [
            (entry["asset_id"], entry["decision"])
            for entry in trace["scanned_candidates"]
        ] == [
            ("clip_high", "saved_freeze_fallback"),
            ("clip_low", "freeze_eligible_not_saved"),
        ]
        assert trace["freeze_fallback_asset_id"] == "clip_high"
        assert trace["final_decision"]["selected_asset_id"] == "clip_high"

    def test_too_short_trace(self) -> None:
        clip = _probed("clip_23", 23)
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        entry = trace["scanned_candidates"][0]
        assert entry["content_gate"] == "passed"
        assert entry["frame_gate"] == "too_short"
        assert entry["decision"] == "too_short"
        assert trace["stop_reason"] == "exhausted_candidates"
        assert trace["final_decision"]["selected_strategy"] == "placeholder"

    def test_character_gate_failure_trace(self) -> None:
        clip = _probed(
            "clip_other",
            96,
            characters=[{"id": "char_lu_chen", "name": "陆辰"}],
            location_id="loc_school_rooftop",
            location_name="学校天台",
            action="独自站立",
            description="林夏独自站在学校天台。",
        )
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        entry = trace["scanned_candidates"][0]
        assert entry["content_gate"] == "failed_character"
        assert entry["frame_gate"] is None
        assert entry["decision"] == "skipped_character_gate"
        assert trace["stop_reason"] == "exhausted_candidates"

    def test_action_gate_failure_trace(self) -> None:
        clip = _probed(
            "clip_action",
            72,
            action="骑车",
            description="路人骑车经过街道。",
        )
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        entry = trace["scanned_candidates"][0]
        assert entry["content_gate"] == "failed_action"
        assert entry["frame_gate"] is None
        assert entry["decision"] == "skipped_action_gate"

    def test_total_early_stop_trace(self) -> None:
        clip = _probed(
            "clip_unrelated",
            96,
            characters=[{"id": "char_other", "name": "路人"}],
            location_id="loc_street",
            location_name="街道",
            action="骑车",
            description="路人在街道上骑车。",
        )
        _selections, audit = retrieve([_requirement()], [clip])
        trace = _trace(audit)
        entry = trace["scanned_candidates"][0]
        assert entry["content_gate"] is None
        assert entry["frame_gate"] is None
        assert entry["decision"] == "stop_total_below_threshold"
        assert trace["stop_reason"] == "total_below_threshold"

    def test_exhausted_candidates_trace(self) -> None:
        first = _probed("clip_a", 23)
        second = _probed("clip_b", 20)
        _selections, audit = retrieve([_requirement()], [first, second])
        trace = _trace(audit)
        assert len(trace["scanned_candidates"]) == 2
        assert trace["stop_reason"] == "exhausted_candidates"
        assert trace["final_decision"]["selected_strategy"] == "placeholder"

    def test_placeholder_final_decision_shape(self) -> None:
        clip = _probed("clip_23", 23)
        _selections, audit = retrieve([_requirement()], [clip])
        final = _trace(audit)["final_decision"]
        assert final == {
            "selected_asset_id": None,
            "selected_global_rank": None,
            "selected_strategy": "placeholder",
            "reason_code": "no_candidate",
            "source_in_frame": 0,
            "source_frame_count": 0,
            "target_frames": 72,
        }

    def test_final_decision_matches_selection(self) -> None:
        clip = _probed("clip_001", 96)
        selections, audit = retrieve([_requirement()], [clip])
        selected = selections["shot_001"]
        final = _trace(audit)["final_decision"]
        assert final["selected_asset_id"] == selected.asset.asset.id
        assert final["selected_global_rank"] == selected.rank
        assert final["reason_code"] == selected.reason_code
        assert final["source_in_frame"] == selected.source_in_frame
        assert final["source_frame_count"] == selected.source_frame_count

    def test_top_3_independent_from_scanned(self) -> None:
        clips = [_probed(f"clip_{i:03d}", 96) for i in range(5)]
        _selections, audit = retrieve([_requirement()], clips)
        shot = audit["shots"][0]
        assert len(shot["top_3"]) == 3
        assert len(shot["selection_trace"]["scanned_candidates"]) == 1

    def test_rank_five_selected_trace_scans_one_to_five(self) -> None:
        freeze = [_probed(f"clip_f{i:02d}", 70) for i in range(1, 5)]
        full = _probed("clip_full", 96)
        selections, audit = retrieve([_requirement()], [*freeze, full])
        trace = _trace(audit)
        assert [entry["global_rank"] for entry in trace["scanned_candidates"]] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert trace["scanned_candidates"][-1]["decision"] == "selected_clip"
        assert trace["final_decision"]["selected_global_rank"] == 5
        assert selections["shot_001"].rank == 5
        assert len(audit["shots"][0]["top_3"]) == 3

    def test_unscanned_candidates_not_in_trace(self) -> None:
        first = _probed("clip_first", 96)
        second = _probed("clip_second", 96)
        third = _probed("clip_third", 96)
        _selections, audit = retrieve(
            [_requirement()],
            [third, second, first],
        )
        trace = _trace(audit)
        assert len(trace["scanned_candidates"]) == 1
        assert trace["scanned_candidates"][0]["asset_id"] == "clip_first"

    def test_repeated_build_trace_byte_identical(self) -> None:
        clips = [
            _probed("clip_f01", 70),
            _probed("clip_f02", 70),
            _probed("clip_full", 96),
        ]
        _first, audit_first = retrieve([_requirement()], clips)
        _second, audit_second = retrieve([_requirement()], clips)
        dumped_first = json.dumps(
            audit_first,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        dumped_second = json.dumps(
            audit_second,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        assert dumped_first == dumped_second

    def test_trace_contains_no_forbidden_fields(self) -> None:
        short = _probed("clip_short", 60)
        full = _probed("clip_full", 96)
        _selections, audit = retrieve([_requirement()], [short, full])
        trace = _trace(audit)
        forbidden = {
            "source_text",
            "dialogue",
            "description",
            "path",
            "sha256",
            "command",
            "stderr",
            "timestamp",
            "uuid",
        }
        text = json.dumps(trace, ensure_ascii=False)
        for key in forbidden:
            assert f'"{key}"' not in text, key
