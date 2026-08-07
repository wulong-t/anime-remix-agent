"""Unit tests for 3B-2 emotion / shot_scale (AGENTS.md v1.12 section 18.8)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from anime_remix.domain.enums import Emotion, ShotScale
from anime_remix.domain.models import (
    AliasesDocument,
    ClipAsset,
    ClipsDocument,
    ScoreBreakdown,
    ShotRequirement,
    quantize_score,
)
from anime_remix.services.clip_retriever import retrieve
from anime_remix.services.input_loader import canonicalize_character_refs
from anime_remix.services.script_parser import (
    EMOTION_ORDER,
    EMOTION_TERMS,
    SHOT_SCALE_TERMS,
    _extract_semantic,
    extract_emotion,
    extract_shot_scale,
    parse_script,
)


def _clip_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "clip_001",
        "path": "clips/clip_001.mp4",
        "characters": [{"id": "char_lin_xia", "name": "林夏"}],
        "location_id": "loc_school_rooftop",
        "location_name": "学校天台",
        "action": "独自站立",
        "description": "林夏独自站在学校天台。",
    }
    base.update(overrides)
    return base


def _requirement_payload(**overrides: object) -> dict[str, object]:
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
    return base


def _clips_doc() -> ClipsDocument:
    doc = ClipsDocument(
        clips=[
            {
                "id": "clip_001",
                "path": "clips/clip_001.mp4",
                "characters": [{"id": "char_lin_xia", "name": "林夏"}],
                "location_id": "loc_school_rooftop",
                "location_name": "学校天台",
                "action": "独自站立",
                "description": "林夏独自站在学校天台。",
            },
            {
                "id": "clip_002",
                "path": "clips/clip_002.mp4",
                "characters": [{"id": "char_lu_chen", "name": "陆辰"}],
                "location_id": "loc_classroom",
                "location_name": "教室",
                "action": "沉默注视",
                "description": "陆辰在教室沉默注视窗外。",
            },
        ]
    )
    return canonicalize_character_refs(doc)


class TestModels:
    @pytest.mark.parametrize("emotion", list(Emotion))
    def test_clip_asset_accepts_each_emotion(
        self,
        emotion: Emotion,
    ) -> None:
        clip = ClipAsset(**_clip_payload(emotion=emotion))
        assert clip.emotion == emotion

    @pytest.mark.parametrize(
        "bad",
        [
            "neutral",
            "unknown",
            "other",
            "mixed",
            "happy!",
            "",
            1,
            True,
        ],
    )
    def test_clip_asset_rejects_invalid_emotion(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(ClipAsset).validate_python(
                _clip_payload(emotion=bad)
            )

    @pytest.mark.parametrize("shot_scale", list(ShotScale))
    def test_clip_asset_accepts_each_shot_scale(
        self,
        shot_scale: ShotScale,
    ) -> None:
        clip = ClipAsset(**_clip_payload(shot_scale=shot_scale))
        assert clip.shot_scale == shot_scale

    @pytest.mark.parametrize(
        "bad",
        [
            "extreme_close_up",
            "medium_close_up",
            "medium_wide",
            "full_shot",
            "extreme_wide",
            "wide ",
            1,
            True,
        ],
    )
    def test_clip_asset_rejects_invalid_shot_scale(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(ClipAsset).validate_python(
                _clip_payload(shot_scale=bad)
            )

    def test_clip_asset_omitted_fields_are_none(self) -> None:
        clip = ClipAsset(**_clip_payload())
        assert clip.emotion is None
        assert clip.shot_scale is None

    def test_shot_requirement_accepts_and_rejects_emotion(self) -> None:
        req = ShotRequirement(
            **_requirement_payload(
                emotion=Emotion.TENSE,
                shot_scale=ShotScale.CLOSE_UP,
            )
        )
        assert req.emotion == Emotion.TENSE
        assert req.shot_scale == ShotScale.CLOSE_UP
        with pytest.raises(ValidationError):
            TypeAdapter(ShotRequirement).validate_python(
                _requirement_payload(emotion="neutral")
            )
        with pytest.raises(ValidationError):
            TypeAdapter(ShotRequirement).validate_python(
                _requirement_payload(shot_scale="extreme_wide")
            )

    def test_shot_requirement_omitted_fields_are_none(self) -> None:
        req = ShotRequirement(**_requirement_payload())
        assert req.emotion is None
        assert req.shot_scale is None

    def test_score_breakdown_accepts_none_and_decimal(self) -> None:
        score = ScoreBreakdown(
            character=None,
            location=None,
            action=Decimal("1.000000"),
            duration=Decimal("1.000000"),
            emotion=None,
            shot_scale=Decimal("0.000000"),
            active_weights={},
            total=Decimal("1.000000"),
        )
        assert score.emotion is None
        assert score.shot_scale == Decimal("0.000000")

    def test_score_breakdown_rejects_bool_semantic_scores(self) -> None:
        base = {
            "action": "1.0",
            "duration": "1.0",
            "active_weights": {},
            "total": "1.0",
        }
        with pytest.raises(ValidationError):
            TypeAdapter(ScoreBreakdown).validate_python(
                {**base, "emotion": True}
            )
        with pytest.raises(ValidationError):
            TypeAdapter(ScoreBreakdown).validate_python(
                {**base, "shot_scale": True}
            )


class TestParser:
    @pytest.mark.parametrize(
        "emotion,term",
        [
            (emotion, term)
            for emotion, terms in EMOTION_TERMS.items()
            for term in terms
        ],
    )
    def test_each_emotion_keyword_extracted(
        self,
        emotion: Emotion,
        term: str,
    ) -> None:
        assert extract_emotion(f"林夏{term}地站在天台。") == emotion

    @pytest.mark.parametrize(
        "shot_scale,term",
        [
            (shot_scale, term)
            for shot_scale, terms in SHOT_SCALE_TERMS.items()
            for term in terms
        ],
    )
    def test_each_shot_scale_keyword_extracted(
        self,
        shot_scale: ShotScale,
        term: str,
    ) -> None:
        assert extract_shot_scale(f"镜头切为{term}。") == shot_scale

    def test_no_match_returns_none(self) -> None:
        assert extract_emotion("林夏愉快地站在天台。") is None
        assert extract_emotion("林夏欢乐地笑了。") is None
        assert extract_shot_scale("画面构图很舒服。") is None

    def test_multiple_emotions_first_text_hit_wins(self) -> None:
        assert extract_emotion("林夏先惊讶，随后开心地笑了。") == (
            Emotion.SURPRISED
        )

    def test_multiple_shot_scales_first_text_hit_wins(self) -> None:
        assert extract_shot_scale("镜头先给特写，再拉到全景。") == (
            ShotScale.CLOSE_UP
        )

    def test_longer_term_at_same_start_wins(self) -> None:
        terms_by_enum: dict[Emotion, tuple[str, ...]] = {
            Emotion.HAPPY: ("开心呀", "开心"),
            Emotion.SAD: ("难过",),
        }
        assert _extract_semantic("开心呀难过", terms_by_enum, EMOTION_ORDER) == (
            Emotion.HAPPY
        )

    def test_enum_tie_break_is_stable(self) -> None:
        # A synthetic identical term under two enums is resolved by the fixed
        # enum order (happy before sad).
        terms_by_enum: dict[Emotion, tuple[str, ...]] = {
            Emotion.HAPPY: ("焦点",),
            Emotion.SAD: ("焦点",),
        }
        assert _extract_semantic("焦点", terms_by_enum, EMOTION_ORDER) == (
            Emotion.HAPPY
        )
        assert _extract_semantic(
            "焦点",
            terms_by_enum,
            (Emotion.SAD, Emotion.HAPPY),
        ) == Emotion.SAD

    def test_emotion_and_character_alias_work_together(self) -> None:
        aliases = AliasesDocument(
            character_aliases=[
                {"target_id": "char_lin_xia", "aliases": ["小夏"]}
            ],
            location_aliases=[
                {"target_id": "loc_school_rooftop", "aliases": ["楼顶"]}
            ],
        )
        requirements = parse_script(
            "小夏难过地站在学校楼顶，望着远方。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            aliases,
        )
        assert requirements[0].characters[0].id == "char_lin_xia"
        assert requirements[0].location_id == "loc_school_rooftop"
        assert requirements[0].emotion == Emotion.SAD

    def test_shot_scale_and_location_alias_work_together(self) -> None:
        aliases = AliasesDocument(
            character_aliases=[
                {"target_id": "char_lin_xia", "aliases": ["小夏"]}
            ],
            location_aliases=[
                {"target_id": "loc_school_rooftop", "aliases": ["楼顶"]}
            ],
        )
        requirements = parse_script(
            "小夏在楼顶紧张地看向远处，镜头切为特写。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            aliases,
        )
        assert requirements[0].characters[0].id == "char_lin_xia"
        assert requirements[0].location_id == "loc_school_rooftop"
        assert requirements[0].emotion == Emotion.TENSE
        assert requirements[0].shot_scale == ShotScale.CLOSE_UP

    def test_parser_uses_no_fuzzy_matching(self) -> None:
        # Synonyms outside the fixed dictionary must never be guessed.
        assert extract_emotion("林夏愉快地站在天台。") is None
        assert extract_emotion("林夏欢乐地笑了。") is None
        assert extract_shot_scale("镜头构图很舒服。") is None


class TestScoring:
    def _probed(
        self,
        clip_id: str,
        frames: int = 96,
        **overrides: object,
    ) -> object:
        from anime_remix.domain.models import ProbedClip

        payload = _clip_payload(id=clip_id, path=f"clips/{clip_id}.mp4")
        for key, value in overrides.items():
            payload[key] = value
        asset = ClipAsset(**payload)
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

    def _requirement(self, **overrides: object) -> ShotRequirement:
        payload = _requirement_payload()
        payload.update(overrides)
        return ShotRequirement(**payload)

    def _score(self, requirement: ShotRequirement, clip: object):
        selections, _ = retrieve([requirement], [clip])
        return selections["shot_001"].score

    def test_requirement_emotion_none_gives_null_score(self) -> None:
        clip = self._probed("clip_001", emotion="calm")
        score = self._score(self._requirement(), clip)
        assert score is not None
        assert score.emotion is None
        assert "emotion" not in score.active_weights

    def test_emotion_exact_match_gives_one(self) -> None:
        clip = self._probed("clip_001", emotion="sad")
        score = self._score(
            self._requirement(emotion="sad"),
            clip,
        )
        assert score is not None
        assert score.emotion == Decimal("1.000000")

    def test_emotion_mismatch_gives_zero(self) -> None:
        clip = self._probed("clip_001", emotion="calm")
        score = self._score(
            self._requirement(emotion="sad"),
            clip,
        )
        assert score is not None
        assert score.emotion == Decimal("0.000000")

    def test_asset_emotion_none_gives_zero(self) -> None:
        clip = self._probed("clip_001", emotion=None)
        score = self._score(
            self._requirement(emotion="sad"),
            clip,
        )
        assert score is not None
        assert score.emotion == Decimal("0.000000")

    def test_shot_scale_symmetric_four_cases(self) -> None:
        # requirement None -> score None
        clip = self._probed("clip_001", shot_scale="wide")
        score = self._score(self._requirement(), clip)
        assert score is not None
        assert score.shot_scale is None
        assert "shot_scale" not in score.active_weights

        # exact -> 1
        score = self._score(
            self._requirement(shot_scale="wide"),
            self._probed("clip_001", shot_scale="wide"),
        )
        assert score is not None
        assert score.shot_scale == Decimal("1.000000")

        # mismatch -> 0
        score = self._score(
            self._requirement(shot_scale="wide"),
            self._probed("clip_001", shot_scale="medium"),
        )
        assert score is not None
        assert score.shot_scale == Decimal("0.000000")

        # asset None -> 0
        score = self._score(
            self._requirement(shot_scale="wide"),
            self._probed("clip_001", shot_scale=None),
        )
        assert score is not None
        assert score.shot_scale == Decimal("0.000000")

    def test_both_inactive_recovers_old_four_weights(self) -> None:
        clip = self._probed("clip_001")
        score = self._score(self._requirement(), clip)
        assert score is not None
        assert score.active_weights == {
            "action": Decimal("0.450000"),
            "character": Decimal("0.250000"),
            "duration": Decimal("0.150000"),
            "location": Decimal("0.150000"),
        }

    def test_only_emotion_active_normalizes_correctly(self) -> None:
        clip = self._probed("clip_001", emotion="sad")
        score = self._score(
            self._requirement(emotion="sad"),
            clip,
        )
        assert score is not None
        assert score.active_weights == {
            "action": Decimal("0.400000"),
            "character": Decimal("0.222222"),
            "duration": Decimal("0.133333"),
            "emotion": Decimal("0.111111"),
            "location": Decimal("0.133333"),
        }

    def test_only_shot_scale_active_normalizes_correctly(self) -> None:
        clip = self._probed("clip_001", shot_scale="wide")
        score = self._score(
            self._requirement(shot_scale="wide"),
            clip,
        )
        assert score is not None
        assert score.active_weights == {
            "action": Decimal("0.400000"),
            "character": Decimal("0.222222"),
            "duration": Decimal("0.133333"),
            "location": Decimal("0.133333"),
            "shot_scale": Decimal("0.111111"),
        }

    def test_both_active_uses_base_weights(self) -> None:
        clip = self._probed("clip_001", emotion="sad", shot_scale="wide")
        score = self._score(
            self._requirement(emotion="sad", shot_scale="wide"),
            clip,
        )
        assert score is not None
        assert score.active_weights == {
            "action": Decimal("0.360000"),
            "character": Decimal("0.200000"),
            "duration": Decimal("0.120000"),
            "emotion": Decimal("0.100000"),
            "location": Decimal("0.120000"),
            "shot_scale": Decimal("0.100000"),
        }

    def test_decimal_half_up_still_applies(self) -> None:
        assert quantize_score(Decimal("0.1234565")) == Decimal("0.123457")
        assert quantize_score(Decimal("0.1234564")) == Decimal("0.123456")

    def test_active_weights_keys_are_stable(self) -> None:
        first = self._score(
            self._requirement(emotion="sad", shot_scale="wide"),
            self._probed("clip_001", emotion="sad", shot_scale="wide"),
        )
        second = self._score(
            self._requirement(emotion="sad", shot_scale="wide"),
            self._probed("clip_001", emotion="sad", shot_scale="wide"),
        )
        assert first is not None and second is not None
        assert list(first.active_weights) == list(second.active_weights)
        assert list(first.active_weights) == sorted(first.active_weights)


class TestRetrieval:
    def _probed(
        self,
        clip_id: str,
        frames: int = 96,
        **overrides: object,
    ) -> object:
        from anime_remix.domain.models import ProbedClip

        payload = _clip_payload(id=clip_id, path=f"clips/{clip_id}.mp4")
        payload.update(overrides)
        return ProbedClip(
            asset=ClipAsset(**payload),
            resolved_path=Path(f"clips/{clip_id}.mp4").resolve(),
            size_bytes=1000,
            width=1280,
            height=720,
            fps_num=24,
            fps_den=1,
            nb_frames=frames,
            duration_seconds=Decimal(frames) / Decimal(24),
        )

    def _requirement(self, **overrides: object) -> ShotRequirement:
        payload = _requirement_payload()
        payload.update(overrides)
        return ShotRequirement(**payload)

    def test_emotion_exact_match_wins_over_tied_candidate(self) -> None:
        exact = self._probed("clip_sad", emotion="sad", shot_scale="medium")
        mismatch = self._probed(
            "clip_calm",
            emotion="calm",
            shot_scale="medium",
        )
        selections, audit = retrieve(
            [self._requirement(emotion="sad")],
            [exact, mismatch],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_sad"
        assert audit["shots"][0]["top_3"][0]["asset_id"] == "clip_sad"

    def test_shot_scale_exact_match_wins_over_tied_candidate(self) -> None:
        wide = self._probed(
            "clip_wide",
            emotion="calm",
            shot_scale="wide",
        )
        medium = self._probed(
            "clip_medium",
            emotion="calm",
            shot_scale="medium",
        )
        selections, audit = retrieve(
            [self._requirement(shot_scale="wide")],
            [medium, wide],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_wide"
        assert audit["shots"][0]["top_3"][0]["asset_id"] == "clip_wide"

    def test_emotion_mismatch_is_not_hard_rejected(self) -> None:
        clip = self._probed("clip_calm", emotion="calm")
        selections, audit = retrieve(
            [self._requirement(emotion="sad")],
            [clip],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_calm"
        assert selected.reason_code == "center_trim"
        assert selected.score is not None
        assert selected.score.emotion == Decimal("0.000000")
        assert audit["shots"][0]["checked_gates"][0]["gates"]["total"] is True

    def test_shot_scale_mismatch_is_not_hard_rejected(self) -> None:
        clip = self._probed("clip_medium", shot_scale="medium")
        selections, _ = retrieve(
            [self._requirement(shot_scale="wide")],
            [clip],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_medium"
        assert selected.score is not None
        assert selected.score.shot_scale == Decimal("0.000000")

    def test_stable_tie_break_unchanged(self) -> None:
        first = self._probed(
            "clip_a",
            emotion="sad",
            shot_scale="wide",
        )
        second = self._probed(
            "clip_b",
            emotion="sad",
            shot_scale="wide",
        )
        selections, audit = retrieve(
            [self._requirement(emotion="sad", shot_scale="wide")],
            [second, first],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_a"
        assert audit["shots"][0]["top_3"][0]["asset_id"] == "clip_a"

    def test_clip_priority_still_beats_higher_rank_freeze(self) -> None:
        short_exact = self._probed(
            "clip_short",
            frames=30,
            emotion="sad",
        )
        full_mismatch = self._probed(
            "clip_full",
            frames=96,
            emotion="calm",
        )
        selections, audit = retrieve(
            [self._requirement(emotion="sad")],
            [short_exact, full_mismatch],
        )
        selected = selections["shot_001"]
        assert selected.asset is not None
        assert selected.asset.asset.id == "clip_full"
        assert selected.reason_code == "center_trim"
        shot = audit["shots"][0]
        assert shot["checked_gates"][0]["frame_gate"] == "freeze_eligible"
        assert shot["selected"]["selected_strategy"] == "clip"
