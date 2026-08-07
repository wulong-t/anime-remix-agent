"""Hand-authored retrieval quality cases for Phase B (AGENTS.md v1.14).

Every expected value is written by a human test author against the locked
first-phase contract. Nothing here is snapshotted from the current retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from anime_remix.domain.models import (
    AliasesDocument,
    CharacterRef,
    ClipAsset,
    ClipsDocument,
    ProbedClip,
    ShotRequirement,
)
from anime_remix.services.script_parser import parse_script

LX = ("char_lin_xia", "林夏")
LC = ("char_lu_chen", "陆辰")
PR = ("char_passerby", "路人")

ROOF = ("loc_school_rooftop", "学校天台")
CLASS = ("loc_classroom", "教室")
GARDEN = ("loc_garden", "花园")
STREET = ("loc_street", "街道")


def _clip(
    clip_id: str,
    frames: int,
    *,
    characters: list[tuple[str, str]] | None = None,
    location_id: str | None = None,
    location_name: str | None = None,
    action: str,
    description: str,
    emotion: str | None = None,
    shot_scale: str | None = None,
) -> ProbedClip:
    refs = [
        CharacterRef(id=char_id, name=char_name)
        for char_id, char_name in (characters or [])
    ]
    asset = ClipAsset(
        id=clip_id,
        path=f"clips/{clip_id}.mp4",
        characters=refs,
        location_id=location_id,
        location_name=location_name,
        action=action,
        description=description,
        emotion=emotion,
        shot_scale=shot_scale,
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


def _req(
    shot_id: str,
    source_text: str,
    *,
    characters: list[tuple[str, str]] | None = None,
    location_id: str | None = None,
    location_name: str | None = None,
    action: str,
    target_frames: int = 72,
    emotion: str | None = None,
    shot_scale: str | None = None,
) -> ShotRequirement:
    refs = [
        CharacterRef(id=char_id, name=char_name)
        for char_id, char_name in (characters or [])
    ]
    return ShotRequirement(
        id=shot_id,
        order=1,
        source_text=source_text,
        characters=refs,
        location_id=location_id,
        location_name=location_name,
        action=action,
        target_frames=target_frames,
        dialogue=None,
        emotion=emotion,
        shot_scale=shot_scale,
    )


def _aliased_requirement(
    script: str,
    clips: list[ProbedClip],
) -> ShotRequirement:
    """Parse the first paragraph through the formal parser with aliases."""

    doc = ClipsDocument(clips=[clip.asset for clip in clips])
    aliases = AliasesDocument(
        character_aliases=[
            {"target_id": "char_lin_xia", "aliases": ["小夏"]},
            {"target_id": "char_lu_chen", "aliases": ["阿辰"]},
        ],
        location_aliases=[
            {"target_id": "loc_school_rooftop", "aliases": ["楼顶", "天台"]},
            {"target_id": "loc_classroom", "aliases": ["课堂"]},
        ],
    )
    return parse_script(script, doc, aliases)[0]


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    requirement: ShotRequirement
    candidates: tuple[ProbedClip, ...]
    expected_selected_asset_id: str | None
    expected_strategy: str
    expected_reason_code: str
    expected_source_in_frame: int | None = None
    expected_source_frame_count: int | None = None
    expected_global_rank: int | None = None
    expected_freeze_fallback_asset_id: str | None = None
    expected_stop_reason: str | None = None
    case_tags: tuple[str, ...] = field(default_factory=tuple)
    expected_rationale: str = ""


def build_cases() -> list[QualityCase]:
    """Return the 30 hand-authored quality cases (first version, <= 50)."""

    cases: list[QualityCase] = []

    cases.append(
        QualityCase(
            case_id="q01_single_person_id_exact",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="沉默注视",
                    description="陆辰在教室沉默注视窗外。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="单人物 ID 精确匹配，clip_a 人物/地点/动作全命中。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q02_multi_person_recall",
            requirement=_req(
                "shot_001",
                "林夏和陆辰站在学校天台。",
                characters=[LX, LC],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_ab",
                    96,
                    characters=[LX, LC],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏和陆辰站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_ab",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="两个人物都在 clip_ab，recall=1，F2 高于只含一人的 clip_a。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q03_precision_penalty",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_ab",
                    96,
                    characters=[LX, LC],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="clip_ab 多带一个人物导致 precision=0.5，F2 被惩罚。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q04_same_name_distinct_ids_no_merge",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_other",
                    96,
                    characters=[("char_other", "林夏")],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="双方都有非空 ID 时只按 ID 匹配，同名不同 ID 不合并。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q05_name_only_fallback",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[("", "林夏")],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="陆辰独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="需求无 ID 时允许规范化 name 精确匹配，命中林夏。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q06_no_person_requirement",
            requirement=_req(
                "shot_001",
                "独自站在学校天台。",
                characters=[],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[PR],
                    location_id=STREET[0],
                    location_name=STREET[1],
                    action="骑车",
                    description="路人在街道上骑车。",
                ),
                _clip(
                    "clip_c",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_c",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("character",),
            expected_rationale="需求未指定人物时 character 维度 inactive，由地点和动作决定。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q07_location_id_exact",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="独自站立",
                    description="林夏在教室独自站立。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("location",),
            expected_rationale="location ID 精确匹配优先，clip_a 得 1。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q08_location_name_similarity",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=None,
                location_name="学校天台",
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=None,
                    location_name="学校天台",
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=None,
                    location_name="教室",
                    action="独自站立",
                    description="林夏在教室独自站立。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("location",),
            expected_rationale="无 location_id 时使用 location_name 文本相似度，clip_a 全等。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q09_no_location_requirement",
            requirement=_req(
                "shot_001",
                "林夏独自站立。",
                characters=[LX],
                location_id=None,
                location_name=None,
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="独自站立",
                    description="林夏在教室独自站立。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("location",),
            expected_rationale="需求无地点时 location 维度 inactive，其余相同则 asset_id 升序平局。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q10_action_exact",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="沉默注视",
                    description="陆辰在教室沉默注视窗外。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("action",),
            expected_rationale="action 精确匹配得 1.0。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q11_action_description_helper",
            requirement=_req(
                "shot_001",
                "陆辰在教室沉默注视窗外。",
                characters=[LC],
                location_id=CLASS[0],
                location_name=CLASS[1],
                action="沉默注视",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="安静",
                    description="陆辰在教室沉默注视窗外。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="看书",
                    description="陆辰在教室看书。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("action",),
            expected_rationale="clip.description 含动作语义，0.9×similarity 辅助匹配胜出。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q12_source_text_description_helper",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台，望着远方。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="站着",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="看书",
                    description="陆辰在教室看书。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("action",),
            expected_rationale="0.8×similarity(source_text, description) 为 clip_a 提供辅助分。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q13_action_mismatch",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="骑车",
                    description="路人在街道上骑车。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="看书",
                    description="陆辰在教室看书。",
                ),
            ),
            expected_selected_asset_id=None,
            expected_strategy="placeholder",
            expected_reason_code="no_candidate",
            expected_source_in_frame=0,
            expected_source_frame_count=0,
            expected_global_rank=None,
            expected_stop_reason="total_below_threshold",
            case_tags=("action",),
            expected_rationale="动作明显不相关，total < 0.55 提前停止，无候选。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q14_exact_length",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    72,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="exact_length",
            expected_source_in_frame=0,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("strategy",),
            expected_rationale="nb_frames == target_frames 时 exact_length。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q15_center_trim",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("strategy",),
            expected_rationale="nb_frames > target_frames 时中心裁剪 (96-72)//2=12。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q16_freeze_frame_fallback",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_short",
                    30,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_short",
            expected_strategy="freeze_frame",
            expected_reason_code="short_source_freeze",
            expected_source_in_frame=0,
            expected_source_frame_count=30,
            expected_global_rank=1,
            expected_freeze_fallback_asset_id="clip_short",
            expected_stop_reason="exhausted_candidates",
            case_tags=("strategy",),
            expected_rationale="无完整 clip 时使用唯一 freeze fallback。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q17_high_rank_freeze_then_full_clip",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_short",
                    60,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏站在学校天台。",
                ),
                _clip(
                    "clip_full",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_full",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=2,
            expected_freeze_fallback_asset_id="clip_short",
            expected_stop_reason="selected_clip",
            case_tags=("strategy",),
            expected_rationale="高排名 freeze 先被保存，随后完整 clip 出现并胜出。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q18_too_short_then_continue",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_short",
                    23,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_full",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="站着",
                    description="林夏站在学校天台。",
                ),
            ),
            expected_selected_asset_id="clip_full",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=2,
            expected_stop_reason="selected_clip",
            case_tags=("strategy",),
            expected_rationale="23 帧 too_short 后继续扫描，后续完整 clip 被选中。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q19_placeholder",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[PR],
                    location_id=STREET[0],
                    location_name=STREET[1],
                    action="骑车",
                    description="路人在街道上骑车。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="看书",
                    description="陆辰在教室看书。",
                ),
            ),
            expected_selected_asset_id=None,
            expected_strategy="placeholder",
            expected_reason_code="no_candidate",
            expected_source_in_frame=0,
            expected_source_frame_count=0,
            expected_global_rank=None,
            expected_stop_reason="total_below_threshold",
            case_tags=("strategy",),
            expected_rationale="无候选通过，placeholder。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q20_character_alias_canonical",
            requirement=_aliased_requirement(
                "小夏独自站在学校天台，望着远方。\n\n"
                "阿辰在课堂上沉默注视窗外。\n\n"
                "小夏转身离开学校楼顶。",
                [
                    _clip(
                        "clip_a",
                        96,
                        characters=[LX],
                        location_id=ROOF[0],
                        location_name=ROOF[1],
                        action="独自站立",
                        description="林夏独自站在学校天台。",
                    ),
                    _clip(
                        "clip_b",
                        96,
                        characters=[LC],
                        location_id=CLASS[0],
                        location_name=CLASS[1],
                        action="沉默注视",
                        description="陆辰在教室沉默注视窗外。",
                    ),
                ],
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="沉默注视",
                    description="陆辰在教室沉默注视窗外。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("aliases", "character"),
            expected_rationale="人物 alias 解析为 canonical char_lin_xia 后正确检索。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q21_location_alias_canonical",
            requirement=_aliased_requirement(
                "小夏独自站在学校楼顶，望着远方。\n\n"
                "阿辰在课堂上沉默注视窗外。\n\n"
                "小夏转身离开学校楼顶。",
                [
                    _clip(
                        "clip_a",
                        96,
                        characters=[LX],
                        location_id=ROOF[0],
                        location_name=ROOF[1],
                        action="独自站立",
                        description="林夏独自站在学校天台。",
                    ),
                    _clip(
                        "clip_b",
                        96,
                        characters=[LC],
                        location_id=CLASS[0],
                        location_name=CLASS[1],
                        action="沉默注视",
                        description="陆辰在教室沉默注视窗外。",
                    ),
                ],
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LC],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="沉默注视",
                    description="陆辰在教室沉默注视窗外。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("aliases", "location"),
            expected_rationale="地点 alias 解析为 canonical loc_school_rooftop 后正确检索。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q22_emotion_exact_lift",
            requirement=_req(
                "shot_001",
                "林夏难过地站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
                emotion="sad",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="sad",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="calm",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("emotion",),
            expected_rationale="emotion exact match 提升 clip_a 总分。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q23_emotion_mismatch_not_hard_reject",
            requirement=_req(
                "shot_001",
                "林夏难过地站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
                emotion="sad",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="calm",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("emotion",),
            expected_rationale="emotion mismatch 只扣分，不产生独立 hard reject。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q24_asset_emotion_none_score_zero",
            requirement=_req(
                "shot_001",
                "林夏难过地站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
                emotion="sad",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion=None,
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("emotion",),
            expected_rationale="素材 emotion 缺失时 emotion score=0，仍可被选中。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q25_shot_scale_exact",
            requirement=_req(
                "shot_001",
                "林夏站在学校天台，镜头里能看到远景。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
                shot_scale="wide",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    shot_scale="wide",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    shot_scale="medium",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("shot_scale",),
            expected_rationale="shot_scale exact match 提升 clip_a。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q26_shot_scale_mismatch_not_hard_reject",
            requirement=_req(
                "shot_001",
                "林夏站在学校天台，镜头里能看到远景。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
                shot_scale="wide",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    shot_scale="medium",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("shot_scale",),
            expected_rationale="shot_scale mismatch 只影响 total，不 hard reject。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q27_character_location_action",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=CLASS[0],
                    location_name=CLASS[1],
                    action="独自站立",
                    description="林夏在教室独自站立。",
                ),
                _clip(
                    "clip_c",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="沉默注视",
                    description="陆辰在教室沉默注视窗外。",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("combination",),
            expected_rationale="character + location + action 三项组合全命中。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q28_aliases_emotion_shot_scale",
            requirement=_aliased_requirement(
                "小夏难过地站在学校楼顶，镜头里能看到远景。\n\n"
                "阿辰在课堂上沉默注视窗外。\n\n"
                "小夏转身离开学校楼顶。",
                [
                    _clip(
                        "clip_a",
                        96,
                        characters=[LX],
                        location_id=ROOF[0],
                        location_name=ROOF[1],
                        action="独自站立",
                        description="林夏独自站在学校天台。",
                        emotion="sad",
                        shot_scale="wide",
                    ),
                    _clip(
                        "clip_b",
                        96,
                        characters=[LX],
                        location_id=ROOF[0],
                        location_name=ROOF[1],
                        action="独自站立",
                        description="林夏独自站在学校天台。",
                        emotion="calm",
                        shot_scale="wide",
                    ),
                    _clip(
                        "clip_c",
                        96,
                        characters=[LX],
                        location_id=ROOF[0],
                        location_name=ROOF[1],
                        action="独自站立",
                        description="林夏独自站在学校天台。",
                        emotion="sad",
                        shot_scale="medium",
                    ),
                ],
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="sad",
                    shot_scale="wide",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="calm",
                    shot_scale="wide",
                ),
                _clip(
                    "clip_c",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                    emotion="sad",
                    shot_scale="medium",
                ),
            ),
            expected_selected_asset_id="clip_a",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=1,
            expected_stop_reason="selected_clip",
            case_tags=("combination", "aliases", "emotion", "shot_scale"),
            expected_rationale="alias 解析 canonical 后，emotion 与 shot_scale 双 exact 命中 clip_a。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q29_high_total_but_character_gate_fail",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    72,
                    characters=[LC],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="独自站立",
                    description="林夏独自站在学校天台。",
                ),
                _clip(
                    "clip_b",
                    96,
                    characters=[LX],
                    location_id=ROOF[0],
                    location_name=ROOF[1],
                    action="站着",
                    description="林夏站着。",
                ),
            ),
            expected_selected_asset_id="clip_b",
            expected_strategy="clip",
            expected_reason_code="center_trim",
            expected_source_in_frame=12,
            expected_source_frame_count=72,
            expected_global_rank=2,
            expected_stop_reason="selected_clip",
            case_tags=("combination",),
            expected_rationale="clip_a 总分高但人物门槛失败被跳过，clip_b 随后被选中。",
        )
    )

    cases.append(
        QualityCase(
            case_id="q30_total_below_threshold_early_stop",
            requirement=_req(
                "shot_001",
                "林夏独自站在学校天台。",
                characters=[LX],
                location_id=ROOF[0],
                location_name=ROOF[1],
                action="独自站立",
            ),
            candidates=(
                _clip(
                    "clip_a",
                    96,
                    characters=[PR],
                    location_id=STREET[0],
                    location_name=STREET[1],
                    action="骑车",
                    description="路人在街道上骑车。",
                ),
            ),
            expected_selected_asset_id=None,
            expected_strategy="placeholder",
            expected_reason_code="no_candidate",
            expected_source_in_frame=0,
            expected_source_frame_count=0,
            expected_global_rank=None,
            expected_stop_reason="total_below_threshold",
            case_tags=("combination",),
            expected_rationale="唯一候选 total < 0.55，触发提前停止，最终 placeholder。",
        )
    )

    return cases
