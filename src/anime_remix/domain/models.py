"""Strict domain models for Anime Remix Agent v1.9."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from anime_remix.domain.enums import Emotion, ShotScale, TimelineStrategy

STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

SCORE_QUANTUM = Decimal("0.000001")
ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ReasonCode = Literal[
    "exact_length",
    "center_trim",
    "short_source_freeze",
    "no_candidate",
]

ContentGateValue = Literal[
    "passed",
    "failed_character",
    "failed_action",
]
FrameGateValue = Literal[
    "clip_eligible",
    "freeze_eligible",
    "too_short",
]
CandidateDecision = Literal[
    "selected_clip",
    "saved_freeze_fallback",
    "freeze_eligible_not_saved",
    "too_short",
    "skipped_character_gate",
    "skipped_action_gate",
    "stop_total_below_threshold",
]
StopReasonValue = Literal[
    "selected_clip",
    "total_below_threshold",
    "exhausted_candidates",
]
SelectedStrategyValue = Literal[
    "clip",
    "freeze_frame",
    "placeholder",
]


def quantize_score(value: Decimal) -> Decimal:
    """Quantize a score to SCORE_QUANTUM using Decimal HALF_UP."""

    return value.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP)


def _coerce_decimal(value: object) -> object:
    if isinstance(value, str):
        try:
            return Decimal(value)
        except ArithmeticError:
            return value
    return value


class CharacterRef(BaseModel):
    model_config = STRICT_CONFIG

    id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _require_id_or_name(self) -> CharacterRef:
        if not (self.id and self.id.strip()) and not (self.name and self.name.strip()):
            raise ValueError("CharacterRef requires at least one of id or name")
        return self


class ClipAsset(BaseModel):
    model_config = STRICT_CONFIG

    id: str
    path: Path
    characters: list[CharacterRef] = Field(default_factory=list)
    location_id: str | None = None
    location_name: str | None = None
    action: str
    description: str
    emotion: Emotion | None = None
    shot_scale: ShotScale | None = None

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value)
        return value

    @field_validator("emotion", mode="before")
    @classmethod
    def _coerce_emotion(cls, value: object) -> object:
        if isinstance(value, str):
            return Emotion(value)
        return value

    @field_validator("shot_scale", mode="before")
    @classmethod
    def _coerce_shot_scale(cls, value: object) -> object:
        if isinstance(value, str):
            return ShotScale(value)
        return value

    @model_validator(mode="after")
    def _validate_id_and_text(self) -> ClipAsset:
        if not ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"invalid clip id {self.id!r}")
        if not self.action.strip():
            raise ValueError("action must be non-empty")
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        return self


class ClipsDocument(BaseModel):
    model_config = STRICT_CONFIG

    schema_version: Literal["1.9"] = "1.9"
    clips: list[ClipAsset]

    @model_validator(mode="after")
    def _validate_clips(self) -> ClipsDocument:
        if len(self.clips) > 50:
            raise ValueError(f"clips must be at most 50, got {len(self.clips)}")
        ids = [clip.id for clip in self.clips]
        if len(ids) != len(set(ids)):
            raise ValueError("clip ids must be globally unique")
        return self


class AliasEntry(BaseModel):
    """One aliases.json entry: a canonical target plus its alias strings."""

    model_config = STRICT_CONFIG

    target_id: str
    aliases: list[str]

    @model_validator(mode="after")
    def _validate_alias_entry(self) -> AliasEntry:
        if not self.target_id.strip():
            raise ValueError("target_id must be non-empty after stripping")
        if not self.aliases:
            raise ValueError("aliases must be a non-empty list")
        if len(self.aliases) > 32:
            raise ValueError(
                f"aliases must be at most 32 per target, got {len(self.aliases)}"
            )
        for alias in self.aliases:
            if not alias.strip():
                raise ValueError("each alias must be non-empty after stripping")
            if len(alias) > 128:
                raise ValueError(
                    f"alias exceeds 128 Unicode code points: {alias!r}"
                )
        return self


class AliasesDocument(BaseModel):
    """Strict aliases.json document (AGENTS.md v1.11 section 7.5)."""

    model_config = STRICT_CONFIG

    schema_version: Literal["1.9"] = "1.9"
    character_aliases: list[AliasEntry] = Field(default_factory=list)
    location_aliases: list[AliasEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_limits(self) -> AliasesDocument:
        if len(self.character_aliases) > 200:
            raise ValueError(
                "character_aliases must be at most 200 entries, "
                f"got {len(self.character_aliases)}"
            )
        if len(self.location_aliases) > 200:
            raise ValueError(
                "location_aliases must be at most 200 entries, "
                f"got {len(self.location_aliases)}"
            )
        return self


class ProbedClip(BaseModel):
    model_config = STRICT_CONFIG

    asset: ClipAsset
    resolved_path: Path
    size_bytes: int
    width: int
    height: int
    fps_num: int
    fps_den: int
    nb_frames: int
    duration_seconds: Decimal
    assumed_color_metadata: bool = False

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _coerce_duration(cls, value: object) -> object:
        return _coerce_decimal(value)


class ShotRequirement(BaseModel):
    model_config = STRICT_CONFIG

    id: str
    order: int = Field(ge=1)
    source_text: str
    characters: list[CharacterRef] = Field(default_factory=list)
    location_id: str | None = None
    location_name: str | None = None
    action: str
    target_frames: int = Field(gt=0)
    dialogue: str | None = None
    emotion: Emotion | None = None
    shot_scale: ShotScale | None = None

    @field_validator("emotion", mode="before")
    @classmethod
    def _coerce_emotion(cls, value: object) -> object:
        if isinstance(value, str):
            return Emotion(value)
        return value

    @field_validator("shot_scale", mode="before")
    @classmethod
    def _coerce_shot_scale(cls, value: object) -> object:
        if isinstance(value, str):
            return ShotScale(value)
        return value

    @model_validator(mode="after")
    def _validate_id(self) -> ShotRequirement:
        if not ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"invalid shot id {self.id!r}")
        if not self.source_text.strip():
            raise ValueError("source_text must be non-empty")
        if not self.action.strip():
            raise ValueError("action must be non-empty")
        return self


class ScoreBreakdown(BaseModel):
    model_config = STRICT_CONFIG

    character: Decimal | None = None
    location: Decimal | None = None
    action: Decimal
    duration: Decimal
    emotion: Decimal | None = None
    shot_scale: Decimal | None = None
    active_weights: dict[str, Decimal]
    total: Decimal

    @field_validator(
        "character",
        "location",
        "action",
        "duration",
        "emotion",
        "shot_scale",
        "total",
        mode="before",
    )
    @classmethod
    def _coerce_score_fields(cls, value: object) -> object:
        return _coerce_decimal(value)

    @field_validator("active_weights", mode="before")
    @classmethod
    def _coerce_active_weights(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: _coerce_decimal(item) for key, item in value.items()
            }
        return value


class ScannedCandidateTrace(BaseModel):
    """One actually scanned candidate inside selection_trace (AGENTS v1.13)."""

    model_config = STRICT_CONFIG

    global_rank: int = Field(ge=1)
    asset_id: str
    total: Decimal
    character: Decimal | None = None
    location: Decimal | None = None
    action: Decimal
    duration: Decimal
    emotion: Decimal | None = None
    shot_scale: Decimal | None = None
    content_gate: ContentGateValue | None = None
    frame_gate: FrameGateValue | None = None
    decision: CandidateDecision

    @field_validator(
        "total",
        "character",
        "location",
        "action",
        "duration",
        "emotion",
        "shot_scale",
        mode="before",
    )
    @classmethod
    def _coerce_score_fields(cls, value: object) -> object:
        return _coerce_decimal(value)

    @model_validator(mode="after")
    def _validate_fields(self) -> ScannedCandidateTrace:
        if not ID_PATTERN.fullmatch(self.asset_id):
            raise ValueError(f"invalid asset_id {self.asset_id!r}")
        if self.content_gate is None and self.frame_gate is not None:
            raise ValueError("frame_gate requires content_gate passed")
        if self.content_gate == "passed" and self.frame_gate is None:
            raise ValueError("passed content_gate requires a frame_gate")
        if self.content_gate in ("failed_character", "failed_action"):
            if self.frame_gate is not None:
                raise ValueError("failed content_gate must not carry frame_gate")
            if self.decision not in (
                "skipped_character_gate",
                "skipped_action_gate",
            ):
                raise ValueError(
                    "failed content_gate requires skipped decision"
                )
        if self.content_gate is None and self.decision != (
            "stop_total_below_threshold"
        ):
            raise ValueError(
                "null content_gate requires stop_total_below_threshold"
            )
        return self


class FinalDecisionTrace(BaseModel):
    """final_decision mirrors the final Selection and timeline item."""

    model_config = STRICT_CONFIG

    selected_asset_id: str | None = None
    selected_global_rank: int | None = None
    selected_strategy: SelectedStrategyValue
    reason_code: ReasonCode
    source_in_frame: int = Field(ge=0)
    source_frame_count: int = Field(ge=0)
    target_frames: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_decision(self) -> FinalDecisionTrace:
        if self.selected_asset_id is None:
            if self.selected_global_rank is not None:
                raise ValueError(
                    "placeholder final decision must not have a rank"
                )
            if (
                self.selected_strategy != "placeholder"
                or self.reason_code != "no_candidate"
                or self.source_in_frame != 0
                or self.source_frame_count != 0
            ):
                raise ValueError(
                    "placeholder final decision must be "
                    "placeholder/no_candidate/0/0"
                )
            return self
        if self.selected_global_rank is None or self.selected_global_rank < 1:
            raise ValueError(
                "selected final decision requires positive global rank"
            )
        if not ID_PATTERN.fullmatch(self.selected_asset_id):
            raise ValueError(
                f"invalid selected_asset_id {self.selected_asset_id!r}"
            )
        if self.selected_strategy not in ("clip", "freeze_frame"):
            raise ValueError(
                "selected source strategy must be clip or freeze_frame"
            )
        return self


class SelectionTrace(BaseModel):
    """Formal deterministic selection trace for one shot (AGENTS v1.13)."""

    model_config = STRICT_CONFIG

    scanned_candidates: list[ScannedCandidateTrace] = Field(
        default_factory=list
    )
    stop_reason: StopReasonValue
    freeze_fallback_asset_id: str | None = None
    final_decision: FinalDecisionTrace

    @model_validator(mode="after")
    def _validate_trace(self) -> SelectionTrace:
        if self.freeze_fallback_asset_id is not None:
            if not ID_PATTERN.fullmatch(self.freeze_fallback_asset_id):
                raise ValueError(
                    f"invalid freeze_fallback_asset_id "
                    f"{self.freeze_fallback_asset_id!r}"
                )
            saved = [
                entry
                for entry in self.scanned_candidates
                if entry.decision == "saved_freeze_fallback"
            ]
            if len(saved) != 1 or saved[0].asset_id != (
                self.freeze_fallback_asset_id
            ):
                raise ValueError(
                    "freeze_fallback_asset_id must match the single "
                    "saved_freeze_fallback entry"
                )
        return self


class RenderProfile(BaseModel):
    model_config = STRICT_CONFIG

    width: Literal[1280] = 1280
    height: Literal[720] = 720
    fps: Literal[24] = 24
    video_codec: Literal["libx264"] = "libx264"
    pixel_format: Literal["yuv420p"] = "yuv420p"
    video_preset: Literal["medium"] = "medium"
    video_crf: Literal[20] = 20
    max_b_frames: Literal[0] = 0
    gop_frames: Literal[48] = 48
    video_track_timescale: Literal[48000] = 48000
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate_kbps: Literal[128] = 128
    audio_sample_rate: Literal[48000] = 48000
    audio_channels: Literal[2] = 2


class TimelineItem(BaseModel):
    model_config = STRICT_CONFIG

    shot_id: str
    order: int
    requirement: ShotRequirement
    strategy: TimelineStrategy

    source_asset_id: str | None = None
    source_path: str | None = None
    source_size_bytes: int | None = None
    source_sha256: str | None = None
    source_in_frame: int = 0
    source_frame_count: int = 0

    target_frames: int
    score: ScoreBreakdown | None = None
    reason_code: ReasonCode
    reason: str

    @field_validator("strategy", mode="before")
    @classmethod
    def _coerce_strategy(cls, value: object) -> object:
        if isinstance(value, str):
            return TimelineStrategy(value)
        return value


class Timeline(BaseModel):
    model_config = STRICT_CONFIG

    schema_version: Literal["1.9"] = "1.9"
    path_base: Literal["timeline_dir"] = "timeline_dir"
    render_profile: RenderProfile
    items: list[TimelineItem]

    @model_validator(mode="after")
    def _validate_timeline(self) -> Timeline:
        if not self.items:
            raise ValueError("timeline must contain at least one item")
        orders = [item.order for item in self.items]
        if sorted(orders) != list(range(1, len(self.items) + 1)):
            raise ValueError("item orders must be unique, starting at 1 and contiguous")
        for index, item in enumerate(self.items):
            # Timeline.items array order is the single source of playback order;
            # item.order must equal the array index + 1.
            if item.order != index + 1:
                raise ValueError(
                    f"item order must equal array index + 1, got order={item.order} "
                    f"at index {index}"
                )
        shot_ids = [item.shot_id for item in self.items]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("shot_ids must be unique")
        for item in self.items:
            req = item.requirement
            if item.shot_id != req.id:
                raise ValueError("item.shot_id must equal requirement.id")
            if item.order != req.order:
                raise ValueError("item.order must equal requirement.order")
            if item.target_frames != req.target_frames:
                raise ValueError("item.target_frames must equal requirement.target_frames")
            if item.target_frames <= 0:
                raise ValueError("target_frames must be positive")
            if item.strategy is TimelineStrategy.CLIP:
                if item.reason_code not in ("exact_length", "center_trim"):
                    raise ValueError(
                        "clip strategy requires reason_code "
                        "exact_length/center_trim, "
                        f"got {item.reason_code!r}"
                    )
                required = (
                    item.source_asset_id,
                    item.source_path,
                    item.source_size_bytes,
                    item.source_sha256,
                )
                if any(v is None for v in required):
                    raise ValueError("clip item requires complete source fields")
                if item.source_in_frame < 0:
                    raise ValueError("clip source_in_frame must be >= 0")
                if item.source_frame_count != item.target_frames:
                    raise ValueError(
                        "clip source_frame_count must equal target_frames"
                    )
                if not SHA256_PATTERN.fullmatch(item.source_sha256 or ""):
                    raise ValueError("clip source_sha256 must be 64 lowercase hex chars")
            elif item.strategy is TimelineStrategy.FREEZE_FRAME:
                if item.reason_code != "short_source_freeze":
                    raise ValueError(
                        "freeze_frame strategy requires reason_code "
                        "short_source_freeze, "
                        f"got {item.reason_code!r}"
                    )
                required = (
                    item.source_asset_id,
                    item.source_path,
                    item.source_size_bytes,
                    item.source_sha256,
                )
                if any(v is None for v in required):
                    raise ValueError(
                        "freeze_frame item requires complete source fields"
                    )
                if item.source_in_frame < 0:
                    raise ValueError(
                        "freeze_frame source_in_frame must be >= 0"
                    )
                if item.source_frame_count < 1:
                    raise ValueError(
                        "freeze_frame source_frame_count must be >= 1"
                    )
                if item.source_frame_count >= item.target_frames:
                    raise ValueError(
                        "freeze_frame source_frame_count must be < target_frames"
                    )
                if not SHA256_PATTERN.fullmatch(item.source_sha256 or ""):
                    raise ValueError(
                        "freeze_frame source_sha256 must be 64 lowercase hex chars"
                    )
            else:  # placeholder
                if item.reason_code != "no_candidate":
                    raise ValueError(
                        "placeholder strategy requires reason_code "
                        "no_candidate, "
                        f"got {item.reason_code!r}"
                    )
                if any(
                    v is not None
                    for v in (
                        item.source_asset_id,
                        item.source_path,
                        item.source_size_bytes,
                        item.source_sha256,
                    )
                ):
                    raise ValueError("placeholder must not carry source fields")
                if item.source_in_frame != 0 or item.source_frame_count != 0:
                    raise ValueError(
                        "placeholder source_in_frame/source_frame_count must be 0"
                    )
        return self
