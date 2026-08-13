"""I2: strict ``shot_plan.json`` contract (research schema ``shot-plan-v1``).

This is the Director LLM output contract used by the I2 experiment.  It is
an independent research JSON: it does not modify the formal legacy Timeline
``1.9`` schema and is not consumed by the legacy Renderer.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from anime_remix.errors import InputValidationError

_SCHEMA_VERSION = "shot-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_MAX_DURATION_SECONDS = 600.0
_MAX_BEATS = 32
_MAX_DIALOGUE_CODEPOINTS = 2000
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

ShotScale = Literal["close_up", "medium", "wide"]


class ShotActionBeat(BaseModel):
    """One timed action beat inside a shot."""

    model_config = _STRICT_CONFIG

    time_seconds: float
    description: str

    @field_validator("description", mode="before")
    @classmethod
    def _description(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("beat description must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("beat description must be non-empty")
        return stripped


class ShotPlanEntry(BaseModel):
    """One Director-designed shot."""

    model_config = _STRICT_CONFIG

    shot_id: str
    scene_id: str
    order: int
    narrative_purpose: str
    duration_seconds: float
    shot_scale: ShotScale
    composition: str
    camera_position: str
    camera_motion: str
    subjects: list[str]
    setting: str
    props: list[str]
    start_state: str
    action_beats: list[ShotActionBeat]
    end_state: str
    emotion_arc: str
    dialogue: str | None = None
    continuity_in: str | None = None
    continuity_out: str | None = None

    @field_validator("shot_id", "scene_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("shot_id/scene_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator(
        "narrative_purpose",
        "composition",
        "camera_position",
        "camera_motion",
        "setting",
        "start_state",
        "end_state",
        "emotion_arc",
        mode="before",
    )
    @classmethod
    def _non_empty_text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("text field must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("text field must be non-empty")
        return stripped

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def _duration(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("duration_seconds must be a number")
        number = float(value)
        if number <= 0:
            raise ValueError("duration_seconds must be positive")
        if number > _MAX_DURATION_SECONDS:
            raise ValueError(
                f"duration_seconds exceeds {_MAX_DURATION_SECONDS}"
            )
        return number

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order must be an integer")
        if value < 1:
            raise ValueError("order must be >= 1")
        return value

    @field_validator("subjects", mode="before")
    @classmethod
    def _subjects(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("subjects must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("list items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("list items must be non-empty")
            cleaned.append(stripped)
        return cleaned

    @field_validator("props", mode="before")
    @classmethod
    def _props(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("props must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("list items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("list items must be non-empty")
            cleaned.append(stripped)
        return cleaned

    @field_validator("action_beats", mode="before")
    @classmethod
    def _beats(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("action_beats must be a list")
        if not value:
            raise ValueError("action_beats must not be empty")
        if len(value) > _MAX_BEATS:
            raise ValueError(f"action_beats exceeds {_MAX_BEATS} items")
        return value

    @field_validator("dialogue", "continuity_in", "continuity_out", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("optional text must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("optional text must be non-empty when provided")
        if len(stripped) > _MAX_DIALOGUE_CODEPOINTS:
            raise ValueError("optional text is too long")
        return stripped

    @model_validator(mode="after")
    def _beats_timing(self) -> ShotPlanEntry:
        times = [beat.time_seconds for beat in self.action_beats]
        if times != sorted(times):
            raise ValueError("action_beats time_seconds must be non-decreasing")
        if any(t < 0 or t > self.duration_seconds for t in times):
            raise ValueError(
                "action_beats time_seconds must lie inside the shot duration"
            )
        if times[0] != 0:
            raise ValueError("first action beat must be at time 0")
        return self


class ShotPlanDocument(BaseModel):
    """Strict top-level shot_plan.json document."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["shot-plan-v1"] = _SCHEMA_VERSION
    shots: list[ShotPlanEntry]

    @model_validator(mode="after")
    def _document(self) -> ShotPlanDocument:
        if not self.shots:
            raise ValueError("shots must not be empty")
        ids = [shot.shot_id for shot in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("shot ids must be globally unique")
        orders = [shot.order for shot in self.shots]
        if orders != list(range(1, len(self.shots) + 1)):
            raise ValueError("shot order must be 1..N contiguous")
        return self


def parse_shot_plan(data: object) -> ShotPlanDocument:
    """Parse and strictly validate a shot_plan.json document."""

    try:
        return TypeAdapter(ShotPlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid shot_plan.json: {exc}") from exc


def load_shot_plan(path: str | object) -> ShotPlanDocument:
    """Load a shot_plan.json file from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_shot_plan(load_json_object(Path(str(path))))
