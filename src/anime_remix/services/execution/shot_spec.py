"""``shot-spec-v1`` research contract: the immutable execution intent truth.

Freeze reference (Round 2/5, 2026-08-11):

- ``shot-spec-v1`` is compiled deterministically from ``shot_plan-v1`` (I2).
  Downstream modules read only this document and never re-read shot_plan-v1.
- It is a frozen, human-reviewed intent contract; execution results are never
  written back into it.
- ``generation_mode`` is a discriminated union: near_match and compose carry
  different required blocks.
- Locks are shot-level structured facts; KeyframePlan may only expand inside
  the space they allow.
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
from anime_remix.services.execution.requirements import Requirement
from anime_remix.services.script.shot_plan import ShotActionBeat

_SCHEMA_VERSION = "shot-spec-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_MAX_DURATION_SECONDS = 600.0
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

GenerationMode = Literal["near_match", "compose"]
ShotScale = Literal["close_up", "medium", "wide"]


class CharacterLocks(BaseModel):
    """Locked character facts (identity lock is inherent to character_id)."""

    model_config = _STRICT_CONFIG

    identity: bool
    hairstyle: bool
    costume_variant: str

    @field_validator("costume_variant", mode="before")
    @classmethod
    def _costume(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("costume_variant must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("costume_variant must be non-empty")
        return stripped


class SceneLocks(BaseModel):
    model_config = _STRICT_CONFIG

    scene_id: str
    time_of_day: str

    @field_validator("scene_id", "time_of_day", mode="before")
    @classmethod
    def _non_empty(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("scene lock fields must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("scene lock fields must be non-empty")
        return stripped


class StyleLocks(BaseModel):
    model_config = _STRICT_CONFIG

    visual_style_id: str

    @field_validator("visual_style_id", mode="before")
    @classmethod
    def _style(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("visual_style_id must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("visual_style_id must be non-empty")
        return stripped


class Locks(BaseModel):
    model_config = _STRICT_CONFIG

    character: CharacterLocks
    scene: SceneLocks
    style: StyleLocks


class ComposeComposition(BaseModel):
    """Composition intent (semantic layer; numbers live in LayoutPlan)."""

    model_config = _STRICT_CONFIG

    shot_scale: ShotScale
    camera_position: str

    @field_validator("camera_position", mode="before")
    @classmethod
    def _camera(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("camera_position must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("camera_position must be non-empty")
        return stripped


class SpatialRelation(BaseModel):
    model_config = _STRICT_CONFIG

    subject: str
    relation: str
    object: str

    @field_validator("subject", "relation", "object", mode="before")
    @classmethod
    def _non_empty(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("spatial relation fields must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("spatial relation fields must be non-empty")
        return stripped


class CharacterRequirementGroup(BaseModel):
    model_config = _STRICT_CONFIG

    character_id: str
    requirements: list[Requirement]

    @field_validator("character_id", mode="before")
    @classmethod
    def _character_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("character_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid character_id {value!r}")
        return stripped

    @field_validator("requirements", mode="before")
    @classmethod
    def _requirements(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("requirements must be a list")
        if not value:
            raise ValueError("character requirements must not be empty")
        return value


class SceneRequirementGroup(BaseModel):
    model_config = _STRICT_CONFIG

    scene_id: str
    requirements: list[Requirement] = []

    @field_validator("scene_id", mode="before")
    @classmethod
    def _scene_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("scene_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid scene_id {value!r}")
        return stripped


class PropRequirementGroup(BaseModel):
    model_config = _STRICT_CONFIG

    prop_id: str
    requirements: list[Requirement] = []

    @field_validator("prop_id", mode="before")
    @classmethod
    def _prop_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("prop_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid prop_id {value!r}")
        return stripped


class ComposeBlock(BaseModel):
    """Required block for ``generation_mode == "compose"``."""

    model_config = _STRICT_CONFIG

    character: CharacterRequirementGroup
    scene: SceneRequirementGroup
    props: list[PropRequirementGroup] = []
    composition: ComposeComposition
    spatial_relations: list[SpatialRelation] = []


class NearMatchSourceRequirements(BaseModel):
    model_config = _STRICT_CONFIG

    character_id: str
    scene_id: str
    shot_scale: ShotScale

    @field_validator("character_id", "scene_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("source requirement ids must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped


class NearMatchBlock(BaseModel):
    """Required block for ``generation_mode == "near_match"``."""

    model_config = _STRICT_CONFIG

    source_requirements: NearMatchSourceRequirements
    preserve: list[str]
    modify: dict[str, str]

    @field_validator("preserve", mode="before")
    @classmethod
    def _preserve(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("preserve must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("preserve items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("preserve items must be non-empty")
            cleaned.append(stripped)
        if not cleaned:
            raise ValueError("preserve must not be empty")
        return cleaned

    @field_validator("modify", mode="before")
    @classmethod
    def _modify(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("modify must be a dict")
        cleaned: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise TypeError("modify keys and values must be strings")
            key_stripped = key.strip()
            item_stripped = item.strip()
            if not key_stripped or not item_stripped:
                raise ValueError("modify keys and values must be non-empty")
            cleaned[key_stripped] = item_stripped
        if not cleaned:
            raise ValueError("modify must not be empty")
        return cleaned


class ShotSpecDocument(BaseModel):
    """Frozen execution intent contract for one shot."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["shot-spec-v1"] = _SCHEMA_VERSION
    shot_id: str
    scene_id: str
    order: int
    narrative_purpose: str
    duration_seconds: float
    camera_motion: str
    emotion_arc: str
    start_state: str
    action_beats: list[ShotActionBeat]
    end_state: str
    dialogue: str | None = None
    continuity_in: str | None = None
    continuity_out: str | None = None
    generation_mode: GenerationMode
    locks: Locks
    near_match: NearMatchBlock | None = None
    compose: ComposeBlock | None = None

    @field_validator("shot_id", "scene_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("shot_id/scene_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order must be an integer")
        if value < 1:
            raise ValueError("order must be >= 1")
        return value

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

    @field_validator(
        "narrative_purpose",
        "camera_motion",
        "emotion_arc",
        "start_state",
        "end_state",
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

    @field_validator(
        "dialogue",
        "continuity_in",
        "continuity_out",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("optional text must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("optional text must be non-empty when provided")
        return stripped

    @model_validator(mode="after")
    def _mode_blocks(self) -> ShotSpecDocument:
        if self.generation_mode == "near_match" and self.near_match is None:
            raise ValueError("near_match block required for near_match mode")
        if self.generation_mode == "compose" and self.compose is None:
            raise ValueError("compose block required for compose mode")
        if self.generation_mode == "near_match" and self.compose is not None:
            raise ValueError("compose block forbidden in near_match mode")
        if self.generation_mode == "compose" and self.near_match is not None:
            raise ValueError("near_match block forbidden in compose mode")
        return self

    @model_validator(mode="after")
    def _locks_consistency(self) -> ShotSpecDocument:
        if self.scene_id != self.locks.scene.scene_id:
            raise ValueError("scene_id must equal locks.scene.scene_id")
        if (
            self.compose is not None
            and self.compose.scene.scene_id != self.scene_id
        ):
            raise ValueError(
                "compose.scene.scene_id must equal the document scene_id"
            )
        return self

    @model_validator(mode="after")
    def _beats_timing(self) -> ShotSpecDocument:
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


def parse_shot_spec(data: object) -> ShotSpecDocument:
    """Parse and strictly validate a shot-spec-v1 document."""

    try:
        return TypeAdapter(ShotSpecDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid shot-spec-v1: {exc}") from exc


def load_shot_spec(path: str | object) -> ShotSpecDocument:
    """Load a shot-spec-v1 file from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_shot_spec(load_json_object(Path(str(path))))
