"""I3: strict ``keyframe_plan.json`` contract (research schema
``keyframe-plan-v1``).

One document plans the keyframes for exactly one shot.  Keyframe count is
decided from content, never fixed; the first keyframe anchors time 0 /
position 0 and the last anchors the shot end.  This is an independent
research JSON and does not touch legacy Timeline 1.9.
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

_SCHEMA_VERSION = "keyframe-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_MIN_KEYFRAMES = 2
_MAX_KEYFRAMES = 16
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

AssetType = Literal[
    "character",
    "background",
    "foreground",
    "prop",
    "style",
]


class RequiredAssetRef(BaseModel):
    """One referenced image asset with its locked role."""

    model_config = _STRICT_CONFIG

    asset_id: str
    asset_type: AssetType
    locked_attributes: list[str]

    @field_validator("asset_id", mode="before")
    @classmethod
    def _asset_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("asset_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid asset_id {value!r}")
        return stripped

    @field_validator("locked_attributes", mode="before")
    @classmethod
    def _locked(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("locked_attributes must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("locked_attributes items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("locked_attributes items must be non-empty")
            cleaned.append(stripped)
        return cleaned


class KeyframeEntry(BaseModel):
    """One keyframe anchored in time and normalized position."""

    model_config = _STRICT_CONFIG

    keyframe_id: str
    shot_id: str
    order: int
    time_seconds: float
    position: float
    visual_description: str
    subject_pose: str
    expression: str
    gaze: str
    composition: str
    camera: str
    background_state: str
    foreground_state: str
    prop_state: str
    required_assets: list[RequiredAssetRef]
    motion_from_previous: str

    @field_validator("keyframe_id", "shot_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("keyframe_id/shot_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator(
        "visual_description",
        "subject_pose",
        "expression",
        "gaze",
        "composition",
        "camera",
        "background_state",
        "foreground_state",
        "prop_state",
        "motion_from_previous",
        mode="before",
    )
    @classmethod
    def _text(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("text field must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("text field must be non-empty")
        return stripped

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order must be an integer")
        if value < 1:
            raise ValueError("order must be >= 1")
        return value

    @field_validator("time_seconds", "position", mode="before")
    @classmethod
    def _number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("time_seconds/position must be numbers")
        return float(value)

    @field_validator("required_assets", mode="before")
    @classmethod
    def _assets(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("required_assets must be a list")
        if not value:
            raise ValueError("required_assets must not be empty")
        return value


class KeyframePlanDocument(BaseModel):
    """Strict top-level keyframe_plan.json for one shot."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["keyframe-plan-v1"] = _SCHEMA_VERSION
    shot_id: str
    shot_duration_seconds: float
    keyframes: list[KeyframeEntry]

    @field_validator("shot_id", mode="before")
    @classmethod
    def _shot_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("shot_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid shot_id {value!r}")
        return stripped

    @field_validator("shot_duration_seconds", mode="before")
    @classmethod
    def _duration(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("shot_duration_seconds must be a number")
        number = float(value)
        if number <= 0:
            raise ValueError("shot_duration_seconds must be positive")
        return number

    @model_validator(mode="after")
    def _keyframes(self) -> KeyframePlanDocument:
        if len(self.keyframes) < _MIN_KEYFRAMES:
            raise ValueError(f"keyframes must be >= {_MIN_KEYFRAMES}")
        if len(self.keyframes) > _MAX_KEYFRAMES:
            raise ValueError(f"keyframes must be <= {_MAX_KEYFRAMES}")
        if any(kf.shot_id != self.shot_id for kf in self.keyframes):
            raise ValueError("all keyframes must reference the document shot_id")
        ids = [kf.keyframe_id for kf in self.keyframes]
        if len(ids) != len(set(ids)):
            raise ValueError("keyframe ids must be unique")
        orders = [kf.order for kf in self.keyframes]
        if orders != list(range(1, len(self.keyframes) + 1)):
            raise ValueError("keyframe order must be 1..N contiguous")
        times = [kf.time_seconds for kf in self.keyframes]
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError("keyframe time_seconds must be strictly increasing")
        if times[0] != 0:
            raise ValueError("first keyframe must be at time 0")
        if times[-1] != self.shot_duration_seconds:
            raise ValueError(
                "last keyframe must be at the shot duration_seconds"
            )
        positions = [kf.position for kf in self.keyframes]
        if any(p < 0 or p > 1 for p in positions):
            raise ValueError("position must be in [0, 1]")
        if positions[0] != 0:
            raise ValueError("first keyframe position must be 0")
        if positions[-1] != 1:
            raise ValueError("last keyframe position must be 1")
        if positions != sorted(positions):
            raise ValueError("position must be non-decreasing")
        return self


def parse_keyframe_plan(data: object) -> KeyframePlanDocument:
    """Parse and strictly validate a keyframe_plan.json document."""

    try:
        return TypeAdapter(KeyframePlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid keyframe_plan.json: {exc}") from exc


def load_keyframe_plan(path: str | object) -> KeyframePlanDocument:
    """Load a keyframe_plan.json file from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_keyframe_plan(load_json_object(Path(str(path))))
