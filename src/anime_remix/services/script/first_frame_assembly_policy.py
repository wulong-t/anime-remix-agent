"""Strict policy contract for high-quality first-frame assembly.

The policy separates raw reference availability from permission to use pixels
in a final render.  In particular, schematic/layout references remain useful
planning evidence but never enter a model request, and future information is
kept out of the first-frame execution graph entirely.
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

_SCHEMA_VERSION = "first-frame-assembly-policy-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

ReferenceAuthority = Literal[
    "final_visual",
    "identity_only",
    "action_only",
    "structure_only",
    "deterministic_overlay",
    "hidden",
]


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be non-empty")
    return cleaned


def _clean_id(value: object, field: str) -> str:
    cleaned = _clean_text(value, field)
    if not _ID_PATTERN.fullmatch(cleaned):
        raise ValueError(f"invalid {field} {value!r}")
    return cleaned


def _clean_ids(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    cleaned = [_clean_id(item, f"{field} item") for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} must contain unique values")
    return cleaned


class ReferenceAuthorityRule(BaseModel):
    """How one bound reference may participate in first-frame assembly."""

    model_config = _STRICT_CONFIG

    asset_id: str
    authority: ReferenceAuthority
    reason: str

    @field_validator("asset_id", mode="before")
    @classmethod
    def _asset_id(cls, value: object) -> object:
        return _clean_id(value, "asset_id")

    @field_validator("reason", mode="before")
    @classmethod
    def _reason(cls, value: object) -> object:
        return _clean_text(value, "reason")


class InteractionRequirement(BaseModel):
    """A contact/spatial relationship that the completed frame must prove."""

    model_config = _STRICT_CONFIG

    interaction_id: str
    actor: str
    target: str
    relation: str
    required_state: str
    evidence_asset_ids: list[str] = []
    hard_gate: bool = True

    @field_validator("interaction_id", mode="before")
    @classmethod
    def _interaction_id(cls, value: object) -> object:
        return _clean_id(value, "interaction_id")

    @field_validator("actor", "target", "relation", "required_state", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("evidence_asset_ids", mode="before")
    @classmethod
    def _evidence(cls, value: object) -> object:
        return _clean_ids(value, "evidence_asset_ids")


class FirstFrameAssemblyPolicy(BaseModel):
    """Optional reviewed policy layered over a validated ReferenceBundle."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["first-frame-assembly-policy-v1"] = _SCHEMA_VERSION
    shot_id: str
    reference_authorities: list[ReferenceAuthorityRule] = []
    interactions: list[InteractionRequirement] = []
    require_production_quality_review: bool = True

    @field_validator("shot_id", mode="before")
    @classmethod
    def _shot_id(cls, value: object) -> object:
        return _clean_id(value, "shot_id")

    @field_validator("reference_authorities", "interactions", mode="before")
    @classmethod
    def _lists(cls, value: object, info) -> object:
        if not isinstance(value, list):
            raise TypeError(f"{info.field_name} must be a list")
        return value

    @model_validator(mode="after")
    def _unique_entries(self) -> FirstFrameAssemblyPolicy:
        asset_ids = [item.asset_id for item in self.reference_authorities]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("reference authority asset ids must be unique")
        interaction_ids = [item.interaction_id for item in self.interactions]
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError("interaction ids must be unique")
        return self


def parse_first_frame_assembly_policy(data: object) -> FirstFrameAssemblyPolicy:
    """Strictly parse a ``first-frame-assembly-policy-v1`` document."""

    try:
        return TypeAdapter(FirstFrameAssemblyPolicy).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid first-frame-assembly-policy-v1: {exc}"
        ) from exc
