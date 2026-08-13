"""Shared requirement language for the Image-First execution contracts.

Freeze reference (Round 3/4, 2026-08-11): ``required/preferred`` belongs to a
Requirement (how important a constraint is), never to a candidate.  The same
Requirement list is consumed by the Asset Retriever, the Renderer Adapter and
the Critic so all three layers share one constraint language.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

Priority = Literal["required", "preferred"]

_REQUIREMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Requirement(BaseModel):
    """One semantic requirement and its importance (not a candidate)."""

    model_config = _STRICT_CONFIG

    requirement_id: str
    constraint: str
    priority: Priority

    @field_validator("requirement_id", mode="before")
    @classmethod
    def _requirement_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("requirement_id must be a string")
        stripped = value.strip()
        if not _REQUIREMENT_ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid requirement_id {value!r}")
        return stripped

    @field_validator("constraint", mode="before")
    @classmethod
    def _constraint(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("constraint must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("constraint must be non-empty")
        return stripped
