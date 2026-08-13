"""``reference-package-v1`` research contract: model-agnostic conditions.

Freeze reference (Round 3, 2026-08-11):

- The package holds *typed visual conditions*, not a bare list of image
  files.  v1 implements ``kind: image | text`` only; future modalities
  (keypoints, depth, embeddings) would bump the schema version.
- ``required/preferred`` belongs to Requirements; Candidates carry
  ``satisfied_constraints`` and ``scores``.
- The package is a candidate pool.  Final selection for one render is made
  by the Renderer Adapter and recorded in the ModelRenderRequest, never in
  this document.
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

_SCHEMA_VERSION = "reference-package-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

Role = Literal[
    "identity",
    "pose",
    "expression",
    "scene",
    "prop",
    "style",
    "source_frame",
    "costume",
]
Kind = Literal["image", "text"]
CandidateScope = Literal["shot", "keyframe"]
GenerationMode = Literal["near_match", "compose"]


class ConditionProvenance(BaseModel):
    model_config = _STRICT_CONFIG

    source_asset_id: str | None = None
    derived_from: str | None = None

    @field_validator("source_asset_id", "derived_from", mode="before")
    @classmethod
    def _optional(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("provenance fields must be strings or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("provenance fields must be non-empty when set")
        return stripped


class ConditionAsset(BaseModel):
    """One typed visual condition candidate."""

    model_config = _STRICT_CONFIG

    condition_id: str
    role: Role
    kind: Kind
    payload_ref: str
    satisfied_constraints: list[str]
    scores: dict[str, float]
    provenance: ConditionProvenance

    @field_validator("condition_id", mode="before")
    @classmethod
    def _condition_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("condition_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid condition_id {value!r}")
        return stripped

    @field_validator("payload_ref", mode="before")
    @classmethod
    def _payload_ref(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("payload_ref must be a string")
        stripped = value.strip()
        if not stripped.startswith(("artifact://", "asset://")):
            raise ValueError(
                "payload_ref must be an artifact:// or asset:// ref"
            )
        return stripped

    @field_validator("satisfied_constraints", mode="before")
    @classmethod
    def _constraints(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("satisfied_constraints must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("satisfied_constraints items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError(
                    "satisfied_constraints items must be non-empty"
                )
            cleaned.append(stripped)
        if not cleaned:
            raise ValueError("satisfied_constraints must not be empty")
        return cleaned

    @field_validator("scores", mode="before")
    @classmethod
    def _scores(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("scores must be a dict")
        if not value:
            raise ValueError("scores must not be empty")
        for key, score in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("score keys must be non-empty strings")
            if isinstance(score, bool) or not isinstance(
                score, (int, float)
            ):
                raise TypeError("score values must be numbers")
            number = float(score)
            if number < 0 or number > 1:
                raise ValueError("score values must be within [0, 1]")
        return value


class CandidateSet(BaseModel):
    """One ranked candidate set bound to a Requirement."""

    model_config = _STRICT_CONFIG

    requirement_id: str
    scope: CandidateScope = "shot"
    keyframe_id: str | None = None
    candidates: list[str]

    @field_validator("requirement_id", mode="before")
    @classmethod
    def _requirement_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("requirement_id must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement_id must be non-empty")
        return stripped

    @field_validator("keyframe_id", mode="before")
    @classmethod
    def _keyframe_id(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("keyframe_id must be a string or null")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid keyframe_id {value!r}")
        return stripped

    @field_validator("candidates", mode="before")
    @classmethod
    def _candidates(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("candidates must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("candidates items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("candidates items must be non-empty")
            cleaned.append(stripped)
        if not cleaned:
            raise ValueError("candidates must not be empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("candidates must be unique")
        return cleaned

    @model_validator(mode="after")
    def _scope(self) -> CandidateSet:
        if self.scope == "keyframe" and self.keyframe_id is None:
            raise ValueError(
                "keyframe_id required when scope == keyframe"
            )
        if self.scope == "shot" and self.keyframe_id is not None:
            raise ValueError(
                "keyframe_id forbidden when scope == shot"
            )
        return self


class ReferencePackageDocument(BaseModel):
    """Model-agnostic candidate pool for one shot."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["reference-package-v1"] = _SCHEMA_VERSION
    package_id: str
    shot_id: str
    generation_mode: GenerationMode
    requirements: list[Requirement]
    conditions: list[ConditionAsset]
    candidate_sets: list[CandidateSet]

    @field_validator("package_id", "shot_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("package_id/shot_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator("requirements", mode="before")
    @classmethod
    def _requirements(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("requirements must be a list")
        if not value:
            raise ValueError("requirements must not be empty")
        return value

    @field_validator("conditions", "candidate_sets", mode="before")
    @classmethod
    def _lists(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("conditions/candidate_sets must be lists")
        return value

    @model_validator(mode="after")
    def _referential_integrity(self) -> ReferencePackageDocument:
        requirement_ids = {req.requirement_id for req in self.requirements}
        if len(requirement_ids) != len(self.requirements):
            raise ValueError("requirement ids must be unique")
        condition_ids = {c.condition_id for c in self.conditions}
        if len(condition_ids) != len(self.conditions):
            raise ValueError("condition ids must be unique")
        set_requirements = [cs.requirement_id for cs in self.candidate_sets]
        if len(set_requirements) != len(set(set_requirements)):
            raise ValueError("candidate_sets must bind distinct requirements")
        for candidate_set in self.candidate_sets:
            if candidate_set.requirement_id not in requirement_ids:
                raise ValueError(
                    "candidate_set references unknown requirement_id "
                    f"{candidate_set.requirement_id!r}"
                )
            for condition_id in candidate_set.candidates:
                if condition_id not in condition_ids:
                    raise ValueError(
                        "candidate_set references unknown condition_id "
                        f"{condition_id!r}"
                    )
        return self

    @model_validator(mode="after")
    def _mode_roles(self) -> ReferencePackageDocument:
        roles = {c.role for c in self.conditions}
        if self.generation_mode == "near_match":
            if "source_frame" not in roles:
                raise ValueError(
                    "near_match package requires a source_frame condition"
                )
        else:
            if "identity" not in roles:
                raise ValueError(
                    "compose package requires an identity condition"
                )
        return self


def parse_reference_package(data: object) -> ReferencePackageDocument:
    """Parse and strictly validate a reference-package-v1 document."""

    try:
        return TypeAdapter(ReferencePackageDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid reference-package-v1: {exc}"
        ) from exc


def load_reference_package(path: str | object) -> ReferencePackageDocument:
    """Load a reference-package-v1 file from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_reference_package(load_json_object(Path(str(path))))
