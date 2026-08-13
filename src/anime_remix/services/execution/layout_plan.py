"""``layout-plan-v1`` research contract: numeric compositing execution contract.

Freeze reference (Round 5, 2026-08-11):

- LayoutPlan is the *post-render* numeric geometry.  It never expresses
  intent (pose/camera/narrative live upstream in LayoutIntent / ShotSpec).
- All coordinates are normalized to the canvas (0..1); ``canvas`` records
  the target size.  ``scene_crop`` is already resolved - no fit/cover verbs
  appear here.
- ``transform`` is the execution instruction; ``bbox`` is the snapshot of
  the transformed visible alpha bounds.
- Occlusion is a canvas-space numeric polygon; execution subtracts the
  occluder mask from the character alpha.
- Mask generation parameters live here; the produced mask refs and hashes
  are recorded as ledger ArtifactInstance facts (immutability).
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

_SCHEMA_VERSION = "layout-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _require_data_ref(value: object, field: str) -> object:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped.startswith(("artifact://", "asset://")):
        raise ValueError(f"{field} must be an artifact:// or asset:// ref")
    return stripped


def _normalized_pair(value: object, field: str) -> object:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    if len(value) != 2:
        raise ValueError(f"{field} must contain exactly 2 values")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field} values must be numbers")
        number = float(item)
        if number < 0 or number > 1:
            raise ValueError(f"{field} values must be within [0, 1]")
    return value


class Canvas(BaseModel):
    model_config = _STRICT_CONFIG

    width: int
    height: int

    @field_validator("width", "height", mode="before")
    @classmethod
    def _positive(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("canvas width/height must be integers")
        if value <= 0:
            raise ValueError("canvas width/height must be positive")
        return value


class SceneCrop(BaseModel):
    """Resolved cover-crop rectangle in normalized source coordinates."""

    model_config = _STRICT_CONFIG

    source_rect: list[float]

    @field_validator("source_rect", mode="before")
    @classmethod
    def _rect(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("source_rect must be a list")
        if len(value) != 4:
            raise ValueError("source_rect must contain exactly 4 values")
        numbers: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("source_rect values must be numbers")
            number = float(item)
            if number < 0 or number > 1:
                raise ValueError("source_rect values must be within [0, 1]")
            numbers.append(number)
        if not (numbers[0] < numbers[2] and numbers[1] < numbers[3]):
            raise ValueError(
                "source_rect must be ordered x0 < x1 and y0 < y1"
            )
        return numbers


class LayerTransform(BaseModel):
    """Execution instruction applied to the source layer."""

    model_config = _STRICT_CONFIG

    scale: float
    translate: list[float]
    rotation_degrees: float = 0

    @field_validator("scale", mode="before")
    @classmethod
    def _scale(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("scale must be a number")
        number = float(value)
        if number <= 0:
            raise ValueError("scale must be positive")
        return number

    @field_validator("translate", mode="before")
    @classmethod
    def _translate(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("translate must be a list")
        if len(value) != 2:
            raise ValueError("translate must contain exactly 2 values")
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("translate values must be numbers")
        return value

    @field_validator("rotation_degrees", mode="before")
    @classmethod
    def _rotation(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("rotation_degrees must be a number")
        number = float(value)
        if number != 0:
            raise ValueError(
                "rotation_degrees must be 0 in v1 (reserved field)"
            )
        return number


class Layer(BaseModel):
    model_config = _STRICT_CONFIG

    layer_id: str
    source_ref: str
    transform: LayerTransform
    bbox: list[float]
    z_index: int

    @field_validator("layer_id", mode="before")
    @classmethod
    def _layer_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("layer_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid layer_id {value!r}")
        return stripped

    @field_validator("source_ref", mode="before")
    @classmethod
    def _source_ref(cls, value: object) -> object:
        return _require_data_ref(value, "source_ref")

    @field_validator("bbox", mode="before")
    @classmethod
    def _bbox(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("bbox must be a list")
        if len(value) != 4:
            raise ValueError("bbox must contain exactly 4 values")
        numbers: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("bbox values must be numbers")
            number = float(item)
            if number < 0 or number > 1:
                raise ValueError("bbox values must be within [0, 1]")
            numbers.append(number)
        if not (numbers[0] < numbers[2] and numbers[1] < numbers[3]):
            raise ValueError("bbox must be ordered x0 < x1 and y0 < y1")
        return numbers

    @field_validator("z_index", mode="before")
    @classmethod
    def _z_index(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("z_index must be an integer")
        return value


class Occlusion(BaseModel):
    """Canvas-space polygon that occludes a target layer."""

    model_config = _STRICT_CONFIG

    occluder_ref: str
    target_layer: str
    polygon: list[list[float]]

    @field_validator("occluder_ref", mode="before")
    @classmethod
    def _occluder(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("occluder_ref must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("occluder_ref must be non-empty")
        return stripped

    @field_validator("target_layer", mode="before")
    @classmethod
    def _target_layer(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("target_layer must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid target_layer {value!r}")
        return stripped

    @field_validator("polygon", mode="before")
    @classmethod
    def _polygon(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("polygon must be a list")
        if len(value) < 3:
            raise ValueError("polygon must contain at least 3 points")
        cleaned: list[list[float]] = []
        for point in value:
            if not isinstance(point, list) or len(point) != 2:
                raise TypeError("polygon points must be [x, y] pairs")
            pair: list[float] = []
            for item in point:
                if isinstance(item, bool) or not isinstance(
                    item, (int, float)
                ):
                    raise TypeError("polygon values must be numbers")
                number = float(item)
                if number < 0 or number > 1:
                    raise ValueError(
                        "polygon values must be within [0, 1]"
                    )
                pair.append(number)
            cleaned.append(pair)
        return cleaned


class Contact(BaseModel):
    """Resolved numeric contact anchors (no human semantics in v1)."""

    model_config = _STRICT_CONFIG

    source_anchor: list[float]
    target_anchor: list[float]
    tolerance: float

    @field_validator("source_anchor", "target_anchor", mode="before")
    @classmethod
    def _anchors(cls, value: object) -> object:
        return _normalized_pair(value, "anchor")

    @field_validator("tolerance", mode="before")
    @classmethod
    def _tolerance(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("tolerance must be a number")
        number = float(value)
        if number < 0:
            raise ValueError("tolerance must be >= 0")
        return number


class MaskGenerationSpec(BaseModel):
    """Mask generation parameters carried by the LayoutPlan.

    The produced composite/inpaint masks are separate immutable artifacts;
    their refs and hashes live in ledger ArtifactInstance facts, not here.
    """

    model_config = _STRICT_CONFIG

    generator_version: str
    parameters: dict[str, int | float | str] = {}

    @field_validator("generator_version", mode="before")
    @classmethod
    def _version(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("generator_version must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("generator_version must be non-empty")
        return stripped


class PolicyRefs(BaseModel):
    model_config = _STRICT_CONFIG

    layout_policy_ref: str
    inpaint_policy_ref: str

    @field_validator("layout_policy_ref", "inpaint_policy_ref", mode="before")
    @classmethod
    def _policy(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("policy refs must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("policy refs must be non-empty")
        return stripped


class LayoutRefs(BaseModel):
    model_config = _STRICT_CONFIG

    character_layer_ref: str
    character_geometry_ref: str
    scene_asset_ref: str
    scene_geometry_ref: str
    layout_intent_ref: str
    keyframe_state_ref: str

    @field_validator(
        "character_layer_ref",
        "character_geometry_ref",
        "scene_asset_ref",
        "scene_geometry_ref",
        "layout_intent_ref",
        "keyframe_state_ref",
        mode="before",
    )
    @classmethod
    def _refs(cls, value: object) -> object:
        return _require_data_ref(value, "layout ref")


class LayoutPlanDocument(BaseModel):
    """Numeric compositing execution contract for one keyframe."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["layout-plan-v1"] = _SCHEMA_VERSION
    plan_id: str
    shot_id: str
    keyframe_id: str
    refs: LayoutRefs
    canvas: Canvas
    scene_crop: SceneCrop
    layers: list[Layer]
    occlusions: list[Occlusion] = []
    contacts: list[Contact] = []
    mask_generation: MaskGenerationSpec
    policy_refs: PolicyRefs

    @field_validator("plan_id", "shot_id", "keyframe_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("plan/shot/keyframe ids must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator("layers", mode="before")
    @classmethod
    def _layers(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("layers must be a list")
        if not value:
            raise ValueError("layers must not be empty")
        return value

    @model_validator(mode="after")
    def _layer_integrity(self) -> LayoutPlanDocument:
        layer_ids = [layer.layer_id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("layer ids must be unique")
        z_indices = [layer.z_index for layer in self.layers]
        if len(z_indices) != len(set(z_indices)):
            raise ValueError("layer z_index values must be unique")
        layer_id_set = set(layer_ids)
        for occlusion in self.occlusions:
            if occlusion.target_layer not in layer_id_set:
                raise ValueError(
                    "occlusion references unknown target_layer "
                    f"{occlusion.target_layer!r}"
                )
        return self


def parse_layout_plan(data: object) -> LayoutPlanDocument:
    """Parse and strictly validate a layout-plan-v1 document."""

    try:
        return TypeAdapter(LayoutPlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid layout-plan-v1: {exc}") from exc


def load_layout_plan(path: str | object) -> LayoutPlanDocument:
    """Load a layout-plan-v1 file from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_layout_plan(load_json_object(Path(str(path))))
