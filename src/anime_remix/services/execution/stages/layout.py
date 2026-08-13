"""``layout`` stage: LayoutIntent + deterministic LayoutPlanner (v1).

Phase 3 scope: uniform scale, translation, rotation = 0, one scene crop,
one character layer, one polygon occlusion and basic contact anchors.
The planner is deliberately dumb but deterministic; no auto-composition.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.shot_spec import ShotSpecDocument

_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

DESIRED_CHARACTER_HEIGHT_RATIO = 0.55
MASK_EDGE_DILATION_PX = 12
LAYOUT_POLICY_REF = "policy://layout-v1"
INPAINT_POLICY_REF = "policy://inpaint-v1"


class LayoutIntent(BaseModel):
    """Semantic layout intent derived from ShotSpec + KeyframeState."""

    model_config = _STRICT_CONFIG

    subject_pose: str
    camera_view: str
    shot_scale: str
    relations: list[str] = []

    @field_validator("subject_pose", "camera_view", "shot_scale", mode="before")
    @classmethod
    def _non_empty(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("layout intent fields must be strings")
        stripped = value.strip()
        if not stripped:
            raise ValueError("layout intent fields must be non-empty")
        return stripped

    @field_validator("relations", mode="before")
    @classmethod
    def _relations(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("relations must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("relations items must be strings")
            stripped = item.strip()
            if not stripped:
                raise ValueError("relations items must be non-empty")
            cleaned.append(stripped)
        return cleaned


def derive_layout_intent(
    shot_spec: ShotSpecDocument, keyframe_state: dict
) -> dict:
    """Derive the semantic LayoutIntent from the frozen intent truth."""

    compose = shot_spec.compose
    if compose is None:
        raise InputValidationError(
            "derive_layout_intent requires a compose ShotSpec"
        )
    intent = LayoutIntent(
        subject_pose=keyframe_state["subject_pose"],
        camera_view=compose.composition.camera_position,
        shot_scale=compose.composition.shot_scale,
        relations=[
            f"{r.subject} {r.relation} {r.object}"
            for r in compose.spatial_relations
        ],
    )
    return intent.model_dump(mode="json")


def map_through_crop(
    point: list[float], source_rect: list[float]
) -> list[float]:
    x0, y0, x1, y1 = source_rect
    px, py = point
    return [
        (px - x0) / (x1 - x0),
        (py - y0) / (y1 - y0),
    ]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def plan_layout(
    *,
    layout_intent: dict,
    character_geometry: dict,
    scene_geometry: dict,
    scene_crop: dict,
    canvas: dict,
    plan_id: str,
    shot_id: str,
    keyframe_id: str,
    character_layer_ref: str,
    character_geometry_ref: str,
    scene_asset_ref: str,
    scene_geometry_ref: str,
    layout_intent_ref: str,
    keyframe_state_ref: str,
) -> dict:
    """Build the numeric LayoutPlan for the fixed v1 test case."""

    alpha_bbox = character_geometry["alpha_bbox"]
    source_w, source_h = (
        int(v) for v in character_geometry["source_size"]
    )
    canvas_w = canvas["width"]
    canvas_h = canvas["height"]
    kx = source_w / canvas_w
    ky = source_h / canvas_h
    x0, y0, x1, y1 = (float(v) for v in alpha_bbox)
    height = y1 - y0
    if height <= 0:
        raise InputValidationError("character geometry height must be positive")
    scale = DESIRED_CHARACTER_HEIGHT_RATIO / (height * ky)

    center_x = (x0 + x1) / 2
    seat = scene_geometry["anchors"]["desk_left_seat"]
    seat_canvas = map_through_crop(seat, scene_crop["source_rect"])
    seat_x, seat_y = seat_canvas
    tx = seat_x - center_x * scale * kx
    ty = seat_y - y1 * scale * ky

    bbox = [
        _clamp01(x0 * scale * kx + tx),
        _clamp01(y0 * scale * ky + ty),
        _clamp01(x1 * scale * kx + tx),
        _clamp01(y1 * scale * ky + ty),
    ]
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise InputValidationError(
            "layout infeasible: transformed character bbox is degenerate"
        )
    if not all(0.0 <= v <= 1.0 for v in bbox):
        raise InputValidationError(
            "layout infeasible: character bbox outside canvas"
        )

    occluder_region = scene_geometry["regions"]["desk_front"]["polygon"]
    occlusion_polygon = [
        map_through_crop(point, scene_crop["source_rect"])
        for point in occluder_region
    ]
    plan = {
        "schema_version": "layout-plan-v1",
        "plan_id": plan_id,
        "shot_id": shot_id,
        "keyframe_id": keyframe_id,
        "refs": {
            "character_layer_ref": character_layer_ref,
            "character_geometry_ref": character_geometry_ref,
            "scene_asset_ref": scene_asset_ref,
            "scene_geometry_ref": scene_geometry_ref,
            "layout_intent_ref": layout_intent_ref,
            "keyframe_state_ref": keyframe_state_ref,
        },
        "canvas": canvas,
        "scene_crop": scene_crop,
        "layers": [
            {
                "layer_id": "scene",
                "source_ref": scene_asset_ref,
                "transform": {
                    "scale": 1.0,
                    "translate": [0.0, 0.0],
                    "rotation_degrees": 0,
                },
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "z_index": 1,
            },
            {
                "layer_id": "character",
                "source_ref": character_layer_ref,
                "transform": {
                    "scale": scale,
                    "translate": [tx, ty],
                    "rotation_degrees": 0,
                },
                "bbox": bbox,
                "z_index": 2,
            },
        ],
        "occlusions": [
            {
                "occluder_ref": "scene.desk_front",
                "target_layer": "character",
                "polygon": occlusion_polygon,
            }
        ],
        "contacts": [
            {
                "source_anchor": [seat_x, seat_y],
                "target_anchor": [seat_x, seat_y],
                "tolerance": 0.03,
            }
        ],
        "mask_generation": {
            "generator_version": "maskgen-v1",
            "parameters": {"edge_dilation_px": MASK_EDGE_DILATION_PX},
        },
        "policy_refs": {
            "layout_policy_ref": LAYOUT_POLICY_REF,
            "inpaint_policy_ref": INPAINT_POLICY_REF,
        },
    }
    return plan
