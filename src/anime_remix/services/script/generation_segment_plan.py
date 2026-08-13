"""Shared-boundary planning for video-model generation segments.

An editorial shot remains one director-level shot.  This research contract
splits it only when an explicitly justified boundary adds control.  Adjacent
generation segments share the exact same boundary anchor: the preceding
segment ends on it and the following segment starts from it.

The canonical first frame comes from an approved ``first-frame-plan-v1``.
Later anchors edit the previous approved frame and may add at most one new
visual reference, because the previous frame already occupies the first of
the two reliable Qwen visual-reference slots.  Text describes only the state
change; visual facts retain explicit reference provenance.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
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
from anime_remix.json_io import load_json_object
from anime_remix.services.script.first_frame_plan import (
    FirstFramePlanDocument,
    parse_first_frame_plan,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry

_SCHEMA_VERSION = "generation-segment-plan-v1"
_INTENTS_SCHEMA_VERSION = "segment-boundary-intents-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
_MAX_ANCHORS = 16
_EPSILON = 1e-6

ControlReason = Literal[
    "information_reveal",
    "motion_phase_change",
    "contact_topology",
    "occlusion_change",
    "camera_or_composition_reset",
    "model_duration_limit",
    "shot_end",
]
AnchorRole = Literal[
    "master_start",
    "continuity",
    "information",
    "action_topology",
    "composition_handoff",
    "final",
]
BoundaryMethod = Literal["edit_previous", "reuse_existing_asset"]
AnchorMethod = Literal[
    "approved_first_frame",
    "edit_previous",
    "reuse_existing_asset",
]
ReferenceRole = Literal[
    "identity",
    "outfit",
    "pose",
    "expression",
    "prop",
    "scene",
    "style",
    "foreground",
]
InformationStatus = Literal[
    "visible_at_start",
    "hidden_known",
    "revealed",
    "unresolved",
]
PlanDecision = Literal["ready", "needs_review", "blocked"]
ReviewStatus = Literal["draft", "approved"]
GenerationRisk = Literal["low", "medium", "high"]

_ANCHOR_WORTHY_REASONS = {
    "information_reveal",
    "contact_topology",
    "camera_or_composition_reset",
    "model_duration_limit",
    "shot_end",
}
_INDEPENDENT_GENERATION_AXES = {
    "information_reveal",
    "contact_topology",
    "camera_or_composition_reset",
}


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


def _clean_list(
    value: object,
    field: str,
    *,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    cleaned = [_clean_text(item, f"{field} item") for item in value]
    if not allow_empty and not cleaned:
        raise ValueError(f"{field} must not be empty")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} must contain unique values")
    return cleaned


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


class SegmentBoundaryIntent(BaseModel):
    """One deliberately selected endpoint after the canonical first frame."""

    model_config = _STRICT_CONFIG

    anchor_id: str
    time_seconds: float
    target_state: str
    composition: str
    camera: str
    process_from_previous: str
    dominant_motion: str
    camera_motion: str
    delta_instruction: str | None
    control_reasons: list[ControlReason]
    reveal_fact_ids: list[str] = []
    generation_method: BoundaryMethod = "edit_previous"
    reference_asset_id: str | None = None
    reference_role: ReferenceRole | None = None
    reference_attributes: list[str] = []
    locked_attributes: list[str] = [
        "identity",
        "outfit",
        "hair",
        "style",
        "scene_continuity",
    ]

    @field_validator("anchor_id", mode="before")
    @classmethod
    def _anchor_id(cls, value: object) -> object:
        return _clean_id(value, "anchor_id")

    @field_validator(
        "target_state",
        "composition",
        "camera",
        "process_from_previous",
        "dominant_motion",
        "camera_motion",
        mode="before",
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("delta_instruction", mode="before")
    @classmethod
    def _delta(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "delta_instruction")

    @field_validator("time_seconds", mode="before")
    @classmethod
    def _time(cls, value: object) -> object:
        number = _number(value, "time_seconds")
        if number <= 0:
            raise ValueError("boundary time_seconds must be > 0")
        return number

    @field_validator("control_reasons", mode="before")
    @classmethod
    def _reasons(cls, value: object) -> object:
        return _clean_list(value, "control_reasons", allow_empty=False)

    @field_validator(
        "reveal_fact_ids",
        "reference_attributes",
        "locked_attributes",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_list(
            value,
            info.field_name,
            allow_empty=info.field_name != "locked_attributes",
        )

    @field_validator("reveal_fact_ids", mode="after")
    @classmethod
    def _fact_ids(cls, value: list[str]) -> list[str]:
        return [_clean_id(item, "reveal_fact_id") for item in value]

    @field_validator("reference_asset_id", mode="before")
    @classmethod
    def _reference(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_id(value, "reference_asset_id")

    @model_validator(mode="after")
    def _method_contract(self) -> SegmentBoundaryIntent:
        has_reference = self.reference_asset_id is not None
        if has_reference != (self.reference_role is not None):
            raise ValueError(
                "reference_asset_id and reference_role must be provided together"
            )
        if self.generation_method == "edit_previous":
            if self.delta_instruction is None:
                raise ValueError("edit_previous requires delta_instruction")
        else:
            if not has_reference:
                raise ValueError("reuse_existing_asset requires reference_asset_id")
            if self.delta_instruction is not None:
                raise ValueError(
                    "reuse_existing_asset must not contain delta_instruction"
                )
            if self.reveal_fact_ids:
                raise ValueError(
                    "reuse_existing_asset cannot claim incremental information; "
                    "the exact asset is the complete target frame"
                )
        if self.reference_attributes and not has_reference:
            raise ValueError("reference_attributes requires a visual reference asset")
        if "information_reveal" in self.control_reasons and not self.reveal_fact_ids:
            raise ValueError("information_reveal requires at least one reveal_fact_id")
        if self.reveal_fact_ids and "information_reveal" not in self.control_reasons:
            raise ValueError(
                "reveal_fact_ids requires the information_reveal control reason"
            )
        return self


class SegmentBoundaryIntentsDocument(BaseModel):
    """Human/LLM-authored boundary decisions consumed by the compiler."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["segment-boundary-intents-v1"] = _INTENTS_SCHEMA_VERSION
    shot_id: str
    boundaries: list[SegmentBoundaryIntent] = []

    @field_validator("shot_id", mode="before")
    @classmethod
    def _shot_id(cls, value: object) -> object:
        return _clean_id(value, "shot_id")

    @model_validator(mode="after")
    def _boundaries(self) -> SegmentBoundaryIntentsDocument:
        if len(self.boundaries) > _MAX_ANCHORS - 1:
            raise ValueError(f"boundaries must be <= {_MAX_ANCHORS - 1}")
        ids = [item.anchor_id for item in self.boundaries]
        if len(ids) != len(set(ids)):
            raise ValueError("boundary anchor ids must be unique")
        times = [item.time_seconds for item in self.boundaries]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("boundary times must be strictly increasing")
        return self


class SegmentInformationFact(BaseModel):
    """Information provenance carried across all anchors in the shot."""

    model_config = _STRICT_CONFIG

    fact_id: str
    attribute: str
    status: InformationStatus
    authority_asset_ids: list[str] = []
    first_visible_anchor_id: str | None = None
    note: str

    @field_validator("fact_id", mode="before")
    @classmethod
    def _fact_id(cls, value: object) -> object:
        return _clean_id(value, "fact_id")

    @field_validator("attribute", "note", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("authority_asset_ids", mode="before")
    @classmethod
    def _assets(cls, value: object) -> object:
        return [
            _clean_id(item, "authority_asset_id")
            for item in _clean_list(value, "authority_asset_ids")
        ]

    @field_validator("first_visible_anchor_id", mode="before")
    @classmethod
    def _first_visible(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_id(value, "first_visible_anchor_id")


class FrameAnchor(BaseModel):
    """One shared image used by both adjacent generation segments."""

    model_config = _STRICT_CONFIG

    anchor_id: str
    order: int
    time_seconds: float
    position: float
    roles: list[AnchorRole]
    control_reasons: list[str]
    target_state: str
    composition: str
    camera: str
    base_anchor_id: str | None
    generation_method: AnchorMethod
    reference_asset_id: str | None = None
    reference_role: ReferenceRole | None = None
    reference_attributes: list[str] = []
    delta_instruction: str | None = None
    locked_attributes: list[str]
    information_added: list[str] = []
    generation_risk: GenerationRisk = "low"
    risk_factors: list[str] = []
    grounding: Literal[
        "approved_first_frame",
        "exact_visual_asset",
        "previous_frame_and_visual_reference",
        "previous_frame_and_action_delta",
        "text_fallback",
    ]

    @field_validator("anchor_id", mode="before")
    @classmethod
    def _anchor_id(cls, value: object) -> object:
        return _clean_id(value, "anchor_id")

    @field_validator("base_anchor_id", "reference_asset_id", mode="before")
    @classmethod
    def _optional_ids(cls, value: object, info) -> object:
        if value is None:
            return value
        return _clean_id(value, info.field_name)

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
    def _numbers(cls, value: object, info) -> object:
        return _number(value, info.field_name)

    @field_validator("target_state", "composition", "camera", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("delta_instruction", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "delta_instruction")

    @field_validator(
        "roles",
        "control_reasons",
        "reference_attributes",
        "locked_attributes",
        "information_added",
        "risk_factors",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_list(
            value,
            info.field_name,
            allow_empty=info.field_name
            in {"reference_attributes", "information_added", "risk_factors"},
        )

    @model_validator(mode="after")
    def _generation_contract(self) -> FrameAnchor:
        has_reference = self.reference_asset_id is not None
        if has_reference != (self.reference_role is not None):
            raise ValueError(
                "reference_asset_id and reference_role must be provided together"
            )
        if self.reference_attributes and not has_reference:
            raise ValueError("reference_attributes requires a visual reference asset")
        if self.generation_method == "approved_first_frame":
            if self.base_anchor_id is not None or self.delta_instruction is not None:
                raise ValueError(
                    "approved_first_frame cannot have a base anchor or delta"
                )
        elif self.generation_method == "edit_previous":
            if self.base_anchor_id is None or self.delta_instruction is None:
                raise ValueError("edit_previous requires a base anchor and delta")
        else:
            if not has_reference or self.delta_instruction is not None:
                raise ValueError(
                    "reuse_existing_asset requires one reference and no delta"
                )
        if self.information_added and "information" not in self.roles:
            raise ValueError("information_added requires the information role")
        if "information" in self.roles and not self.information_added:
            raise ValueError("information role requires information_added")
        return self


class GenerationSegment(BaseModel):
    """One video-model invocation bounded by two shared frame anchors."""

    model_config = _STRICT_CONFIG

    segment_id: str
    order: int
    start_anchor_id: str
    end_anchor_id: str
    duration_seconds: float
    process_description: str
    dominant_motion: str
    camera_motion: str
    end_control_reasons: list[ControlReason]
    required_visible_fact_ids: list[str] = []
    continuity_mode: Literal["shared_boundary"] = "shared_boundary"

    @field_validator("segment_id", "start_anchor_id", "end_anchor_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

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
        number = _number(value, "duration_seconds")
        if number <= 0:
            raise ValueError("duration_seconds must be positive")
        return number

    @field_validator(
        "process_description", "dominant_motion", "camera_motion", mode="before"
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("end_control_reasons", mode="before")
    @classmethod
    def _reasons(cls, value: object) -> object:
        return _clean_list(value, "end_control_reasons", allow_empty=False)

    @field_validator("required_visible_fact_ids", mode="before")
    @classmethod
    def _facts(cls, value: object) -> object:
        return [
            _clean_id(item, "required_visible_fact_id")
            for item in _clean_list(value, "required_visible_fact_ids")
        ]


class GenerationSegmentPlanDocument(BaseModel):
    """Strict plan for a single editorial shot and its generation segments."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["generation-segment-plan-v1"] = _SCHEMA_VERSION
    plan_id: str
    shot_id: str
    shot_duration_seconds: float
    first_frame_plan_id: str
    policy: Literal["shared-boundary-reference-first-v1"] = (
        "shared-boundary-reference-first-v1"
    )
    review_status: ReviewStatus = "draft"
    decision: PlanDecision
    information_ledger: list[SegmentInformationFact]
    anchors: list[FrameAnchor]
    segments: list[GenerationSegment]
    warnings: list[str] = []

    @field_validator("plan_id", "shot_id", "first_frame_plan_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("shot_duration_seconds", mode="before")
    @classmethod
    def _duration(cls, value: object) -> object:
        number = _number(value, "shot_duration_seconds")
        if number <= 0:
            raise ValueError("shot_duration_seconds must be positive")
        return number

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings(cls, value: object) -> object:
        return _clean_list(value, "warnings")

    @model_validator(mode="after")
    def _graph(self) -> GenerationSegmentPlanDocument:
        if not 2 <= len(self.anchors) <= _MAX_ANCHORS:
            raise ValueError(f"anchors must contain 2..{_MAX_ANCHORS} items")
        if len(self.segments) != len(self.anchors) - 1:
            raise ValueError("segments must contain exactly anchors - 1 items")
        anchor_ids = [item.anchor_id for item in self.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("anchor ids must be unique")
        if [item.order for item in self.anchors] != list(
            range(1, len(self.anchors) + 1)
        ):
            raise ValueError("anchor order must be 1..N contiguous")
        times = [item.time_seconds for item in self.anchors]
        positions = [item.position for item in self.anchors]
        if times[0] != 0 or positions[0] != 0:
            raise ValueError("first anchor must be at time/position 0")
        if not math.isclose(times[-1], self.shot_duration_seconds):
            raise ValueError("last anchor must be at the shot duration")
        if not math.isclose(positions[-1], 1.0):
            raise ValueError("last anchor position must be 1")
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("anchor times must be strictly increasing")
        if positions != sorted(positions) or len(positions) != len(set(positions)):
            raise ValueError("anchor positions must be strictly increasing")
        if any(position < 0 or position > 1 for position in positions):
            raise ValueError("anchor positions must lie in [0, 1]")
        first = self.anchors[0]
        if first.roles != ["master_start"]:
            raise ValueError("first anchor must have only the master_start role")
        if first.generation_method != "approved_first_frame":
            raise ValueError("first anchor must use the approved first frame")
        if first.base_anchor_id is not None:
            raise ValueError("first anchor cannot have a base_anchor_id")
        if "final" not in self.anchors[-1].roles:
            raise ValueError("last anchor must have the final role")
        if any("final" in anchor.roles for anchor in self.anchors[:-1]):
            raise ValueError("only the last anchor may have the final role")
        for index, anchor in enumerate(self.anchors[1:], start=1):
            if anchor.base_anchor_id != self.anchors[index - 1].anchor_id:
                raise ValueError(
                    "every generated anchor must use the immediately previous "
                    "shared anchor as its base"
                )
            if anchor.generation_method == "approved_first_frame":
                raise ValueError("only the first anchor may use approved_first_frame")
            if anchor.generation_method == "edit_previous":
                if anchor.delta_instruction is None:
                    raise ValueError("edit_previous anchor requires a delta")
            elif anchor.reference_asset_id is None:
                raise ValueError("reuse_existing_asset requires a reference asset")
            expected_position = anchor.time_seconds / self.shot_duration_seconds
            if not math.isclose(anchor.position, expected_position):
                raise ValueError("anchor position must equal time divided by duration")
        segment_ids = [item.segment_id for item in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment ids must be unique")
        if [item.order for item in self.segments] != list(
            range(1, len(self.segments) + 1)
        ):
            raise ValueError("segment order must be 1..N contiguous")
        fact_ids = {item.fact_id for item in self.information_ledger}
        if len(fact_ids) != len(self.information_ledger):
            raise ValueError("information fact ids must be unique")
        for index, segment in enumerate(self.segments):
            start = self.anchors[index]
            end = self.anchors[index + 1]
            if (
                segment.start_anchor_id != start.anchor_id
                or segment.end_anchor_id != end.anchor_id
            ):
                raise ValueError(
                    "segments must connect each adjacent anchor exactly once"
                )
            expected = end.time_seconds - start.time_seconds
            if not math.isclose(segment.duration_seconds, expected):
                raise ValueError("segment duration must equal its anchor interval")
            if not set(segment.required_visible_fact_ids) <= fact_ids:
                raise ValueError("segment references unknown information facts")
            if list(segment.end_control_reasons) != end.control_reasons:
                raise ValueError(
                    "segment end_control_reasons must match its end anchor"
                )
        anchor_set = set(anchor_ids)
        for fact in self.information_ledger:
            if (
                fact.first_visible_anchor_id is not None
                and fact.first_visible_anchor_id not in anchor_set
            ):
                raise ValueError("information fact references an unknown anchor")
        for anchor in self.anchors:
            if not set(anchor.information_added) <= fact_ids:
                raise ValueError("anchor adds an unknown information fact")
        added_at: dict[str, list[str]] = {fact_id: [] for fact_id in fact_ids}
        for anchor in self.anchors:
            for fact_id in anchor.information_added:
                added_at[fact_id].append(anchor.anchor_id)
        for fact in self.information_ledger:
            additions = added_at[fact.fact_id]
            if fact.status == "visible_at_start":
                if fact.first_visible_anchor_id != first.anchor_id or additions:
                    raise ValueError(
                        "visible_at_start facts must originate at the first anchor"
                    )
            elif fact.status == "revealed":
                if additions != [fact.first_visible_anchor_id]:
                    raise ValueError(
                        "revealed facts must be added exactly at first_visible_anchor_id"
                    )
            elif fact.first_visible_anchor_id is not None or additions:
                raise ValueError(
                    "hidden or unresolved facts cannot have a visible anchor"
                )
        if self.review_status == "approved" and self.decision == "blocked":
            raise ValueError("a blocked segment plan cannot be approved")
        return self


def parse_segment_boundary_intents(
    data: object,
) -> SegmentBoundaryIntentsDocument:
    try:
        return TypeAdapter(SegmentBoundaryIntentsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid segment boundary intents: {exc}") from exc


def parse_generation_segment_plan(data: object) -> GenerationSegmentPlanDocument:
    try:
        return TypeAdapter(GenerationSegmentPlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid generation_segment_plan.json: {exc}"
        ) from exc


def load_generation_segment_plan(
    path: str | Path,
) -> GenerationSegmentPlanDocument:
    return parse_generation_segment_plan(load_json_object(Path(path)))


def _roles(reasons: list[ControlReason], *, final: bool) -> list[AnchorRole]:
    roles: list[AnchorRole] = []
    if "information_reveal" in reasons:
        roles.append("information")
    if "contact_topology" in reasons:
        roles.append("action_topology")
    if "camera_or_composition_reset" in reasons:
        roles.append("composition_handoff")
    if not roles:
        roles.append("continuity")
    if final:
        roles.append("final")
    return roles


def _generation_risk(
    reasons: list[ControlReason],
) -> tuple[GenerationRisk, list[str]]:
    """Classify how many independent visual problems one anchor must solve."""

    factors = [reason for reason in reasons if reason in _INDEPENDENT_GENERATION_AXES]
    if len(factors) >= 2:
        return "high", factors
    if factors or "model_duration_limit" in reasons:
        return "medium", factors or ["model_duration_limit"]
    return "low", []


def _default_end_intent(
    shot: ShotPlanEntry,
    *,
    after_time: float,
) -> SegmentBoundaryIntent:
    remaining = [
        beat.description
        for beat in shot.action_beats
        if beat.time_seconds > after_time + _EPSILON
    ]
    process = "; ".join(remaining) if remaining else shot.end_state
    return SegmentBoundaryIntent(
        anchor_id=f"{shot.shot_id}_end",
        time_seconds=shot.duration_seconds,
        target_state=shot.end_state,
        composition=shot.composition,
        camera=shot.camera_position,
        process_from_previous=process,
        dominant_motion=process,
        camera_motion=shot.camera_motion,
        delta_instruction=(
            "Change only the action and state needed to reach the specified "
            f"endpoint: {shot.end_state}. Preserve all locked visual facts."
        ),
        control_reasons=["shot_end"],
        generation_method="edit_previous",
    )


def _choose_information_reference(
    *,
    intent: SegmentBoundaryIntent,
    ledger_by_id: dict[str, SegmentInformationFact],
    bound_assets: set[str],
) -> tuple[str | None, ReferenceRole | None, list[str], list[str]]:
    reference_id = intent.reference_asset_id
    role = intent.reference_role
    attributes = list(intent.reference_attributes)
    warnings: list[str] = []
    if reference_id is not None and reference_id not in bound_assets:
        raise InputValidationError(
            f"boundary {intent.anchor_id} references unbound asset {reference_id!r}"
        )
    if not intent.reveal_fact_ids:
        return reference_id, role, attributes, warnings

    facts = [ledger_by_id.get(fact_id) for fact_id in intent.reveal_fact_ids]
    unknown = [
        fact_id
        for fact_id, fact in zip(intent.reveal_fact_ids, facts, strict=True)
        if fact is None
    ]
    if unknown:
        raise InputValidationError(
            f"boundary {intent.anchor_id} reveals unknown facts: {unknown}"
        )
    typed_facts = [fact for fact in facts if fact is not None]
    repeated = [fact.fact_id for fact in typed_facts if fact.status == "revealed"]
    visible = [
        fact.fact_id for fact in typed_facts if fact.status == "visible_at_start"
    ]
    if repeated or visible:
        raise InputValidationError(
            f"boundary {intent.anchor_id} contains redundant information "
            f"(already revealed={repeated}, visible_at_start={visible})"
        )
    known_sets = [set(fact.authority_asset_ids) for fact in typed_facts]
    nonempty_sets = [item for item in known_sets if item]
    common = set.intersection(*nonempty_sets) if nonempty_sets else set()
    if reference_id is None and common:
        reference_id = min(common)
        role = "expression"
    if reference_id is not None:
        uncovered = [
            fact.fact_id
            for fact in typed_facts
            if fact.authority_asset_ids and reference_id not in fact.authority_asset_ids
        ]
        if uncovered:
            raise InputValidationError(
                f"boundary {intent.anchor_id} reference {reference_id!r} is "
                f"not authoritative for facts: {uncovered}"
            )
    elif nonempty_sets:
        raise InputValidationError(
            f"boundary {intent.anchor_id} reveals facts that do not share one "
            "visual authority; split the boundary or choose one scoped fact"
        )
    else:
        warnings.append(
            f"boundary {intent.anchor_id} reveals information without a visual "
            "authority; text fallback may drift"
        )
    if reference_id is not None and role is None:
        role = "expression"
    if reference_id is not None and not attributes:
        attributes = list(dict.fromkeys(fact.attribute for fact in typed_facts))
    return reference_id, role, attributes, warnings


def build_generation_segment_plan(
    shot: ShotPlanEntry,
    *,
    first_frame_plan: FirstFramePlanDocument | dict,
    boundary_intents: SegmentBoundaryIntentsDocument | dict | None = None,
) -> GenerationSegmentPlanDocument:
    """Compile explicit boundaries into a minimal shared-anchor plan.

    With no boundary intents this deliberately emits one generation segment:
    start to end.  Action beats are process evidence, not automatic cut
    points.  Intermediate segments therefore exist only when a reviewed
    boundary intent supplies a non-empty control reason.
    """

    first = (
        first_frame_plan
        if isinstance(first_frame_plan, FirstFramePlanDocument)
        else parse_first_frame_plan(first_frame_plan)
    )
    if first.shot_id != shot.shot_id:
        raise InputValidationError("first-frame plan shot_id does not match shot")
    if first.review_status != "approved":
        raise InputValidationError(
            "first-frame plan must be approved before segment planning"
        )
    if first.decision == "blocked":
        raise InputValidationError("blocked first-frame plan cannot anchor segments")
    intents = (
        SegmentBoundaryIntentsDocument(
            shot_id=shot.shot_id,
            boundaries=[],
        )
        if boundary_intents is None
        else (
            boundary_intents
            if isinstance(boundary_intents, SegmentBoundaryIntentsDocument)
            else parse_segment_boundary_intents(boundary_intents)
        )
    )
    if intents.shot_id != shot.shot_id:
        raise InputValidationError("boundary intents shot_id does not match shot")
    for intent in intents.boundaries:
        if intent.time_seconds > shot.duration_seconds + _EPSILON:
            raise InputValidationError(
                f"boundary {intent.anchor_id} lies after the shot duration"
            )
        if (
            intent.time_seconds < shot.duration_seconds - _EPSILON
            and "shot_end" in intent.control_reasons
        ):
            raise InputValidationError(
                f"intermediate boundary {intent.anchor_id} cannot use shot_end"
            )
        if (
            intent.time_seconds < shot.duration_seconds - _EPSILON
            and not set(intent.control_reasons) & _ANCHOR_WORTHY_REASONS
        ):
            raise InputValidationError(
                f"boundary {intent.anchor_id} has no anchor-worthy control "
                "reason; ordinary motion phases and occlusion changes belong "
                "in the segment process description"
            )

    boundary_list = list(intents.boundaries)
    if (
        not boundary_list
        or boundary_list[-1].time_seconds < shot.duration_seconds - _EPSILON
    ):
        after_time = boundary_list[-1].time_seconds if boundary_list else 0.0
        boundary_list.append(_default_end_intent(shot, after_time=after_time))
    else:
        final = boundary_list[-1]
        if not math.isclose(final.time_seconds, shot.duration_seconds):
            raise InputValidationError("last boundary must equal shot duration")
        if "shot_end" not in final.control_reasons:
            boundary_list[-1] = final.model_copy(
                update={"control_reasons": [*final.control_reasons, "shot_end"]}
            )

    if len(boundary_list) + 1 > _MAX_ANCHORS:
        raise InputValidationError(f"segment plan exceeds {_MAX_ANCHORS} anchors")
    if any(item.anchor_id == first.keyframe_id for item in boundary_list):
        raise InputValidationError("boundary anchor id conflicts with first frame id")

    ledger: list[SegmentInformationFact] = []
    for fact in first.information_coverage:
        status: InformationStatus
        first_visible: str | None
        if fact.status == "visible":
            status = "visible_at_start"
            first_visible = first.keyframe_id
        elif fact.status in {"known_from_reference", "occluded"}:
            status = "hidden_known"
            first_visible = None
        else:
            status = "unresolved"
            first_visible = None
        ledger.append(
            SegmentInformationFact(
                fact_id=fact.fact_id,
                attribute=fact.attribute,
                status=status,
                authority_asset_ids=list(fact.source_asset_ids),
                first_visible_anchor_id=first_visible,
                note=fact.note,
            )
        )
    ledger_by_id = {item.fact_id: item for item in ledger}
    bound_assets = set(first.bound_asset_ids)
    warnings: list[str] = []
    anchors: list[FrameAnchor] = [
        FrameAnchor(
            anchor_id=first.keyframe_id,
            order=1,
            time_seconds=0.0,
            position=0.0,
            roles=["master_start"],
            control_reasons=["canonical shot anchor"],
            target_state=first.intent.start_state,
            composition=first.intent.composition,
            camera=first.intent.camera,
            base_anchor_id=None,
            generation_method="approved_first_frame",
            locked_attributes=[
                "identity",
                "outfit",
                "hair",
                "style",
                "scene_continuity",
            ],
            grounding="approved_first_frame",
        )
    ]
    visible_fact_ids = {
        fact.fact_id for fact in ledger if fact.status == "visible_at_start"
    }
    segments: list[GenerationSegment] = []
    for intent in boundary_list:
        final = math.isclose(intent.time_seconds, shot.duration_seconds)
        reference_id, reference_role, reference_attributes, new_warnings = (
            _choose_information_reference(
                intent=intent,
                ledger_by_id=ledger_by_id,
                bound_assets=bound_assets,
            )
        )
        warnings.extend(new_warnings)
        for fact_id in intent.reveal_fact_ids:
            fact = ledger_by_id[fact_id]
            fact.status = "revealed"
            fact.first_visible_anchor_id = intent.anchor_id
            visible_fact_ids.add(fact_id)
        method: AnchorMethod = intent.generation_method
        generation_risk, risk_factors = _generation_risk(intent.control_reasons)
        if generation_risk == "high":
            warnings.append(
                f"boundary {intent.anchor_id} combines independent visual "
                f"constraints ({', '.join(risk_factors)}); keep only "
                "inseparable changes or move a camera reset to an adjacent "
                "anchor"
            )
        if method == "reuse_existing_asset":
            grounding = "exact_visual_asset"
        elif reference_id is not None:
            grounding = "previous_frame_and_visual_reference"
        elif intent.reveal_fact_ids:
            grounding = "text_fallback"
        else:
            grounding = "previous_frame_and_action_delta"
        previous = anchors[-1]
        anchor = FrameAnchor(
            anchor_id=intent.anchor_id,
            order=len(anchors) + 1,
            time_seconds=intent.time_seconds,
            position=intent.time_seconds / shot.duration_seconds,
            roles=_roles(intent.control_reasons, final=final),
            control_reasons=list(intent.control_reasons),
            target_state=intent.target_state,
            composition=intent.composition,
            camera=intent.camera,
            base_anchor_id=previous.anchor_id,
            generation_method=method,
            reference_asset_id=reference_id,
            reference_role=reference_role,
            reference_attributes=reference_attributes,
            delta_instruction=intent.delta_instruction,
            locked_attributes=list(intent.locked_attributes),
            information_added=list(intent.reveal_fact_ids),
            generation_risk=generation_risk,
            risk_factors=risk_factors,
            grounding=grounding,
        )
        anchors.append(anchor)
        segments.append(
            GenerationSegment(
                segment_id=f"{shot.shot_id}_seg_{len(segments) + 1:03d}",
                order=len(segments) + 1,
                start_anchor_id=previous.anchor_id,
                end_anchor_id=anchor.anchor_id,
                duration_seconds=anchor.time_seconds - previous.time_seconds,
                process_description=intent.process_from_previous,
                dominant_motion=intent.dominant_motion,
                camera_motion=intent.camera_motion,
                end_control_reasons=list(intent.control_reasons),
                required_visible_fact_ids=sorted(visible_fact_ids),
            )
        )

    decision: PlanDecision = "needs_review" if warnings else "ready"
    return GenerationSegmentPlanDocument(
        plan_id=f"{shot.shot_id}-generation-segments",
        shot_id=shot.shot_id,
        shot_duration_seconds=shot.duration_seconds,
        first_frame_plan_id=first.plan_id,
        review_status="draft",
        decision=decision,
        information_ledger=ledger,
        anchors=anchors,
        segments=segments,
        warnings=warnings,
    )


def approve_generation_segment_plan(
    value: GenerationSegmentPlanDocument | dict,
) -> GenerationSegmentPlanDocument:
    plan = (
        value
        if isinstance(value, GenerationSegmentPlanDocument)
        else parse_generation_segment_plan(value)
    )
    if plan.decision == "blocked":
        raise InputValidationError("blocked generation segment plan cannot be approved")
    return parse_generation_segment_plan(
        plan.model_copy(update={"review_status": "approved"}).model_dump(mode="json")
    )
