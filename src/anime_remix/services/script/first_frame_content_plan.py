"""Model-independent visual truth for a shot's canonical first frame.

The content plan answers *what must be visible in K0* before reference
selection or image-model execution is considered.  It is intentionally
editable and requires explicit approval: the deterministic builder can
scaffold a useful plan from ``ShotPlanEntry``, but it cannot safely invent
contact relationships, occlusion or exact blocking that the Director did not
state.
"""

from __future__ import annotations

import re
import unicodedata
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
from anime_remix.services.script.first_frame_assembly_policy import (
    FirstFrameAssemblyPolicy,
    parse_first_frame_assembly_policy,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry

_SCHEMA_VERSION = "first-frame-content-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

ReviewStatus = Literal["draft", "approved"]
PlanDecision = Literal["ready", "needs_review", "blocked"]
LayerKind = Literal["scene", "set_piece", "character", "prop", "foreground"]
InformationState = Literal["visible", "occluded", "deferred"]
GateTier = Literal["reject", "local_repair", "acceptable_variation"]


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


def _clean_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    cleaned = [_clean_text(item, f"{field} item") for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} must contain unique values")
    return cleaned


class FirstFrameCameraTruth(BaseModel):
    model_config = _STRICT_CONFIG

    shot_scale: Literal["close_up", "medium", "wide"]
    position: str
    composition: str
    initial_motion: str

    @field_validator("position", "composition", "initial_motion", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class FirstFrameLayerTruth(BaseModel):
    """One back-to-front layer in the intended final frame."""

    model_config = _STRICT_CONFIG

    layer_id: str
    order: int
    component_id: str
    kind: LayerKind
    subject: str
    target_state: str
    spatial_relationship: str

    @field_validator("layer_id", "component_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator(
        "subject", "target_state", "spatial_relationship", mode="before"
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order must be an integer")
        if value < 0:
            raise ValueError("order must be non-negative")
        return value


class CharacterStateTruth(BaseModel):
    model_config = _STRICT_CONFIG

    component_id: str
    subject: str
    action_pose: str
    expression: str
    gaze: str
    screen_position: str
    identity_attributes_to_preserve: list[str]
    facing_direction: str | None = None

    @field_validator("component_id", mode="before")
    @classmethod
    def _component_id(cls, value: object) -> object:
        return _clean_id(value, "component_id")

    @field_validator(
        "subject", "action_pose", "expression", "gaze", "screen_position",
        mode="before",
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("identity_attributes_to_preserve", mode="before")
    @classmethod
    def _attributes(cls, value: object) -> object:
        return _clean_list(value, "identity_attributes_to_preserve")

    @field_validator("facing_direction", mode="before")
    @classmethod
    def _optional_facing(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "facing_direction")


class PropFunctionalAffordanceTruth(BaseModel):
    """Functional topology that must survive action-component generation."""

    model_config = _STRICT_CONFIG

    grip_zone: str
    active_end: str
    native_action_axis: str

    @field_validator("grip_zone", "active_end", "native_action_axis", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class PropStateTruth(BaseModel):
    model_config = _STRICT_CONFIG

    component_id: str
    subject: str
    state: str
    screen_position: str
    functional_affordance: PropFunctionalAffordanceTruth | None = None

    @field_validator("component_id", mode="before")
    @classmethod
    def _component_id(cls, value: object) -> object:
        return _clean_id(value, "component_id")

    @field_validator("subject", "state", "screen_position", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class NormalizedFramePoint(BaseModel):
    """Human-approved layout anchor in normalized final-frame coordinates."""

    model_config = _STRICT_CONFIG

    x: float
    y: float

    @field_validator("x", "y", mode="before")
    @classmethod
    def _coordinate(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{info.field_name} must be a number")
        coordinate = float(value)
        if not 0.0 <= coordinate <= 1.0:
            raise ValueError(f"{info.field_name} must be within 0..1")
        return coordinate


class ExternalAttachmentTruth(BaseModel):
    """A component-to-scene approach/alignment that must read in K0."""

    model_config = _STRICT_CONFIG

    attachment_id: str
    source_component_id: str
    source_anchor: str
    source_anchor_position: NormalizedFramePoint
    target_component_id: str
    target_anchor: str
    target_anchor_position: NormalizedFramePoint
    relation: str
    action_axis: str
    initial_gap: str
    required_visible_state: str
    must_remain_visible: list[str]
    source_must_remain_visible: list[str] = []
    target_must_remain_visible: list[str] = []
    hard_gate: bool = True

    @field_validator(
        "attachment_id", "source_component_id", "target_component_id", mode="before"
    )
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator(
        "source_anchor",
        "target_anchor",
        "relation",
        "action_axis",
        "initial_gap",
        "required_visible_state",
        mode="before",
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator(
        "must_remain_visible",
        "source_must_remain_visible",
        "target_must_remain_visible",
        mode="before",
    )
    @classmethod
    def _visibility(cls, value: object, info) -> object:
        visible = _clean_list(value, info.field_name)
        if info.field_name == "must_remain_visible" and not visible:
            raise ValueError("must_remain_visible must not be empty")
        return visible

    @model_validator(mode="after")
    def _distinct_components(self) -> ExternalAttachmentTruth:
        if self.source_component_id == self.target_component_id:
            raise ValueError("external attachment requires distinct components")
        split_visibility = {
            *self.source_must_remain_visible,
            *self.target_must_remain_visible,
        }
        if split_visibility and split_visibility != set(self.must_remain_visible):
            raise ValueError(
                "source and target visibility must partition must_remain_visible"
            )
        return self


class ContactRelationshipTruth(BaseModel):
    """Atomic visual relationship; co-presence is not sufficient evidence."""

    model_config = _STRICT_CONFIG

    interaction_id: str
    actor_component_id: str
    target_component_id: str
    relation: str
    required_visible_state: str
    hard_gate: bool = True

    @field_validator(
        "interaction_id", "actor_component_id", "target_component_id", mode="before"
    )
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("relation", "required_visible_state", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class FirstFrameInformationTruth(BaseModel):
    model_config = _STRICT_CONFIG

    fact_id: str
    component_id: str
    state: InformationState
    fact: str
    reason: str

    @field_validator("fact_id", "component_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("fact", "reason", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class FirstFrameAcceptanceGate(BaseModel):
    model_config = _STRICT_CONFIG

    gate_id: str
    tier: GateTier
    criterion: str

    @field_validator("gate_id", mode="before")
    @classmethod
    def _gate_id(cls, value: object) -> object:
        return _clean_id(value, "gate_id")

    @field_validator("criterion", mode="before")
    @classmethod
    def _criterion(cls, value: object) -> object:
        return _clean_text(value, "criterion")


class FirstFrameContentPlanDocument(BaseModel):
    """Reviewed K0 visual truth, independent from assets and model prompts."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["first-frame-content-plan-v1"] = _SCHEMA_VERSION
    plan_id: str
    shot_id: str
    review_status: ReviewStatus
    decision: PlanDecision
    narrative_purpose: str
    camera: FirstFrameCameraTruth
    layers: list[FirstFrameLayerTruth]
    character_states: list[CharacterStateTruth]
    prop_states: list[PropStateTruth]
    contact_graph: list[ContactRelationshipTruth]
    attachment_graph: list[ExternalAttachmentTruth] = []
    information: list[FirstFrameInformationTruth]
    continuity_in: str
    motion_runway: str
    acceptance_gates: list[FirstFrameAcceptanceGate]
    warnings: list[str] = []

    @field_validator("plan_id", "shot_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator(
        "narrative_purpose", "continuity_in", "motion_runway", mode="before"
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings(cls, value: object) -> object:
        return _clean_list(value, "warnings")

    @model_validator(mode="after")
    def _integrity(self) -> FirstFrameContentPlanDocument:
        if not self.layers:
            raise ValueError("content plan requires at least one layer")
        orders = [item.order for item in self.layers]
        if orders != list(range(len(self.layers))):
            raise ValueError("layer order must be contiguous from 0")
        layer_ids = [item.layer_id for item in self.layers]
        component_ids = [item.component_id for item in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("layer ids must be unique")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("each component must occupy exactly one planned layer")
        if self.layers[0].kind != "scene":
            raise ValueError("the backmost layer must be the scene")
        component_set = set(component_ids)
        state_ids = [item.component_id for item in self.character_states]
        state_ids += [item.component_id for item in self.prop_states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("character/prop state component ids must be unique")
        if not set(state_ids) <= component_set:
            raise ValueError("state references a component absent from the layer stack")
        for item in self.contact_graph:
            if {
                item.actor_component_id,
                item.target_component_id,
            } - component_set:
                raise ValueError("contact relationship references an unknown component")
            if item.actor_component_id == item.target_component_id:
                raise ValueError("contact relationship requires distinct components")
        attachment_ids = [item.attachment_id for item in self.attachment_graph]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("external attachment ids must be unique")
        for item in self.attachment_graph:
            if {
                item.source_component_id,
                item.target_component_id,
            } - component_set:
                raise ValueError("external attachment references an unknown component")
        if self.review_status == "approved":
            props = {item.component_id: item for item in self.prop_states}
            missing_affordance = sorted(
                item.source_component_id
                for item in self.attachment_graph
                if item.hard_gate
                and item.source_component_id in props
                and props[item.source_component_id].functional_affordance is None
            )
            if missing_affordance:
                raise ValueError(
                    "approved hard external attachments require prop functional "
                    f"affordance: {missing_affordance}"
                )
            incomplete_visibility = sorted(
                item.attachment_id
                for item in self.attachment_graph
                if item.hard_gate
                and (
                    not item.source_must_remain_visible
                    or not item.target_must_remain_visible
                )
            )
            if incomplete_visibility:
                raise ValueError(
                    "approved hard external attachments require separate source "
                    f"and target visibility: {incomplete_visibility}"
                )
        fact_ids = [item.fact_id for item in self.information]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("information fact ids must be unique")
        if any(item.component_id not in component_set for item in self.information):
            raise ValueError("information fact references an unknown component")
        gate_ids = [item.gate_id for item in self.acceptance_gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("acceptance gate ids must be unique")
        if not any(item.tier == "reject" for item in self.acceptance_gates):
            raise ValueError("content plan requires at least one reject-tier gate")
        if self.review_status == "approved" and self.decision == "blocked":
            raise ValueError("a blocked content plan cannot be approved")
        return self


def parse_first_frame_content_plan(data: object) -> FirstFrameContentPlanDocument:
    try:
        return TypeAdapter(FirstFrameContentPlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid first-frame-content-plan-v1: {exc}"
        ) from exc


def approve_first_frame_content_plan(data: object) -> FirstFrameContentPlanDocument:
    plan = parse_first_frame_content_plan(data)
    payload = plan.model_dump(mode="json")
    payload["review_status"] = "approved"
    return parse_first_frame_content_plan(payload)


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace(" ", "")


def _entity_root(value: str) -> str:
    return _norm(value.split(".", maxsplit=1)[0].replace("_", " "))


def _eyes_are_hidden(start_state: str) -> bool:
    value = _norm(start_state)
    return any(
        marker in value
        for marker in ("闭眼", "闭着眼", "合眼", "eyesclosed", "closedeyes")
    )


def build_first_frame_content_plan(
    shot: ShotPlanEntry | dict,
    *,
    assembly_policy: FirstFrameAssemblyPolicy | object | None = None,
) -> FirstFrameContentPlanDocument:
    """Scaffold an editable K0 content plan from reviewed Director facts."""

    try:
        shot_doc = (
            shot
            if isinstance(shot, ShotPlanEntry)
            else TypeAdapter(ShotPlanEntry).validate_python(shot)
        )
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid shot for first-frame content: {exc}") from exc
    policy = None
    if assembly_policy is not None:
        policy = (
            assembly_policy
            if isinstance(assembly_policy, FirstFrameAssemblyPolicy)
            else parse_first_frame_assembly_policy(assembly_policy)
        )
        if policy.shot_id != shot_doc.shot_id:
            raise InputValidationError("assembly policy shot_id does not match shot")

    layers: list[FirstFrameLayerTruth] = [
        FirstFrameLayerTruth(
            layer_id="layer.scene",
            order=0,
            component_id="scene",
            kind="scene",
            subject=shot_doc.setting,
            target_state="the location at the exact start of the shot",
            spatial_relationship=shot_doc.composition,
        )
    ]
    prop_states: list[PropStateTruth] = []
    for index, prop in enumerate(shot_doc.props, start=1):
        component_id = f"prop_{index:03d}"
        layers.append(
            FirstFrameLayerTruth(
                layer_id=f"layer.{component_id}",
                order=len(layers),
                component_id=component_id,
                kind="prop",
                subject=prop,
                target_state=shot_doc.start_state,
                spatial_relationship=shot_doc.composition,
            )
        )
        prop_states.append(
            PropStateTruth(
                component_id=component_id,
                subject=prop,
                state=shot_doc.start_state,
                screen_position=shot_doc.composition,
            )
        )

    character_states: list[CharacterStateTruth] = []
    for index, subject in enumerate(shot_doc.subjects, start=1):
        component_id = f"character_{index:03d}"
        layers.append(
            FirstFrameLayerTruth(
                layer_id=f"layer.{component_id}",
                order=len(layers),
                component_id=component_id,
                kind="character",
                subject=subject,
                target_state=shot_doc.start_state,
                spatial_relationship=shot_doc.composition,
            )
        )
        character_states.append(
            CharacterStateTruth(
                component_id=component_id,
                subject=subject,
                action_pose=shot_doc.start_state,
                expression=f"expression implied by: {shot_doc.start_state}",
                gaze=f"gaze implied by: {shot_doc.start_state}",
                screen_position=shot_doc.composition,
                identity_attributes_to_preserve=[
                    "face",
                    "hair",
                    "outfit",
                    "body proportions",
                ],
            )
        )

    component_by_subject = {
        _entity_root(item.subject): item.component_id
        for item in [*character_states, *prop_states]
    }
    contact_graph: list[ContactRelationshipTruth] = []
    warnings = [
        "review the default back-to-front layer order and exact screen positions",
        "review character expression and gaze; ShotPlan start_state may be underspecified",
    ]
    if policy is not None:
        for requirement in policy.interactions:
            actor = component_by_subject.get(_entity_root(requirement.actor))
            target = component_by_subject.get(_entity_root(requirement.target))
            if actor is None or target is None:
                warnings.append(
                    f"interaction {requirement.interaction_id} could not be mapped "
                    "to content components"
                )
                continue
            contact_graph.append(
                ContactRelationshipTruth(
                    interaction_id=requirement.interaction_id,
                    actor_component_id=actor,
                    target_component_id=target,
                    relation=requirement.relation,
                    required_visible_state=requirement.required_state,
                    hard_gate=requirement.hard_gate,
                )
            )
    if not contact_graph and (shot_doc.subjects and shot_doc.props):
        warnings.append(
            "no atomic contact relationship is declared; add one if a subject "
            "must touch, hold or operate a prop in K0"
        )
    if shot_doc.props:
        warnings.append(
            "declare an external attachment when a directional prop must point, "
            "approach or align with a fixed scene target; include functional prop "
            "affordance and normalized source/target anchors"
        )

    information = [
        FirstFrameInformationTruth(
            fact_id="scene.visible",
            component_id="scene",
            state="visible",
            fact=shot_doc.setting,
            reason="the first frame must establish its location",
        )
    ]
    for item in character_states:
        information.append(
            FirstFrameInformationTruth(
                fact_id=f"{item.component_id}.identity",
                component_id=item.component_id,
                state="visible",
                fact=f"canonical identity of {item.subject}",
                reason="K0 anchors character continuity for the entire shot",
            )
        )
        if _eyes_are_hidden(shot_doc.start_state):
            information.append(
                FirstFrameInformationTruth(
                    fact_id=f"{item.component_id}.eyes",
                    component_id=item.component_id,
                    state="deferred",
                    fact=f"iris color and eye shape of {item.subject}",
                    reason="eyes are closed in K0; reveal this in a later information frame",
                )
            )
    for item in prop_states:
        information.append(
            FirstFrameInformationTruth(
                fact_id=f"{item.component_id}.state",
                component_id=item.component_id,
                state="visible",
                fact=f"{item.subject}: {item.state}",
                reason="the prop state participates in the initial visual truth",
            )
        )
    if shot_doc.end_state != shot_doc.start_state:
        information.append(
            FirstFrameInformationTruth(
                fact_id="scene.future_state",
                component_id="scene",
                state="deferred",
                fact=shot_doc.end_state,
                reason="the end state must not leak into the first-frame image",
            )
        )

    first_future_beat = next(
        (beat.description for beat in shot_doc.action_beats if beat.time_seconds > 0),
        shot_doc.end_state,
    )
    gates = [
        FirstFrameAcceptanceGate(
            gate_id="gate.reject.identity",
            tier="reject",
            criterion="reject any wrong face, hair, outfit or body proportions",
        ),
        FirstFrameAcceptanceGate(
            gate_id="gate.reject.contact",
            tier="reject",
            criterion="reject malformed anatomy, floating props or false contact",
        ),
        FirstFrameAcceptanceGate(
            gate_id="gate.reject.hidden",
            tier="reject",
            criterion="reject leakage of information explicitly deferred from K0",
        ),
        FirstFrameAcceptanceGate(
            gate_id="gate.reject.composition",
            tier="reject",
            criterion="reject wrong framing, subject placement or layer ordering",
        ),
        FirstFrameAcceptanceGate(
            gate_id="gate.repair.integration",
            tier="local_repair",
            criterion="locally repair small lighting, edge or material integration defects",
        ),
        FirstFrameAcceptanceGate(
            gate_id="gate.accept.microtexture",
            tier="acceptable_variation",
            criterion="accept unimportant microtexture variation after hard facts pass",
        ),
    ]
    return FirstFrameContentPlanDocument(
        plan_id=f"{shot_doc.shot_id}-first-frame-content",
        shot_id=shot_doc.shot_id,
        review_status="draft",
        decision="needs_review",
        narrative_purpose=shot_doc.narrative_purpose,
        camera=FirstFrameCameraTruth(
            shot_scale=shot_doc.shot_scale,
            position=shot_doc.camera_position,
            composition=shot_doc.composition,
            initial_motion=shot_doc.camera_motion,
        ),
        layers=layers,
        character_states=character_states,
        prop_states=prop_states,
        contact_graph=contact_graph,
        attachment_graph=[],
        information=information,
        continuity_in=shot_doc.continuity_in or "no incoming continuity constraint",
        motion_runway=(
            f"K0 must leave believable physical and compositional room for: "
            f"{first_future_beat}"
        ),
        acceptance_gates=gates,
        warnings=warnings,
    )
