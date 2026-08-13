"""Reference-first planning for a shot's canonical first frame.

``first-frame-plan-v1`` is deliberately independent from the historical
multi-keyframe planner.  It turns one reviewed ``ShotPlanEntry`` and one
validated reference bundle into a reviewable sequence of small fusion
stages.  A stage may use at most two primary images, while the complete
sequence may consume as many relevant, explicitly bound assets as needed.

The plan never stores media bytes or filesystem paths.  It records asset ids,
visual-information coverage, minimal text fallbacks and an ordered stage DAG.
The execution layer resolves only the exact selected asset ids.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection
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
from anime_remix.services.image_assets import (
    ImageAssetCatalog,
    ImageAssetRecord,
)
from anime_remix.services.script.first_frame_assembly_policy import (
    FirstFrameAssemblyPolicy,
    ReferenceAuthority,
    parse_first_frame_assembly_policy,
)
from anime_remix.services.script.first_frame_content_plan import (
    FirstFrameContentPlanDocument,
    parse_first_frame_content_plan,
)
from anime_remix.services.script.prepared_component_plan import (
    PreparedComponentPlanDocument,
    PreparedComponentTask,
    parse_prepared_component_plan,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry

_SCHEMA_VERSION = "first-frame-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

ComponentKind = Literal["scene", "character", "foreground", "prop", "style"]
CoverageStatus = Literal[
    "visible",
    "known_from_reference",
    "occluded",
    "unresolved",
]
StageOperation = Literal[
    "adopt_anchor",
    "synthesize_base",
    "fuse_component",
    "apply_text_delta",
    "composite_overlay",
]
StageInputType = Literal["asset", "stage_output"]
StageInputRole = Literal[
    "base",
    "scene",
    "style",
    "identity",
    "outfit",
    "pose",
    "expression",
    "prop",
    "foreground",
    "source_frame",
]
PlanDecision = Literal["ready", "needs_review", "blocked"]
ReviewStatus = Literal["draft", "approved"]
ReferenceDisposition = Literal[
    "selected_visual",
    "control_only",
    "deferred_hidden",
    "unused",
]
InteractionGrounding = Literal[
    "visually_grounded",
    "structure_only",
    "text_only",
    "unresolved",
]
QualityGateKind = Literal[
    "interaction",
    "attachment",
    "hidden_information",
    "production_quality",
]

_TIER_RANK = {
    "canonical": 0,
    "derived": 1,
    "approved_generated": 2,
    "generated_candidate": 3,
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


def _clean_string_list(
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


class FirstFrameIntent(BaseModel):
    """Human-readable visual intent; references remain the visual truth."""

    model_config = _STRICT_CONFIG

    setting: str
    composition: str
    camera: str
    start_state: str
    subjects: list[str]
    props: list[str]

    @field_validator("setting", "composition", "camera", "start_state", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("subjects", "props", mode="before")
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_string_list(value, info.field_name)


class FirstFrameComponent(BaseModel):
    """One semantic layer/component that contributes to the frame."""

    model_config = _STRICT_CONFIG

    component_id: str
    kind: ComponentKind
    subject: str
    target_state: str
    spatial_instruction: str | None = None
    reference_asset_ids: list[str] = []
    reference_attributes: list[str] = []

    @field_validator("component_id", mode="before")
    @classmethod
    def _component_id(cls, value: object) -> object:
        return _clean_id(value, "component_id")

    @field_validator("subject", "target_state", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("spatial_instruction", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "spatial_instruction")

    @field_validator("reference_asset_ids", "reference_attributes", mode="before")
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_string_list(value, info.field_name)


class InformationCoverageFact(BaseModel):
    """Provenance and visibility state for one visual fact."""

    model_config = _STRICT_CONFIG

    fact_id: str
    component_id: str
    attribute: str
    status: CoverageStatus
    source_asset_ids: list[str] = []
    note: str

    @field_validator("fact_id", "component_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("attribute", "note", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator("source_asset_ids", mode="before")
    @classmethod
    def _sources(cls, value: object) -> object:
        return _clean_string_list(value, "source_asset_ids")

    @model_validator(mode="after")
    def _status_sources(self) -> InformationCoverageFact:
        if (
            self.status in {"visible", "known_from_reference", "occluded"}
            and not self.source_asset_ids
            and self.attribute
            not in {
                "action_state",
                "spatial_relation",
            }
        ):
            raise ValueError(f"coverage fact {self.fact_id!r} requires a source asset")
        if self.status == "unresolved" and self.source_asset_ids:
            raise ValueError(
                f"unresolved coverage fact {self.fact_id!r} cannot have sources"
            )
        return self


class FirstFrameReferenceAdmission(BaseModel):
    """Pixel-use decision for one asset in the bound reference bundle."""

    model_config = _STRICT_CONFIG

    asset_id: str
    authority: ReferenceAuthority
    disposition: ReferenceDisposition
    reason: str

    @field_validator("asset_id", mode="before")
    @classmethod
    def _asset_id(cls, value: object) -> object:
        return _clean_id(value, "asset_id")

    @field_validator("reason", mode="before")
    @classmethod
    def _reason(cls, value: object) -> object:
        return _clean_text(value, "reason")

    @model_validator(mode="after")
    def _authority_disposition(self) -> FirstFrameReferenceAdmission:
        if self.authority == "structure_only" and self.disposition != "control_only":
            raise ValueError("structure_only references must remain control_only")
        if self.authority == "hidden" and self.disposition != "deferred_hidden":
            raise ValueError("hidden references must remain deferred_hidden")
        if self.disposition == "selected_visual" and self.authority in {
            "structure_only",
            "hidden",
        }:
            raise ValueError("non-visual authority cannot be selected for rendering")
        return self


class FirstFrameInteractionUnit(BaseModel):
    """A relationship whose visual truth cannot be inferred from co-presence."""

    model_config = _STRICT_CONFIG

    interaction_id: str
    actor: str
    target: str
    relation: str
    required_state: str
    participant_component_ids: list[str]
    evidence_asset_ids: list[str] = []
    grounding: InteractionGrounding
    hard_gate: bool = True

    @field_validator("interaction_id", mode="before")
    @classmethod
    def _interaction_id(cls, value: object) -> object:
        return _clean_id(value, "interaction_id")

    @field_validator("actor", "target", "relation", "required_state", mode="before")
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)

    @field_validator(
        "participant_component_ids",
        "evidence_asset_ids",
        mode="before",
    )
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_string_list(
            value,
            info.field_name,
            allow_empty=True,
        )


class FirstFrameAttachmentUnit(BaseModel):
    """Directional connection between a prepared component and scene target."""

    model_config = _STRICT_CONFIG

    attachment_id: str
    source_component_id: str
    source_anchor: str
    source_anchor_x: float
    source_anchor_y: float
    target_component_id: str
    target_anchor: str
    target_anchor_x: float
    target_anchor_y: float
    relation: str
    action_axis: str
    initial_gap: str
    required_visible_state: str
    must_remain_visible: list[str]
    source_must_remain_visible: list[str]
    target_must_remain_visible: list[str]
    evidence_asset_ids: list[str] = []
    grounding: InteractionGrounding
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
        "evidence_asset_ids",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_string_list(
            value,
            info.field_name,
            allow_empty=info.field_name == "evidence_asset_ids",
        )

    @model_validator(mode="after")
    def _visibility_partition(self) -> FirstFrameAttachmentUnit:
        if {
            *self.source_must_remain_visible,
            *self.target_must_remain_visible,
        } != set(self.must_remain_visible):
            raise ValueError(
                "source and target visibility must partition must_remain_visible"
            )
        return self

    @field_validator(
        "source_anchor_x",
        "source_anchor_y",
        "target_anchor_x",
        "target_anchor_y",
        mode="before",
    )
    @classmethod
    def _coordinate(cls, value: object, info) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{info.field_name} must be a number")
        coordinate = float(value)
        if not 0.0 <= coordinate <= 1.0:
            raise ValueError(f"{info.field_name} must be within 0..1")
        return coordinate


class FirstFrameQualityGate(BaseModel):
    """Mandatory manual judgement attached to a concrete stage output."""

    model_config = _STRICT_CONFIG

    gate_id: str
    kind: QualityGateKind
    description: str
    review_stage_id: str
    interaction_id: str | None = None
    attachment_id: str | None = None
    asset_ids: list[str] = []

    @field_validator("gate_id", "review_stage_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("interaction_id", mode="before")
    @classmethod
    def _optional_id(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_id(value, "interaction_id")

    @field_validator("attachment_id", mode="before")
    @classmethod
    def _optional_attachment_id(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_id(value, "attachment_id")

    @field_validator("description", mode="before")
    @classmethod
    def _description(cls, value: object) -> object:
        return _clean_text(value, "description")

    @field_validator("asset_ids", mode="before")
    @classmethod
    def _asset_ids(cls, value: object) -> object:
        return _clean_string_list(value, "asset_ids")


class FirstFrameStageInput(BaseModel):
    model_config = _STRICT_CONFIG

    slot: int
    source_type: StageInputType
    source_id: str
    role: StageInputRole

    @field_validator("slot", mode="before")
    @classmethod
    def _slot(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("slot must be an integer")
        if value not in (1, 2):
            raise ValueError("slot must be 1 or 2")
        return value

    @field_validator("source_id", mode="before")
    @classmethod
    def _source_id(cls, value: object) -> object:
        return _clean_id(value, "source_id")


class FirstFrameStage(BaseModel):
    """One deterministic adoption or paid-capable model fusion stage."""

    model_config = _STRICT_CONFIG

    stage_id: str
    order: int
    operation: StageOperation
    component_ids: list[str]
    inputs: list[FirstFrameStageInput]
    instruction: str | None = None
    reference_attributes: list[str] = []
    text_fallbacks: dict[str, str] = {}
    quality_gate_ids: list[str] = []

    @field_validator("stage_id", mode="before")
    @classmethod
    def _stage_id(cls, value: object) -> object:
        return _clean_id(value, "stage_id")

    @field_validator("order", mode="before")
    @classmethod
    def _order(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("order must be an integer")
        if value < 1:
            raise ValueError("order must be positive")
        return value

    @field_validator(
        "component_ids",
        "reference_attributes",
        "quality_gate_ids",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_string_list(
            value,
            info.field_name,
            allow_empty=info.field_name
            in {
                "reference_attributes",
                "quality_gate_ids",
            },
        )

    @field_validator("inputs", mode="before")
    @classmethod
    def _inputs(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("inputs must be a list")
        if len(value) > 2:
            raise ValueError("one fusion stage may use at most two images")
        return value

    @field_validator("instruction", mode="before")
    @classmethod
    def _instruction(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "instruction")

    @field_validator("text_fallbacks", mode="before")
    @classmethod
    def _fallbacks(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("text_fallbacks must be a dict")
        return {
            _clean_text(key, "text_fallback key"): _clean_text(
                item, "text_fallback value"
            )
            for key, item in value.items()
        }

    @model_validator(mode="after")
    def _operation_contract(self) -> FirstFrameStage:
        slots = [item.slot for item in self.inputs]
        if slots != list(range(1, len(self.inputs) + 1)):
            raise ValueError("input slots must be contiguous from 1")
        if self.operation == "adopt_anchor":
            if len(self.inputs) != 1 or self.inputs[0].source_type != "asset":
                raise ValueError("adopt_anchor requires exactly one asset input")
            if self.instruction or self.text_fallbacks:
                raise ValueError("adopt_anchor cannot alter the selected image")
        elif self.operation == "synthesize_base":
            if any(item.source_type != "asset" for item in self.inputs):
                raise ValueError("synthesize_base accepts only asset inputs")
            if not self.instruction:
                raise ValueError("synthesize_base requires an instruction")
        elif self.operation == "fuse_component":
            if len(self.inputs) != 2:
                raise ValueError("fuse_component requires canvas + one reference")
            if self.inputs[0].source_type != "stage_output":
                raise ValueError("fuse_component slot 1 must be the prior canvas")
            if self.inputs[1].source_type != "asset":
                raise ValueError("fuse_component slot 2 must be an asset")
            if not self.instruction:
                raise ValueError("fuse_component requires an instruction")
        elif self.operation == "apply_text_delta":
            if len(self.inputs) != 1:
                raise ValueError("apply_text_delta requires the prior canvas")
            if self.inputs[0].source_type != "stage_output":
                raise ValueError("apply_text_delta input must be a stage output")
            if not self.instruction or not self.text_fallbacks:
                raise ValueError(
                    "apply_text_delta requires instruction and text fallbacks"
                )
        elif self.operation == "composite_overlay":
            if len(self.inputs) != 2:
                raise ValueError("composite_overlay requires canvas + overlay")
            if self.inputs[0].source_type != "stage_output":
                raise ValueError("composite_overlay slot 1 must be the prior canvas")
            if self.inputs[1].source_type != "asset":
                raise ValueError("composite_overlay slot 2 must be an asset")
            if self.inputs[1].role != "foreground":
                raise ValueError("composite_overlay slot 2 must be foreground")
            if self.instruction or self.text_fallbacks:
                raise ValueError("composite_overlay is deterministic")
        return self


class FirstFramePlanDocument(BaseModel):
    """Strict, reviewable contract for constructing one first frame."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["first-frame-plan-v1"] = _SCHEMA_VERSION
    plan_id: str
    shot_id: str
    keyframe_id: str
    content_plan_id: str | None = None
    prepared_component_plan_id: str | None = None
    prepared_component_asset_ids: list[str] = []
    review_status: ReviewStatus
    decision: PlanDecision
    max_primary_visual_references_per_model_call: Literal[2] = 2
    intent: FirstFrameIntent
    components: list[FirstFrameComponent]
    information_coverage: list[InformationCoverageFact]
    bound_asset_ids: list[str]
    selected_asset_ids: list[str]
    control_asset_ids: list[str] = []
    deferred_asset_ids: list[str] = []
    unused_bound_asset_ids: list[str]
    reference_admissions: list[FirstFrameReferenceAdmission] = []
    interaction_units: list[FirstFrameInteractionUnit] = []
    attachment_units: list[FirstFrameAttachmentUnit] = []
    quality_gates: list[FirstFrameQualityGate] = []
    stages: list[FirstFrameStage]
    warnings: list[str] = []

    @field_validator("plan_id", "shot_id", "keyframe_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("content_plan_id", "prepared_component_plan_id", mode="before")
    @classmethod
    def _optional_ids(cls, value: object, info) -> object:
        if value is None:
            return value
        return _clean_id(value, info.field_name)

    @field_validator(
        "bound_asset_ids",
        "selected_asset_ids",
        "control_asset_ids",
        "deferred_asset_ids",
        "unused_bound_asset_ids",
        "prepared_component_asset_ids",
        "warnings",
        mode="before",
    )
    @classmethod
    def _lists(cls, value: object, info) -> object:
        return _clean_string_list(value, info.field_name)

    @model_validator(mode="after")
    def _referential_integrity(self) -> FirstFramePlanDocument:
        component_ids = [item.component_id for item in self.components]
        if not component_ids or len(component_ids) != len(set(component_ids)):
            raise ValueError("component ids must be non-empty and unique")
        fact_ids = [item.fact_id for item in self.information_coverage]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("coverage fact ids must be unique")
        if any(
            item.component_id not in set(component_ids)
            for item in self.information_coverage
        ):
            raise ValueError("coverage fact references an unknown component")

        bound = set(self.bound_asset_ids)
        selected = set(self.selected_asset_ids)
        control = set(self.control_asset_ids)
        deferred = set(self.deferred_asset_ids)
        unused = set(self.unused_bound_asset_ids)
        prepared = set(self.prepared_component_asset_ids)
        if (
            selected & control
            or selected & deferred
            or selected & unused
            or control & deferred
            or control & unused
            or deferred & unused
            or selected | control | deferred | unused != bound
        ):
            raise ValueError(
                "selected, control, deferred and unused assets must partition "
                "bound_asset_ids"
            )
        if not prepared <= selected:
            raise ValueError("prepared component assets must be selected visuals")
        if prepared and not self.prepared_component_plan_id:
            raise ValueError(
                "prepared component assets require prepared_component_plan_id"
            )
        if self.prepared_component_plan_id and not self.content_plan_id:
            raise ValueError(
                "prepared component plan requires the reviewed content plan id"
            )
        for component in self.components:
            if not set(component.reference_asset_ids) <= selected:
                raise ValueError("component references an unselected asset")
        for fact in self.information_coverage:
            if not set(fact.source_asset_ids) <= selected | control | deferred:
                raise ValueError(
                    "coverage fact references neither a selected, control nor "
                    "deferred asset"
                )

        if self.reference_admissions:
            admission_ids = [item.asset_id for item in self.reference_admissions]
            if len(admission_ids) != len(set(admission_ids)):
                raise ValueError("reference admission asset ids must be unique")
            if set(admission_ids) != bound:
                raise ValueError("reference admissions must cover every bound asset")
            expected_dispositions = {
                "selected_visual": selected,
                "control_only": control,
                "deferred_hidden": deferred,
                "unused": unused,
            }
            for disposition, expected_ids in expected_dispositions.items():
                actual_ids = {
                    item.asset_id
                    for item in self.reference_admissions
                    if item.disposition == disposition
                }
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"reference admission disposition {disposition!r} does "
                        "not match the asset partition"
                    )

        if not self.stages:
            raise ValueError("first-frame plan requires at least one stage")
        orders = [item.order for item in self.stages]
        if orders != list(range(1, len(self.stages) + 1)):
            raise ValueError("stage order must be contiguous from 1")
        stage_ids = [item.stage_id for item in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("stage ids must be unique")
        used_assets: set[str] = set()
        for index, stage in enumerate(self.stages):
            if not set(stage.component_ids) <= set(component_ids):
                raise ValueError("stage references an unknown component")
            for item in stage.inputs:
                if item.source_type == "asset":
                    if item.source_id not in selected:
                        raise ValueError("stage references an unselected asset")
                    used_assets.add(item.source_id)
                else:
                    if index == 0 or item.source_id != self.stages[index - 1].stage_id:
                        raise ValueError(
                            "stage output input must reference the immediately "
                            "preceding stage"
                        )
        if used_assets != selected:
            raise ValueError("every selected asset must be consumed by a stage")

        interaction_ids = [item.interaction_id for item in self.interaction_units]
        if len(interaction_ids) != len(set(interaction_ids)):
            raise ValueError("interaction ids must be unique")
        component_set = set(component_ids)
        for interaction in self.interaction_units:
            if not set(interaction.participant_component_ids) <= component_set:
                raise ValueError("interaction references an unknown component")
            if not set(interaction.evidence_asset_ids) <= bound:
                raise ValueError("interaction evidence is not a bound asset")

        attachment_ids = [item.attachment_id for item in self.attachment_units]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("attachment ids must be unique")
        for attachment in self.attachment_units:
            if {
                attachment.source_component_id,
                attachment.target_component_id,
            } - component_set:
                raise ValueError("attachment references an unknown component")
            if not set(attachment.evidence_asset_ids) <= bound:
                raise ValueError("attachment evidence is not a bound asset")

        gate_ids = [item.gate_id for item in self.quality_gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("quality gate ids must be unique")
        gate_set = set(gate_ids)
        for gate in self.quality_gates:
            if gate.review_stage_id not in set(stage_ids):
                raise ValueError("quality gate references an unknown stage")
            if gate.interaction_id not in set(interaction_ids) | {None}:
                raise ValueError("quality gate references an unknown interaction")
            if gate.attachment_id not in set(attachment_ids) | {None}:
                raise ValueError("quality gate references an unknown attachment")
            if not set(gate.asset_ids) <= bound:
                raise ValueError("quality gate references an unbound asset")
        staged_gate_ids = {
            gate_id for stage in self.stages for gate_id in stage.quality_gate_ids
        }
        if staged_gate_ids != gate_set:
            raise ValueError("every quality gate must be attached to exactly one stage")
        for stage in self.stages:
            if len(stage.quality_gate_ids) != len(set(stage.quality_gate_ids)):
                raise ValueError("stage quality gate ids must be unique")
            for gate_id in stage.quality_gate_ids:
                gate = next(
                    item for item in self.quality_gates if item.gate_id == gate_id
                )
                if gate.review_stage_id != stage.stage_id:
                    raise ValueError("quality gate is attached to the wrong stage")
        if self.review_status == "approved" and self.decision == "blocked":
            raise ValueError("a blocked first-frame plan cannot be approved")
        return self


def parse_first_frame_plan(data: object) -> FirstFramePlanDocument:
    """Strictly parse a ``first-frame-plan-v1`` document."""

    try:
        return TypeAdapter(FirstFramePlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid first-frame-plan-v1: {exc}") from exc


def approve_first_frame_plan(data: object) -> FirstFramePlanDocument:
    """Return the same reviewed plan with its explicit approval gate set."""

    plan = parse_first_frame_plan(data)
    payload = plan.model_dump(mode="json")
    payload["review_status"] = "approved"
    return parse_first_frame_plan(payload)


def _norm(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold().strip()


def _record_rank(record: ImageAssetRecord, term: str = "") -> tuple:
    needle = _norm(term)
    haystacks = (
        _norm(record.subject_or_scene_id),
        _norm(record.asset_id),
        _norm(record.quality_notes),
    )
    score = 0
    if needle:
        if needle in haystacks[0] or haystacks[0] in needle:
            score = 3
        elif needle in haystacks[1]:
            score = 2
        elif needle in haystacks[2]:
            score = 1
    return (
        -score,
        _TIER_RANK.get(record.source_tier, len(_TIER_RANK)),
        0 if record.analysis_status == "analyzed" else 1,
        record.asset_id,
    )


def _matches(record: ImageAssetRecord, term: str) -> bool:
    needle = _norm(term)
    if not needle:
        return False
    return any(
        needle in value or value in needle
        for value in (
            _norm(record.subject_or_scene_id),
            _norm(record.asset_id),
            _norm(record.quality_notes),
        )
        if value
    )


def _reference_attributes(record: ImageAssetRecord) -> list[str]:
    roles = set(record.reference_roles)
    attributes: list[str] = []
    if "identity_reference" in roles or (
        record.asset_type == "character" and not roles
    ):
        attributes.extend(["identity", "face", "hair", "body proportions"])
    if "outfit_reference" in roles or record.outfit:
        attributes.extend(["outfit", "accessories"])
    if "pose_reference" in roles or record.pose:
        attributes.append("pose")
    if "expression_reference" in roles or record.expression:
        attributes.extend(["expression", "gaze", "eye state"])
    if "scene_reference" in roles or record.asset_type == "background":
        attributes.extend(["scene identity", "spatial structure", "lighting"])
    if "style_reference" in roles or record.asset_type == "style":
        attributes.extend(["palette", "linework", "shading", "texture"])
    if "prop_reference" in roles or record.asset_type == "prop":
        attributes.extend(["prop identity", "prop appearance"])
    if record.asset_type == "foreground":
        attributes.extend(["foreground appearance", "occlusion"])
    return list(dict.fromkeys(attributes or ["referenced appearance"]))


def _prepared_reference_attributes(
    record: ImageAssetRecord,
    task: PreparedComponentTask | None,
) -> list[str]:
    """Describe every visual fact approved in one atomic component plate.

    A prepared interaction plate is registered as one catalog asset type even
    when it contains several planned components.  Its final-frame authority
    must therefore come from the completed preparation task, not only from the
    catalog asset type.  Otherwise a character-typed character+prop plate loses
    the approved prop and contact authority during final fusion.
    """

    attributes = _reference_attributes(record)
    if task is None:
        return attributes
    if any(component_id.startswith("character_") for component_id in task.component_ids):
        attributes.extend(
            ["identity", "face", "hair", "body proportions", "outfit", "accessories"]
        )
    if any(component_id.startswith("prop_") for component_id in task.component_ids):
        attributes.extend(["prop identity", "prop appearance"])
    if task.kind == "interaction_plate":
        attributes.extend(
            ["interaction/contact geometry", "relative participant placement"]
        )
    if task.prop_affordances:
        attributes.append("prop functional topology")
    if task.external_attachments:
        attributes.extend(
            ["external attachment geometry", "final-frame canvas placement"]
        )
    return list(dict.fromkeys(attributes))


def _prepared_placement_instruction(task: PreparedComponentTask | None) -> str:
    if task is None or not task.external_attachments:
        return ""
    details = []
    for item in task.external_attachments:
        details.append(
            f"preserve {item.source_anchor} at normalized position "
            f"({item.source_anchor_x:.3f}, {item.source_anchor_y:.3f}) and keep it "
            f"{item.relation} {item.target_subject} {item.target_anchor} at "
            f"({item.target_anchor_x:.3f}, {item.target_anchor_y:.3f}) along "
            f"{item.action_axis}, with {item.initial_gap}; keep visible: "
            f"{', '.join(item.must_remain_visible)}"
        )
        details.append(
            f"the target anchor region around ({item.target_anchor_x:.3f}, "
            f"{item.target_anchor_y:.3f}) and the approach gap must remain fully "
            "visible and unobstructed by the source component, the character's "
            "body, arm, hand or the key"
        )
    return " Approved external-attachment geometry: " + "; ".join(details) + "."


def _stage_role(record: ImageAssetRecord) -> StageInputRole:
    roles = set(record.reference_roles)
    if record.asset_type == "background":
        return "scene"
    if record.asset_type == "style":
        return "style"
    if record.asset_type == "prop":
        return "prop"
    if record.asset_type == "foreground":
        return "foreground"
    if "pose_reference" in roles and "identity_reference" not in roles:
        return "pose"
    if "expression_reference" in roles and "identity_reference" not in roles:
        return "expression"
    if "outfit_reference" in roles and "identity_reference" not in roles:
        return "outfit"
    return "identity"


def _default_authority(record: ImageAssetRecord) -> ReferenceAuthority:
    roles = set(record.reference_roles)
    action_roles = {"pose_reference", "expression_reference"}
    identity_roles = {"identity_reference", "outfit_reference"}
    if roles and roles <= action_roles:
        return "action_only"
    if record.asset_type == "character" or roles & identity_roles:
        return "identity_only"
    return "final_visual"


def _normalize_entity_selector(value: str) -> str:
    root = value.split(".", maxsplit=1)[0]
    return _norm(root).replace("_", "").replace(" ", "")


def _component_for_selector(
    selector: str,
    components: list[FirstFrameComponent],
) -> str | None:
    needle = _normalize_entity_selector(selector)
    matches = [
        item.component_id
        for item in components
        if needle
        in {
            _normalize_entity_selector(item.subject),
            _normalize_entity_selector(item.component_id),
        }
    ]
    return matches[0] if len(matches) == 1 else None


def _bundle_records(
    bundle: object,
    *,
    shot_id: str,
    catalog: ImageAssetCatalog,
) -> tuple[list[ImageAssetRecord], dict[str, str]]:
    if not isinstance(bundle, dict):
        raise InputValidationError("reference bundle must be an object")
    if bundle.get("schema_version") != "reference-bundle-v1":
        raise InputValidationError("unsupported reference bundle schema")
    if bundle.get("shot_id") != shot_id:
        raise InputValidationError("reference bundle shot_id does not match shot")
    raw_references = bundle.get("references")
    if not isinstance(raw_references, list) or not raw_references:
        raise InputValidationError("reference bundle must contain references")
    records: list[ImageAssetRecord] = []
    notes: dict[str, str] = {}
    seen: set[str] = set()
    for item in raw_references:
        if not isinstance(item, dict):
            raise InputValidationError("reference bundle entries must be objects")
        asset_id = item.get("asset_id")
        asset_type = item.get("asset_type")
        if not isinstance(asset_id, str) or asset_id in seen:
            raise InputValidationError("reference bundle asset ids must be unique")
        record = catalog.get(asset_id)
        if record is None:
            raise InputValidationError(
                f"reference bundle uses unknown asset {asset_id!r}",
                asset_id=asset_id,
            )
        if asset_type != record.asset_type:
            raise InputValidationError(
                f"reference bundle type mismatch for {asset_id!r}",
                asset_id=asset_id,
            )
        note = item.get("note", "")
        if not isinstance(note, str):
            raise InputValidationError("reference bundle note must be a string")
        records.append(record)
        notes[asset_id] = note.strip()
        seen.add(asset_id)
    return records, notes


def _closed_eyes(text: str) -> bool:
    normalized = _norm(text).replace(" ", "")
    markers = ("闭眼", "闭着眼", "合眼", "eyesclosed", "closedeyes")
    return any(marker in normalized for marker in markers)


def build_first_frame_plan(
    shot: ShotPlanEntry | dict,
    *,
    reference_bundle: object,
    catalog: ImageAssetCatalog,
    full_frame_anchor_asset_id: str | None = None,
    deferred_asset_ids: Collection[str] = (),
    assembly_policy: FirstFrameAssemblyPolicy | object | None = None,
    content_plan: FirstFrameContentPlanDocument | object | None = None,
    prepared_component_plan: PreparedComponentPlanDocument | object | None = None,
) -> FirstFramePlanDocument:
    """Build a deterministic, reviewable first-frame fusion plan.

    Every relevant selected reference is consumed by exactly one ordered
    stage.  Additional references are staged against the latest approved
    canvas instead of being silently discarded or converted into prose.
    """

    try:
        shot_doc = (
            shot
            if isinstance(shot, ShotPlanEntry)
            else TypeAdapter(ShotPlanEntry).validate_python(shot)
        )
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid shot for first frame: {exc}") from exc

    content = None
    if content_plan is not None:
        content = (
            content_plan
            if isinstance(content_plan, FirstFrameContentPlanDocument)
            else parse_first_frame_content_plan(content_plan)
        )
        if content.shot_id != shot_doc.shot_id:
            raise InputValidationError("content plan shot_id does not match shot")
        if content.review_status != "approved":
            raise InputValidationError("first-frame content plan must be approved")
        scene_layer = next(item for item in content.layers if item.kind == "scene")
        initial_states = "; ".join(
            f"{item.subject}: {item.target_state}" for item in content.layers
        )
        shot_doc = shot_doc.model_copy(
            update={
                "shot_scale": content.camera.shot_scale,
                "composition": content.camera.composition,
                "camera_position": content.camera.position,
                "camera_motion": content.camera.initial_motion,
                "subjects": [item.subject for item in content.character_states],
                "setting": scene_layer.subject,
                "props": [item.subject for item in content.prop_states],
                "start_state": initial_states,
                "narrative_purpose": content.narrative_purpose,
                "continuity_in": content.continuity_in,
            }
        )
    prepared = None
    if prepared_component_plan is not None:
        if content is None:
            raise InputValidationError(
                "prepared component plan requires an approved content plan"
            )
        prepared = (
            prepared_component_plan
            if isinstance(prepared_component_plan, PreparedComponentPlanDocument)
            else parse_prepared_component_plan(prepared_component_plan)
        )
        if prepared.shot_id != shot_doc.shot_id:
            raise InputValidationError(
                "prepared component plan shot_id does not match shot"
            )
        if prepared.content_plan_id != content.plan_id:
            raise InputValidationError(
                "prepared component plan does not belong to the content plan"
            )
        if prepared.completion_status != "completed":
            raise InputValidationError(
                "prepared component plan must be completed before final assembly"
            )

    all_records, bundle_notes = _bundle_records(
        reference_bundle,
        shot_id=shot_doc.shot_id,
        catalog=catalog,
    )
    bound_ids = [item.asset_id for item in all_records]
    prepared_output_ids = (
        prepared.approved_output_asset_ids if prepared is not None else []
    )
    missing_prepared_outputs = sorted(set(prepared_output_ids) - set(bound_ids))
    if missing_prepared_outputs:
        raise InputValidationError(
            "prepared component outputs must be bound for final assembly: "
            f"{missing_prepared_outputs}"
        )
    preparation_sources_bound = sorted(
        set(prepared.source_asset_ids) & set(bound_ids)
        if prepared is not None
        else set()
    )
    if preparation_sources_bound:
        raise InputValidationError(
            "final assembly bundle must contain approved component outputs, not "
            f"their raw preparation sources: {preparation_sources_bound}"
        )
    invalid_prepared_outputs = sorted(
        asset_id
        for asset_id in prepared_output_ids
        if (
            catalog.get(asset_id) is None
            or catalog.get(asset_id).source_tier != "approved_generated"
        )
    )
    if invalid_prepared_outputs:
        raise InputValidationError(
            "prepared component outputs must be registered approved_generated "
            f"assets: {invalid_prepared_outputs}"
        )
    policy = None
    if assembly_policy is not None:
        policy = (
            assembly_policy
            if isinstance(assembly_policy, FirstFrameAssemblyPolicy)
            else parse_first_frame_assembly_policy(assembly_policy)
        )
        if policy.shot_id != shot_doc.shot_id:
            raise InputValidationError("assembly policy shot_id does not match shot")
    if content is not None:
        existing_interactions = {
            item.interaction_id: item
            for item in (policy.interactions if policy is not None else [])
        }
        component_subjects = {
            item.component_id: item.subject
            for item in [*content.character_states, *content.prop_states]
        }
        prepared_evidence: dict[str, list[str]] = {}
        if prepared is not None:
            for task in prepared.tasks:
                for interaction in content.contact_graph:
                    participants = {
                        interaction.actor_component_id,
                        interaction.target_component_id,
                    }
                    if participants <= set(task.component_ids):
                        prepared_evidence.setdefault(
                            interaction.interaction_id, []
                        ).append(task.output_asset_id)
        interaction_payloads = []
        for interaction in content.contact_graph:
            existing = existing_interactions.get(interaction.interaction_id)
            actor = component_subjects[interaction.actor_component_id]
            target = component_subjects[interaction.target_component_id]
            if existing is not None and (
                _norm(existing.actor) != _norm(actor)
                or _norm(existing.target) != _norm(target)
                or existing.relation != interaction.relation
                or existing.required_state != interaction.required_visible_state
            ):
                raise InputValidationError(
                    f"assembly policy interaction {interaction.interaction_id!r} "
                    "conflicts with approved content truth"
                )
            interaction_payloads.append(
                {
                    "interaction_id": interaction.interaction_id,
                    "actor": actor,
                    "target": target,
                    "relation": interaction.relation,
                    "required_state": interaction.required_visible_state,
                    "evidence_asset_ids": prepared_evidence.get(
                        interaction.interaction_id,
                        list(existing.evidence_asset_ids) if existing else [],
                    ),
                    "hard_gate": interaction.hard_gate,
                }
            )
        content_ids = {item.interaction_id for item in content.contact_graph}
        interaction_payloads.extend(
            item.model_dump(mode="json")
            for item in existing_interactions.values()
            if item.interaction_id not in content_ids
        )
        policy = parse_first_frame_assembly_policy(
            {
                "schema_version": "first-frame-assembly-policy-v1",
                "shot_id": shot_doc.shot_id,
                "reference_authorities": (
                    [
                        item.model_dump(mode="json")
                        for item in policy.reference_authorities
                    ]
                    if policy is not None
                    else []
                ),
                "interactions": interaction_payloads,
                "require_production_quality_review": True,
            }
        )
    explicit_rules = (
        {item.asset_id: item for item in policy.reference_authorities}
        if policy is not None
        else {}
    )
    unknown_rules = sorted(set(explicit_rules) - set(bound_ids))
    if unknown_rules:
        raise InputValidationError(
            f"assembly policy references assets not bound to the shot: {unknown_rules}"
        )
    authorities: dict[str, ReferenceAuthority] = {
        item.asset_id: (
            explicit_rules[item.asset_id].authority
            if item.asset_id in explicit_rules
            else _default_authority(item)
        )
        for item in all_records
    }
    records_by_bound_id = {item.asset_id: item for item in all_records}
    invalid_identity = sorted(
        asset_id
        for asset_id, authority in authorities.items()
        if authority in {"identity_only", "action_only"}
        and records_by_bound_id[asset_id].asset_type != "character"
    )
    if invalid_identity:
        raise InputValidationError(
            "identity_only/action_only authority requires character assets: "
            f"{invalid_identity}"
        )
    invalid_overlays = sorted(
        asset_id
        for asset_id, authority in authorities.items()
        if authority == "deterministic_overlay"
        and records_by_bound_id[asset_id].asset_type != "foreground"
    )
    if invalid_overlays:
        raise InputValidationError(
            "deterministic_overlay authority requires foreground assets: "
            f"{invalid_overlays}"
        )
    legacy_deferred = list(dict.fromkeys(deferred_asset_ids))
    unknown_legacy_deferred = sorted(set(legacy_deferred) - set(bound_ids))
    if unknown_legacy_deferred:
        raise InputValidationError(
            f"deferred first-frame assets are not bound: {unknown_legacy_deferred}"
        )
    conflicts = sorted(
        asset_id
        for asset_id in legacy_deferred
        if asset_id in explicit_rules and explicit_rules[asset_id].authority != "hidden"
    )
    if conflicts:
        raise InputValidationError(
            "legacy deferred assets conflict with assembly-policy authority: "
            f"{conflicts}"
        )
    for asset_id in legacy_deferred:
        authorities[asset_id] = "hidden"
    deferred = [
        asset_id for asset_id in bound_ids if authorities.get(asset_id) == "hidden"
    ]
    control = [
        asset_id
        for asset_id in bound_ids
        if authorities.get(asset_id) == "structure_only"
    ]
    deferred_set = set(deferred)
    excluded_from_render = deferred_set | set(control)
    records = [
        item for item in all_records if item.asset_id not in excluded_from_render
    ]
    records_by_id = {item.asset_id: item for item in records}
    prepared_task_by_output = (
        {task.output_asset_id: task for task in prepared.tasks}
        if prepared is not None
        else {}
    )
    prepared_record_by_component = {
        component_id: records_by_id[task.output_asset_id]
        for task in (prepared.tasks if prepared is not None else [])
        for component_id in task.component_ids
    }
    warnings: list[str] = []
    for asset_id in control:
        reason = explicit_rules.get(asset_id)
        warnings.append(
            f"structure-only reference {asset_id} is retained as planning "
            "evidence and will not be uploaded to the image model"
            + (f": {reason.reason}" if reason is not None else "")
        )
    selected: list[str] = []
    components: list[FirstFrameComponent] = []
    coverage: list[InformationCoverageFact] = []
    stages: list[FirstFrameStage] = []

    def select(record: ImageAssetRecord) -> None:
        if record.asset_id not in selected:
            selected.append(record.asset_id)

    def add_stage(
        *,
        operation: StageOperation,
        component_ids: list[str],
        asset: ImageAssetRecord | None = None,
        base_assets: list[ImageAssetRecord] | None = None,
        instruction: str | None = None,
        attributes: list[str] | None = None,
        text_fallbacks: dict[str, str] | None = None,
    ) -> None:
        order = len(stages) + 1
        stage_id = f"stage_{order:03d}"
        inputs: list[dict] = []
        if operation == "adopt_anchor":
            assert asset is not None
            inputs.append(
                {
                    "slot": 1,
                    "source_type": "asset",
                    "source_id": asset.asset_id,
                    "role": "base",
                }
            )
        elif operation == "synthesize_base":
            for slot, item in enumerate(base_assets or [], start=1):
                inputs.append(
                    {
                        "slot": slot,
                        "source_type": "asset",
                        "source_id": item.asset_id,
                        "role": _stage_role(item),
                    }
                )
        else:
            inputs.append(
                {
                    "slot": 1,
                    "source_type": "stage_output",
                    "source_id": stages[-1].stage_id,
                    "role": "source_frame",
                }
            )
            if operation in {"fuse_component", "composite_overlay"}:
                assert asset is not None
                inputs.append(
                    {
                        "slot": 2,
                        "source_type": "asset",
                        "source_id": asset.asset_id,
                        "role": _stage_role(asset),
                    }
                )
        stages.append(
            FirstFrameStage(
                stage_id=stage_id,
                order=order,
                operation=operation,
                component_ids=component_ids,
                inputs=inputs,
                instruction=instruction,
                reference_attributes=attributes or [],
                text_fallbacks=text_fallbacks or {},
            )
        )

    backgrounds = sorted(
        [item for item in records if item.asset_type == "background"],
        key=lambda item: _record_rank(item, shot_doc.setting),
    )
    styles = sorted(
        [item for item in records if item.asset_type == "style"],
        key=_record_rank,
    )
    anchor: ImageAssetRecord | None = None
    if full_frame_anchor_asset_id is not None:
        if authorities.get(full_frame_anchor_asset_id) in {
            "structure_only",
            "hidden",
        }:
            raise InputValidationError(
                "full-frame anchor must be admitted as renderable visual truth",
                asset_id=full_frame_anchor_asset_id,
            )
        anchor = records_by_id.get(full_frame_anchor_asset_id)
        if anchor is None:
            raise InputValidationError(
                "full_frame_anchor_asset_id is not in the bound reference bundle",
                asset_id=full_frame_anchor_asset_id,
            )
        if anchor.asset_type not in {"background", "style"}:
            raise InputValidationError(
                "full-frame anchor must be a background or style asset",
                asset_id=full_frame_anchor_asset_id,
            )

    scene_assets: list[ImageAssetRecord] = []
    if anchor is not None:
        select(anchor)
        scene_assets.append(anchor)
        add_stage(
            operation="adopt_anchor",
            component_ids=["scene"],
            asset=anchor,
        )
        for style in styles:
            if style.asset_id != anchor.asset_id:
                warnings.append(
                    f"style asset {style.asset_id} was not applied because the "
                    "explicit full-frame anchor is the style authority"
                )
    else:
        base_inputs: list[ImageAssetRecord] = []
        if backgrounds:
            base_inputs.append(backgrounds[0])
        if styles and styles[0] not in base_inputs:
            base_inputs.append(styles[0])
        for item in base_inputs:
            select(item)
            scene_assets.append(item)
        base_fallbacks = {
            "composition": shot_doc.composition,
            "camera": shot_doc.camera_position,
        }
        if not backgrounds:
            base_fallbacks["setting"] = shot_doc.setting
        if not shot_doc.subjects:
            base_fallbacks["starting_state"] = shot_doc.start_state
        add_stage(
            operation="synthesize_base",
            component_ids=["scene"],
            base_assets=base_inputs,
            instruction=(
                "Establish the first-frame scene canvas. Preserve every visual "
                "fact supplied by the selected scene/style references and apply "
                "only the uncovered camera and layout facts."
            ),
            attributes=[
                attribute
                for item in base_inputs
                for attribute in _reference_attributes(item)
            ],
            text_fallbacks=base_fallbacks,
        )
        if not base_inputs:
            warnings.append(
                "no scene or style reference is available; the base requires "
                "text-only synthesis and has elevated style-drift risk"
            )

    additional_backgrounds = [
        item
        for item in backgrounds
        if item.asset_id not in {asset.asset_id for asset in scene_assets}
    ]
    for item in additional_backgrounds:
        select(item)
        scene_assets.append(item)
        note = bundle_notes.get(item.asset_id)
        detail = f" Bound intent: {note}." if note else ""
        add_stage(
            operation="fuse_component",
            component_ids=["scene"],
            asset=item,
            instruction=(
                "Incorporate only the additional location-specific scene "
                "information from the component reference into the current "
                f"canvas; preserve all existing composition and style.{detail}"
            ),
            attributes=_reference_attributes(item),
        )

    components.append(
        FirstFrameComponent(
            component_id="scene",
            kind="scene",
            subject=shot_doc.setting,
            target_state=shot_doc.start_state,
            spatial_instruction=(
                f"{shot_doc.composition}; camera {shot_doc.camera_position}"
            ),
            reference_asset_ids=[item.asset_id for item in scene_assets],
            reference_attributes=list(
                dict.fromkeys(
                    attribute
                    for item in scene_assets
                    for attribute in _reference_attributes(item)
                )
            ),
        )
    )
    coverage.append(
        InformationCoverageFact(
            fact_id="scene.identity",
            component_id="scene",
            attribute="scene_identity",
            status="visible" if scene_assets else "unresolved",
            source_asset_ids=[item.asset_id for item in scene_assets],
            note=(
                "scene reference is the spatial visual authority"
                if scene_assets
                else "no scene reference is available"
            ),
        )
    )
    style_sources = [
        item.asset_id
        for item in scene_assets
        if item.asset_type == "style" or "style_reference" in item.reference_roles
    ]
    if not style_sources and anchor is not None:
        style_sources = [anchor.asset_id]
    coverage.append(
        InformationCoverageFact(
            fact_id="scene.style",
            component_id="scene",
            attribute="visual_style",
            status="visible" if style_sources else "unresolved",
            source_asset_ids=style_sources,
            note=(
                "selected visual reference controls frame style"
                if style_sources
                else "no explicit style authority is available"
            ),
        )
    )
    if anchor is not None:
        coverage.extend(
            [
                InformationCoverageFact(
                    fact_id="scene.composition",
                    component_id="scene",
                    attribute="composition",
                    status="visible",
                    source_asset_ids=[anchor.asset_id],
                    note="the explicit full-frame anchor controls composition",
                ),
                InformationCoverageFact(
                    fact_id="scene.camera",
                    component_id="scene",
                    attribute="camera",
                    status="visible",
                    source_asset_ids=[anchor.asset_id],
                    note="the explicit full-frame anchor controls camera",
                ),
            ]
        )
    else:
        coverage.extend(
            [
                InformationCoverageFact(
                    fact_id="scene.composition",
                    component_id="scene",
                    attribute="composition",
                    status="unresolved",
                    source_asset_ids=[],
                    note="composition is an uncovered structured/text fallback",
                ),
                InformationCoverageFact(
                    fact_id="scene.camera",
                    component_id="scene",
                    attribute="camera",
                    status="unresolved",
                    source_asset_ids=[],
                    note="camera is an uncovered structured/text fallback",
                ),
            ]
        )

    character_records = [item for item in records if item.asset_type == "character"]
    deferred_character_records = [
        item
        for item in all_records
        if item.asset_type == "character" and item.asset_id in deferred_set
    ]
    has_foreground_reference = any(item.asset_type == "foreground" for item in records)
    assigned_character_ids: set[str] = set()
    ambiguous_characters = False
    for index, subject in enumerate(shot_doc.subjects, start=1):
        component_id = f"character_{index:03d}"
        content_character = (
            next(
                (
                    item
                    for item in content.character_states
                    if item.component_id == component_id
                ),
                None,
            )
            if content is not None
            else None
        )
        placement_constraint = (
            f" Placement constraint: {content_character.screen_position}."
            if content_character is not None and content_character.screen_position
            else ""
        )
        prepared_record = prepared_record_by_component.get(component_id)
        matches = [
            item
            for item in character_records
            if item.asset_id not in assigned_character_ids and _matches(item, subject)
        ]
        if prepared_record is not None:
            matches = [prepared_record]
        if not matches and len(shot_doc.subjects) == 1:
            matches = [
                item
                for item in character_records
                if item.asset_id not in assigned_character_ids
            ]
            if matches:
                warnings.append(
                    f"character references for {subject!r} had no metadata "
                    "match and were assigned because it is the only subject"
                )
        matches.sort(key=lambda item: _record_rank(item, subject))
        attributes: list[str] = []
        for item in matches:
            was_selected = item.asset_id in selected
            select(item)
            assigned_character_ids.add(item.asset_id)
            prepared_task = prepared_task_by_output.get(item.asset_id)
            item_attributes = _prepared_reference_attributes(item, prepared_task)
            placement_detail = _prepared_placement_instruction(prepared_task)
            attributes.extend(item_attributes)
            note = bundle_notes.get(item.asset_id)
            detail = f" Bound intent: {note}." if note else ""
            occlusion_staging = (
                f" Fully render {subject}'s canonical appearance and required "
                "placement now even if the final start state will obscure part "
                "of the character; do not omit the character. A later foreground "
                "stage will apply the intended occlusion."
                if has_foreground_reference
                else ""
            )
            if not was_selected:
                stage_component_ids = (
                    list(prepared_task.component_ids)
                    if prepared_task is not None
                    else [component_id]
                )
                deterministic_layout_overlay = bool(
                    prepared_task is not None
                    and prepared_task.external_attachments
                    and item.asset_type == "foreground"
                )
                if deterministic_layout_overlay:
                    add_stage(
                        operation="composite_overlay",
                        component_ids=stage_component_ids,
                        asset=item,
                        attributes=item_attributes,
                    )
                    add_stage(
                        operation="fuse_component",
                        component_ids=stage_component_ids,
                        asset=item,
                        instruction=(
                            "Harmonize only the deterministic component overlay's "
                            "edges, lighting and material integration. Preserve its "
                            "approved scale, screen position, facing, action axis, "
                            "contact geometry and external target relationship."
                            f"{placement_detail}{detail}"
                        ),
                        attributes=item_attributes,
                    )
                else:
                    add_stage(
                        operation="fuse_component",
                        component_ids=stage_component_ids,
                        asset=item,
                        instruction=(
                            f"Place or update {subject} only as required by the "
                            f"shot's starting action/state: {shot_doc.start_state}. "
                            "Keep the established scene, camera, lighting and style "
                            f"unchanged.{placement_constraint}"
                            f"{occlusion_staging}{placement_detail}{detail}"
                        ),
                        attributes=item_attributes,
                    )
        if not matches:
            warnings.append(
                f"no visual character reference is available for {subject!r}; "
                "the character requires a text fallback"
            )
            add_stage(
                operation="apply_text_delta",
                component_ids=[component_id],
                instruction=(
                    f"Add {subject} only in the required starting action/state "
                    "without changing the established scene or style."
                ),
                text_fallbacks={"starting_state": shot_doc.start_state},
            )
        components.append(
            FirstFrameComponent(
                component_id=component_id,
                kind="character",
                subject=subject,
                target_state=shot_doc.start_state,
                spatial_instruction=shot_doc.composition,
                reference_asset_ids=[item.asset_id for item in matches],
                reference_attributes=list(dict.fromkeys(attributes)),
            )
        )
        identity_sources = [item.asset_id for item in matches]
        coverage.append(
            InformationCoverageFact(
                fact_id=f"{component_id}.identity",
                component_id=component_id,
                attribute="identity_and_appearance",
                status="visible" if identity_sources else "unresolved",
                source_asset_ids=identity_sources,
                note=(
                    "selected character references are the identity authority"
                    if identity_sources
                    else "no character identity reference is available"
                ),
            )
        )
        eye_sources = [
            item.asset_id
            for item in matches
            if "identity_reference" in item.reference_roles
            or "expression_reference" in item.reference_roles
            or not item.reference_roles
        ]
        eye_sources.extend(
            item.asset_id
            for item in deferred_character_records
            if _matches(item, subject)
            and (
                "identity_reference" in item.reference_roles
                or "expression_reference" in item.reference_roles
                or not item.reference_roles
            )
        )
        eye_sources = list(dict.fromkeys(eye_sources))
        coverage.append(
            InformationCoverageFact(
                fact_id=f"{component_id}.eyes",
                component_id=component_id,
                attribute="eye_shape_and_colour",
                status=(
                    "occluded"
                    if eye_sources and _closed_eyes(shot_doc.start_state)
                    else ("known_from_reference" if eye_sources else "unresolved")
                ),
                source_asset_ids=eye_sources,
                note=(
                    "eyes are intentionally closed in the first frame; the "
                    "bound character references retain the hidden eye information"
                    if eye_sources and _closed_eyes(shot_doc.start_state)
                    else (
                        "eye information is retained by the bound character references"
                        if eye_sources
                        else "eye information is not available"
                    )
                ),
            )
        )
        visible_pose_sources = [
            item.asset_id
            for item in matches
            if "pose_reference" in item.reference_roles
            or item.pose
            or item.asset_id in prepared_task_by_output
        ]
        deferred_pose_sources = [
            item.asset_id
            for item in deferred_character_records
            if _matches(item, subject)
            and ("pose_reference" in item.reference_roles or item.pose)
        ]
        pose_sources = list(
            dict.fromkeys([*visible_pose_sources, *deferred_pose_sources])
        )
        coverage.append(
            InformationCoverageFact(
                fact_id=f"{component_id}.pose",
                component_id=component_id,
                attribute="pose",
                status=(
                    "visible"
                    if visible_pose_sources
                    else (
                        "known_from_reference"
                        if deferred_pose_sources
                        else "unresolved"
                    )
                ),
                source_asset_ids=pose_sources,
                note=(
                    "pose is controlled by a visual reference"
                    if visible_pose_sources
                    else (
                        "future pose information is retained by a deferred "
                        "character reference"
                        if deferred_pose_sources
                        else "pose is supplied only by the starting-state instruction"
                    )
                ),
            )
        )

    unassigned_characters = [
        item
        for item in character_records
        if item.asset_id not in assigned_character_ids
    ]
    if unassigned_characters:
        ambiguous_characters = bool(shot_doc.subjects)
        warnings.append(
            "unassigned character references require review: "
            + ", ".join(item.asset_id for item in unassigned_characters)
        )

    prop_records = [item for item in records if item.asset_type == "prop"]
    assigned_props: set[str] = set()
    for index, prop in enumerate(shot_doc.props, start=1):
        component_id = f"prop_{index:03d}"
        prepared_record = prepared_record_by_component.get(component_id)
        matches = [
            item
            for item in prop_records
            if item.asset_id not in assigned_props and _matches(item, prop)
        ]
        if prepared_record is not None:
            matches = [prepared_record]
        if not matches and len(shot_doc.props) == 1:
            matches = [
                item for item in prop_records if item.asset_id not in assigned_props
            ]
        matches.sort(key=lambda item: _record_rank(item, prop))
        for item in matches:
            was_selected = item.asset_id in selected
            select(item)
            assigned_props.add(item.asset_id)
            prepared_task = prepared_task_by_output.get(item.asset_id)
            if not was_selected:
                add_stage(
                    operation="fuse_component",
                    component_ids=(
                        list(prepared_task.component_ids)
                        if prepared_task is not None
                        else [component_id]
                    ),
                    asset=item,
                    instruction=(
                        f"Add or correct only the required prop {prop}; preserve the "
                        "current people, scene, composition, lighting and style."
                    ),
                    attributes=_prepared_reference_attributes(item, prepared_task),
                )
        if not matches:
            warnings.append(
                f"no visual prop reference is available for {prop!r}; using text"
            )
            add_stage(
                operation="apply_text_delta",
                component_ids=[component_id],
                instruction=(
                    f"Add only the required prop {prop}; preserve all existing "
                    "visual facts."
                ),
                text_fallbacks={"prop": prop},
            )
        components.append(
            FirstFrameComponent(
                component_id=component_id,
                kind="prop",
                subject=prop,
                target_state=shot_doc.start_state,
                spatial_instruction=shot_doc.composition,
                reference_asset_ids=[item.asset_id for item in matches],
                reference_attributes=list(
                    dict.fromkeys(
                        attribute
                        for item in matches
                        for attribute in _prepared_reference_attributes(
                            item, prepared_task_by_output.get(item.asset_id)
                        )
                    )
                ),
            )
        )
        coverage.append(
            InformationCoverageFact(
                fact_id=f"{component_id}.appearance",
                component_id=component_id,
                attribute="prop_appearance",
                status="visible" if matches else "unresolved",
                source_asset_ids=[item.asset_id for item in matches],
                note=(
                    "prop appearance is controlled by visual references"
                    if matches
                    else "prop appearance has no visual reference"
                ),
            )
        )

    foreground_records = sorted(
        [
            item
            for item in records
            if item.asset_type == "foreground"
            and item.asset_id not in set(prepared_output_ids)
        ],
        key=_record_rank,
    )
    for index, item in enumerate(foreground_records, start=1):
        component_id = f"foreground_{index:03d}"
        select(item)
        if authorities[item.asset_id] == "deterministic_overlay":
            add_stage(
                operation="composite_overlay",
                component_ids=[component_id],
                asset=item,
                attributes=_reference_attributes(item),
            )
        else:
            add_stage(
                operation="fuse_component",
                component_ids=[component_id],
                asset=item,
                instruction=(
                    "Add only the foreground and occlusion information from the "
                    "component reference; preserve all established identities, "
                    "scene geometry, camera, lighting and style."
                ),
                attributes=_reference_attributes(item),
            )
        components.append(
            FirstFrameComponent(
                component_id=component_id,
                kind="foreground",
                subject=item.subject_or_scene_id or item.asset_id,
                target_state=bundle_notes.get(item.asset_id) or "foreground layer",
                spatial_instruction=shot_doc.composition,
                reference_asset_ids=[item.asset_id],
                reference_attributes=_reference_attributes(item),
            )
        )
        coverage.append(
            InformationCoverageFact(
                fact_id=f"{component_id}.appearance",
                component_id=component_id,
                attribute="foreground_and_occlusion",
                status="visible",
                source_asset_ids=[item.asset_id],
                note="foreground reference controls appearance and occlusion",
            )
        )

    if content is not None and len(stages) > 1:
        layer_rank = {item.component_id: item.order for item in content.layers}
        original_index = {item.stage_id: index for index, item in enumerate(stages)}
        ordered_stages = [stages[0]] + sorted(
            stages[1:],
            key=lambda item: (
                min(
                    (layer_rank.get(component_id, len(layer_rank) + 1)
                     for component_id in item.component_ids),
                    default=len(layer_rank) + 1,
                ),
                original_index[item.stage_id],
            ),
        )
        rewritten_stages: list[FirstFrameStage] = []
        for index, stage in enumerate(ordered_stages, start=1):
            payload = stage.model_dump(mode="json")
            payload["stage_id"] = f"stage_{index:03d}"
            payload["order"] = index
            if index > 1:
                for item in payload["inputs"]:
                    if item["source_type"] == "stage_output":
                        item["source_id"] = f"stage_{index - 1:03d}"
            rewritten_stages.append(FirstFrameStage.model_validate(payload))
        stages = rewritten_stages

    selected_set = set(selected)
    unselected_prepared_outputs = sorted(set(prepared_output_ids) - selected_set)
    if unselected_prepared_outputs:
        raise InputValidationError(
            "prepared component outputs did not map to final frame components: "
            f"{unselected_prepared_outputs}"
        )
    unused = [
        asset_id
        for asset_id in bound_ids
        if asset_id not in selected_set
        and asset_id not in set(control)
        and asset_id not in set(deferred)
    ]
    if unused:
        warnings.append(
            "bound references not consumed because they were ambiguous or "
            "conflicted with the selected full-frame authority: " + ", ".join(unused)
        )
    admissions: list[FirstFrameReferenceAdmission] = []
    for asset_id in bound_ids:
        authority = authorities[asset_id]
        if asset_id in selected_set:
            disposition: ReferenceDisposition = "selected_visual"
        elif asset_id in set(control):
            disposition = "control_only"
        elif asset_id in set(deferred):
            disposition = "deferred_hidden"
        else:
            disposition = "unused"
        rule = explicit_rules.get(asset_id)
        admissions.append(
            FirstFrameReferenceAdmission(
                asset_id=asset_id,
                authority=authority,
                disposition=disposition,
                reason=(
                    rule.reason
                    if rule is not None
                    else (
                        "legacy deferred first-frame information"
                        if asset_id in legacy_deferred
                        else "authority derived from asset type and reference roles"
                    )
                ),
            )
        )

    interactions: list[FirstFrameInteractionUnit] = []
    blocked_interaction = False
    if policy is not None:
        for requirement in policy.interactions:
            unknown_evidence = sorted(
                set(requirement.evidence_asset_ids) - set(bound_ids)
            )
            if unknown_evidence:
                raise InputValidationError(
                    f"interaction {requirement.interaction_id!r} uses unbound "
                    f"evidence assets: {unknown_evidence}"
                )
            actor_component = _component_for_selector(
                requirement.actor,
                components,
            )
            target_component = _component_for_selector(
                requirement.target,
                components,
            )
            participant_ids = list(
                dict.fromkeys(
                    item
                    for item in (actor_component, target_component)
                    if item is not None
                )
            )
            selected_evidence = [
                asset_id
                for asset_id in requirement.evidence_asset_ids
                if asset_id in selected_set
                and authorities[asset_id]
                in {"final_visual", "identity_only", "action_only"}
            ]
            structure_evidence = [
                asset_id
                for asset_id in requirement.evidence_asset_ids
                if asset_id in set(control)
            ]
            if actor_component is None or target_component is None:
                grounding: InteractionGrounding = "unresolved"
                blocked_interaction = blocked_interaction or requirement.hard_gate
            elif selected_evidence:
                grounding = "visually_grounded"
            elif structure_evidence:
                grounding = "structure_only"
            else:
                grounding = "text_only"
            interactions.append(
                FirstFrameInteractionUnit(
                    interaction_id=requirement.interaction_id,
                    actor=requirement.actor,
                    target=requirement.target,
                    relation=requirement.relation,
                    required_state=requirement.required_state,
                    participant_component_ids=participant_ids,
                    evidence_asset_ids=list(requirement.evidence_asset_ids),
                    grounding=grounding,
                    hard_gate=requirement.hard_gate,
                )
            )
            if grounding != "visually_grounded":
                warnings.append(
                    f"interaction {requirement.interaction_id} is {grounding}; "
                    "co-presence does not prove the required contact relationship"
                )

    attachments: list[FirstFrameAttachmentUnit] = []
    blocked_attachment = False
    if content is not None:
        prepared_attachment_evidence: dict[str, list[str]] = {}
        if prepared is not None:
            for task in prepared.tasks:
                for attachment in task.external_attachments:
                    prepared_attachment_evidence.setdefault(
                        attachment.attachment_id,
                        [],
                    ).append(task.output_asset_id)
        for attachment in content.attachment_graph:
            evidence = [
                asset_id
                for asset_id in prepared_attachment_evidence.get(
                    attachment.attachment_id,
                    [],
                )
                if asset_id in selected_set
            ]
            grounding: InteractionGrounding = (
                "visually_grounded" if evidence else "text_only"
            )
            if attachment.hard_gate and grounding != "visually_grounded":
                blocked_attachment = True
            attachments.append(
                FirstFrameAttachmentUnit(
                    attachment_id=attachment.attachment_id,
                    source_component_id=attachment.source_component_id,
                    source_anchor=attachment.source_anchor,
                    source_anchor_x=attachment.source_anchor_position.x,
                    source_anchor_y=attachment.source_anchor_position.y,
                    target_component_id=attachment.target_component_id,
                    target_anchor=attachment.target_anchor,
                    target_anchor_x=attachment.target_anchor_position.x,
                    target_anchor_y=attachment.target_anchor_position.y,
                    relation=attachment.relation,
                    action_axis=attachment.action_axis,
                    initial_gap=attachment.initial_gap,
                    required_visible_state=attachment.required_visible_state,
                    must_remain_visible=list(attachment.must_remain_visible),
                    source_must_remain_visible=list(
                        attachment.source_must_remain_visible
                    ),
                    target_must_remain_visible=list(
                        attachment.target_must_remain_visible
                    ),
                    evidence_asset_ids=evidence,
                    grounding=grounding,
                    hard_gate=attachment.hard_gate,
                )
            )
            if grounding != "visually_grounded":
                warnings.append(
                    f"external attachment {attachment.attachment_id} is {grounding}; "
                    "the final-frame action axis lacks an approved component plate"
                )

    quality_gates: list[FirstFrameQualityGate] = []
    if policy is not None:
        review_stage_id = stages[-1].stage_id
        if policy.require_production_quality_review:
            quality_gates.append(
                FirstFrameQualityGate(
                    gate_id="gate.production_quality",
                    kind="production_quality",
                    description=(
                        "Reject storyboard-like geometry, malformed anatomy, "
                        "floating objects and visibly unintegrated layers."
                    ),
                    review_stage_id=review_stage_id,
                )
            )
        if content is not None:
            quality_gates.extend(
                FirstFrameQualityGate(
                    gate_id=f"gate.content.{gate.gate_id}",
                    kind="production_quality",
                    description=gate.criterion,
                    review_stage_id=review_stage_id,
                )
                for gate in content.acceptance_gates
                if gate.tier == "reject"
            )
        for interaction in interactions:
            if interaction.hard_gate:
                quality_gates.append(
                    FirstFrameQualityGate(
                        gate_id=f"gate.interaction.{interaction.interaction_id}",
                        kind="interaction",
                        description=(
                            f"Visually confirm {interaction.required_state}; "
                            "mere co-presence is insufficient."
                        ),
                        review_stage_id=review_stage_id,
                        interaction_id=interaction.interaction_id,
                        asset_ids=list(interaction.evidence_asset_ids),
                    )
                )
        for attachment in attachments:
            if attachment.hard_gate:
                quality_gates.append(
                    FirstFrameQualityGate(
                        gate_id=f"gate.attachment.{attachment.attachment_id}",
                        kind="attachment",
                        description=(
                            f"Visually confirm {attachment.source_anchor} "
                            f"{attachment.relation} {attachment.target_anchor} along "
                            f"{attachment.action_axis}, the gap remains "
                            f"{attachment.initial_gap}, and these are visible: "
                            f"{', '.join(attachment.must_remain_visible)}."
                        ),
                        review_stage_id=review_stage_id,
                        attachment_id=attachment.attachment_id,
                        asset_ids=list(attachment.evidence_asset_ids),
                    )
                )
        for asset_id in deferred:
            quality_gates.append(
                FirstFrameQualityGate(
                    gate_id=f"gate.hidden.{asset_id}",
                    kind="hidden_information",
                    description=(
                        f"Confirm hidden asset {asset_id} contributed no visible "
                        "information to the first frame."
                    ),
                    review_stage_id=review_stage_id,
                    asset_ids=[asset_id],
                )
            )
        if content is not None:
            quality_gates.extend(
                FirstFrameQualityGate(
                    gate_id=f"gate.content.hidden.{fact.fact_id}",
                    kind="hidden_information",
                    description=(
                        f"Confirm deferred K0 information remains invisible: {fact.fact}. "
                        f"Reason: {fact.reason}"
                    ),
                    review_stage_id=review_stage_id,
                )
                for fact in content.information
                if fact.state == "deferred"
            )
        if quality_gates:
            final_stage = stages[-1].model_dump(mode="json")
            final_stage["quality_gate_ids"] = [item.gate_id for item in quality_gates]
            stages[-1] = FirstFrameStage.model_validate(final_stage)

    unresolved = any(item.status == "unresolved" for item in coverage)
    if blocked_interaction or blocked_attachment:
        decision: PlanDecision = "blocked"
    else:
        decision = (
            "needs_review"
            if warnings or unresolved or ambiguous_characters or quality_gates
            else "ready"
        )
    return FirstFramePlanDocument(
        plan_id=f"{shot_doc.shot_id}-first-frame",
        shot_id=shot_doc.shot_id,
        keyframe_id=f"{shot_doc.shot_id}_first",
        content_plan_id=content.plan_id if content is not None else None,
        prepared_component_plan_id=(prepared.plan_id if prepared is not None else None),
        prepared_component_asset_ids=prepared_output_ids,
        review_status="draft",
        decision=decision,
        intent=FirstFrameIntent(
            setting=shot_doc.setting,
            composition=shot_doc.composition,
            camera=shot_doc.camera_position,
            start_state=shot_doc.start_state,
            subjects=list(shot_doc.subjects),
            props=list(shot_doc.props),
        ),
        components=components,
        information_coverage=coverage,
        bound_asset_ids=bound_ids,
        selected_asset_ids=selected,
        control_asset_ids=control,
        deferred_asset_ids=deferred,
        unused_bound_asset_ids=unused,
        reference_admissions=admissions,
        interaction_units=interactions,
        attachment_units=attachments,
        quality_gates=quality_gates,
        stages=stages,
        warnings=warnings,
    )
