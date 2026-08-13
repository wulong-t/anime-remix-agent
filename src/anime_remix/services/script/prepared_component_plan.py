"""WHO/HOW preparation contract for production-quality first-frame plates.

Preparation is deliberately separate from final frame assembly.  A task may
upload at most two visual references.  Identity/appearance always comes from
visual references; text fallbacks are restricted to action, contact and
placement.  Generated plates must be registered and manually approved before
the plan can be completed and consumed by ``first-frame-plan-v1``.
"""

from __future__ import annotations

import hashlib
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
from anime_remix.services.image_assets import ImageAssetCatalog, ImageAssetRecord
from anime_remix.services.script.first_frame_assembly_policy import (
    FirstFrameAssemblyPolicy,
    ReferenceAuthority,
    parse_first_frame_assembly_policy,
)
from anime_remix.services.script.first_frame_content_plan import (
    ExternalAttachmentTruth,
    FirstFrameContentPlanDocument,
    PropFunctionalAffordanceTruth,
    parse_first_frame_content_plan,
)

_SCHEMA_VERSION = "prepared-component-plan-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_ASSET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

TaskKind = Literal["character_plate", "interaction_plate"]
InputFunction = Literal["who", "how", "prop_visual"]
TaskResult = Literal["pending", "approved", "rejected"]
GateResult = Literal["pass", "fail"]
ReviewStatus = Literal["draft", "approved"]
CompletionStatus = Literal["pending", "completed"]
PlanDecision = Literal["ready", "needs_review", "blocked"]
_ALLOWED_TEXT_KEYS = frozenset(
    {
        "action",
        "contact_relation",
        "spatial_placement",
        "occlusion",
        "prop_affordance",
        "facing_direction",
        "action_axis",
        "external_target",
        "required_visibility",
    }
)


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be non-empty")
    return cleaned


def _clean_id(value: object, field: str, *, asset: bool = False) -> str:
    cleaned = _clean_text(value, field)
    pattern = _ASSET_ID_PATTERN if asset else _ID_PATTERN
    if not pattern.fullmatch(cleaned):
        raise ValueError(f"invalid {field} {value!r}")
    return cleaned


def _clean_ids(value: object, field: str, *, asset: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    cleaned = [_clean_id(item, f"{field} item", asset=asset) for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} must contain unique values")
    return cleaned


class PreparedComponentInput(BaseModel):
    model_config = _STRICT_CONFIG

    asset_id: str
    function: InputFunction

    @field_validator("asset_id", mode="before")
    @classmethod
    def _asset_id(cls, value: object) -> object:
        return _clean_id(value, "asset_id", asset=True)


class PreparedPropAffordance(BaseModel):
    model_config = _STRICT_CONFIG

    component_id: str
    subject: str
    grip_zone: str
    active_end: str
    native_action_axis: str

    @field_validator("component_id", mode="before")
    @classmethod
    def _component_id(cls, value: object) -> object:
        return _clean_id(value, "component_id")

    @field_validator(
        "subject", "grip_zone", "active_end", "native_action_axis", mode="before"
    )
    @classmethod
    def _text(cls, value: object, info) -> object:
        return _clean_text(value, info.field_name)


class PreparedExternalAttachment(BaseModel):
    model_config = _STRICT_CONFIG

    attachment_id: str
    source_component_id: str
    source_anchor: str
    source_anchor_x: float
    source_anchor_y: float
    target_component_id: str
    target_subject: str
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
    hard_gate: bool = True

    @field_validator(
        "attachment_id", "source_component_id", "target_component_id", mode="before"
    )
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator(
        "source_anchor",
        "target_subject",
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

    @field_validator(
        "must_remain_visible",
        "source_must_remain_visible",
        "target_must_remain_visible",
        mode="before",
    )
    @classmethod
    def _visible(cls, value: object, info) -> object:
        if not isinstance(value, list):
            raise TypeError(f"{info.field_name} must be a list")
        visible = [_clean_text(item, f"{info.field_name} item") for item in value]
        if not visible or len(visible) != len(set(visible)):
            raise ValueError(f"{info.field_name} must be non-empty and unique")
        return visible

    @model_validator(mode="after")
    def _visibility_partition(self) -> PreparedExternalAttachment:
        if {
            *self.source_must_remain_visible,
            *self.target_must_remain_visible,
        } != set(self.must_remain_visible):
            raise ValueError(
                "source and target visibility must partition must_remain_visible"
            )
        return self


class PreparedComponentReviewGate(BaseModel):
    model_config = _STRICT_CONFIG

    gate_id: str
    criterion: str

    @field_validator("gate_id", mode="before")
    @classmethod
    def _gate_id(cls, value: object) -> object:
        return _clean_id(value, "gate_id")

    @field_validator("criterion", mode="before")
    @classmethod
    def _criterion(cls, value: object) -> object:
        return _clean_text(value, "criterion")


class PreparedComponentGateResult(BaseModel):
    model_config = _STRICT_CONFIG

    gate_id: str
    result: GateResult
    note: str

    @field_validator("gate_id", mode="before")
    @classmethod
    def _gate_id(cls, value: object) -> object:
        return _clean_id(value, "gate_id")

    @field_validator("note", mode="before")
    @classmethod
    def _note(cls, value: object) -> object:
        return _clean_text(value, "note")


class PreparedComponentTask(BaseModel):
    model_config = _STRICT_CONFIG

    task_id: str
    kind: TaskKind
    component_ids: list[str]
    subjects: list[str]
    target_state: str
    model_inputs: list[PreparedComponentInput]
    control_evidence_asset_ids: list[str] = []
    preserve_attributes: list[str]
    allowed_text_fallbacks: dict[str, str] = {}
    prop_affordances: list[PreparedPropAffordance] = []
    external_attachments: list[PreparedExternalAttachment] = []
    review_gates: list[PreparedComponentReviewGate] = []
    output_asset_id: str
    result: TaskResult = "pending"
    result_review_notes: str | None = None
    gate_results: list[PreparedComponentGateResult] = []

    @field_validator("task_id", mode="before")
    @classmethod
    def _task_id(cls, value: object) -> object:
        return _clean_id(value, "task_id")

    @field_validator("output_asset_id", mode="before")
    @classmethod
    def _output_asset_id(cls, value: object) -> object:
        return _clean_id(value, "output_asset_id", asset=True)

    @field_validator("component_ids", mode="before")
    @classmethod
    def _component_ids(cls, value: object) -> object:
        return _clean_ids(value, "component_ids")

    @field_validator("control_evidence_asset_ids", mode="before")
    @classmethod
    def _control_ids(cls, value: object) -> object:
        return _clean_ids(value, "control_evidence_asset_ids", asset=True)

    @field_validator("subjects", "preserve_attributes", mode="before")
    @classmethod
    def _text_lists(cls, value: object, info) -> object:
        if not isinstance(value, list):
            raise TypeError(f"{info.field_name} must be a list")
        cleaned = [_clean_text(item, f"{info.field_name} item") for item in value]
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ValueError(f"{info.field_name} must be non-empty and unique")
        return cleaned

    @field_validator("target_state", mode="before")
    @classmethod
    def _target_state(cls, value: object) -> object:
        return _clean_text(value, "target_state")

    @field_validator("model_inputs", mode="before")
    @classmethod
    def _inputs(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("model_inputs must be a list")
        if not 1 <= len(value) <= 2:
            raise ValueError("component preparation requires one or two visual inputs")
        return value

    @field_validator("allowed_text_fallbacks", mode="before")
    @classmethod
    def _fallbacks(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("allowed_text_fallbacks must be a dict")
        unknown = set(value) - _ALLOWED_TEXT_KEYS
        if unknown:
            raise ValueError(
                "text fallbacks may describe only action/contact/placement/occlusion; "
                f"found {sorted(unknown)}"
            )
        return {
            _clean_text(key, "text fallback key"): _clean_text(item, "text fallback")
            for key, item in value.items()
        }

    @field_validator("result_review_notes", mode="before")
    @classmethod
    def _review_notes(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "result_review_notes")

    @model_validator(mode="after")
    def _task_contract(self) -> PreparedComponentTask:
        input_ids = [item.asset_id for item in self.model_inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("model input asset ids must be unique")
        if set(input_ids) & set(self.control_evidence_asset_ids):
            raise ValueError("control evidence must never be uploaded as model input")
        functions = [item.function for item in self.model_inputs]
        if self.kind == "character_plate" and "who" not in functions:
            raise ValueError("character plate requires a WHO visual reference")
        if self.kind == "interaction_plate" and not any(
            item in functions for item in ("who", "prop_visual")
        ):
            raise ValueError("interaction plate requires visual participant identity")
        affordance_ids = [item.component_id for item in self.prop_affordances]
        if len(affordance_ids) != len(set(affordance_ids)):
            raise ValueError("prop affordance component ids must be unique")
        if not set(affordance_ids) <= set(self.component_ids):
            raise ValueError("prop affordance references a component outside the task")
        attachment_ids = [item.attachment_id for item in self.external_attachments]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("external attachment ids must be unique within a task")
        if any(
            item.source_component_id not in set(self.component_ids)
            for item in self.external_attachments
        ):
            raise ValueError("external attachment source must belong to the task")
        gate_ids = [item.gate_id for item in self.review_gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("component review gate ids must be unique")
        result_ids = [item.gate_id for item in self.gate_results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("component gate result ids must be unique")
        if not set(result_ids) <= set(gate_ids):
            raise ValueError("component gate result references an unknown review gate")
        if self.result == "approved" and not self.result_review_notes:
            raise ValueError("approved component result requires review notes")
        if self.result == "approved" and self.review_gates:
            if set(result_ids) != set(gate_ids):
                raise ValueError(
                    "approved component result requires a result for every review gate"
                )
            if any(item.result != "pass" for item in self.gate_results):
                raise ValueError("approved component result requires every gate to pass")
        return self


class PreparedComponentPlanDocument(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["prepared-component-plan-v1"] = _SCHEMA_VERSION
    plan_id: str
    shot_id: str
    content_plan_id: str
    review_status: ReviewStatus
    completion_status: CompletionStatus
    decision: PlanDecision
    max_primary_visual_references_per_model_call: Literal[2] = 2
    source_asset_ids: list[str]
    tasks: list[PreparedComponentTask]
    warnings: list[str] = []

    @field_validator("plan_id", "shot_id", "content_plan_id", mode="before")
    @classmethod
    def _ids(cls, value: object, info) -> object:
        return _clean_id(value, info.field_name)

    @field_validator("source_asset_ids", mode="before")
    @classmethod
    def _source_ids(cls, value: object) -> object:
        return _clean_ids(value, "source_asset_ids", asset=True)

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("warnings must be a list")
        return [_clean_text(item, "warning") for item in value]

    @model_validator(mode="after")
    def _integrity(self) -> PreparedComponentPlanDocument:
        task_ids = [item.task_id for item in self.tasks]
        output_ids = [item.output_asset_id for item in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("component task ids must be unique")
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("component output asset ids must be unique")
        used_sources = {
            item.asset_id for task in self.tasks for item in task.model_inputs
        } | {
            item for task in self.tasks for item in task.control_evidence_asset_ids
        }
        if used_sources != set(self.source_asset_ids):
            raise ValueError("source_asset_ids must exactly cover task evidence")
        if self.review_status == "approved" and self.decision == "blocked":
            raise ValueError("a blocked preparation plan cannot be approved")
        if self.completion_status == "completed":
            if self.review_status != "approved":
                raise ValueError("completed preparation plan must be approved")
            if any(item.result != "approved" for item in self.tasks):
                raise ValueError("completed plan requires every component result approved")
        return self

    @property
    def approved_output_asset_ids(self) -> list[str]:
        return [item.output_asset_id for item in self.tasks if item.result == "approved"]


def parse_prepared_component_plan(data: object) -> PreparedComponentPlanDocument:
    try:
        return TypeAdapter(PreparedComponentPlanDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid prepared-component-plan-v1: {exc}") from exc


def approve_prepared_component_plan(data: object) -> PreparedComponentPlanDocument:
    plan = parse_prepared_component_plan(data)
    payload = plan.model_dump(mode="json")
    payload["review_status"] = "approved"
    return parse_prepared_component_plan(payload)


def complete_prepared_component_plan(
    data: object,
    *,
    catalog: ImageAssetCatalog,
) -> PreparedComponentPlanDocument:
    """Complete a reviewed plan only after every output is approved and registered."""

    plan = parse_prepared_component_plan(data)
    if plan.review_status != "approved":
        raise InputValidationError("component preparation plan must be approved first")
    if not plan.tasks:
        raise InputValidationError("component preparation plan has no tasks to complete")
    for task in plan.tasks:
        if task.result != "approved":
            raise InputValidationError(
                f"component task {task.task_id!r} is not manually approved"
            )
        record = catalog.get(task.output_asset_id)
        if record is None:
            raise InputValidationError(
                f"approved component output is not registered: {task.output_asset_id}"
            )
        if record.source_tier != "approved_generated":
            raise InputValidationError(
                f"component output {task.output_asset_id!r} must use "
                "source_tier=approved_generated"
            )
        if record.analysis_status != "analyzed":
            raise InputValidationError(
                f"component output {task.output_asset_id!r} must be analyzed"
            )
        if not any(_matches(record, subject) for subject in task.subjects):
            raise InputValidationError(
                f"component output {task.output_asset_id!r} metadata does not "
                "identify any planned subject"
            )
    payload = plan.model_dump(mode="json")
    payload["completion_status"] = "completed"
    return parse_prepared_component_plan(payload)


def _norm(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").casefold().strip()


def _matches(record: ImageAssetRecord, subject: str) -> bool:
    needle = _norm(subject)
    values = (_norm(record.subject_or_scene_id), _norm(record.asset_id))
    return any(needle in value or value in needle for value in values if value)


def _bundle_records(
    bundle: object,
    *,
    shot_id: str,
    catalog: ImageAssetCatalog,
) -> list[ImageAssetRecord]:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "reference-bundle-v1":
        raise InputValidationError("invalid reference-bundle-v1 document")
    if bundle.get("shot_id") != shot_id:
        raise InputValidationError("reference bundle shot_id does not match content plan")
    references = bundle.get("references")
    if not isinstance(references, list):
        raise InputValidationError("reference bundle references must be a list")
    records: list[ImageAssetRecord] = []
    seen: set[str] = set()
    for item in references:
        if not isinstance(item, dict) or not isinstance(item.get("asset_id"), str):
            raise InputValidationError("invalid reference bundle item")
        asset_id = item["asset_id"]
        if asset_id in seen:
            raise InputValidationError("reference bundle asset ids must be unique")
        record = catalog.get(asset_id)
        if record is None:
            raise InputValidationError(f"bound asset is not in catalog: {asset_id}")
        records.append(record)
        seen.add(asset_id)
    return records


def _authority_map(
    records: list[ImageAssetRecord],
    policy: FirstFrameAssemblyPolicy | None,
) -> dict[str, ReferenceAuthority]:
    explicit = (
        {item.asset_id: item.authority for item in policy.reference_authorities}
        if policy is not None
        else {}
    )
    unknown = set(explicit) - {item.asset_id for item in records}
    if unknown:
        raise InputValidationError(
            f"assembly policy references assets outside preparation bundle: {sorted(unknown)}"
        )
    result: dict[str, ReferenceAuthority] = {}
    for record in records:
        if record.asset_id in explicit:
            result[record.asset_id] = explicit[record.asset_id]
        elif record.asset_type == "character" and "pose_reference" in record.reference_roles:
            result[record.asset_id] = "action_only"
        elif record.asset_type == "character":
            result[record.asset_id] = "identity_only"
        elif record.asset_type == "prop":
            result[record.asset_id] = "final_visual"
        else:
            result[record.asset_id] = "structure_only"
    return result


def _output_id(shot_id: str, suffix: str) -> str:
    raw = f"prep_{shot_id}_{suffix}".replace(".", "_").replace("-", "_")
    if len(raw) <= 64:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"prep_{shot_id[:16]}_{suffix[:30]}_{digest}".rstrip("_")


def _prepared_affordance(
    *,
    component_id: str,
    subject: str,
    affordance: PropFunctionalAffordanceTruth,
) -> PreparedPropAffordance:
    return PreparedPropAffordance(
        component_id=component_id,
        subject=subject,
        grip_zone=affordance.grip_zone,
        active_end=affordance.active_end,
        native_action_axis=affordance.native_action_axis,
    )


def _prepared_attachment(
    attachment: ExternalAttachmentTruth,
    *,
    target_subject: str,
) -> PreparedExternalAttachment:
    return PreparedExternalAttachment(
        attachment_id=attachment.attachment_id,
        source_component_id=attachment.source_component_id,
        source_anchor=attachment.source_anchor,
        source_anchor_x=attachment.source_anchor_position.x,
        source_anchor_y=attachment.source_anchor_position.y,
        target_component_id=attachment.target_component_id,
        target_subject=target_subject,
        target_anchor=attachment.target_anchor,
        target_anchor_x=attachment.target_anchor_position.x,
        target_anchor_y=attachment.target_anchor_position.y,
        relation=attachment.relation,
        action_axis=attachment.action_axis,
        initial_gap=attachment.initial_gap,
        required_visible_state=(
            "Plate-scope state: "
            + ", ".join(attachment.source_must_remain_visible)
            + " remain visible; the target scene object and its anchor must be "
            "absent from the plate"
        ),
        must_remain_visible=list(attachment.must_remain_visible),
        source_must_remain_visible=list(attachment.source_must_remain_visible),
        target_must_remain_visible=list(attachment.target_must_remain_visible),
        hard_gate=attachment.hard_gate,
    )


def _attachment_fallbacks(
    attachments: list[PreparedExternalAttachment],
    affordances: list[PreparedPropAffordance],
) -> dict[str, str]:
    fallbacks: dict[str, str] = {}
    if affordances:
        fallbacks["prop_affordance"] = "; ".join(
            f"{item.subject}: grip only at {item.grip_zone}; active end is "
            f"{item.active_end}; native axis is {item.native_action_axis}"
            for item in affordances
        )
    if attachments:
        fallbacks["action_axis"] = "; ".join(
            f"{item.source_anchor} at ({item.source_anchor_x:.3f}, "
            f"{item.source_anchor_y:.3f}) follows {item.action_axis} toward "
            f"({item.target_anchor_x:.3f}, {item.target_anchor_y:.3f})"
            for item in attachments
        )
        fallbacks["external_target"] = "; ".join(
            f"aim toward layout-only target coordinate ({item.target_anchor_x:.3f}, "
            f"{item.target_anchor_y:.3f}); intended final-frame gap: "
            f"{item.initial_gap} (not rendered in this plate); do not render, "
            "invent or represent the target scene object, its anchor or any scene "
            "pixels in this component plate; the target must remain absent from "
            "the plate and exists only as a layout coordinate"
            for item in attachments
        )
        fallbacks["required_visibility"] = "; ".join(
            ", ".join(item.source_must_remain_visible) for item in attachments
        )
    return fallbacks


def _attachment_review_gates(
    attachments: list[PreparedExternalAttachment],
    affordances: list[PreparedPropAffordance],
) -> list[PreparedComponentReviewGate]:
    gates = [
        PreparedComponentReviewGate(
            gate_id=f"gate.affordance.{item.component_id}",
            criterion=(
                f"Confirm {item.subject} is held only at {item.grip_zone}, its "
                f"active end is {item.active_end}, and its functional topology "
                "has not been reversed."
            ),
        )
        for item in affordances
    ]
    gates.extend(
        PreparedComponentReviewGate(
            gate_id=f"gate.attachment.{item.attachment_id}",
            criterion=(
                f"Confirm {item.source_anchor} follows {item.action_axis} toward "
                f"layout-only target coordinate ({item.target_anchor_x:.3f}, "
                f"{item.target_anchor_y:.3f}); the target scene object and its "
                f"anchor must be absent from the plate; these source-group elements "
                f"remain visible: {', '.join(item.source_must_remain_visible)}."
            ),
        )
        for item in attachments
        if item.hard_gate
    )
    return gates


def build_prepared_component_plan(
    content_plan: FirstFrameContentPlanDocument | object,
    *,
    reference_bundle: object,
    catalog: ImageAssetCatalog,
    assembly_policy: FirstFrameAssemblyPolicy | object | None = None,
) -> PreparedComponentPlanDocument:
    """Plan high-quality atomic character/interaction plates before assembly."""

    content = (
        content_plan
        if isinstance(content_plan, FirstFrameContentPlanDocument)
        else parse_first_frame_content_plan(content_plan)
    )
    if content.review_status != "approved":
        raise InputValidationError("first-frame content plan must be approved")
    policy = None
    if assembly_policy is not None:
        policy = (
            assembly_policy
            if isinstance(assembly_policy, FirstFrameAssemblyPolicy)
            else parse_first_frame_assembly_policy(assembly_policy)
        )
        if policy.shot_id != content.shot_id:
            raise InputValidationError("assembly policy shot_id does not match content")
    records = _bundle_records(
        reference_bundle,
        shot_id=content.shot_id,
        catalog=catalog,
    )
    authorities = _authority_map(records, policy)
    components = {
        item.component_id: item
        for item in [*content.character_states, *content.prop_states]
    }
    layer_subjects = {item.component_id: item.subject for item in content.layers}
    prop_states = {item.component_id: item for item in content.prop_states}
    character_states = {
        item.component_id: item for item in content.character_states
    }
    tasks: list[PreparedComponentTask] = []
    warnings: list[str] = []
    prepared_components: set[str] = set()
    hard_contact_counts: dict[str, int] = {}
    for interaction in content.contact_graph:
        if not interaction.hard_gate:
            continue
        for component_id in {
            interaction.actor_component_id,
            interaction.target_component_id,
        }:
            hard_contact_counts[component_id] = hard_contact_counts.get(component_id, 0) + 1
    overlapping_hard_components = sorted(
        component_id
        for component_id, count in hard_contact_counts.items()
        if count > 1
    )
    if overlapping_hard_components:
        warnings.append(
            "overlapping hard interactions require one reviewed combined HOW plate "
            "or a revised K0 contact graph; components: "
            + ", ".join(overlapping_hard_components)
        )

    for interaction in content.contact_graph:
        participants = [
            components[interaction.actor_component_id],
            components[interaction.target_component_id],
        ]
        candidates: list[PreparedComponentInput] = []
        for participant in participants:
            participant_records = [
                item
                for item in records
                if _matches(item, participant.subject)
                and authorities[item.asset_id]
                in {"identity_only", "final_visual", "action_only"}
            ]
            participant_records.sort(
                key=lambda item: (
                    0 if authorities[item.asset_id] == "identity_only" else 1,
                    item.asset_id,
                )
            )
            if participant_records:
                selected = participant_records[0]
                function: InputFunction = (
                    "prop_visual" if selected.asset_type == "prop" else "who"
                )
                if selected.asset_id not in {item.asset_id for item in candidates}:
                    candidates.append(
                        PreparedComponentInput(
                            asset_id=selected.asset_id,
                            function=function,
                        )
                    )
        if not candidates:
            warnings.append(
                f"interaction {interaction.interaction_id} has no visual participant evidence"
            )
            continue
        candidates = candidates[:2]
        control = [
            item.asset_id
            for item in records
            if authorities[item.asset_id] == "structure_only"
            and any(_matches(item, participant.subject) for participant in participants)
        ]
        participant_ids = {
            interaction.actor_component_id,
            interaction.target_component_id,
        }
        attachment_truths = [
            item
            for item in content.attachment_graph
            if item.source_component_id in participant_ids
        ]
        external_attachments = [
            _prepared_attachment(
                item,
                target_subject=layer_subjects[item.target_component_id],
            )
            for item in attachment_truths
        ]
        affordances = [
            _prepared_affordance(
                component_id=component_id,
                subject=prop_states[component_id].subject,
                affordance=prop_states[component_id].functional_affordance,
            )
            for component_id in participant_ids
            if component_id in prop_states
            and prop_states[component_id].functional_affordance is not None
            and any(
                item.source_component_id == component_id
                for item in attachment_truths
            )
        ]
        placement_fallbacks = _attachment_fallbacks(
            external_attachments,
            affordances,
        )
        facing = [
            f"{character_states[component_id].subject}: "
            f"{character_states[component_id].facing_direction}"
            for component_id in participant_ids
            if component_id in character_states
            and character_states[component_id].facing_direction is not None
        ]
        if facing:
            placement_fallbacks["facing_direction"] = "; ".join(facing)
        if external_attachments:
            spatial_placement = "; ".join(
                [
                    character_states[component_id].screen_position
                    for component_id in participant_ids
                    if component_id in character_states
                ]
                + [
                    f"{item.source_anchor} at normalized final-frame position "
                    f"({item.source_anchor_x:.3f}, {item.source_anchor_y:.3f})"
                    for item in external_attachments
                ]
            )
        else:
            spatial_placement = content.camera.composition
        tasks.append(
            PreparedComponentTask(
                task_id=f"task.interaction.{interaction.interaction_id}",
                kind="interaction_plate",
                component_ids=[
                    interaction.actor_component_id,
                    interaction.target_component_id,
                ],
                subjects=[item.subject for item in participants],
                target_state=interaction.required_visible_state,
                model_inputs=candidates,
                control_evidence_asset_ids=control,
                preserve_attributes=[
                    "participant identities",
                    "canonical clothing and prop appearance",
                    *(
                        [
                            "prop functional topology",
                            "external attachment geometry",
                            "final-frame canvas placement",
                        ]
                        if external_attachments
                        else []
                    ),
                ],
                allowed_text_fallbacks={
                    "contact_relation": interaction.required_visible_state,
                    "spatial_placement": spatial_placement,
                    **placement_fallbacks,
                },
                prop_affordances=affordances,
                external_attachments=external_attachments,
                review_gates=(
                    [
                        PreparedComponentReviewGate(
                            gate_id=f"gate.contact.{interaction.interaction_id}",
                            criterion=(
                                f"Confirm the atomic contact is visually true: "
                                f"{interaction.required_visible_state}"
                            ),
                        )
                    ]
                    + _attachment_review_gates(
                        external_attachments,
                        affordances,
                    )
                    if external_attachments
                    else []
                ),
                output_asset_id=_output_id(
                    content.shot_id, f"interaction_{interaction.interaction_id}"
                ),
            )
        )
        prepared_components.update(
            {interaction.actor_component_id, interaction.target_component_id}
        )

    for character in content.character_states:
        if character.component_id in prepared_components:
            continue
        matches = [
            item
            for item in records
            if item.asset_type == "character" and _matches(item, character.subject)
        ]
        who = sorted(
            [
                item
                for item in matches
                if authorities[item.asset_id] in {"identity_only", "final_visual"}
            ],
            key=lambda item: item.asset_id,
        )
        how = sorted(
            [item for item in matches if authorities[item.asset_id] == "action_only"],
            key=lambda item: item.asset_id,
        )
        if not who:
            warnings.append(
                f"character {character.subject!r} has no WHO reference; no text-only "
                "appearance task was created"
            )
            continue
        inputs = [PreparedComponentInput(asset_id=who[0].asset_id, function="who")]
        if how and how[0].asset_id != who[0].asset_id:
            inputs.append(PreparedComponentInput(asset_id=how[0].asset_id, function="how"))
        attachment_truths = [
            item
            for item in content.attachment_graph
            if item.source_component_id == character.component_id
        ]
        external_attachments = [
            _prepared_attachment(
                item,
                target_subject=layer_subjects[item.target_component_id],
            )
            for item in attachment_truths
        ]
        attachment_fallbacks = _attachment_fallbacks(external_attachments, [])
        if character.facing_direction is not None:
            attachment_fallbacks["facing_direction"] = character.facing_direction
        tasks.append(
            PreparedComponentTask(
                task_id=f"task.character.{character.component_id}",
                kind="character_plate",
                component_ids=[character.component_id],
                subjects=[character.subject],
                target_state=character.action_pose,
                model_inputs=inputs,
                preserve_attributes=list(character.identity_attributes_to_preserve),
                allowed_text_fallbacks=(
                    {
                        "spatial_placement": character.screen_position,
                        **attachment_fallbacks,
                    }
                    if len(inputs) == 2
                    else {
                        "action": character.action_pose,
                        "spatial_placement": character.screen_position,
                        **attachment_fallbacks,
                    }
                ),
                external_attachments=external_attachments,
                review_gates=_attachment_review_gates(external_attachments, []),
                output_asset_id=_output_id(content.shot_id, character.component_id),
            )
        )

    used_source_ids = sorted(
        {
            item.asset_id
            for task in tasks
            for item in task.model_inputs
        }
        | {
            item for task in tasks for item in task.control_evidence_asset_ids
        }
    )
    missing_characters = {
        item.component_id for item in content.character_states
    } - {component_id for task in tasks for component_id in task.component_ids}
    hard_missing = any(
        item.hard_gate
        and not {
            item.actor_component_id,
            item.target_component_id,
        }
        <= {component_id for task in tasks for component_id in task.component_ids}
        for item in content.contact_graph
    )
    prepared_attachment_ids = {
        item.attachment_id
        for task in tasks
        for item in task.external_attachments
    }
    hard_attachment_missing = any(
        item.hard_gate and item.attachment_id not in prepared_attachment_ids
        for item in content.attachment_graph
    )
    if hard_attachment_missing:
        warnings.append(
            "one or more hard external attachments are not covered by a prepared "
            "component task"
        )
    if hard_missing or hard_attachment_missing or overlapping_hard_components:
        decision: PlanDecision = "blocked"
    elif warnings or missing_characters:
        decision = "needs_review"
    else:
        decision = "ready"
    return PreparedComponentPlanDocument(
        plan_id=f"{content.shot_id}-prepared-components",
        shot_id=content.shot_id,
        content_plan_id=content.plan_id,
        review_status="draft",
        completion_status="pending",
        decision=decision,
        source_asset_ids=used_source_ids,
        tasks=tasks,
        warnings=warnings,
    )
