"""``execution-ledger-v1`` research contract: immutable execution facts.

Freeze reference (Round 8..10, 2026-08-11):

- The ledger stores immutable *facts*, never mutable entity snapshots.
  NodeRun / RenderAttempt are lifecycle events
  (started -> artifact_registered -> finished).
- ``seq`` is the serialization order assigned by the single LedgerWriter;
  causality is expressed only by ``causal_refs``.
- Ref scopes: ``ledger://`` (records), ``artifact://`` (runtime artifacts),
  ``asset://`` (asset library).  ``blob://`` is storage-internal only and
  must never be consumed directly by executors.
- Relation usage is whitelisted per record type; the LedgerWriter enforces
  it together with target types at append time (Phase 2).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from anime_remix.errors import InputValidationError

_SCHEMA_VERSION = "execution-ledger-v1"
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

RecordType = Literal[
    "run_started",
    "run_finished",
    "plan_instantiated",
    "node_run_started",
    "node_run_finished",
    "artifact_registered",
    "port_bound",
    "render_intent_created",
    "model_render_request_created",
    "render_attempt_started",
    "render_attempt_finished",
    "validation_result",
    "review_decision",
    "failure_event",
    "repair_decision",
]

Relation = Literal[
    "input",
    "derived_from",
    "produced_by",
    "triggered_by",
    "supersedes",
    "selected_by",
]

FailureClass = Literal[
    "runtime",
    "contract",
    "content",
    "feasibility",
    "dependency",
]

FailureCategory = Literal[
    "identity",
    "pose",
    "reference",
    "geometry",
    "layout",
    "mask",
    "compositing",
    "scene_planning",
    "artifact_integrity",
    "resource",
    "unknown",
]

_RECORD_ID_RE = re.compile(r"^rec_[0-9]+$")
_ARTIFACT_ID_RE = re.compile(r"^art_[0-9]+$")
_LEDGER_REF_RE = re.compile(r"^ledger://[A-Za-z0-9_.-]+/rec_[0-9]+$")
_ARTIFACT_REF_RE = re.compile(
    r"^artifact://[A-Za-z0-9_.-]+/art_[0-9]+$"
)
_ASSET_REF_RE = re.compile(
    r"^asset://[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
    r"[A-Za-z0-9_.-]+@v[0-9]+$"
)
_BLOB_REF_RE = re.compile(r"^blob://sha256:[0-9a-f]{64}$")
_PORT_REF_RE = re.compile(
    r"^plan://[A-Za-z0-9_.-]+/ports/[A-Za-z0-9_.-]+$"
)


def _pattern(pattern: re.Pattern[str]) -> Callable[[str], str]:
    def check(value: str) -> str:
        if not pattern.fullmatch(value):
            raise ValueError(f"invalid ref/identifier {value!r}")
        return value

    return check


def _non_empty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("field must be non-empty")
    return stripped


def _data_or_ledger_ref(value: str) -> str:
    if not (
        _LEDGER_REF_RE.fullmatch(value)
        or _ARTIFACT_REF_RE.fullmatch(value)
        or _ASSET_REF_RE.fullmatch(value)
    ):
        raise ValueError(f"invalid causal ref {value!r}")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
RecordId = Annotated[str, AfterValidator(_pattern(_RECORD_ID_RE))]
ArtifactId = Annotated[str, AfterValidator(_pattern(_ARTIFACT_ID_RE))]
LedgerRef = Annotated[str, AfterValidator(_pattern(_LEDGER_REF_RE))]
ArtifactRef = Annotated[str, AfterValidator(_pattern(_ARTIFACT_REF_RE))]
AssetRef = Annotated[str, AfterValidator(_pattern(_ASSET_REF_RE))]
BlobRef = Annotated[str, AfterValidator(_pattern(_BLOB_REF_RE))]
PortRef = Annotated[str, AfterValidator(_pattern(_PORT_REF_RE))]
CausalRefTarget = Annotated[str, AfterValidator(_data_or_ledger_ref)]


def ref_scope(ref: str) -> str:
    """Return the scope of a reference: ledger/artifact/asset/blob."""

    if _LEDGER_REF_RE.fullmatch(ref):
        return "ledger"
    if _ARTIFACT_REF_RE.fullmatch(ref):
        return "artifact"
    if _ASSET_REF_RE.fullmatch(ref):
        return "asset"
    if _BLOB_REF_RE.fullmatch(ref):
        return "blob"
    raise InputValidationError(f"unknown ref scope: {ref!r}")


ALLOWED_RELATIONS: dict[RecordType, tuple[Relation, ...]] = {
    "run_started": (),
    "run_finished": (),
    "plan_instantiated": ("input", "triggered_by"),
    "node_run_started": ("input", "triggered_by"),
    "node_run_finished": (),
    "artifact_registered": (),
    "port_bound": ("input", "selected_by", "supersedes", "triggered_by"),
    "render_intent_created": ("input", "triggered_by"),
    "model_render_request_created": ("input", "triggered_by"),
    "render_attempt_started": ("input", "triggered_by"),
    "render_attempt_finished": (),
    "validation_result": ("input", "derived_from"),
    "review_decision": ("input", "derived_from"),
    "failure_event": ("derived_from",),
    "repair_decision": ("triggered_by",),
}

RecordTypeTuple = tuple[RecordType, ...]

# Target-type whitelist per (record_type, relation).  ``None`` means any
# ledger record type is an acceptable target.  Enforced by the LedgerWriter
# at append time (Phase 2); this is the only contract refinement carried
# over from Phase 1.
ALLOWED_TARGET_TYPES: dict[RecordType, dict[Relation, RecordTypeTuple | None]] = {
    "run_started": {},
    "run_finished": {},
    "plan_instantiated": {
        "input": None,
        "triggered_by": ("run_started", "repair_decision"),
    },
    "node_run_started": {
        "input": None,
        "triggered_by": ("plan_instantiated", "repair_decision"),
    },
    "node_run_finished": {},
    "artifact_registered": {},
    "port_bound": {
        "input": None,
        "selected_by": (
            "repair_decision",
            "node_run_finished",
            "render_attempt_finished",
        ),
        "supersedes": ("port_bound",),
        "triggered_by": ("node_run_finished", "repair_decision"),
    },
    "render_intent_created": {
        "input": None,
        "triggered_by": ("plan_instantiated", "repair_decision"),
    },
    "model_render_request_created": {
        "input": None,
        "triggered_by": ("render_intent_created", "repair_decision"),
    },
    "render_attempt_started": {
        "input": None,
        "triggered_by": ("model_render_request_created", "repair_decision"),
    },
    "render_attempt_finished": {},
    "validation_result": {
        "input": None,
        "derived_from": ("node_run_finished", "render_attempt_finished"),
    },
    "review_decision": {
        "input": None,
        "derived_from": (
            "node_run_finished",
            "validation_result",
            "render_attempt_finished",
        ),
    },
    "failure_event": {
        "derived_from": (
            "review_decision",
            "validation_result",
            "node_run_finished",
            "render_attempt_finished",
        ),
    },
    "repair_decision": {
        "triggered_by": ("failure_event",),
    },
}


def validate_target_type(
    record_type: RecordType,
    relation: Relation,
    target_record_type: RecordType,
) -> None:
    """Check that ``relation`` on ``record_type`` may target the record type."""

    allowed = ALLOWED_TARGET_TYPES[record_type].get(relation)
    if allowed is None:
        return
    if target_record_type not in allowed:
        raise InputValidationError(
            f"relation {relation!r} on {record_type} cannot target "
            f"{target_record_type}"
        )


def validate_causal_refs(
    record_type: RecordType,
    causal_refs: list[CausalRef],
) -> None:
    """Validate the relation whitelist and target scope for causal refs."""

    allowed = ALLOWED_RELATIONS[record_type]
    for ref in causal_refs:
        if ref.relation not in allowed:
            raise InputValidationError(
                f"relation {ref.relation!r} not allowed on {record_type}"
            )
        if ref.relation != "input" and ref_scope(ref.record_ref) != "ledger":
            raise InputValidationError(
                f"relation {ref.relation!r} must target a ledger ref"
            )


class CausalRef(BaseModel):
    model_config = _STRICT_CONFIG

    record_ref: CausalRefTarget
    relation: Relation


class RunStartedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    run_id: NonEmptyStr
    execution_template_ref: NonEmptyStr
    policy_refs: list[NonEmptyStr] = []
    started_at: NonEmptyStr


class RunFinishedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    run_id: NonEmptyStr
    status: Literal["completed", "blocked_human", "stopped"]
    finished_at: NonEmptyStr


class PlanInstantiatedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    plan_id: NonEmptyStr
    template_ref: NonEmptyStr
    shot_id: NonEmptyStr
    keyframe_id: str | None = None
    shot_spec_ref: NonEmptyStr
    keyframe_plan_ref: NonEmptyStr
    reference_package_ref: NonEmptyStr
    policy_refs: list[NonEmptyStr] = []

    @field_validator("keyframe_id", mode="before")
    @classmethod
    def _keyframe_id(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("keyframe_id must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("keyframe_id must be non-empty when provided")
        return stripped


class NodeRunStartedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    instance_id: NonEmptyStr
    plan_id: NonEmptyStr
    node_id: NonEmptyStr
    operation: NonEmptyStr
    node_type: Literal["model", "deterministic", "gate", "review"]
    inputs: list[NonEmptyStr] = []
    started_at: NonEmptyStr


class NodeRunFinishedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    instance_id: NonEmptyStr
    started_ref: LedgerRef
    outputs: list[ArtifactRef] = []
    status: Literal["success", "runtime_failure", "memoized", "skipped"]
    finished_at: NonEmptyStr


class ArtifactRegisteredPayload(BaseModel):
    model_config = _STRICT_CONFIG

    artifact_id: ArtifactId
    blob_ref: BlobRef
    artifact_kind: NonEmptyStr
    schema_version: NonEmptyStr
    producer_started_ref: LedgerRef
    size_bytes: int = 0

    @field_validator("size_bytes", mode="before")
    @classmethod
    def _size(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("size_bytes must be an integer")
        if value < 0:
            raise ValueError("size_bytes must be >= 0")
        return value


class PortBoundPayload(BaseModel):
    model_config = _STRICT_CONFIG

    binding_id: NonEmptyStr
    logical_port_ref: PortRef
    artifact_ref: ArtifactRef
    supersedes_binding_ref: LedgerRef | None = None
    bound_by_ref: LedgerRef | None = None


class RenderIntentCreatedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    intent_id: NonEmptyStr
    operation: NonEmptyStr
    shot_id: NonEmptyStr
    keyframe_id: NonEmptyStr
    requirements: list[NonEmptyStr] = []
    reference_package_ref: NonEmptyStr
    constraint_set_ref: NonEmptyStr


class RequestCondition(BaseModel):
    model_config = _STRICT_CONFIG

    slot: int
    condition_ref: NonEmptyStr

    @field_validator("slot", mode="before")
    @classmethod
    def _slot(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("slot must be an integer")
        if value < 1:
            raise ValueError("slot must be >= 1")
        return value


class ModelRenderRequestCreatedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    request_id: NonEmptyStr
    intent_ref: LedgerRef
    adapter_id: NonEmptyStr
    model_id: NonEmptyStr
    revision: NonEmptyStr
    conditions: list[RequestCondition] = []
    prompt: NonEmptyStr
    parameters: dict[str, StrictBool | int | float | str] = {}
    input_hash: NonEmptyStr


class RenderAttemptStartedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    attempt_id: NonEmptyStr
    request_ref: LedgerRef
    started_at: NonEmptyStr


class RuntimeInfo(BaseModel):
    model_config = _STRICT_CONFIG

    device: str | None = None
    duration_ms: float | None = None

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _duration(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("duration_ms must be a number or null")
        number = float(value)
        if number < 0:
            raise ValueError("duration_ms must be >= 0")
        return number


class RenderAttemptFinishedPayload(BaseModel):
    model_config = _STRICT_CONFIG

    attempt_id: NonEmptyStr
    started_ref: LedgerRef
    request_ref: LedgerRef
    status: Literal["success", "runtime_failure"]
    output_artifact_ref: ArtifactRef | None = None
    runtime: RuntimeInfo | None = None
    finished_at: NonEmptyStr


class ValidationCheck(BaseModel):
    model_config = _STRICT_CONFIG

    check_id: NonEmptyStr
    passed: bool
    detail: str | None = None

    @field_validator("detail", mode="before")
    @classmethod
    def _detail(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("detail must be a string or null")
        return value


class ValidationResultPayload(BaseModel):
    model_config = _STRICT_CONFIG

    result_id: NonEmptyStr
    node_id: NonEmptyStr
    valid: bool
    checks: list[ValidationCheck] = []
    failure_category: NonEmptyStr | None = None


class ReviewFinding(BaseModel):
    model_config = _STRICT_CONFIG

    keyframe_id: NonEmptyStr
    requirement_id: NonEmptyStr
    result: Literal["pass", "borderline", "fail"]
    failure_category: NonEmptyStr


class ReviewDecisionPayload(BaseModel):
    model_config = _STRICT_CONFIG

    review_id: NonEmptyStr
    shot_id: NonEmptyStr
    status: Literal["approved", "rejected"]
    findings: list[ReviewFinding] = []


class FailureFinding(BaseModel):
    model_config = _STRICT_CONFIG

    finding_id: NonEmptyStr
    keyframe_id: str | None = None
    node_id: str | None = None
    failure_class: FailureClass
    failure_category: FailureCategory
    reason_code: NonEmptyStr
    evidence_refs: list[NonEmptyStr] = []

    @field_validator("keyframe_id", "node_id", mode="before")
    @classmethod
    def _optional_ids(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("finding ids must be strings or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("finding ids must be non-empty when provided")
        return stripped


class FailureEventPayload(BaseModel):
    model_config = _STRICT_CONFIG

    event_id: NonEmptyStr
    source_kind: Literal["runtime", "validator", "critic", "human_review", "planner"]
    source_ref: NonEmptyStr
    findings: list[FailureFinding] = []


class RepairAction(BaseModel):
    model_config = _STRICT_CONFIG

    action: Literal["retry_attempt"]
    target_node_id: str | None = None
    attempt_ref: LedgerRef | None = None

    @field_validator("target_node_id", mode="before")
    @classmethod
    def _target(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("target_node_id must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("target_node_id must be non-empty when provided")
        return stripped


class RepairDecisionPayload(BaseModel):
    model_config = _STRICT_CONFIG

    decision_id: NonEmptyStr
    failure_event_ref: LedgerRef
    actions: list[RepairAction] = []
    disposition: Literal["retry", "blocked_human", "stopped"]
    reason_code: NonEmptyStr


class _LedgerRecordBase(BaseModel):
    model_config = _STRICT_CONFIG

    record_id: RecordId
    seq: int
    ledger_schema_version: Literal["execution-ledger-v1"] = _SCHEMA_VERSION
    run_ref: NonEmptyStr
    recorded_at: NonEmptyStr
    causal_refs: list[CausalRef] = []

    @field_validator("seq", mode="before")
    @classmethod
    def _seq(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("seq must be an integer")
        if value < 1:
            raise ValueError("seq must be >= 1")
        return value


class RunStartedRecord(_LedgerRecordBase):
    record_type: Literal["run_started"]
    payload: RunStartedPayload


class RunFinishedRecord(_LedgerRecordBase):
    record_type: Literal["run_finished"]
    payload: RunFinishedPayload


class PlanInstantiatedRecord(_LedgerRecordBase):
    record_type: Literal["plan_instantiated"]
    payload: PlanInstantiatedPayload


class NodeRunStartedRecord(_LedgerRecordBase):
    record_type: Literal["node_run_started"]
    payload: NodeRunStartedPayload


class NodeRunFinishedRecord(_LedgerRecordBase):
    record_type: Literal["node_run_finished"]
    payload: NodeRunFinishedPayload


class ArtifactRegisteredRecord(_LedgerRecordBase):
    record_type: Literal["artifact_registered"]
    payload: ArtifactRegisteredPayload


class PortBoundRecord(_LedgerRecordBase):
    record_type: Literal["port_bound"]
    payload: PortBoundPayload


class RenderIntentCreatedRecord(_LedgerRecordBase):
    record_type: Literal["render_intent_created"]
    payload: RenderIntentCreatedPayload


class ModelRenderRequestCreatedRecord(_LedgerRecordBase):
    record_type: Literal["model_render_request_created"]
    payload: ModelRenderRequestCreatedPayload


class RenderAttemptStartedRecord(_LedgerRecordBase):
    record_type: Literal["render_attempt_started"]
    payload: RenderAttemptStartedPayload


class RenderAttemptFinishedRecord(_LedgerRecordBase):
    record_type: Literal["render_attempt_finished"]
    payload: RenderAttemptFinishedPayload


class ValidationResultRecord(_LedgerRecordBase):
    record_type: Literal["validation_result"]
    payload: ValidationResultPayload


class ReviewDecisionRecord(_LedgerRecordBase):
    record_type: Literal["review_decision"]
    payload: ReviewDecisionPayload


class FailureEventRecord(_LedgerRecordBase):
    record_type: Literal["failure_event"]
    payload: FailureEventPayload


class RepairDecisionRecord(_LedgerRecordBase):
    record_type: Literal["repair_decision"]
    payload: RepairDecisionPayload


LedgerRecord = Annotated[
    RunStartedRecord
    | RunFinishedRecord
    | PlanInstantiatedRecord
    | NodeRunStartedRecord
    | NodeRunFinishedRecord
    | ArtifactRegisteredRecord
    | PortBoundRecord
    | RenderIntentCreatedRecord
    | ModelRenderRequestCreatedRecord
    | RenderAttemptStartedRecord
    | RenderAttemptFinishedRecord
    | ValidationResultRecord
    | ReviewDecisionRecord
    | FailureEventRecord
    | RepairDecisionRecord,
    Field(discriminator="record_type"),
]

_LEDGER_RECORD_ADAPTER = TypeAdapter(LedgerRecord)


def parse_ledger_record(data: object) -> LedgerRecord:
    """Parse, strictly validate and relation-check one ledger record."""

    try:
        record = _LEDGER_RECORD_ADAPTER.validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid execution-ledger record: {exc}"
        ) from exc
    validate_causal_refs(record.record_type, record.causal_refs)
    return record


def load_ledger_record(path: str | object) -> LedgerRecord:
    """Load one ledger record from disk and validate it."""

    from pathlib import Path

    from anime_remix.json_io import load_json_object

    return parse_ledger_record(load_json_object(Path(str(path))))
