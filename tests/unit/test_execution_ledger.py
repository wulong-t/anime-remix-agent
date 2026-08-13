"""Unit tests for the execution-ledger-v1 research contract."""

from __future__ import annotations

import json
from typing import get_args

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.execution_ledger import (
    ALLOWED_RELATIONS,
    ALLOWED_TARGET_TYPES,
    RecordType,
    parse_ledger_record,
    ref_scope,
)


def _run_started() -> dict:
    return {
        "record_id": "rec_000001",
        "seq": 1,
        "record_type": "run_started",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:00+08:00",
        "causal_refs": [],
        "payload": {
            "run_id": "run_001",
            "execution_template_ref": "template://compose-v1",
            "policy_refs": [],
            "started_at": "2026-08-11T12:00:00+08:00",
        },
    }


def _node_run_started() -> dict:
    return {
        "record_id": "rec_000002",
        "seq": 2,
        "record_type": "node_run_started",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:01+08:00",
        "causal_refs": [
            {
                "record_ref": "asset://anime-remix/scene/classroom_01@v1",
                "relation": "input",
            }
        ],
        "payload": {
            "instance_id": "layout-run-17",
            "plan_id": "shot003-kf002-plan-v1",
            "node_id": "layout",
            "operation": "layout",
            "node_type": "deterministic",
            "inputs": ["artifact://run_001/art_000121"],
            "started_at": "2026-08-11T12:00:01+08:00",
        },
    }


def _artifact_registered() -> dict:
    return {
        "record_id": "rec_000003",
        "seq": 3,
        "record_type": "artifact_registered",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:02+08:00",
        "causal_refs": [],
        "payload": {
            "artifact_id": "art_000121",
            "blob_ref": "blob://sha256:" + "a" * 64,
            "artifact_kind": "layout_plan",
            "schema_version": "layout-plan-v1",
            "producer_started_ref": "ledger://run_001/rec_000002",
            "size_bytes": 4213,
        },
    }


def _node_run_finished() -> dict:
    return {
        "record_id": "rec_000004",
        "seq": 4,
        "record_type": "node_run_finished",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:03+08:00",
        "causal_refs": [],
        "payload": {
            "instance_id": "layout-run-17",
            "started_ref": "ledger://run_001/rec_000002",
            "outputs": ["artifact://run_001/art_000121"],
            "status": "success",
            "finished_at": "2026-08-11T12:00:03+08:00",
        },
    }


def _model_render_request() -> dict:
    return {
        "record_id": "rec_000006",
        "seq": 6,
        "record_type": "model_render_request_created",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:05+08:00",
        "causal_refs": [
            {
                "record_ref": "ledger://run_001/rec_000005",
                "relation": "input",
            }
        ],
        "payload": {
            "request_id": "req_0091",
            "intent_ref": "ledger://run_001/rec_000100",
            "adapter_id": "qwen-edit-2511-adapter-v1",
            "model_id": "Qwen/Qwen-Image-Edit-2511",
            "revision": "main",
            "conditions": [{"slot": 1, "condition_ref": "cond_001"}],
            "prompt": "Edit this anime illustration into a new keyframe.",
            "parameters": {"seed": 0, "steps": 40},
            "input_hash": "abc123",
        },
    }


def _repair_decision() -> dict:
    return {
        "record_id": "rec_000007",
        "seq": 7,
        "record_type": "repair_decision",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:06+08:00",
        "causal_refs": [
            {
                "record_ref": "ledger://run_001/rec_000200",
                "relation": "triggered_by",
            }
        ],
        "payload": {
            "decision_id": "repair_001",
            "failure_event_ref": "ledger://run_001/rec_000200",
            "actions": [
                {"action": "retry_attempt", "target_node_id": "local_inpaint"}
            ],
            "disposition": "retry",
            "reason_code": "runtime_oom",
        },
    }


def test_run_started_parses() -> None:
    record = parse_ledger_record(_run_started())
    assert record.record_type == "run_started"
    assert record.payload.run_id == "run_001"


def test_lifecycle_sequence_parses() -> None:
    assert parse_ledger_record(_node_run_started()).record_type == (
        "node_run_started"
    )
    registered = parse_ledger_record(_artifact_registered())
    assert registered.payload.blob_ref == "blob://sha256:" + "a" * 64
    assert parse_ledger_record(_node_run_finished()).payload.status == "success"


def test_model_render_request_parses_with_inline_prompt() -> None:
    record = parse_ledger_record(_model_render_request())
    assert "anime illustration" in record.payload.prompt
    assert record.payload.parameters["steps"] == 40


def test_model_render_request_parameters_accept_strict_booleans() -> None:
    document = _model_render_request()
    document["payload"]["parameters"] = {
        "seed": 0,
        "n": 1,
        "size": "1280*720",
        "prompt_extend": False,
        "prompt_extend_mode": "direct",
        "watermark": False,
        "enabled": True,
    }
    record = parse_ledger_record(document)
    parameters = record.payload.parameters
    assert parameters["prompt_extend"] is False
    assert parameters["watermark"] is False
    assert parameters["enabled"] is True
    assert type(parameters["seed"]) is int
    assert type(parameters["n"]) is int
    assert type(parameters["size"]) is str
    assert type(parameters["prompt_extend_mode"]) is str


def test_model_render_request_parameters_bool_json_round_trip() -> None:
    document = _model_render_request()
    document["payload"]["parameters"] = {
        "prompt_extend": False,
        "watermark": False,
        "enabled": True,
    }
    record = parse_ledger_record(json.loads(json.dumps(document)))
    parameters = record.payload.parameters
    assert parameters["prompt_extend"] is False
    assert parameters["watermark"] is False
    assert parameters["enabled"] is True


def test_model_render_request_parameters_int_stays_int_not_bool() -> None:
    document = _model_render_request()
    document["payload"]["parameters"] = {"seed": 1, "steps": 40}
    record = parse_ledger_record(document)
    assert type(record.payload.parameters["seed"]) is int
    assert record.payload.parameters["seed"] is not True
    assert type(record.payload.parameters["steps"]) is int


@pytest.mark.parametrize("bad_value", [None, [], {}, ["x"]])
def test_model_render_request_parameters_reject_non_scalar(
    bad_value: object,
) -> None:
    document = _model_render_request()
    document["payload"]["parameters"] = {"seed": bad_value}
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_repair_decision_parses() -> None:
    record = parse_ledger_record(_repair_decision())
    assert record.payload.disposition == "retry"
    assert record.payload.actions[0].action == "retry_attempt"


def test_disallowed_relation_rejected() -> None:
    document = _run_started()
    document["causal_refs"] = [
        {"record_ref": "ledger://run_001/rec_000999", "relation": "input"}
    ]
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_supersedes_must_target_ledger_ref() -> None:
    document = {
        "record_id": "rec_000008",
        "seq": 8,
        "record_type": "port_bound",
        "ledger_schema_version": "execution-ledger-v1",
        "run_ref": "run_001",
        "recorded_at": "2026-08-11T12:00:07+08:00",
        "causal_refs": [
            {
                "record_ref": "artifact://run_001/art_000999",
                "relation": "supersedes",
            }
        ],
        "payload": {
            "binding_id": "bind_002",
            "logical_port_ref": "plan://shot003-kf002-plan-v1/ports/layout_plan",
            "artifact_ref": "artifact://run_001/art_000207",
            "supersedes_binding_ref": "ledger://run_001/rec_000005",
        },
    }
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_invalid_blob_ref_rejected() -> None:
    document = _artifact_registered()
    document["payload"]["blob_ref"] = "blob://sha256:nothex"
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_invalid_record_id_rejected() -> None:
    document = _run_started()
    document["record_id"] = "rec_abc"
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_unknown_record_type_rejected() -> None:
    document = _run_started()
    document["record_type"] = "mystery_event"
    with pytest.raises(InputValidationError):
        parse_ledger_record(document)


def test_all_record_types_have_relation_policy() -> None:
    expected = {
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
    }
    assert set(ALLOWED_RELATIONS) == expected
    for record_type in expected:
        assert isinstance(record_type, str)


def test_target_type_table_matches_relation_whitelist() -> None:
    for record_type, relations in ALLOWED_RELATIONS.items():
        assert record_type in ALLOWED_TARGET_TYPES
        assert set(ALLOWED_TARGET_TYPES[record_type]) == set(relations)


def test_ref_scope_detection() -> None:
    assert ref_scope("ledger://run_001/rec_000001") == "ledger"
    assert ref_scope("artifact://run_001/art_000001") == "artifact"
    assert ref_scope("asset://anime-remix/scene/classroom_01@v1") == "asset"
    assert ref_scope("blob://sha256:" + "b" * 64) == "blob"
    with pytest.raises(InputValidationError):
        ref_scope("file://local/thing.png")


def test_record_type_literal_has_policy_entry() -> None:
    valid_types = set(get_args(RecordType))
    assert set(ALLOWED_RELATIONS) == valid_types
