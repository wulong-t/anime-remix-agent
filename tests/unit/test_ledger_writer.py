"""Unit tests for the Phase 2 LedgerWriter."""

from __future__ import annotations

import json

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.execution_ledger import (
    parse_ledger_record,
)
from anime_remix.services.execution.ledger_writer import (
    LedgerWriter,
    read_complete_records,
)

RUN = "run_001"
PORT = "plan://shot003-kf002-plan-v1/ports/layout_plan"


def _blob(seed: str = "a") -> str:
    return "blob://sha256:" + seed * 64


def _run_started_draft() -> dict:
    return {
        "record_type": "run_started",
        "causal_refs": [],
        "payload": {
            "run_id": RUN,
            "execution_template_ref": "template://compose-v1",
            "policy_refs": [],
            "started_at": "2026-08-11T12:00:00+08:00",
        },
    }


def _node_run_started_draft(
    instance_id: str = "layout-run-17",
    inputs: list[str] | None = None,
) -> dict:
    return {
        "record_type": "node_run_started",
        "causal_refs": [
            {
                "record_ref": "asset://anime-remix/scene/classroom_01@v1",
                "relation": "input",
            }
        ],
        "payload": {
            "instance_id": instance_id,
            "plan_id": "shot003-kf002-plan-v1",
            "node_id": "layout",
            "operation": "layout",
            "node_type": "deterministic",
            "inputs": inputs or [],
            "started_at": "2026-08-11T12:00:01+08:00",
        },
    }


def _node_run_finished_draft(
    instance_id: str,
    started_ref: str,
    outputs: list[str],
) -> dict:
    return {
        "record_type": "node_run_finished",
        "causal_refs": [],
        "payload": {
            "instance_id": instance_id,
            "started_ref": started_ref,
            "outputs": outputs,
            "status": "success",
            "finished_at": "2026-08-11T12:00:02+08:00",
        },
    }


def _artifact_registered_draft(
    blob_ref: str,
    producer_started_ref: str,
) -> dict:
    return {
        "record_type": "artifact_registered",
        "causal_refs": [],
        "payload": {
            "blob_ref": blob_ref,
            "artifact_kind": "layout_plan",
            "schema_version": "layout-plan-v1",
            "producer_started_ref": producer_started_ref,
            "size_bytes": 10,
        },
    }


def _port_bound_draft(
    artifact_ref: str,
    supersedes_binding_ref: str | None = None,
) -> dict:
    return {
        "record_type": "port_bound",
        "causal_refs": [],
        "payload": {
            "binding_id": "bind_001",
            "logical_port_ref": PORT,
            "artifact_ref": artifact_ref,
            "supersedes_binding_ref": supersedes_binding_ref,
        },
    }


def _bind_one_artifact(
    writer: LedgerWriter, instance_id: str, blob_seed: str
) -> tuple[str, str]:
    started = writer.append(_node_run_started_draft(instance_id))
    artifact = writer.append(
        _artifact_registered_draft(
            _blob(blob_seed), f"ledger://{RUN}/{started.record_id}"
        )
    )
    artifact_ref = f"artifact://{RUN}/{artifact.payload.artifact_id}"
    writer.append(
        _node_run_finished_draft(
            instance_id,
            f"ledger://{RUN}/{started.record_id}",
            [artifact_ref],
        )
    )
    return artifact_ref, started.record_id


def test_append_happy_path(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    writer = LedgerWriter(ledger_path, RUN)
    run_started = writer.append(_run_started_draft())
    started = writer.append(_node_run_started_draft())
    artifact = writer.append(
        _artifact_registered_draft(
            _blob("a"), f"ledger://{RUN}/{started.record_id}"
        )
    )
    artifact_ref = f"artifact://{RUN}/{artifact.payload.artifact_id}"
    writer.append(
        _node_run_finished_draft(
            "layout-run-17",
            f"ledger://{RUN}/{started.record_id}",
            [artifact_ref],
        )
    )
    bound = writer.append(_port_bound_draft(artifact_ref))
    assert [run_started.seq, started.seq, artifact.seq, bound.seq] == [
        1,
        2,
        3,
        5,
    ]
    assert run_started.record_id == "rec_000001"
    assert artifact.payload.artifact_id == "art_000003"
    records = read_complete_records(ledger_path)
    assert len(records) == 5


def test_seq_monotonic_and_resume(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    writer = LedgerWriter(ledger_path, RUN)
    writer.append(_run_started_draft())
    writer.append(_node_run_started_draft())
    resumed = LedgerWriter(ledger_path, RUN)
    artifact = resumed.append(
        _artifact_registered_draft(_blob("a"), "ledger://run_001/rec_000002")
    )
    assert artifact.seq == 3
    assert artifact.record_id == "rec_000003"


def test_forward_reference_rejected(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    writer.append(_run_started_draft())
    draft = _node_run_started_draft()
    draft["causal_refs"] = [
        {"record_ref": "ledger://run_001/rec_999999", "relation": "input"}
    ]
    with pytest.raises(InputValidationError):
        writer.append(draft)


def test_unregistered_artifact_input_rejected(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    writer.append(_run_started_draft())
    draft = _node_run_started_draft(
        inputs=["artifact://run_001/art_999999"]
    )
    with pytest.raises(InputValidationError):
        writer.append(draft)


def test_stale_supersedes_rejected(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    writer.append(_run_started_draft())
    artifact_1, _ = _bind_one_artifact(writer, "layout-run-1", "a")
    bind_1 = writer.append(_port_bound_draft(artifact_1))
    artifact_2, _ = _bind_one_artifact(writer, "layout-run-2", "b")
    bind_2 = writer.append(
        _port_bound_draft(
            artifact_2,
            supersedes_binding_ref=f"ledger://{RUN}/{bind_1.record_id}",
        )
    )
    assert bind_2.seq > bind_1.seq
    # stale: bind_1 is no longer the terminal binding
    artifact_3, _ = _bind_one_artifact(writer, "layout-run-3", "c")
    with pytest.raises(InputValidationError):
        writer.append(
            _port_bound_draft(
                artifact_3,
                supersedes_binding_ref=f"ledger://{RUN}/{bind_1.record_id}",
            )
        )
    # rebind without supersedes when already bound
    with pytest.raises(InputValidationError):
        writer.append(_port_bound_draft(artifact_3))


def test_target_type_whitelist_enforced(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    writer.append(_run_started_draft())
    started = writer.append(_node_run_started_draft())
    artifact = writer.append(
        _artifact_registered_draft(
            _blob("a"), f"ledger://{RUN}/{started.record_id}"
        )
    )
    artifact_ref = f"artifact://{RUN}/{artifact.payload.artifact_id}"
    finished = writer.append(
        _node_run_finished_draft(
            "layout-run-17",
            f"ledger://{RUN}/{started.record_id}",
            [artifact_ref],
        )
    )
    draft = _port_bound_draft(artifact_ref)
    draft["causal_refs"] = [
        {
            "record_ref": f"ledger://{RUN}/{finished.record_id}",
            "relation": "supersedes",
        }
    ]
    with pytest.raises(InputValidationError):
        writer.append(draft)


def test_first_record_must_be_run_started(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    with pytest.raises(InputValidationError):
        writer.append(_node_run_started_draft())


def test_run_started_run_id_mismatch(tmp_path) -> None:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    draft = _run_started_draft()
    draft["payload"]["run_id"] = "run_other"
    with pytest.raises(InputValidationError):
        writer.append(draft)


def test_duplicate_record_in_file_rejected(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    writer = LedgerWriter(ledger_path, RUN)
    record = writer.append(_run_started_draft())
    parsed = parse_ledger_record(
        {
            "record_id": record.record_id,
            "seq": record.seq,
            "record_type": "run_started",
            "ledger_schema_version": "execution-ledger-v1",
            "run_ref": RUN,
            "recorded_at": record.recorded_at,
            "causal_refs": [],
            "payload": record.payload.model_dump(mode="json"),
        }
    )
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                parsed.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    with pytest.raises(InputValidationError):
        LedgerWriter(ledger_path, RUN)
