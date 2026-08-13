"""Unit tests for the Phase 2 read-only Resolver.current()."""

from __future__ import annotations

from anime_remix.services.execution.ledger_writer import LedgerWriter
from anime_remix.services.execution.resolver import Resolver

RUN = "run_001"
PORT = "plan://shot003-kf002-plan-v1/ports/layout_plan"


def _blob(seed: str = "a") -> str:
    return "blob://sha256:" + seed * 64


def _bind_one(
    writer: LedgerWriter, instance_id: str, blob_seed: str
) -> str:
    started = writer.append(
        {
            "record_type": "node_run_started",
            "causal_refs": [],
            "payload": {
                "instance_id": instance_id,
                "plan_id": "shot003-kf002-plan-v1",
                "node_id": "layout",
                "operation": "layout",
                "node_type": "deterministic",
                "inputs": [],
                "started_at": "2026-08-11T12:00:01+08:00",
            },
        }
    )
    artifact = writer.append(
        {
            "record_type": "artifact_registered",
            "causal_refs": [],
            "payload": {
                "blob_ref": _blob(blob_seed),
                "artifact_kind": "layout_plan",
                "schema_version": "layout-plan-v1",
                "producer_started_ref": f"ledger://{RUN}/{started.record_id}",
                "size_bytes": 10,
            },
        }
    )
    artifact_ref = f"artifact://{RUN}/{artifact.payload.artifact_id}"
    writer.append(
        {
            "record_type": "node_run_finished",
            "causal_refs": [],
            "payload": {
                "instance_id": instance_id,
                "started_ref": f"ledger://{RUN}/{started.record_id}",
                "outputs": [artifact_ref],
                "status": "success",
                "finished_at": "2026-08-11T12:00:02+08:00",
            },
        }
    )
    return artifact_ref


def _seed_writer(tmp_path) -> tuple[LedgerWriter, str]:
    writer = LedgerWriter(tmp_path / "ledger.jsonl", RUN)
    writer.append(
        {
            "record_type": "run_started",
            "causal_refs": [],
            "payload": {
                "run_id": RUN,
                "execution_template_ref": "template://compose-v1",
                "policy_refs": [],
                "started_at": "2026-08-11T12:00:00+08:00",
            },
        }
    )
    return writer


def test_current_folds_ledger_to_latest_binding(tmp_path) -> None:
    writer = _seed_writer(tmp_path)
    artifact_1 = _bind_one(writer, "layout-run-1", "a")
    bind_1 = writer.append(
        {
            "record_type": "port_bound",
            "causal_refs": [],
            "payload": {
                "binding_id": "bind_001",
                "logical_port_ref": PORT,
                "artifact_ref": artifact_1,
            },
        }
    )
    artifact_2 = _bind_one(writer, "layout-run-2", "b")
    bind_2 = writer.append(
        {
            "record_type": "port_bound",
            "causal_refs": [],
            "payload": {
                "binding_id": "bind_002",
                "logical_port_ref": PORT,
                "artifact_ref": artifact_2,
                "supersedes_binding_ref": f"ledger://{RUN}/{bind_1.record_id}",
            },
        }
    )

    resolver = Resolver(tmp_path / "ledger.jsonl")
    assert resolver.current(PORT) == artifact_2
    assert resolver.current_binding(PORT) == (
        f"ledger://{RUN}/{bind_2.record_id}"
    )


def test_current_unknown_port_returns_none(tmp_path) -> None:
    writer = _seed_writer(tmp_path)
    artifact = _bind_one(writer, "layout-run-1", "a")
    writer.append(
        {
            "record_type": "port_bound",
            "causal_refs": [],
            "payload": {
                "binding_id": "bind_001",
                "logical_port_ref": PORT,
                "artifact_ref": artifact,
            },
        }
    )
    resolver = Resolver(tmp_path / "ledger.jsonl")
    assert resolver.current("plan://other/ports/nonexistent") is None
