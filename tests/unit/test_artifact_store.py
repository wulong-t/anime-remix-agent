"""Unit tests for the Phase 2 ArtifactStore and registration path."""

from __future__ import annotations

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.artifact_store import (
    ArtifactStore,
    canonical_json_bytes,
    register_artifact,
    sha256_hex,
)
from anime_remix.services.execution.ledger_writer import LedgerWriter

RUN = "run_001"


def _writer_with_producer(tmp_path, instance_id: str = "layout-run-1"):
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
    producer = writer.append(
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
    return writer, f"ledger://{RUN}/{producer.record_id}"


def test_same_bytes_same_blob_dedup(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.store_bytes(b"same bytes")
    second = store.store_bytes(b"same bytes")
    assert first == second
    blobs = list((tmp_path / "artifacts" / "blobs").iterdir())
    assert len(blobs) == 1


def test_structured_canonicalization(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref_a, bytes_a = store.store_structured({"a": 1, "b": 2})
    ref_b, bytes_b = store.store_structured({"b": 2, "a": 1})
    assert ref_a == ref_b
    assert bytes_a == bytes_b == canonical_json_bytes({"a": 1, "b": 2})
    assert ref_a == "blob://sha256:" + sha256_hex(bytes_a)


def test_hash_not_trusted_from_caller(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    writer, producer_ref = _writer_with_producer(tmp_path)
    data = {"layout_plan": "resolved numbers"}
    artifact_ref, record = register_artifact(
        store,
        writer,
        artifact_kind="layout_plan",
        schema_version="layout-plan-v1",
        producer_started_ref=producer_ref,
        data=data,
        canonicalize=True,
    )
    expected_bytes = canonical_json_bytes(data)
    assert record.payload.blob_ref == "blob://sha256:" + sha256_hex(
        expected_bytes
    )
    assert record.payload.size_bytes == len(expected_bytes)
    assert artifact_ref.startswith(f"artifact://{RUN}/art_")
    assert store.read_bytes(record.payload.blob_ref) == expected_bytes


def test_same_blob_different_artifact_instances(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    writer, producer_1 = _writer_with_producer(tmp_path, "layout-run-1")
    producer_2 = writer.append(
        {
            "record_type": "node_run_started",
            "causal_refs": [],
            "payload": {
                "instance_id": "layout-run-2",
                "plan_id": "shot003-kf002-plan-v1",
                "node_id": "layout",
                "operation": "layout",
                "node_type": "deterministic",
                "inputs": [],
                "started_at": "2026-08-11T12:00:02+08:00",
            },
        }
    )
    producer_2_ref = f"ledger://{RUN}/{producer_2.record_id}"
    data = {"payload": "identical content"}
    ref_1, record_1 = register_artifact(
        store,
        writer,
        artifact_kind="layout_plan",
        schema_version="layout-plan-v1",
        producer_started_ref=producer_1,
        data=data,
        canonicalize=True,
    )
    ref_2, record_2 = register_artifact(
        store,
        writer,
        artifact_kind="layout_plan",
        schema_version="layout-plan-v1",
        producer_started_ref=producer_2_ref,
        data=data,
        canonicalize=True,
    )
    assert ref_1 != ref_2
    assert record_1.payload.artifact_id != record_2.payload.artifact_id
    assert record_1.payload.blob_ref == record_2.payload.blob_ref


def test_register_bytes_roundtrip(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    writer, producer_ref = _writer_with_producer(tmp_path)
    _ref, record = register_artifact(
        store,
        writer,
        artifact_kind="character_layer",
        schema_version="png",
        producer_started_ref=producer_ref,
        data=b"\x89PNG\r\n\x1a\n" + b"payload",
    )
    assert store.read_bytes(record.payload.blob_ref) == (
        b"\x89PNG\r\n\x1a\n" + b"payload"
    )


def test_unsupported_canonicalizer_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(InputValidationError):
        store.store_structured(
            {"a": 1}, canonicalizer_version="canonical-json-v2"
        )
