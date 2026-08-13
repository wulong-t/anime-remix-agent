"""ArtifactStore: content-addressed Blob storage + ArtifactInstance registry.

Freeze reference (Round 9/10, 2026-08-11):

- Blob identity is computed by the trusted storage/registration boundary,
  never from a caller-supplied hash.
- Structured artifacts are canonically serialized before hashing
  (``canonicalize(data, canonicalizer_version)`` then SHA-256).
- ``ContentBlob`` is deduplicated; every production still creates its own
  ``ArtifactInstance`` (ledger ``artifact_registered`` fact).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.execution_ledger import LedgerRecord
from anime_remix.services.execution.ledger_writer import LedgerWriter

_CANONICAL_JSON_V1 = "canonical-json-v1"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(data: object) -> bytes:
    """Canonical JSON serialization (canonicalizer_version=canonical-json-v1)."""

    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ArtifactStore:
    """Content-addressed blob store under ``<root>/artifacts/blobs``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blobs_dir = self.root / "artifacts" / "blobs"

    def blob_path(self, blob_ref: str) -> Path:
        if not blob_ref.startswith("blob://sha256:"):
            raise InputValidationError(f"invalid blob ref {blob_ref!r}")
        digest = blob_ref.removeprefix("blob://sha256:")
        return self.blobs_dir / digest

    def has_blob(self, blob_ref: str) -> bool:
        return self.blob_path(blob_ref).exists()

    def size_of(self, blob_ref: str) -> int:
        return self.blob_path(blob_ref).stat().st_size

    def store_bytes(self, data: bytes) -> str:
        digest = sha256_hex(data)
        blob_ref = f"blob://sha256:{digest}"
        path = self.blobs_dir / digest
        if not path.exists():
            self.blobs_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return blob_ref

    def store_text(self, text: str) -> str:
        return self.store_bytes(text.encode("utf-8"))

    def store_structured(
        self,
        data: object,
        canonicalizer_version: str = _CANONICAL_JSON_V1,
    ) -> tuple[str, bytes]:
        if canonicalizer_version != _CANONICAL_JSON_V1:
            raise InputValidationError(
                "unsupported canonicalizer_version "
                f"{canonicalizer_version!r}"
            )
        canonical = canonical_json_bytes(data)
        return self.store_bytes(canonical), canonical

    def read_bytes(self, blob_ref: str) -> bytes:
        path = self.blob_path(blob_ref)
        if not path.exists():
            raise FileNotFoundError(f"blob not found: {blob_ref}")
        return path.read_bytes()


def register_artifact(
    store: ArtifactStore,
    writer: LedgerWriter,
    *,
    artifact_kind: str,
    schema_version: str,
    producer_started_ref: str,
    data: object,
    canonicalize: bool = False,
) -> tuple[str, LedgerRecord]:
    """Store bytes, compute the real content hash, and register an instance.

    The caller never supplies a hash: ``store`` computes the canonical Blob
    identity and ``writer`` assigns the ArtifactInstance id.
    """

    if canonicalize:
        blob_ref, _ = store.store_structured(data)
    elif isinstance(data, bytes):
        blob_ref = store.store_bytes(data)
    elif isinstance(data, str):
        blob_ref = store.store_text(data)
    else:
        raise TypeError(
            "register_artifact data must be bytes/str, or dict/list "
            "with canonicalize=True"
        )
    size_bytes = store.size_of(blob_ref)
    record = writer.append(
        {
            "record_type": "artifact_registered",
            "causal_refs": [],
            "payload": {
                "blob_ref": blob_ref,
                "artifact_kind": artifact_kind,
                "schema_version": schema_version,
                "producer_started_ref": producer_started_ref,
                "size_bytes": size_bytes,
            },
        }
    )
    artifact_ref = (
        f"artifact://{writer.run_ref}/{record.payload.artifact_id}"
    )
    return artifact_ref, record
