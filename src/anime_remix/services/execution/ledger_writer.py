"""Single LedgerWriter: the only write path for ``execution-ledger.jsonl``.

Freeze reference (Round 8..10, 2026-08-11):

- All components submit record drafts; only the LedgerWriter assigns
  ``seq`` / ``record_id`` / ``recorded_at``, validates schema, refs,
  relations and target types, forbids forward references, and enforces
  PortBinding CAS (supersedes must target the current terminal binding).
- The writer never trusts caller-provided content hashes; artifact
  registration flows through ArtifactStore which computes them.
- On startup, an incomplete JSON tail (crash residue) is truncated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.execution_ledger import (
    LedgerRecord,
    parse_ledger_record,
    ref_scope,
    validate_target_type,
)

_UTC = UTC


def _now_iso() -> str:
    return datetime.now(_UTC).isoformat()


def _json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _record_id_of(ref: str) -> str:
    """Normalize a ledger ref (or bare record id) to its record id."""

    return ref.rsplit("/", 1)[-1]


def _read_ledger_with_offsets(
    path: Path,
) -> tuple[list[LedgerRecord], int, bool]:
    """Read complete records; return (records, good_bytes, had_tail)."""

    if not path.exists():
        return [], 0, False
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    records: list[LedgerRecord] = []
    offset = 0
    had_tail = False
    for line in lines:
        if line.strip() == b"":
            offset += len(line) + 1
            continue
        next_offset = offset + len(line) + 1
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            had_tail = True
            break
        records.append(parse_ledger_record(data))
        offset = next_offset
    if not had_tail:
        offset = len(raw)
    return records, min(offset, len(raw)), had_tail


def read_complete_records(path: str | Path) -> list[LedgerRecord]:
    """Read all complete ledger records (used by the read-only Resolver)."""

    records, _offset, _tail = _read_ledger_with_offsets(Path(path))
    return records


class LedgerWriter:
    """Append-only ledger writer for one run."""

    def __init__(self, ledger_path: str | Path, run_ref: str) -> None:
        self.ledger_path = Path(ledger_path)
        self.run_ref = run_ref
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        records, offset, had_tail = _read_ledger_with_offsets(self.ledger_path)
        if had_tail:
            with self.ledger_path.open("r+b") as fh:
                fh.truncate(offset)
        self._records: dict[str, LedgerRecord] = {}
        self._artifacts: set[str] = set()
        self._ports: dict[str, str] = {}
        self._max_seq = 0
        for record in records:
            self._index_record(record)
        self._next_seq = self._max_seq + 1

    def append(self, draft: dict) -> LedgerRecord:
        """Validate a draft, assign identity, and append one fact."""

        if not isinstance(draft, dict):
            raise InputValidationError("draft must be a dict")
        record_type = draft.get("record_type")
        payload = draft.get("payload")
        causal_refs = draft.get("causal_refs", [])
        if not isinstance(record_type, str):
            raise InputValidationError("draft requires record_type")
        if not isinstance(payload, dict):
            raise InputValidationError("draft requires payload")
        if not isinstance(causal_refs, list):
            raise InputValidationError("causal_refs must be a list")
        if not self._records and record_type != "run_started":
            raise InputValidationError(
                "first ledger record must be run_started"
            )

        seq = self._next_seq
        full = {
            "record_id": f"rec_{seq:06d}",
            "seq": seq,
            "record_type": record_type,
            "ledger_schema_version": "execution-ledger-v1",
            "run_ref": self.run_ref,
            "recorded_at": _now_iso(),
            "causal_refs": causal_refs,
            "payload": payload,
        }
        if record_type == "artifact_registered":
            expected_artifact_id = f"art_{seq:06d}"
            provided = payload.get("artifact_id")
            if provided is not None and provided != expected_artifact_id:
                raise InputValidationError(
                    "artifact_id must be assigned by the writer: "
                    f"{expected_artifact_id}"
                )
            payload["artifact_id"] = expected_artifact_id

        parsed = parse_ledger_record(full)
        self._validate_writer_refs(parsed)
        self._apply_cas(parsed)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(_json_line(full) + "\n")
            fh.flush()
        self._index_record(parsed)
        self._next_seq = seq + 1
        return parsed

    def _index_record(self, record: LedgerRecord) -> None:
        if record.run_ref != self.run_ref:
            raise InputValidationError(
                f"ledger run_ref {record.run_ref!r} != writer run_ref "
                f"{self.run_ref!r}"
            )
        if record.record_id in self._records:
            raise InputValidationError(
                f"duplicate record_id {record.record_id}"
            )
        if record.seq <= self._max_seq:
            raise InputValidationError("seq must be strictly increasing")
        self._max_seq = record.seq
        if record.record_type == "port_bound":
            self._ports[record.payload.logical_port_ref] = record.record_id
        if record.record_type == "artifact_registered":
            self._artifacts.add(record.payload.artifact_id)
        self._records[record.record_id] = record

    def _validate_writer_refs(self, record: LedgerRecord) -> None:
        for ref in record.causal_refs:
            scope = ref_scope(ref.record_ref)
            if scope == "ledger":
                target = self._records.get(_record_id_of(ref.record_ref))
                if target is None:
                    raise InputValidationError(
                        "forward reference: causal ref "
                        f"{ref.record_ref!r} does not exist"
                    )
                validate_target_type(
                    record.record_type, ref.relation, target.record_type
                )
            elif scope == "artifact":
                artifact_id = ref.record_ref.rsplit("/", 1)[-1]
                if artifact_id not in self._artifacts:
                    raise InputValidationError(
                        "forward reference: artifact "
                        f"{ref.record_ref!r} is not registered"
                    )
        payload = record.payload
        record_type = record.record_type
        if record_type == "run_started":
            if payload.run_id != self.run_ref:
                raise InputValidationError(
                    f"run_started run_id {payload.run_id!r} != writer "
                    f"run_ref {self.run_ref!r}"
                )
        elif record_type == "node_run_started":
            for ref in payload.inputs:
                self._require_input_ref(ref, "node_run_started.inputs")
        elif record_type == "node_run_finished":
            self._require_ledger_ref(
                payload.started_ref, "node_run_finished.started_ref"
            )
            for ref in payload.outputs:
                self._require_artifact_ref(
                    ref, "node_run_finished.outputs"
                )
        elif record_type == "artifact_registered":
            self._require_ledger_ref(
                payload.producer_started_ref,
                "artifact_registered.producer_started_ref",
            )
        elif record_type == "port_bound":
            self._require_artifact_ref(
                payload.artifact_ref, "port_bound.artifact_ref"
            )
            if payload.supersedes_binding_ref is not None:
                self._require_ledger_ref(
                    payload.supersedes_binding_ref,
                    "port_bound.supersedes_binding_ref",
                )
            if payload.bound_by_ref is not None:
                self._require_ledger_ref(
                    payload.bound_by_ref, "port_bound.bound_by_ref"
                )
        elif record_type == "render_intent_created":
            self._require_ref_if_uri(
                payload.reference_package_ref,
                "render_intent_created.reference_package_ref",
            )
            self._require_ref_if_uri(
                payload.constraint_set_ref,
                "render_intent_created.constraint_set_ref",
            )
        elif record_type == "model_render_request_created":
            self._require_ledger_ref(
                payload.intent_ref,
                "model_render_request_created.intent_ref",
            )
        elif record_type == "render_attempt_started":
            self._require_ledger_ref(
                payload.request_ref, "render_attempt_started.request_ref"
            )
        elif record_type == "render_attempt_finished":
            self._require_ledger_ref(
                payload.started_ref,
                "render_attempt_finished.started_ref",
            )
            self._require_ledger_ref(
                payload.request_ref,
                "render_attempt_finished.request_ref",
            )
            if payload.output_artifact_ref is not None:
                self._require_artifact_ref(
                    payload.output_artifact_ref,
                    "render_attempt_finished.output_artifact_ref",
                )
        elif record_type == "repair_decision":
            self._require_ledger_ref(
                payload.failure_event_ref,
                "repair_decision.failure_event_ref",
            )

    def _require_input_ref(self, ref: str, context: str) -> None:
        scope = ref_scope(ref)
        if scope == "artifact":
            artifact_id = ref.rsplit("/", 1)[-1]
            if artifact_id not in self._artifacts:
                raise InputValidationError(
                    f"forward reference: {context} artifact {ref!r} "
                    "is not registered"
                )
        elif scope == "ledger" and _record_id_of(ref) not in self._records:
            raise InputValidationError(
                f"forward reference: {context} {ref!r} does not exist"
            )

    def _require_ref_if_uri(self, ref: str, context: str) -> None:
        if ref.startswith(("ledger://", "artifact://", "asset://", "blob://")):
            self._require_input_ref(ref, context)

    def _require_ledger_ref(self, ref: str, context: str) -> None:
        if _record_id_of(ref) not in self._records:
            raise InputValidationError(
                f"forward reference: {context} {ref!r} does not exist"
            )

    def _require_artifact_ref(self, ref: str, context: str) -> None:
        artifact_id = ref.rsplit("/", 1)[-1]
        if artifact_id not in self._artifacts:
            raise InputValidationError(
                f"forward reference: {context} artifact {ref!r} "
                "is not registered"
            )

    def _apply_cas(self, record: LedgerRecord) -> None:
        if record.record_type != "port_bound":
            return
        payload = record.payload
        current = self._ports.get(payload.logical_port_ref)
        if payload.supersedes_binding_ref is None:
            if current is not None:
                raise InputValidationError(
                    f"port {payload.logical_port_ref!r} already bound; "
                    "new binding must supersede the current binding"
                )
            return
        supersedes_id = _record_id_of(payload.supersedes_binding_ref)
        if supersedes_id not in self._records:
            raise InputValidationError(
                "supersedes_binding_ref must target an existing record"
            )
        target = self._records[supersedes_id]
        if target.record_type != "port_bound":
            raise InputValidationError(
                "supersedes_binding_ref must target a port_bound record"
            )
        if target.payload.logical_port_ref != payload.logical_port_ref:
            raise InputValidationError(
                "supersedes target must bind the same logical port"
            )
        if current != supersedes_id:
            raise InputValidationError(
                "stale supersedes: target is not the current terminal "
                f"binding (current={current!r})"
            )
