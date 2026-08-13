"""Read-only Resolver: folds an execution ledger into queryable state.

Freeze reference (Round 10, 2026-08-11): ``Resolver.current()`` reconstructs
the current artifact of any logical port from the ledger alone.  This is the
minimum query that proves the PortBinding design; ``why()`` / ``replay()``
are deliberately out of scope for Phase 2.
"""

from __future__ import annotations

from pathlib import Path

from anime_remix.services.execution.execution_ledger import LedgerRecord
from anime_remix.services.execution.ledger_writer import read_complete_records


class Resolver:
    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self._records: dict[str, LedgerRecord] = {}
        self._ports: dict[str, str] = {}
        for record in read_complete_records(self.ledger_path):
            self._records[record.record_id] = record
            if record.record_type == "port_bound":
                self._ports[record.payload.logical_port_ref] = (
                    record.record_id
                )

    def current(self, logical_port_ref: str) -> str | None:
        """Return the current artifact ref bound to a logical port."""

        binding_id = self._ports.get(logical_port_ref)
        if binding_id is None:
            return None
        record = self._records[binding_id]
        if record.record_type != "port_bound":
            return None
        return record.payload.artifact_ref

    def current_binding(self, logical_port_ref: str) -> str | None:
        """Return the ledger ref of the terminal binding for a port."""

        binding_id = self._ports.get(logical_port_ref)
        if binding_id is None:
            return None
        return f"ledger://{self._run_id()}/{binding_id}"

    def record(self, record_ref: str) -> LedgerRecord | None:
        record_id = record_ref.rsplit("/", 1)[-1]
        return self._records.get(record_id)

    def _run_id(self) -> str:
        if not self._records:
            return ""
        first = next(iter(self._records.values()))
        return first.run_ref
