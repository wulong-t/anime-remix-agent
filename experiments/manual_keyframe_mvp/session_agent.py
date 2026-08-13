#!/usr/bin/env python
"""G1-MK4-L: thin manual-shot session coordinator (local state machine).

Coordinates the already-PASS manual-keyframe tools without duplicating their
validation/media logic: ``init -> awaiting_approval -> awaiting_remote ->
finalize+QA -> complete``.  Local-only (no SSH/SCP/Remote/AniSora/FFmpeg/
shell); stores only minimal session truth; ``next_action`` is derived from
phase.  Transitions run in verified temporary child/siblings and publish only
after the existing PASS validators succeed, cleaning only coordinator-owned
new paths on failure so the prior phase stays retryable.  Never reads real
media, secrets, environment values, Remote files or ``.tmp``/``runs`` content.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from anime_remix.json_io import dump_json_atomic
from experiments.manual_keyframe_mvp import manual_keyframe_mvp as harness
from experiments.manual_keyframe_mvp import qa_evidence, remote_sample

SESSION_SCHEMA = "g1-mk4-manual-shot-session-v1"
SESSION_FILE = "session.json"
PHASE_AWAITING_APPROVAL = "awaiting_approval"
PHASE_AWAITING_REMOTE = "awaiting_remote"
PHASE_COMPLETE = "complete"
SESSION_PHASES = (PHASE_AWAITING_APPROVAL, PHASE_AWAITING_REMOTE, PHASE_COMPLETE)

# Fixed relative workspace artifact names (no discovery/enumeration).
INSPECTION_DIR = "inspection"
PACKAGE_DIR = "package"
FINALIZED_DIR = "finalized"
QA_DIR = "qa"
INSPECTION_JSON = f"{INSPECTION_DIR}/inspection.json"
APPROVAL_JSON = f"{INSPECTION_DIR}/approval.json"
PACKAGE_MANIFEST = f"{PACKAGE_DIR}/package_manifest.json"
GENERATION_MANIFEST = f"{FINALIZED_DIR}/generation_manifest.json"
QA_METRICS = f"{QA_DIR}/metrics.json"
QA_ARTIFACTS = f"{QA_DIR}/artifacts.json"

# Exact successful remote-output files consumed by finalize + QA.
REMOTE_RAW = "raw_shot.mp4"
REMOTE_RECEIPT = "sampling_receipt.json"
REMOTE_FILES = (REMOTE_RAW, REMOTE_RECEIPT)

SESSION_FIELDS = {"schema_version", "request", "sample_steps", "phase",
                  "package_manifest_sha256", "completion"}
REQUEST_RECORD_FIELDS = {"path", "id", "sha256"}
COMPLETION_RECORD_FIELDS = {"generation_manifest_sha256", "qa_metrics_sha256",
                            "qa_artifacts_sha256"}
NEXT_ACTIONS = {
    PHASE_AWAITING_APPROVAL: (
        "approve", APPROVAL_JSON,
        "edit and approve inspection/approval.json, then run advance to package",
    ),
    PHASE_AWAITING_REMOTE: (
        "provide_remote_output", None,
        "run remote worker on package/, then pass one successful remote-output dir",
    ),
    PHASE_COMPLETE: ("none", None, "session complete; advance is read-only status"),
}


def _require_exact_keys(data: Any, fields: set[str], what: str) -> dict[str, Any]:
    data = harness._require_object(data, what, "evidence_incomplete")
    if set(data) == fields:
        return data
    missing = sorted(fields - set(data))
    unknown = sorted(set(data) - fields)
    detail = [f"missing keys: {', '.join(missing)}"] if missing else []
    if unknown:
        detail.append(f"unknown keys: {', '.join(unknown)}")
    raise harness.HarnessError("evidence_incomplete", f"{what}: " + "; ".join(detail))


def _validate_sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or not harness.SHA256_RE.fullmatch(value):
        raise harness.HarnessError(
            "evidence_incomplete", f"{what} must be 64 lowercase hex characters"
        )
    return value


def _capture_bytes(path: Path, what: str) -> bytes:
    if path.is_symlink():
        raise harness.HarnessError(
            "evidence_incomplete", f"{what} must not be a symlink"
        )
    return harness._capture_bytes(path, "evidence_incomplete")


def _file_sha(path: Path, what: str) -> str:
    return harness._sha256_bytes(_capture_bytes(path, what))


def _read_json(path: Path, what: str) -> Any:
    return harness._load_json_bytes(
        _capture_bytes(path, what), what, "evidence_incomplete"
    )


def _next_action(phase: str) -> dict[str, Any]:
    action, target, description = NEXT_ACTIONS[phase]
    return {"action": action, "target": target, "description": description}


def _status_payload(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SESSION_SCHEMA,
        "request_id": session["request"]["id"],
        "request_sha256": session["request"]["sha256"],
        "sample_steps": session["sample_steps"],
        "phase": session["phase"],
        "package_manifest_sha256": session["package_manifest_sha256"],
        "completion": session["completion"],
        "next_action": _next_action(session["phase"]),
    }


def _validate_request_record(record: dict[str, Any]) -> None:
    request_bytes = _capture_bytes(Path(record["path"]), "request.json")
    if harness._sha256_bytes(request_bytes) != record["sha256"]:
        raise harness.HarnessError(
            "evidence_incomplete", "session request path/hash drift"
        )
    data = harness._require_object(
        harness._load_json_bytes(request_bytes, "request", "evidence_incomplete"),
        "request",
        "evidence_incomplete",
    )
    if (
        data.get("schema_version") != harness.REQUEST_SCHEMA
        or data.get("request_id") != record["id"]
    ):
        raise harness.HarnessError("evidence_incomplete", "request schema/id drift")


def _validate_package_manifest(
    workspace: Path, stored_sha: str, request_id: str
) -> None:
    path = workspace / PACKAGE_MANIFEST
    if _file_sha(path, "package_manifest.json") != stored_sha:
        raise harness.HarnessError(
            "evidence_incomplete", "package_manifest.json hash drift"
        )
    data = _read_json(path, "package_manifest.json")
    if (
        data.get("schema_version") != harness.PACKAGE_MANIFEST_SCHEMA
        or data.get("request_id") != request_id
    ):
        raise harness.HarnessError(
            "evidence_incomplete", "package_manifest.json schema/id drift"
        )


def _validate_completion_record(
    workspace: Path, completion: dict[str, Any], request_id: str
) -> None:
    bindings = (
        (GENERATION_MANIFEST, "generation_manifest_sha256", harness.GENERATION_MANIFEST_SCHEMA),
        (QA_METRICS, "qa_metrics_sha256", qa_evidence.QA_EVIDENCE_SCHEMA),
        (QA_ARTIFACTS, "qa_artifacts_sha256", qa_evidence.QA_ARTIFACTS_SCHEMA),
    )
    for relative, field, schema in bindings:
        path = workspace / relative
        if _file_sha(path, relative) != completion[field]:
            raise harness.HarnessError("evidence_incomplete", f"{relative} hash drift")
        data = _read_json(path, relative)
        if data.get("schema_version") != schema or data.get("request_id") != request_id:
            raise harness.HarnessError("evidence_incomplete", f"{relative} schema/id drift")


def _validate_session(data: Any, workspace: Path) -> dict[str, Any]:
    session = _require_exact_keys(data, SESSION_FIELDS, "session.json")
    if session["schema_version"] != SESSION_SCHEMA:
        raise harness.HarnessError("evidence_incomplete", "session schema drift")
    request = _require_exact_keys(
        session["request"], REQUEST_RECORD_FIELDS, "session.request"
    )
    if not isinstance(request["path"], str) or not request["path"]:
        raise harness.HarnessError(
            "evidence_incomplete", "session.request.path must be non-empty"
        )
    request_id = request["id"]
    if not isinstance(request_id, str) or not harness.REQUEST_ID_RE.fullmatch(
        request_id
    ):
        raise harness.HarnessError("evidence_incomplete", "session.request.id drift")
    _validate_sha256(request["sha256"], "session.request.sha256")
    harness._validate_sample_steps(session["sample_steps"], "evidence_incomplete")
    phase = session["phase"]
    if phase not in SESSION_PHASES:
        raise harness.HarnessError(
            "evidence_incomplete", f"unknown session phase: {phase}"
        )
    package_sha = session["package_manifest_sha256"]
    if package_sha is not None:
        _validate_sha256(package_sha, "session.package_manifest_sha256")
    completion = session["completion"]
    if completion is not None:
        completion = _require_exact_keys(
            completion, COMPLETION_RECORD_FIELDS, "session.completion"
        )
        for field in COMPLETION_RECORD_FIELDS:
            _validate_sha256(completion[field], f"session.completion.{field}")
    if (
        (phase == PHASE_AWAITING_APPROVAL and (package_sha or completion))
        or (phase == PHASE_AWAITING_REMOTE and (package_sha is None or completion))
        or (phase == PHASE_COMPLETE and (package_sha is None or completion is None))
    ):
        raise harness.HarnessError(
            "evidence_incomplete", "artifact/phase drift in session records"
        )
    _validate_request_record(request)
    if package_sha is not None:
        _validate_package_manifest(workspace, package_sha, request_id)
    if completion is not None:
        _validate_completion_record(workspace, completion, request_id)
    return session


def _load_session(workspace: Path) -> dict[str, Any]:
    workspace = Path(workspace)
    if workspace.is_symlink() or not workspace.is_dir():
        raise harness.HarnessError(
            "input_contract",
            f"workspace must be an exact existing directory: {workspace}",
        )
    return _validate_session(
        _read_json(workspace / SESSION_FILE, "session.json"), workspace
    )


def _publish_dir(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise harness.HarnessError("input_contract", f"output already exists: {output}")
    try:
        os.replace(staging, output)
    except OSError:
        try:
            os.rename(staging, output)
        except OSError as exc:
            raise harness.HarnessError(
                "evidence_incomplete", f"atomic publication failed: {exc}"
            ) from exc


def _remove_exact_child(parent: Path, child: Path) -> None:
    if not child.exists() and not child.is_symlink():
        return
    if child.is_symlink():
        raise harness.HarnessError(
            "evidence_incomplete", f"refusing to remove symlink: {child}"
        )
    parent_root = os.path.normcase(str(parent.resolve()))
    child_root = child.resolve()
    if os.path.normcase(str(child_root.parent)) != parent_root:
        raise harness.HarnessError(
            "evidence_incomplete", f"refusing to remove outside parent: {child}"
        )
    shutil.rmtree(child_root)


def _write_session(workspace: Path, session: dict[str, Any]) -> None:
    dump_json_atomic(workspace / SESSION_FILE, session)


def _session_update(
    session: dict[str, Any], phase: str, package_sha: str | None, completion: Any
) -> dict[str, Any]:
    return {
        "schema_version": SESSION_SCHEMA,
        "request": dict(session["request"]),
        "sample_steps": session["sample_steps"],
        "phase": phase,
        "package_manifest_sha256": package_sha,
        "completion": completion,
    }


def cmd_init(
    request: Path,
    workspace: Path,
    sample_steps: int = harness.SAMPLE_STEPS_DEFAULT,
) -> Path:
    """Validate a request via ``cmd_inspect`` and start a session atomically."""

    sample_steps = harness._validate_sample_steps(sample_steps)
    request = Path(request)
    workspace = Path(workspace)
    if workspace.exists() or workspace.is_symlink():
        raise harness.HarnessError(
            "input_contract", f"workspace already exists: {workspace}"
        )
    request_bytes = harness._capture_bytes(request, "input_contract")
    request_sha256 = harness._sha256_bytes(request_bytes)
    parent = workspace.resolve().parent
    temp = Path(tempfile.mkdtemp(prefix=f".{workspace.name}.init-staging-", dir=parent))
    try:
        harness.cmd_inspect(request, temp / INSPECTION_DIR)
        if harness._sha256_bytes(
            harness._capture_bytes(request, "input_contract")
        ) != request_sha256:
            raise harness.HarnessError("evidence_incomplete", "request changed during init")
        request_id = _read_json(temp / INSPECTION_JSON, "inspection.json")["request_id"]
        session = {
            "schema_version": SESSION_SCHEMA,
            "request": {
                "path": str(request.resolve()),
                "id": request_id,
                "sha256": request_sha256,
            },
            "sample_steps": sample_steps,
            "phase": PHASE_AWAITING_APPROVAL,
            "package_manifest_sha256": None,
            "completion": None,
        }
        _write_session(temp, session)
        _publish_dir(temp, workspace)
    except BaseException:
        _remove_exact_child(parent, temp)
        raise
    return workspace


def cmd_status(workspace: Path) -> dict[str, Any]:
    """Read-only machine-readable status with exact binding validation."""

    return _status_payload(_load_session(workspace))


def _advance_approval(workspace: Path, session: dict[str, Any]) -> dict[str, Any]:
    stage = Path(tempfile.mkdtemp(prefix=".package-staging-", dir=workspace))
    published: Path | None = None
    try:
        temp_package = stage / "package"
        harness.cmd_package(
            Path(session["request"]["path"]),
            workspace / INSPECTION_JSON,
            workspace / APPROVAL_JSON,
            temp_package,
            session["sample_steps"],
        )
        package_sha = remote_sample.validate_package(temp_package)[
            "package_manifest_sha256"
        ]
        _publish_dir(temp_package, workspace / PACKAGE_DIR)
        published = workspace / PACKAGE_DIR
        updated = _session_update(
            session, PHASE_AWAITING_REMOTE, package_sha, None
        )
        _write_session(workspace, updated)
        _remove_exact_child(workspace, stage)
    except BaseException:
        if published is not None:
            _remove_exact_child(workspace, published)
        _remove_exact_child(workspace, stage)
        raise
    return _status_payload(updated)


def _advance_remote(
    workspace: Path,
    session: dict[str, Any],
    remote_output: Path | None,
) -> dict[str, Any]:
    if remote_output is None:
        payload = _status_payload(session)
        payload["missing"] = {
            "option": "--remote-output",
            "required": "exact successful remote-output directory",
            "files": list(REMOTE_FILES),
        }
        return payload
    remote = Path(remote_output)
    if remote.is_symlink() or not remote.is_dir():
        raise harness.HarnessError(
            "input_contract",
            f"remote output must be an exact existing directory: {remote}",
        )
    if any(
        (workspace / name).exists() or (workspace / name).is_symlink()
        for name in (FINALIZED_DIR, QA_DIR)
    ):
        raise harness.HarnessError(
            "input_contract", "workspace/finalized or workspace/qa already exists"
        )
    raw = remote / REMOTE_RAW
    receipt = remote / REMOTE_RECEIPT
    finalize_stage = Path(tempfile.mkdtemp(prefix=".finalize-staging-", dir=workspace))
    qa_stage = Path(tempfile.mkdtemp(prefix=".qa-staging-", dir=workspace))
    published: list[Path] = []
    try:
        temp_finalized = finalize_stage / "finalized"
        temp_qa = qa_stage / "qa"
        harness.cmd_finalize(workspace / PACKAGE_DIR, raw, receipt, temp_finalized)
        qa_evidence.cmd_qa(workspace / PACKAGE_DIR, raw, temp_finalized, temp_qa)
        completion = {
            "generation_manifest_sha256": _file_sha(
                temp_finalized / "generation_manifest.json",
                "generation_manifest.json",
            ),
            "qa_metrics_sha256": _file_sha(
                temp_qa / "metrics.json", "qa/metrics.json"
            ),
            "qa_artifacts_sha256": _file_sha(
                temp_qa / "artifacts.json", "qa/artifacts.json"
            ),
        }
        _publish_dir(temp_finalized, workspace / FINALIZED_DIR)
        published.append(workspace / FINALIZED_DIR)
        _publish_dir(temp_qa, workspace / QA_DIR)
        published.append(workspace / QA_DIR)
        updated = _session_update(
            session, PHASE_COMPLETE, session["package_manifest_sha256"], completion
        )
        _write_session(workspace, updated)
        _remove_exact_child(workspace, finalize_stage)
        _remove_exact_child(workspace, qa_stage)
    except BaseException:
        for path in published:
            _remove_exact_child(workspace, path)
        _remove_exact_child(workspace, finalize_stage)
        _remove_exact_child(workspace, qa_stage)
        raise
    return _status_payload(updated)


def cmd_advance(
    workspace: Path, remote_output: Path | None = None
) -> dict[str, Any]:
    """Advance the session state machine (complete is idempotent status)."""

    session = _load_session(workspace)
    phase = session["phase"]
    if phase == PHASE_AWAITING_APPROVAL:
        return _advance_approval(workspace, session)
    if phase == PHASE_AWAITING_REMOTE:
        return _advance_remote(workspace, session, remote_output)
    return _status_payload(session)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="session_agent",
                                     description="thin local session coordinator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser(
        "init", help="validate a request and start a new session"
    )
    init_parser.add_argument("--request", required=True, type=Path)
    init_parser.add_argument("--workspace", required=True, type=Path)
    init_parser.add_argument(
        "--sample-steps", type=int, default=harness.SAMPLE_STEPS_DEFAULT, metavar="1..100",
        help=f"packaged sampling steps, 1..100 (default {harness.SAMPLE_STEPS_DEFAULT})",
    )
    subparsers.add_parser("status", help="read-only session status").add_argument(
        "--workspace", required=True, type=Path
    )
    advance_parser = subparsers.add_parser("advance", help="advance the session")
    advance_parser.add_argument("--workspace", required=True, type=Path)
    advance_parser.add_argument(
        "--remote-output",
        type=Path,
        default=None,
        help="exact successful remote-output directory (finalize+QA step)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "init":
            workspace = cmd_init(args.request, args.workspace, args.sample_steps)
            payload = _status_payload(_load_session(workspace))
            payload["status"] = "ok"
        elif args.command == "status":
            payload = cmd_status(args.workspace)
        else:
            payload = cmd_advance(args.workspace, args.remote_output)
    except harness.HarnessError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
