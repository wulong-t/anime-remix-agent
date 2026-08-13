#!/usr/bin/env python3
"""Minimal Local Codex -> SSH -> Remote Codex stage orchestrator.

This tool is deliberately isolated from ``src/anime_remix``.  It supports a
strictly linear pipeline, one remote Git worktree per stage, structured stage
results, and a local PASS gate.  It never merges, rebases, or pushes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    tomllib = None  # type: ignore[assignment]


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_ROOT = TOOL_DIR / ".state"
RESULT_SCHEMA_PATH = TOOL_DIR / "stage-result.schema.json"

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
RESULT_STATUSES = {"pass", "borderline", "fail", "blocked", "needs_user_review"}
NEXT_ACTIONS = {"continue", "stop", "review"}
PLACEHOLDER_HOSTS = {"replace-me", "your-ssh-host", "fill-me", "example-host"}

ERROR_EXIT_CODES = {
    "CONFIG_ERROR": 2,
    "SSH_ERROR": 3,
    "REMOTE_GIT_ERROR": 4,
    "CODEX_ERROR": 5,
    "INVALID_RESULT": 6,
    "STAGE_FAILED": 7,
    "STAGE_BLOCKED": 8,
    "DIRTY_WORKTREE": 9,
    "UNEXPECTED_BRANCH": 10,
    "USER_REVIEW_REQUIRED": 11,
}


class OrchestratorError(RuntimeError):
    """Expected, classified orchestration failure."""

    def __init__(self, code: str, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    repo: PurePosixPath
    worktree_root: PurePosixPath
    identity_file: Path | None = None
    allow_dirty_primary: bool = False
    connect_timeout_seconds: int = 15
    stage_timeout_seconds: int = 900


@dataclass(frozen=True)
class StageConfig:
    id: str
    prompt_declared: str
    prompt_path: Path
    prompt_sha256: str
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    id: str
    source_path: Path
    source_sha256: str
    remote: RemoteConfig
    base_branch: str
    push: bool
    stages: tuple[StageConfig, ...]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OrchestratorError("CONFIG_ERROR", f"{label} must be a TOML table")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrchestratorError("CONFIG_ERROR", f"{label} must be a non-empty string")
    return value.strip()


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise OrchestratorError("CONFIG_ERROR", f"{label} must be a boolean")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise OrchestratorError("CONFIG_ERROR", f"{label} must be a positive integer")
    return value


def _reject_unknown_keys(table: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise OrchestratorError(
            "CONFIG_ERROR",
            f"{label} contains unsupported keys: {', '.join(unknown)}",
        )


def validate_safe_id(value: str, label: str = "stage id") -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise OrchestratorError(
            "CONFIG_ERROR",
            f"{label} must match [A-Za-z0-9_-]+: {value!r}",
        )
    return value


def validate_host(value: str) -> str:
    if value.startswith("-") or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise OrchestratorError(
            "CONFIG_ERROR", "remote.host contains unsafe characters"
        )
    return value


def validate_branch_ref(value: str, label: str = "branch") -> str:
    forbidden = ("..", "@{", "\\", " ", "~", "^", ":", "?", "*", "[")
    if (
        not value
        or value.startswith(("-", "/", "."))
        or value.endswith(("/", ".", ".lock"))
        or "//" in value
        or any(token in value for token in forbidden)
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise OrchestratorError("CONFIG_ERROR", f"unsafe {label}: {value!r}")
    return value


def validate_remote_path(value: str, label: str) -> PurePosixPath:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise OrchestratorError("CONFIG_ERROR", f"{label} contains a control character")
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise OrchestratorError(
            "CONFIG_ERROR",
            f"{label} must be a specific absolute POSIX path without '..'",
        )
    return path


def _is_path_prefix(parent: PurePosixPath, child: PurePosixPath) -> bool:
    parent_parts = parent.parts
    return (
        len(parent_parts) < len(child.parts)
        and child.parts[: len(parent_parts)] == parent_parts
    )


def _resolve_prompt(pipeline_path: Path, declared: str) -> Path:
    declared_path = Path(declared)
    if declared_path.is_absolute():
        raise OrchestratorError(
            "CONFIG_ERROR", "stage prompt must be relative to the pipeline"
        )
    pipeline_dir = pipeline_path.parent.resolve()
    candidate = (pipeline_dir / declared_path).resolve()
    try:
        candidate.relative_to(pipeline_dir)
    except ValueError as exc:
        raise OrchestratorError(
            "CONFIG_ERROR",
            f"stage prompt escapes the pipeline directory: {declared!r}",
        ) from exc
    if not candidate.is_file():
        raise OrchestratorError(
            "CONFIG_ERROR", f"stage prompt does not exist: {candidate}"
        )
    return candidate


def _resolve_identity_file(value: Any) -> Path | None:
    if value is None:
        return None
    declared = _require_string(value, "remote.identity_file")
    candidate = Path(declared).expanduser()
    if not candidate.is_absolute():
        raise OrchestratorError(
            "CONFIG_ERROR", "remote.identity_file must be an absolute local path"
        )
    # Do not stat or read a private key during config parsing. On managed Windows
    # hosts the orchestrator sandbox may be denied metadata access while the
    # approved ssh.exe process can still use the key correctly.
    return candidate.resolve(strict=False)


def load_pipeline(path: Path | str) -> PipelineConfig:
    if tomllib is None:
        raise OrchestratorError(
            "CONFIG_ERROR", "Python 3.11+ is required (tomllib missing)"
        )

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise OrchestratorError(
            "CONFIG_ERROR", f"pipeline file does not exist: {source_path}"
        )
    raw = source_path.read_bytes()
    try:
        payload = tomllib.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise OrchestratorError(
            "CONFIG_ERROR", f"invalid pipeline TOML: {exc}"
        ) from exc

    root = _require_mapping(payload, "pipeline document")
    _reject_unknown_keys(root, {"remote", "pipeline", "stages"}, "pipeline document")
    remote_raw = _require_mapping(root.get("remote"), "[remote]")
    pipeline_raw = _require_mapping(root.get("pipeline"), "[pipeline]")
    stages_raw = root.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise OrchestratorError(
            "CONFIG_ERROR", "at least one [[stages]] entry is required"
        )

    _reject_unknown_keys(
        remote_raw,
        {
            "host",
            "repo",
            "worktree_root",
            "identity_file",
            "allow_dirty_primary",
            "connect_timeout_seconds",
            "stage_timeout_seconds",
        },
        "[remote]",
    )
    _reject_unknown_keys(pipeline_raw, {"id", "base_branch", "push"}, "[pipeline]")

    host = validate_host(_require_string(remote_raw.get("host"), "remote.host"))
    repo = validate_remote_path(
        _require_string(remote_raw.get("repo"), "remote.repo"), "remote.repo"
    )
    worktree_root = validate_remote_path(
        _require_string(remote_raw.get("worktree_root"), "remote.worktree_root"),
        "remote.worktree_root",
    )
    identity_file = _resolve_identity_file(remote_raw.get("identity_file"))
    allow_dirty_primary = _require_bool(
        remote_raw.get("allow_dirty_primary", False),
        "remote.allow_dirty_primary",
    )
    if (
        repo == worktree_root
        or _is_path_prefix(repo, worktree_root)
        or _is_path_prefix(worktree_root, repo)
    ):
        raise OrchestratorError(
            "CONFIG_ERROR",
            "remote.repo and remote.worktree_root must be separate, non-nested paths",
        )

    connect_timeout = _require_positive_int(
        remote_raw.get("connect_timeout_seconds", 15),
        "remote.connect_timeout_seconds",
    )
    stage_timeout = _require_positive_int(
        remote_raw.get("stage_timeout_seconds", 900),
        "remote.stage_timeout_seconds",
    )
    if connect_timeout > 120 or stage_timeout > 86400:
        raise OrchestratorError(
            "CONFIG_ERROR", "configured timeout exceeds the bounded v1 limit"
        )

    pipeline_id = validate_safe_id(
        _require_string(pipeline_raw.get("id"), "pipeline.id"),
        "pipeline id",
    )
    base_branch = validate_branch_ref(
        _require_string(pipeline_raw.get("base_branch"), "pipeline.base_branch"),
        "base branch",
    )
    push = _require_bool(pipeline_raw.get("push", False), "pipeline.push")
    if push:
        raise OrchestratorError(
            "CONFIG_ERROR", "pipeline.push=true is unsupported in v1"
        )

    stages: list[StageConfig] = []
    seen: set[str] = set()
    for index, raw_stage in enumerate(stages_raw):
        stage_table = _require_mapping(raw_stage, f"stages[{index}]")
        _reject_unknown_keys(
            stage_table, {"id", "prompt", "depends_on"}, f"stages[{index}]"
        )
        stage_id = validate_safe_id(
            _require_string(stage_table.get("id"), f"stages[{index}].id")
        )
        if stage_id in seen:
            raise OrchestratorError("CONFIG_ERROR", f"duplicate stage id: {stage_id}")
        seen.add(stage_id)
        prompt_declared = _require_string(
            stage_table.get("prompt"),
            f"stages[{index}].prompt",
        )
        prompt_path = _resolve_prompt(source_path, prompt_declared)
        dependencies = stage_table.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in dependencies
        ):
            raise OrchestratorError(
                "CONFIG_ERROR",
                f"stages[{index}].depends_on must be an array of stage ids",
            )
        depends_on = tuple(dependencies)
        expected = () if index == 0 else (stages[index - 1].id,)
        if depends_on != expected:
            raise OrchestratorError(
                "CONFIG_ERROR",
                f"stage {stage_id!r} must depend exactly on {list(expected)!r}; v1 supports only a linear pipeline",
            )
        stages.append(
            StageConfig(
                id=stage_id,
                prompt_declared=prompt_declared,
                prompt_path=prompt_path,
                prompt_sha256=sha256_bytes(prompt_path.read_bytes()),
                depends_on=depends_on,
            )
        )

    return PipelineConfig(
        id=pipeline_id,
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        remote=RemoteConfig(
            host=host,
            repo=repo,
            worktree_root=worktree_root,
            identity_file=identity_file,
            allow_dirty_primary=allow_dirty_primary,
            connect_timeout_seconds=connect_timeout,
            stage_timeout_seconds=stage_timeout,
        ),
        base_branch=base_branch,
        push=push,
        stages=tuple(stages),
    )


def branch_for(stage: StageConfig) -> str:
    branch = f"codex/{stage.id}"
    return validate_branch_ref(branch)


def worktree_for(config: PipelineConfig, stage: StageConfig) -> PurePosixPath:
    return config.remote.worktree_root / stage.id


def control_dir_for(config: PipelineConfig, stage: StageConfig) -> PurePosixPath:
    return config.remote.worktree_root / ".orchestrator-state" / config.id / stage.id


def base_ref_for(config: PipelineConfig, index: int) -> str:
    if index == 0:
        return config.base_branch
    return branch_for(config.stages[index - 1])


def build_dry_run_plan(config: PipelineConfig) -> dict[str, Any]:
    return {
        "pipeline_id": config.id,
        "remote_host": config.remote.host,
        "remote_repo": str(config.remote.repo),
        "worktree_root": str(config.remote.worktree_root),
        "identity_file_configured": config.remote.identity_file is not None,
        "primary_checkout_policy": (
            "preserve_exact_snapshot"
            if config.remote.allow_dirty_primary
            else "require_clean"
        ),
        "base_branch": config.base_branch,
        "push": False,
        "stages": [
            {
                "id": stage.id,
                "base_ref": base_ref_for(config, index),
                "branch": branch_for(stage),
                "worktree": str(worktree_for(config, stage)),
                "prompt": str(stage.prompt_path),
                "prompt_sha256": stage.prompt_sha256,
            }
            for index, stage in enumerate(config.stages)
        ],
        "side_effects": False,
        "automatic_merge": False,
        "automatic_push": False,
    }


def _validate_relative_path(value: str, label: str) -> str:
    if not value or "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise OrchestratorError("INVALID_RESULT", f"unsafe {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise OrchestratorError("INVALID_RESULT", f"unsafe {label}: {value!r}")
    return value


def validate_stage_result(
    payload: Any,
    *,
    stage: StageConfig,
    expected_branch: str,
    expected_base_sha: str,
    terminal: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OrchestratorError("INVALID_RESULT", "stage result must be a JSON object")
    required = {
        "stage",
        "status",
        "branch",
        "base_commit",
        "head_commit",
        "commit_created",
        "summary",
        "tests",
        "artifacts",
        "changed_files",
        "blocking_issue",
        "recommended_next_action",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise OrchestratorError(
            "INVALID_RESULT",
            f"stage result keys mismatch; missing={missing}, extra={extra}",
        )

    for key in (
        "stage",
        "status",
        "branch",
        "base_commit",
        "head_commit",
        "summary",
        "recommended_next_action",
    ):
        if not isinstance(payload[key], str):
            raise OrchestratorError(
                "INVALID_RESULT", f"stage result {key} must be a string"
            )
    if type(payload["commit_created"]) is not bool:
        raise OrchestratorError("INVALID_RESULT", "commit_created must be a boolean")
    if payload["stage"] != stage.id:
        raise OrchestratorError(
            "INVALID_RESULT", "stage result identifies an unexpected stage"
        )
    if payload["branch"] != expected_branch:
        raise OrchestratorError(
            "INVALID_RESULT", "stage result identifies an unexpected branch"
        )
    if payload["base_commit"] != expected_base_sha:
        raise OrchestratorError(
            "INVALID_RESULT", "stage result base_commit differs from the actual base"
        )
    if not SAFE_SHA_RE.fullmatch(payload["base_commit"]) or not SAFE_SHA_RE.fullmatch(
        payload["head_commit"]
    ):
        raise OrchestratorError(
            "INVALID_RESULT", "base_commit/head_commit must be lowercase Git object ids"
        )
    if payload["status"] not in RESULT_STATUSES:
        raise OrchestratorError(
            "INVALID_RESULT", f"unsupported stage status: {payload['status']!r}"
        )
    if payload["recommended_next_action"] not in NEXT_ACTIONS:
        raise OrchestratorError("INVALID_RESULT", "unsupported recommended_next_action")
    if not payload["summary"].strip():
        raise OrchestratorError("INVALID_RESULT", "summary must not be empty")
    if payload["blocking_issue"] is not None and not isinstance(
        payload["blocking_issue"], str
    ):
        raise OrchestratorError(
            "INVALID_RESULT", "blocking_issue must be string or null"
        )

    tests = payload["tests"]
    if not isinstance(tests, list):
        raise OrchestratorError("INVALID_RESULT", "tests must be an array")
    for index, test in enumerate(tests):
        if not isinstance(test, dict) or set(test) != {"command", "passed"}:
            raise OrchestratorError(
                "INVALID_RESULT", f"tests[{index}] has invalid shape"
            )
        if not isinstance(test["command"], str) or not test["command"].strip():
            raise OrchestratorError(
                "INVALID_RESULT", f"tests[{index}].command is invalid"
            )
        if type(test["passed"]) is not bool:
            raise OrchestratorError(
                "INVALID_RESULT", f"tests[{index}].passed must be boolean"
            )

    for key in ("artifacts", "changed_files"):
        values = payload[key]
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise OrchestratorError(
                "INVALID_RESULT", f"{key} must be an array of strings"
            )
        if len(values) != len(set(values)):
            raise OrchestratorError("INVALID_RESULT", f"{key} contains duplicates")
        for value in values:
            _validate_relative_path(value, key)

    if payload["status"] == "pass":
        if not payload["commit_created"]:
            raise OrchestratorError(
                "INVALID_RESULT", "PASS requires commit_created=true"
            )
        if payload["head_commit"] == payload["base_commit"]:
            raise OrchestratorError(
                "INVALID_RESULT", "PASS requires a commit after base_commit"
            )
        if payload["blocking_issue"] is not None:
            raise OrchestratorError(
                "INVALID_RESULT", "PASS requires blocking_issue=null"
            )
        allowed_pass_actions = {"continue", "stop"} if terminal else {"continue"}
        if payload["recommended_next_action"] not in allowed_pass_actions:
            raise OrchestratorError(
                "INVALID_RESULT",
                "PASS requires recommended_next_action=continue"
                + (" (or stop for the terminal Stage)" if terminal else ""),
            )
        if any(not item["passed"] for item in tests):
            raise OrchestratorError(
                "INVALID_RESULT", "PASS cannot report a failed test"
            )
    elif payload["recommended_next_action"] == "continue":
        raise OrchestratorError(
            "INVALID_RESULT", "non-PASS status cannot recommend continue"
        )

    return payload


def should_continue(result: dict[str, Any]) -> bool:
    return result.get("status") == "pass"


def redact_text(value: str) -> str:
    value = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+",
        r"\1<redacted>",
        value,
    )
    value = re.sub(
        r"(?i)((?:OPENAI|CODEX)_API_KEY\s*[=:]\s*)[^\s]+",
        r"\1<redacted>",
        value,
    )
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "<redacted-api-key>", value)


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(redact_value(payload), ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def _clip_log(value: str, limit: int = 6000) -> str:
    redacted = redact_text(value)
    if len(redacted) <= limit:
        return redacted
    return redacted[:limit] + f"\n... <truncated {len(redacted) - limit} chars>"


class SSHClient:
    def __init__(self, remote: RemoteConfig, logger: Callable[[str], None]) -> None:
        self.remote = remote
        self.logger = logger

    def _arguments(self, remote_command: str) -> list[str]:
        arguments = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.remote.connect_timeout_seconds}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]
        if self.remote.identity_file is not None:
            arguments.extend(
                [
                    "-o",
                    "IdentitiesOnly=yes",
                    "-i",
                    str(self.remote.identity_file),
                ]
            )
        arguments.extend([self.remote.host, remote_command])
        return arguments

    def run(
        self,
        remote_command: str,
        *,
        purpose: str,
        timeout: int = 120,
        input_text: str | None = None,
        check: bool = True,
        error_code: str = "REMOTE_GIT_ERROR",
    ) -> CommandResult:
        self.logger(f"ssh start: {purpose}")
        try:
            completed = subprocess.run(
                self._arguments(remote_command),
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise OrchestratorError(
                "SSH_ERROR", "ssh executable was not found"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise OrchestratorError(
                "SSH_ERROR", f"SSH timed out during {purpose}"
            ) from exc

        result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
        self.logger(
            f"ssh end: {purpose}; rc={result.returncode}; "
            f"stdout={_clip_log(result.stdout)!r}; stderr={_clip_log(result.stderr)!r}"
        )
        if check and result.returncode != 0:
            code = "SSH_ERROR" if result.returncode == 255 else error_code
            raise OrchestratorError(
                code,
                f"remote command failed during {purpose} (rc={result.returncode})",
                details={"stderr": _clip_log(result.stderr, 2000)},
            )
        return result

    def run_codex(
        self,
        remote_command: str,
        *,
        prompt: str,
        timeout: int,
        events_path: Path,
        stderr_path: Path,
    ) -> int:
        self.logger("ssh start: remote codex exec")
        events_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                self._arguments(remote_command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise OrchestratorError(
                "SSH_ERROR", "ssh executable was not found"
            ) from exc

        pump_errors: list[Exception] = []

        def pump(stream: Any, destination: Path) -> None:
            try:
                with destination.open("w", encoding="utf-8", newline="\n") as handle:
                    for line in iter(stream.readline, ""):
                        handle.write(redact_text(line))
                        handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, UnicodeError, ValueError) as exc:  # pragma: no cover
                pump_errors.append(exc)
            finally:
                stream.close()

        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=pump,
            args=(process.stdout, events_path),
            name="remote-codex-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump,
            args=(process.stderr, stderr_path),
            name="remote-codex-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        assert process.stdin is not None
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except BrokenPipeError:
            pass

        timeout_error: subprocess.TimeoutExpired | None = None
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timeout_error = exc
            process.kill()
            process.wait(timeout=30)

        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise OrchestratorError(
                "CODEX_ERROR", "could not finish capturing Remote Codex logs"
            )
        if pump_errors:
            raise OrchestratorError(
                "CODEX_ERROR",
                f"could not write Remote Codex logs: {pump_errors[0]}",
            )
        stderr = (
            stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
        )
        if timeout_error is not None:
            raise OrchestratorError(
                "CODEX_ERROR", "remote Codex exceeded the stage timeout"
            ) from timeout_error
        self.logger(
            f"ssh end: remote codex exec; rc={process.returncode}; "
            f"events={events_path.name}; stderr={_clip_log(stderr)!r}"
        )
        if process.returncode == 255:
            raise OrchestratorError(
                "SSH_ERROR", "SSH connection failed while Remote Codex was running"
            )
        if process.returncode != 0:
            raise OrchestratorError(
                "CODEX_ERROR",
                f"remote codex exec failed (rc={process.returncode})",
                details={"stderr": _clip_log(stderr, 2000)},
            )
        return process.returncode


def _q(value: str | PurePosixPath) -> str:
    return shlex.quote(str(value))


def _git(repo: PurePosixPath, *args: str) -> str:
    return " ".join(["git", "-C", _q(repo), *(_q(arg) for arg in args)])


def _primary_content_fingerprint(
    ssh: SSHClient, repo: PurePosixPath, *, purpose: str
) -> str:
    script = r"""
import hashlib
import os
import pathlib
import subprocess
import sys

repo = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()

def add(value):
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)

for command in (
    ["git", "-C", os.fspath(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
    ["git", "-C", os.fspath(repo), "diff", "--binary", "HEAD"],
):
    add(subprocess.check_output(command))

untracked = subprocess.check_output(
    ["git", "-C", os.fspath(repo), "ls-files", "--others", "--exclude-standard", "-z"]
)
for encoded in sorted(item for item in untracked.split(b"\0") if item):
    add(encoded)
    path = repo / os.fsdecode(encoded)
    if path.is_symlink():
        add(b"L" + os.fsencode(os.readlink(path)))
    elif path.is_file():
        add(b"F")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    else:
        add(b"O")

print(digest.hexdigest())
""".strip()
    result = ssh.run(
        f"python3 -c {_q(script)} {_q(repo)}",
        purpose=purpose,
        error_code="REMOTE_GIT_ERROR",
        timeout=600,
    )
    fingerprint = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise OrchestratorError(
            "REMOTE_GIT_ERROR", "remote primary fingerprint was not a SHA256 value"
        )
    return fingerprint


class StateStore:
    def __init__(self, config: PipelineConfig, root: Path = DEFAULT_STATE_ROOT) -> None:
        self.config = config
        self.pipeline_dir = root.resolve() / config.id
        self.meta_path = self.pipeline_dir / "pipeline-state.json"
        self.pipeline_log = self.pipeline_dir / "orchestrator.log"

    def stage_dir(self, stage: StageConfig) -> Path:
        return self.pipeline_dir / stage.id

    def stage_state_path(self, stage: StageConfig) -> Path:
        return self.stage_dir(stage) / "stage-state.json"

    def read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestratorError(
                "USER_REVIEW_REQUIRED", f"cannot read local state: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise OrchestratorError(
                "USER_REVIEW_REQUIRED", f"local state is not an object: {path}"
            )
        return value

    def append_log(self, path: Path, message: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{utc_now()} {redact_text(message)}\n")

    def log_pipeline(self, message: str) -> None:
        self.append_log(self.pipeline_log, message)

    def log_stage(self, stage: StageConfig, message: str) -> None:
        self.append_log(self.stage_dir(stage) / "orchestrator.log", message)

    def initialize(self) -> dict[str, Any]:
        existing = self.read_json(self.meta_path)
        prompt_hashes = {stage.id: stage.prompt_sha256 for stage in self.config.stages}
        if existing is not None:
            if existing.get("pipeline_sha256") != self.config.source_sha256:
                raise OrchestratorError(
                    "USER_REVIEW_REQUIRED",
                    "pipeline TOML changed after state was created; review or use a new pipeline id",
                )
            if existing.get("prompt_sha256") != prompt_hashes:
                raise OrchestratorError(
                    "USER_REVIEW_REQUIRED",
                    "a stage prompt changed after state was created; review or use a new pipeline id",
                )
            return existing
        payload = {
            "schema_version": 1,
            "pipeline_id": self.config.id,
            "pipeline_file": str(self.config.source_path),
            "pipeline_sha256": self.config.source_sha256,
            "prompt_sha256": prompt_hashes,
            "status": "starting",
            "current_stage": None,
            "started_at": utc_now(),
            "ended_at": None,
            "error": None,
        }
        atomic_write_json(self.meta_path, payload)
        return payload

    def update_pipeline(self, **updates: Any) -> None:
        current = self.read_json(self.meta_path) or {}
        current.update(updates)
        atomic_write_json(self.meta_path, current)

    def write_stage_state(self, stage: StageConfig, payload: dict[str, Any]) -> None:
        atomic_write_json(self.stage_state_path(stage), payload)


def _remote_preflight(config: PipelineConfig, ssh: SSHClient) -> dict[str, Any]:
    tool_check = "command -v git >/dev/null 2>&1 && command -v codex >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1"
    ssh.run(tool_check, purpose="check git/codex/timeout", error_code="CODEX_ERROR")

    top = ssh.run(
        _git(config.remote.repo, "rev-parse", "--show-toplevel"),
        purpose="verify remote repository",
    ).stdout.strip()
    if top.rstrip("/") != str(config.remote.repo).rstrip("/"):
        raise OrchestratorError(
            "REMOTE_GIT_ERROR",
            f"configured remote repo resolves to unexpected top-level: {top!r}",
        )
    status = ssh.run(
        _git(config.remote.repo, "status", "--porcelain=v1", "--untracked-files=all"),
        purpose="check remote main checkout cleanliness",
    ).stdout
    if status.strip() and not config.remote.allow_dirty_primary:
        raise OrchestratorError(
            "DIRTY_WORKTREE",
            "remote primary checkout is not clean; refusing to create stage worktrees",
            details={"status": _clip_log(status, 3000)},
        )
    if status.strip():
        ssh.logger(
            "remote primary checkout has a reviewed dirty baseline; "
            "the exact status snapshot must remain unchanged"
        )

    branch = ssh.run(
        _git(config.remote.repo, "branch", "--show-current"),
        purpose="read remote primary checkout branch",
    ).stdout.strip()
    head = ssh.run(
        _git(config.remote.repo, "rev-parse", "HEAD"),
        purpose="read remote primary checkout HEAD",
    ).stdout.strip()
    base_sha = ssh.run(
        _git(
            config.remote.repo,
            "rev-parse",
            "--verify",
            f"refs/heads/{config.base_branch}^{{commit}}",
        ),
        purpose="verify configured base branch",
    ).stdout.strip()
    common_dir = ssh.run(
        _git(
            config.remote.repo,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ),
        purpose="read remote common git directory",
    ).stdout.strip()
    codex_version = ssh.run(
        "codex --version", purpose="read remote Codex version", error_code="CODEX_ERROR"
    ).stdout.strip()
    codex_help = ssh.run(
        "codex exec --help",
        purpose="audit remote codex exec flags",
        error_code="CODEX_ERROR",
    ).stdout
    required_flags = (
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-schema",
        "--output-last-message",
    )
    missing = [flag for flag in required_flags if flag not in codex_help]
    if missing:
        raise OrchestratorError(
            "CODEX_ERROR",
            f"remote codex exec lacks required flags: {', '.join(missing)}",
        )
    primary_content_fingerprint = _primary_content_fingerprint(
        ssh,
        config.remote.repo,
        purpose="fingerprint remote primary checkout baseline",
    )
    return {
        "remote_primary_branch": branch,
        "remote_primary_head": head,
        "remote_primary_status": status,
        "remote_primary_content_sha256": primary_content_fingerprint,
        "remote_common_git_dir": common_dir,
        "configured_base_sha": base_sha,
        "codex_version": codex_version,
        "codex_auto_review": "--approve-for-me" in codex_help,
        "checked_at": utc_now(),
    }


def _verify_primary_unchanged(
    config: PipelineConfig, ssh: SSHClient, snapshot: dict[str, Any]
) -> None:
    branch = ssh.run(
        _git(config.remote.repo, "branch", "--show-current"),
        purpose="recheck remote primary branch",
    ).stdout.strip()
    head = ssh.run(
        _git(config.remote.repo, "rev-parse", "HEAD"),
        purpose="recheck remote primary HEAD",
    ).stdout.strip()
    status = ssh.run(
        _git(config.remote.repo, "status", "--porcelain=v1", "--untracked-files=all"),
        purpose="recheck remote primary status",
    ).stdout
    content_fingerprint = _primary_content_fingerprint(
        ssh,
        config.remote.repo,
        purpose="recheck remote primary checkout fingerprint",
    )
    if (
        branch != snapshot["remote_primary_branch"]
        or head != snapshot["remote_primary_head"]
        or status != snapshot["remote_primary_status"]
        or content_fingerprint != snapshot["remote_primary_content_sha256"]
    ):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            "remote primary checkout changed during orchestration",
            details={"branch": branch, "head": head, "status": _clip_log(status)},
        )


def _probe_exists(ssh: SSHClient, path: PurePosixPath, purpose: str) -> bool:
    result = ssh.run(f"test -e {_q(path)}", purpose=purpose, check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    code = "SSH_ERROR" if result.returncode == 255 else "REMOTE_GIT_ERROR"
    raise OrchestratorError(code, f"could not probe remote path during {purpose}")


def _branch_exists(ssh: SSHClient, repo: PurePosixPath, branch: str) -> bool:
    result = ssh.run(
        _git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"),
        purpose=f"probe branch {branch}",
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    code = "SSH_ERROR" if result.returncode == 255 else "REMOTE_GIT_ERROR"
    raise OrchestratorError(code, f"could not probe branch {branch}")


def _verify_worktree_identity(
    config: PipelineConfig,
    ssh: SSHClient,
    *,
    worktree: PurePosixPath,
    branch: str,
    expected_common_dir: str,
) -> dict[str, str]:
    top = ssh.run(
        _git(worktree, "rev-parse", "--show-toplevel"),
        purpose=f"verify worktree top-level for {branch}",
    ).stdout.strip()
    actual_branch = ssh.run(
        _git(worktree, "branch", "--show-current"),
        purpose=f"verify worktree branch for {branch}",
    ).stdout.strip()
    common_dir = ssh.run(
        _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        purpose=f"verify worktree repository for {branch}",
    ).stdout.strip()
    if top.rstrip("/") != str(worktree).rstrip("/"):
        raise OrchestratorError(
            "UNEXPECTED_BRANCH", f"unexpected worktree top-level: {top!r}"
        )
    if actual_branch != branch:
        raise OrchestratorError(
            "UNEXPECTED_BRANCH",
            f"worktree uses {actual_branch!r}, expected {branch!r}",
        )
    if common_dir.rstrip("/") != expected_common_dir.rstrip("/"):
        raise OrchestratorError(
            "UNEXPECTED_BRANCH",
            "worktree belongs to a different Git repository",
        )
    head = ssh.run(
        _git(worktree, "rev-parse", "HEAD"),
        purpose=f"read worktree HEAD for {branch}",
    ).stdout.strip()
    return {"top": top, "branch": actual_branch, "common_dir": common_dir, "head": head}


def _create_stage_worktree(
    config: PipelineConfig,
    stage: StageConfig,
    ssh: SSHClient,
    *,
    base_ref: str,
    expected_common_dir: str,
) -> tuple[PurePosixPath, str, str]:
    branch = branch_for(stage)
    worktree = worktree_for(config, stage)
    check_ref = ssh.run(
        f"git check-ref-format --branch {_q(branch)}",
        purpose=f"validate remote branch name {branch}",
        check=False,
    )
    if check_ref.returncode != 0:
        code = "SSH_ERROR" if check_ref.returncode == 255 else "REMOTE_GIT_ERROR"
        raise OrchestratorError(code, f"remote Git rejected branch name: {branch}")
    if _branch_exists(ssh, config.remote.repo, branch):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"branch already exists without a completed local stage state: {branch}",
        )
    if _probe_exists(ssh, worktree, f"probe worktree path for {stage.id}"):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"worktree path already exists: {worktree}",
        )

    base_sha = ssh.run(
        _git(
            config.remote.repo,
            "rev-parse",
            "--verify",
            f"refs/heads/{base_ref}^{{commit}}",
        ),
        purpose=f"resolve stage base {base_ref}",
    ).stdout.strip()
    ssh.run(
        f"mkdir -p {_q(config.remote.worktree_root)}",
        purpose="create configured worktree root",
    )
    ssh.run(
        _git(
            config.remote.repo, "worktree", "add", "-b", branch, str(worktree), base_ref
        ),
        purpose=f"create worktree for {stage.id}",
    )
    identity = _verify_worktree_identity(
        config,
        ssh,
        worktree=worktree,
        branch=branch,
        expected_common_dir=expected_common_dir,
    )
    if identity["head"] != base_sha:
        raise OrchestratorError(
            "REMOTE_GIT_ERROR", "new worktree HEAD differs from resolved base"
        )
    return worktree, branch, base_sha


def _write_remote_file(
    ssh: SSHClient, path: PurePosixPath, content: str, purpose: str
) -> None:
    command = f"umask 077 && mkdir -p {_q(path.parent)} && cat > {_q(path)}"
    ssh.run(command, purpose=purpose, input_text=content, error_code="CODEX_ERROR")


def _build_stage_prompt(
    stage: StageConfig,
    *,
    worktree: PurePosixPath,
    branch: str,
    base_sha: str,
) -> str:
    user_prompt = stage.prompt_path.read_text(encoding="utf-8-sig")
    return f"""你正在执行远程 Stage: {stage.id}

当前工作目录：
{worktree}

当前 branch：
{branch}

Base commit：
{base_sha}

固定契约：
1. 只操作当前 worktree；开始前完整读取仓库根目录 AGENTS.md。
2. 不修改其他 worktree 或远程主 checkout。
3. 不 merge，不 rebase main，不 push，不 force push。
4. 不删除共享模型、checkpoint 或其他共享目录。
5. 只完成本 Stage；不得自动开始下一 Stage。
6. 完成后运行 Stage 要求的测试。
7. status=pass 前必须在当前 branch 创建至少一个本地 Git commit。
8. status=pass 前必须确保 git status --porcelain 没有未提交内容。
9. 最终响应必须只包含符合已提供 JSON Schema 的 JSON 对象，不要 Markdown 代码围栏。
10. stage 必须是 {stage.id!r}，branch 必须是 {branch!r}，base_commit 必须是 {base_sha!r}。
11. head_commit 必须取自最终 git rev-parse HEAD；changed_files 必须与 base..HEAD 的提交差异一致。
12. FAIL/BLOCKED/BORDERLINE/NEEDS_USER_REVIEW 时不得伪造 PASS，也不得建议 continue。
13. 不读取、打印或写入密码、私钥、token、API key 或完整环境变量。

本 Stage 的实际任务：

{user_prompt.rstrip()}
"""


def _remote_codex_command(
    config: PipelineConfig,
    *,
    worktree: PurePosixPath,
    schema_path: PurePosixPath,
    result_path: PurePosixPath,
    auto_review: bool,
) -> str:
    argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        str(config.remote.stage_timeout_seconds),
        "codex",
        "exec",
    ]
    if auto_review:
        argv.append("--approve-for-me")
    else:
        argv.extend(["--sandbox", "workspace-write"])
    argv.extend(
        [
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
            "-",
        ]
    )
    return f"cd {_q(worktree)} && exec " + " ".join(_q(item) for item in argv)


def _read_remote_result(ssh: SSHClient, path: PurePosixPath) -> Any:
    result = ssh.run(
        f"cat {_q(path)}",
        purpose="fetch stage-result.json",
        error_code="INVALID_RESULT",
    )
    if len(result.stdout.encode("utf-8")) > 1_000_000:
        raise OrchestratorError("INVALID_RESULT", "stage result exceeds 1 MB")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OrchestratorError(
            "INVALID_RESULT", f"stage result is not valid JSON: {exc}"
        ) from exc


def _collect_review(
    config: PipelineConfig,
    stage: StageConfig,
    ssh: SSHClient,
    *,
    worktree: PurePosixPath,
    branch: str,
    base_sha: str,
    expected_common_dir: str,
) -> dict[str, Any]:
    identity = _verify_worktree_identity(
        config,
        ssh,
        worktree=worktree,
        branch=branch,
        expected_common_dir=expected_common_dir,
    )
    status = ssh.run(
        _git(worktree, "status", "--porcelain=v1", "--untracked-files=all"),
        purpose=f"collect status for {stage.id}",
    ).stdout
    ancestry = ssh.run(
        _git(
            worktree,
            "merge-base",
            "--is-ancestor",
            base_sha,
            identity["head"],
        ),
        purpose=f"verify commit ancestry for {stage.id}",
        check=False,
    )
    if ancestry.returncode != 0:
        code = "SSH_ERROR" if ancestry.returncode == 255 else "REMOTE_GIT_ERROR"
        raise OrchestratorError(
            code,
            f"Stage HEAD is not descended from its recorded base: {stage.id}",
        )
    diff_stat = ssh.run(
        _git(worktree, "diff", "--stat", f"{base_sha}..{identity['head']}"),
        purpose=f"collect diff stat for {stage.id}",
    ).stdout
    changed_output = ssh.run(
        " ".join(
            [
                "git",
                "-c",
                "core.quotepath=false",
                "-C",
                _q(worktree),
                "diff",
                "--name-only",
                _q(f"{base_sha}..{identity['head']}"),
            ]
        ),
        purpose=f"collect changed files for {stage.id}",
    ).stdout
    log_line = ssh.run(
        _git(worktree, "log", "-1", "--oneline"),
        purpose=f"collect latest commit for {stage.id}",
    ).stdout.strip()
    changed_files = [line for line in changed_output.splitlines() if line]
    return {
        "stage": stage.id,
        "branch": identity["branch"],
        "base_commit": base_sha,
        "head_commit": identity["head"],
        "git_status_porcelain": status,
        "diff_stat": diff_stat,
        "changed_files": changed_files,
        "log_1_oneline": log_line,
        "checked_at": utc_now(),
    }


def _verify_result_against_git(
    result: dict[str, Any],
    review: dict[str, Any],
    *,
    config: PipelineConfig,
    ssh: SSHClient,
    worktree: PurePosixPath,
) -> None:
    if result["head_commit"] != review["head_commit"]:
        raise OrchestratorError(
            "INVALID_RESULT", "reported head_commit differs from actual Git HEAD"
        )
    if result["branch"] != review["branch"]:
        raise OrchestratorError(
            "UNEXPECTED_BRANCH", "reported branch differs from actual worktree branch"
        )
    if set(result["changed_files"]) != set(review["changed_files"]):
        raise OrchestratorError(
            "INVALID_RESULT",
            "reported changed_files differs from committed base..HEAD diff",
            details={
                "reported": result["changed_files"],
                "actual": review["changed_files"],
            },
        )
    for artifact in result["artifacts"]:
        artifact_path = worktree / PurePosixPath(artifact)
        if not _probe_exists(
            ssh, artifact_path, f"verify declared artifact {artifact}"
        ):
            raise OrchestratorError(
                "INVALID_RESULT", f"declared artifact does not exist: {artifact}"
            )
    if result["status"] == "pass":
        if review["head_commit"] == review["base_commit"]:
            raise OrchestratorError(
                "INVALID_RESULT", "PASS stage did not create a commit"
            )
        if review["git_status_porcelain"].strip():
            raise OrchestratorError(
                "DIRTY_WORKTREE",
                "PASS stage left the worktree dirty",
                details={"status": _clip_log(review["git_status_porcelain"], 3000)},
            )


def _verify_completed_stage(
    config: PipelineConfig,
    stage: StageConfig,
    ssh: SSHClient,
    state: dict[str, Any],
    *,
    expected_common_dir: str,
) -> None:
    branch = branch_for(stage)
    worktree = worktree_for(config, stage)
    if not _branch_exists(ssh, config.remote.repo, branch) or not _probe_exists(
        ssh,
        worktree,
        f"verify resumed worktree {stage.id}",
    ):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"completed local state for {stage.id} no longer matches remote branch/worktree existence",
        )
    identity = _verify_worktree_identity(
        config,
        ssh,
        worktree=worktree,
        branch=branch,
        expected_common_dir=expected_common_dir,
    )
    if identity["head"] != state.get("head_commit"):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"remote HEAD for completed stage {stage.id} changed after local recording",
        )
    status = ssh.run(
        _git(worktree, "status", "--porcelain=v1", "--untracked-files=all"),
        purpose=f"verify resumed worktree cleanliness for {stage.id}",
    ).stdout
    if status.strip():
        raise OrchestratorError(
            "DIRTY_WORKTREE", f"completed stage worktree became dirty: {stage.id}"
        )


def _stage_error_code(status: str) -> str:
    if status == "blocked":
        return "STAGE_BLOCKED"
    if status in {"borderline", "needs_user_review"}:
        return "USER_REVIEW_REQUIRED"
    return "STAGE_FAILED"


def _verify_retry_preflight_unchanged(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    protected = (
        "remote_primary_branch",
        "remote_primary_head",
        "remote_primary_status",
        "remote_primary_content_sha256",
        "remote_common_git_dir",
        "configured_base_sha",
    )
    changed = [key for key in protected if previous.get(key) != current.get(key)]
    if changed:
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            "remote primary checkout changed since the failed Stage; retry refused",
            details={"changed_fields": changed},
        )


def _prepare_retry_worktree(
    config: PipelineConfig,
    stage: StageConfig,
    ssh: SSHClient,
    existing: dict[str, Any],
    *,
    expected_common_dir: str,
) -> tuple[PurePosixPath, str, str, str]:
    if existing.get("status") != "error":
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"retry only supports a Stage stopped with status=error: {stage.id}",
        )
    retry_count = existing.get("retry_count", 0)
    if type(retry_count) is not int or retry_count != 0:
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"the single automatic retry has already been used for {stage.id}",
        )
    base_sha = existing.get("base_commit")
    if not isinstance(base_sha, str) or not SAFE_SHA_RE.fullmatch(base_sha):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"failed Stage has no trustworthy base commit: {stage.id}",
        )
    branch = branch_for(stage)
    worktree = worktree_for(config, stage)
    if not _branch_exists(ssh, config.remote.repo, branch) or not _probe_exists(
        ssh, worktree, f"verify retry worktree {stage.id}"
    ):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"retry branch/worktree is missing for {stage.id}",
        )
    identity = _verify_worktree_identity(
        config,
        ssh,
        worktree=worktree,
        branch=branch,
        expected_common_dir=expected_common_dir,
    )
    status = ssh.run(
        _git(worktree, "status", "--porcelain=v1", "--untracked-files=all"),
        purpose=f"verify retry worktree cleanliness for {stage.id}",
    ).stdout
    if status.strip():
        raise OrchestratorError(
            "DIRTY_WORKTREE",
            "v1 retry refuses a failed worktree containing uncommitted changes",
            details={"status": _clip_log(status, 3000)},
        )
    result_remote = control_dir_for(config, stage) / "stage-result.json"
    result_exists = _probe_exists(
        ssh, result_remote, f"probe recoverable result for {stage.id}"
    )
    if identity["head"] == base_sha and not result_exists:
        return worktree, branch, base_sha, "rerun"
    if identity["head"] != base_sha and result_exists:
        return worktree, branch, base_sha, "recover_result"
    raise OrchestratorError(
        "USER_REVIEW_REQUIRED",
        "failed Stage is neither a clean pre-execution retry nor a committed-result recovery",
        details={
            "head_equals_base": identity["head"] == base_sha,
            "result_exists": result_exists,
        },
    )


def _finalize_stage_result(
    config: PipelineConfig,
    stage: StageConfig,
    store: StateStore,
    ssh: SSHClient,
    snapshot: dict[str, Any],
    *,
    worktree: PurePosixPath,
    branch: str,
    base_sha: str,
    stage_state: dict[str, Any],
    raw_result: Any,
    terminal: bool,
) -> None:
    stage_dir = store.stage_dir(stage)
    result = validate_stage_result(
        raw_result,
        stage=stage,
        expected_branch=branch,
        expected_base_sha=base_sha,
        terminal=terminal,
    )
    atomic_write_json(stage_dir / "stage-result.json", result)
    review = _collect_review(
        config,
        stage,
        ssh,
        worktree=worktree,
        branch=branch,
        base_sha=base_sha,
        expected_common_dir=snapshot["remote_common_git_dir"],
    )
    atomic_write_json(stage_dir / "review.json", review)
    _verify_result_against_git(
        result,
        review,
        config=config,
        ssh=ssh,
        worktree=worktree,
    )
    _verify_primary_unchanged(config, ssh, snapshot)

    stage_state.update(
        status=result["status"],
        head_commit=review["head_commit"],
        ended_at=utc_now(),
        result=redact_value(result),
        review=review,
    )
    store.write_stage_state(stage, stage_state)
    print(
        f"[gate] {stage.id}: {result['status'].upper()} "
        f"head={review['head_commit'][:12]}",
        flush=True,
    )
    if not should_continue(result):
        raise OrchestratorError(
            _stage_error_code(result["status"]),
            f"stage {stage.id} stopped the pipeline with status={result['status']}",
            details={"blocking_issue": result["blocking_issue"]},
        )


def _execute_stage(
    config: PipelineConfig,
    stage: StageConfig,
    store: StateStore,
    ssh: SSHClient,
    snapshot: dict[str, Any],
    *,
    worktree: PurePosixPath,
    branch: str,
    base_sha: str,
    stage_state: dict[str, Any],
    terminal: bool,
) -> None:
    stage_dir = store.stage_dir(stage)
    control_dir = control_dir_for(config, stage)
    schema_remote = control_dir / "stage-result.schema.json"
    result_remote = control_dir / "stage-result.json"
    if _probe_exists(ssh, result_remote, f"probe prior result for {stage.id}"):
        raise OrchestratorError(
            "USER_REVIEW_REQUIRED",
            f"remote result path already exists for {stage.id}",
        )
    schema_text = RESULT_SCHEMA_PATH.read_text(encoding="utf-8")
    _write_remote_file(
        ssh,
        schema_remote,
        schema_text,
        f"upload result schema for {stage.id}",
    )
    effective_prompt = _build_stage_prompt(
        stage,
        worktree=worktree,
        branch=branch,
        base_sha=base_sha,
    )
    atomic_write_text(stage_dir / "effective-prompt.md", effective_prompt)
    codex_command = _remote_codex_command(
        config,
        worktree=worktree,
        schema_path=schema_remote,
        result_path=result_remote,
        auto_review=bool(snapshot["codex_auto_review"]),
    )
    print(
        f"[stage] {stage.id}: Remote Codex running "
        f"(sandbox=workspace-write, timeout={config.remote.stage_timeout_seconds}s)",
        flush=True,
    )
    ssh.run_codex(
        codex_command,
        prompt=effective_prompt,
        timeout=config.remote.stage_timeout_seconds + 90,
        events_path=stage_dir / "codex-events.jsonl",
        stderr_path=stage_dir / "codex-stderr.log",
    )
    raw_result = _read_remote_result(ssh, result_remote)
    _finalize_stage_result(
        config,
        stage,
        store,
        ssh,
        snapshot,
        worktree=worktree,
        branch=branch,
        base_sha=base_sha,
        stage_state=stage_state,
        raw_result=raw_result,
        terminal=terminal,
    )


def _recover_existing_stage_result(
    config: PipelineConfig,
    stage: StageConfig,
    store: StateStore,
    ssh: SSHClient,
    snapshot: dict[str, Any],
    *,
    worktree: PurePosixPath,
    branch: str,
    base_sha: str,
    stage_state: dict[str, Any],
    terminal: bool,
) -> None:
    result_remote = control_dir_for(config, stage) / "stage-result.json"
    raw_result = _read_remote_result(ssh, result_remote)
    _finalize_stage_result(
        config,
        stage,
        store,
        ssh,
        snapshot,
        worktree=worktree,
        branch=branch,
        base_sha=base_sha,
        stage_state=stage_state,
        raw_result=raw_result,
        terminal=terminal,
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    retry_stage_id: str | None = None,
    state_root: Path = DEFAULT_STATE_ROOT,
    ssh_factory: Callable[[RemoteConfig, Callable[[str], None]], SSHClient] = SSHClient,
) -> dict[str, Any]:
    if retry_stage_id is not None:
        validate_safe_id(retry_stage_id, "retry stage id")
        if retry_stage_id not in {stage.id for stage in config.stages}:
            raise OrchestratorError(
                "CONFIG_ERROR",
                f"retry stage is not part of pipeline {config.id}: {retry_stage_id}",
            )
        if dry_run:
            raise OrchestratorError(
                "CONFIG_ERROR", "retry cannot be combined with dry_run"
            )
    if dry_run:
        return build_dry_run_plan(config)
    if (
        config.remote.host.lower() in PLACEHOLDER_HOSTS
        or config.remote.host.upper().startswith(("REPLACE_", "YOUR_"))
    ):
        raise OrchestratorError(
            "CONFIG_ERROR", "replace the example remote.host before run"
        )
    if not RESULT_SCHEMA_PATH.is_file():
        raise OrchestratorError(
            "CONFIG_ERROR", f"missing result schema: {RESULT_SCHEMA_PATH}"
        )

    store = StateStore(config, state_root)
    initial_pipeline_state = store.initialize()
    previous_preflight = initial_pipeline_state.get("remote_preflight")
    store.update_pipeline(status="preflight", current_stage=None, error=None)
    preflight_ssh = ssh_factory(config.remote, store.log_pipeline)
    try:
        snapshot = _remote_preflight(config, preflight_ssh)
        if previous_preflight is not None:
            if not isinstance(previous_preflight, dict):
                raise OrchestratorError(
                    "USER_REVIEW_REQUIRED",
                    "saved remote preflight snapshot is not a JSON object",
                )
            _verify_retry_preflight_unchanged(previous_preflight, snapshot)
        elif retry_stage_id is not None:
            raise OrchestratorError(
                "USER_REVIEW_REQUIRED",
                "retry requires the saved preflight snapshot from the failed run",
            )
        atomic_write_json(store.pipeline_dir / "remote-preflight.json", snapshot)
        store.update_pipeline(status="running", remote_preflight=snapshot)

        retry_consumed = retry_stage_id is None
        for index, stage in enumerate(config.stages):
            existing = store.read_json(store.stage_state_path(stage))
            if existing is not None:
                if existing.get("status") == "pass":
                    if stage.id == retry_stage_id:
                        raise OrchestratorError(
                            "USER_REVIEW_REQUIRED",
                            f"stage {stage.id} is already PASS and cannot be retried",
                        )
                    stage_ssh = ssh_factory(
                        config.remote,
                        lambda message, s=stage: store.log_stage(s, message),
                    )
                    _verify_completed_stage(
                        config,
                        stage,
                        stage_ssh,
                        existing,
                        expected_common_dir=snapshot["remote_common_git_dir"],
                    )
                    print(
                        f"[resume] {stage.id}: already PASS; remote state verified",
                        flush=True,
                    )
                    continue

                if stage.id != retry_stage_id:
                    raise OrchestratorError(
                        "USER_REVIEW_REQUIRED",
                        f"stage {stage.id} has non-PASS local state; use an explicit eligible retry or review it",
                    )
                if retry_consumed:
                    raise OrchestratorError(
                        "USER_REVIEW_REQUIRED",
                        f"unexpected second retry target encountered: {stage.id}",
                    )

                store.update_pipeline(status="running", current_stage=stage.id)
                stage_ssh = ssh_factory(
                    config.remote, lambda message, s=stage: store.log_stage(s, message)
                )
                worktree, branch, base_sha, retry_mode = _prepare_retry_worktree(
                    config,
                    stage,
                    stage_ssh,
                    existing,
                    expected_common_dir=snapshot["remote_common_git_dir"],
                )
                stage_dir = store.stage_dir(stage)
                retry_number = 1
                prior_state_path = (
                    stage_dir / f"retry-{retry_number:03d}-prior-state.json"
                )
                if prior_state_path.exists():
                    raise OrchestratorError(
                        "USER_REVIEW_REQUIRED",
                        f"retry archive already exists for {stage.id}",
                    )
                atomic_write_json(prior_state_path, existing)
                previous_errors = []
                if existing.get("error") is not None:
                    previous_errors.append(existing["error"])
                stage_state = {
                    "schema_version": 1,
                    "pipeline_id": config.id,
                    "stage": stage.id,
                    "status": "running",
                    "remote_branch": branch,
                    "remote_worktree": str(worktree),
                    "base_ref": existing.get("base_ref", base_ref_for(config, index)),
                    "base_commit": base_sha,
                    "head_commit": None,
                    "started_at": utc_now(),
                    "ended_at": None,
                    "result": None,
                    "error": None,
                    "retry_count": retry_number,
                    "retry_mode": retry_mode,
                    "previous_errors": previous_errors,
                }
                store.write_stage_state(stage, stage_state)
                print(
                    f"[retry] {stage.id}: {retry_mode} in verified clean branch/worktree",
                    flush=True,
                )
                try:
                    if retry_mode == "recover_result":
                        _recover_existing_stage_result(
                            config,
                            stage,
                            store,
                            stage_ssh,
                            snapshot,
                            worktree=worktree,
                            branch=branch,
                            base_sha=base_sha,
                            stage_state=stage_state,
                            terminal=index == len(config.stages) - 1,
                        )
                    else:
                        _execute_stage(
                            config,
                            stage,
                            store,
                            stage_ssh,
                            snapshot,
                            worktree=worktree,
                            branch=branch,
                            base_sha=base_sha,
                            stage_state=stage_state,
                            terminal=index == len(config.stages) - 1,
                        )
                except OrchestratorError as exc:
                    latest = (
                        store.read_json(store.stage_state_path(stage)) or stage_state
                    )
                    if latest.get("status") not in RESULT_STATUSES:
                        latest.update(
                            status="error",
                            ended_at=utc_now(),
                            error={
                                "code": exc.code,
                                "message": exc.message,
                                "details": exc.details,
                            },
                        )
                        store.write_stage_state(stage, latest)
                    raise
                retry_consumed = True
                continue

            if not retry_consumed:
                raise OrchestratorError(
                    "USER_REVIEW_REQUIRED",
                    f"retry target {retry_stage_id} has no failed local state",
                )

            store.update_pipeline(status="running", current_stage=stage.id)
            stage_dir = store.stage_dir(stage)
            stage_dir.mkdir(parents=True, exist_ok=True)
            started = utc_now()
            stage_state = {
                "schema_version": 1,
                "pipeline_id": config.id,
                "stage": stage.id,
                "status": "starting",
                "remote_branch": branch_for(stage),
                "remote_worktree": str(worktree_for(config, stage)),
                "base_ref": base_ref_for(config, index),
                "base_commit": None,
                "head_commit": None,
                "started_at": started,
                "ended_at": None,
                "result": None,
                "error": None,
            }
            store.write_stage_state(stage, stage_state)
            stage_ssh = ssh_factory(
                config.remote, lambda message, s=stage: store.log_stage(s, message)
            )
            print(f"[stage] {stage.id}: creating isolated branch/worktree", flush=True)
            try:
                worktree, branch, base_sha = _create_stage_worktree(
                    config,
                    stage,
                    stage_ssh,
                    base_ref=base_ref_for(config, index),
                    expected_common_dir=snapshot["remote_common_git_dir"],
                )
                stage_state.update(status="running", base_commit=base_sha)
                store.write_stage_state(stage, stage_state)
                _execute_stage(
                    config,
                    stage,
                    store,
                    stage_ssh,
                    snapshot,
                    worktree=worktree,
                    branch=branch,
                    base_sha=base_sha,
                    stage_state=stage_state,
                    terminal=index == len(config.stages) - 1,
                )
            except OrchestratorError as exc:
                latest = store.read_json(store.stage_state_path(stage)) or stage_state
                if latest.get("status") not in RESULT_STATUSES:
                    latest.update(
                        status="error",
                        ended_at=utc_now(),
                        error={
                            "code": exc.code,
                            "message": exc.message,
                            "details": exc.details,
                        },
                    )
                    store.write_stage_state(stage, latest)
                raise

        if not retry_consumed:
            raise OrchestratorError(
                "USER_REVIEW_REQUIRED",
                f"retry target was not reached: {retry_stage_id}",
            )

        _verify_primary_unchanged(config, preflight_ssh, snapshot)
        store.update_pipeline(
            status="complete",
            current_stage=None,
            ended_at=utc_now(),
            error=None,
        )
        return {
            "pipeline_id": config.id,
            "status": "complete",
            "stages": [
                {
                    "id": stage.id,
                    "branch": branch_for(stage),
                    "state": str(store.stage_state_path(stage)),
                }
                for stage in config.stages
            ],
            "automatic_merge": False,
            "automatic_push": False,
        }
    except OrchestratorError as exc:
        store.update_pipeline(
            status="stopped",
            ended_at=utc_now(),
            error={"code": exc.code, "message": exc.message, "details": exc.details},
        )
        store.log_pipeline(f"pipeline stopped: {exc.code}: {exc.message}")
        raise


def summarize_last_jsonl_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_line = ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        modified = path.stat().st_mtime
        payload = json.loads(last_line) if last_line else None
    except OSError:
        return {"type": "unreadable"}
    except json.JSONDecodeError:
        payload = None

    summary: dict[str, Any] = {
        "updated_at": datetime.fromtimestamp(modified, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    }
    if isinstance(payload, dict):
        summary["type"] = payload.get("type")
        item = payload.get("item")
        if isinstance(item, dict):
            summary["item_type"] = item.get("type")
            summary["item_status"] = item.get("status")
    else:
        summary["type"] = "unparseable"
    return summary


def read_status(
    config: PipelineConfig, state_root: Path = DEFAULT_STATE_ROOT
) -> dict[str, Any]:
    store = StateStore(config, state_root)
    pipeline_state = store.read_json(store.meta_path)
    stages = []
    for stage in config.stages:
        state = store.read_json(store.stage_state_path(stage))
        events_path = store.stage_dir(stage) / "codex-events.jsonl"
        stages.append(
            {
                "id": stage.id,
                "branch": branch_for(stage),
                "worktree": str(worktree_for(config, stage)),
                "status": None if state is None else state.get("status"),
                "base_commit": None if state is None else state.get("base_commit"),
                "head_commit": None if state is None else state.get("head_commit"),
                "state_file": str(store.stage_state_path(stage)),
                "events_file": str(events_path),
                "last_event": summarize_last_jsonl_event(events_path),
            }
        )
    return {
        "pipeline_id": config.id,
        "status": "not_started"
        if pipeline_state is None
        else pipeline_state.get("status"),
        "pipeline_state": pipeline_state,
        "stages": stages,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--pipeline", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--pipeline", required=True, type=Path)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact stage plan without SSH or local state writes",
    )
    retry = subparsers.add_parser("retry")
    retry.add_argument("--pipeline", required=True, type=Path)
    retry.add_argument(
        "stage_id",
        help="retry one pre-execution error after verifying unchanged local/remote state",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_pipeline(args.pipeline)
        if args.command == "validate":
            output = {
                "status": "valid",
                "pipeline_id": config.id,
                "stage_order": [stage.id for stage in config.stages],
                "linear": True,
                "push": False,
                "automatic_merge": False,
            }
        elif args.command == "status":
            output = read_status(config)
        elif args.command == "retry":
            output = run_pipeline(config, retry_stage_id=args.stage_id)
        else:
            output = run_pipeline(config, dry_run=args.dry_run)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except OrchestratorError as exc:
        error = {
            "status": "error",
            "error_code": exc.code,
            "message": exc.message,
            "details": redact_value(exc.details),
        }
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return ERROR_EXIT_CODES.get(exc.code, 1)


if __name__ == "__main__":
    raise SystemExit(main())
