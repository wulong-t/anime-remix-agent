"""Strict JSON and atomic file I/O."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from anime_remix.errors import InputValidationError


def read_text(path: Path, *, encoding: str = "utf-8-sig") -> str:
    """Read text with BOM tolerance and normalize line endings to \\n."""

    try:
        raw = path.read_text(encoding=encoding)
    except OSError as exc:
        raise InputValidationError(f"cannot read {path}", actual=str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"cannot decode {path}", actual=str(exc)) from exc
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a formal JSON document. Top-level bare lists are rejected."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputValidationError(f"cannot read {path}", actual=str(exc)) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"invalid JSON in {path}",
            field=f"line {exc.lineno} col {exc.colno}",
            actual=exc.msg,
        ) from exc
    if not isinstance(data, dict):
        raise InputValidationError(
            f"formal JSON must be an object, got {type(data).__name__}",
            actual=path,
        )
    return data


def dump_json_atomic(
    path: Path,
    data: Any,
    *,
    sort_keys: bool = False,
) -> None:
    """Atomically write stable UTF-8 JSON with a single trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"cannot serialize JSON for {path}",
            actual=str(exc),
        ) from exc
    if not payload.endswith("\n"):
        payload += "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise InputValidationError(
            f"cannot write {path}",
            actual=str(exc),
        ) from exc


def sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file."""

    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputValidationError(f"cannot hash {path}", actual=str(exc)) from exc
    return digest.hexdigest()
