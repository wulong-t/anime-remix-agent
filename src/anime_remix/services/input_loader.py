"""Load and validate script / clips inputs with strict path safety."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from pydantic import TypeAdapter

from anime_remix.domain.models import CharacterRef, ClipAsset, ClipsDocument
from anime_remix.errors import InputValidationError, UnsafePathError
from anime_remix.json_io import load_json_object, read_text

_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_URL_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")
_WINDOWS_UNC = re.compile(r"^\\\\")


def load_script_text(path: Path) -> str:
    """Read a script with BOM tolerance and normalize line endings."""

    text = read_text(path)
    if not text.strip():
        raise InputValidationError("script must not be empty", actual=path)
    return text


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().lower()


def canonicalize_character_refs(doc: ClipsDocument) -> ClipsDocument:
    """Apply CharacterRef canonical merge rules (AGENTS.md 7.3)."""

    by_id: dict[str, str] = {}
    name_to_id: dict[str, str] = {}

    for clip in doc.clips:
        for ref in clip.characters:
            if ref.id:
                name = (ref.name or "").strip()
                existing = by_id.get(ref.id)
                if name:
                    if existing and _normalize_name(existing) != _normalize_name(
                        name
                    ):
                        raise InputValidationError(
                            f"same character id {ref.id!r} has conflicting names",
                            asset_id=clip.id,
                            field="characters",
                            actual=[existing, name],
                        )
                    by_id[ref.id] = name
                else:
                    by_id.setdefault(ref.id, "")
                if name:
                    key = _normalize_name(name)
                    previous = name_to_id.get(key)
                    if previous is not None and previous != ref.id:
                        raise InputValidationError(
                            f"normalized name {name!r} maps to multiple ids",
                            asset_id=clip.id,
                            field="characters",
                            actual=[previous, ref.id],
                        )
                    name_to_id[key] = ref.id

    rebuilt: list[ClipAsset] = []
    for clip in doc.clips:
        seen: set[str] = set()
        characters: list[CharacterRef] = []
        for ref in clip.characters:
            resolved: CharacterRef
            if ref.id:
                resolved = CharacterRef(
                    id=ref.id,
                    name=by_id.get(ref.id) or None,
                )
            else:
                mapped_id = name_to_id.get(_normalize_name(ref.name or ""))
                resolved = (
                    CharacterRef(id=mapped_id, name=ref.name)
                    if mapped_id is not None
                    else CharacterRef(name=ref.name)
                )
            key = (
                f"id:{resolved.id}"
                if resolved.id
                else f"name:{_normalize_name(resolved.name or '')}"
            )
            if key in seen:
                continue
            seen.add(key)
            characters.append(resolved)
        rebuilt.append(clip.model_copy(update={"characters": characters}))
    return doc.model_copy(update={"clips": rebuilt})


def _reject_bad_path(
    raw: str,
    *,
    clip_id: str | None = None,
    allow_parent: bool = False,
) -> None:
    if not raw:
        raise UnsafePathError("path must not be empty", asset_id=clip_id)
    if _URL_PREFIX.match(raw):
        raise UnsafePathError("URL paths are rejected", asset_id=clip_id, actual=raw)
    if (
        Path(raw).is_absolute()
        or _WINDOWS_DRIVE.match(raw)
        or _WINDOWS_UNC.match(raw)
    ):
        raise UnsafePathError("absolute/device paths are rejected", asset_id=clip_id, actual=raw)
    if "://" in raw:
        raise UnsafePathError("protocol paths are rejected", asset_id=clip_id, actual=raw)
    parts = Path(raw).parts
    if not allow_parent and any(part == ".." for part in parts):
        raise UnsafePathError("parent traversal is rejected", asset_id=clip_id, actual=raw)
    name = Path(raw).name.upper()
    if name in _DEVICE_NAMES or re.fullmatch(r"CON[0-9]|COM[0-9]|LPT[0-9]", name):
        raise UnsafePathError("device paths are rejected", asset_id=clip_id, actual=raw)


def validate_clip_path(base_dir: Path, raw: str, *, clip_id: str) -> Path:
    """Validate a clips.json path: relative, inside base dir, regular file."""

    _reject_bad_path(raw, clip_id=clip_id)
    base = base_dir.resolve()
    candidate = (base / raw).resolve(strict=False)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise UnsafePathError(
            "clip file does not exist or is unreadable",
            asset_id=clip_id,
            actual=raw,
        ) from exc
    if not _is_within(resolved, base):
        raise UnsafePathError(
            "clip path escapes clips.json directory",
            asset_id=clip_id,
            actual=raw,
        )
    if not resolved.is_file():
        raise UnsafePathError("clip path is not a regular file", asset_id=clip_id, actual=raw)
    return resolved


def validate_timeline_source_path(base_dir: Path, raw: str) -> Path:
    """Resolve a timeline source_path relative to the timeline directory.

    Unlike clips.json paths, relative parent traversal is allowed so run
    directories can reference the original project assets.
    """

    _reject_bad_path(raw, allow_parent=True)
    if Path(raw).is_absolute():
        raise UnsafePathError("timeline source_path must be relative", actual=raw)
    try:
        resolved = (base_dir.resolve() / raw).resolve(strict=True)
    except OSError as exc:
        raise UnsafePathError(
            "timeline source file does not exist",
            actual=raw,
        ) from exc
    if not resolved.is_file():
        raise UnsafePathError(
            "timeline source is not a regular file",
            actual=raw,
        )
    return resolved


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def load_clips_document(path: Path) -> ClipsDocument:
    """Load clips.json with strict schema, canonical characters and safe paths."""

    data = load_json_object(path)
    try:
        doc = TypeAdapter(ClipsDocument).validate_python(data)
    except Exception as exc:  # pydantic ValidationError
        raise InputValidationError(
            f"invalid clips.json schema: {exc}",
            actual=path,
        ) from exc
    doc = canonicalize_character_refs(doc)
    base_dir = path.resolve().parent
    for clip in doc.clips:
        validate_clip_path(base_dir, os.fspath(clip.path), clip_id=clip.id)
    return doc
