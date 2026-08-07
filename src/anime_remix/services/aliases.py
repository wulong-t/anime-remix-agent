"""Load and validate aliases.json (AGENTS.md v1.11 section 7.5 / 3B-1)."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from anime_remix.domain.models import (
    AliasesDocument,
    ClipsDocument,
)
from anime_remix.errors import InputValidationError
from anime_remix.json_io import read_text

MAX_ALIASES_FILE_BYTES = 1024 * 1024


def alias_key(value: str) -> str:
    """Stable uniqueness key: Unicode NFKC -> strip -> casefold."""

    return unicodedata.normalize("NFKC", value).strip().casefold()


@dataclass(frozen=True)
class CanonicalCharacter:
    id: str
    name: str


@dataclass(frozen=True)
class CanonicalLocation:
    id: str
    name: str


def build_canonical_characters(doc: ClipsDocument) -> dict[str, CanonicalCharacter]:
    """Stable canonical character table: non-empty character ID -> ref."""

    table: dict[str, CanonicalCharacter] = {}
    for clip in doc.clips:
        for ref in clip.characters:
            if not (ref.id and ref.id.strip()):
                continue
            table.setdefault(
                ref.id,
                CanonicalCharacter(
                    id=ref.id,
                    name=(ref.name or "").strip(),
                ),
            )
    return dict(sorted(table.items()))


def build_canonical_locations(doc: ClipsDocument) -> dict[str, CanonicalLocation]:
    """Stable canonical location table: non-empty location_id -> name."""

    table: dict[str, CanonicalLocation] = {}
    for clip in doc.clips:
        if not (clip.location_id and clip.location_id.strip()):
            continue
        table.setdefault(
            clip.location_id,
            CanonicalLocation(
                id=clip.location_id,
                name=(clip.location_name or "").strip(),
            ),
        )
    return dict(sorted(table.items()))


def _canonical_terms(
    table: dict[str, CanonicalCharacter | CanonicalLocation],
) -> dict[str, str]:
    """alias_key -> original canonical term (ID and non-empty name)."""

    terms: dict[str, str] = {}
    for canonical in table.values():
        for term in (canonical.id, canonical.name):
            if not term:
                continue
            terms.setdefault(alias_key(term), term)
    return terms


def _validate_category(
    *,
    category: str,
    entries: list[Any],
    table: dict[str, CanonicalCharacter | CanonicalLocation],
) -> None:
    canonical_terms = _canonical_terms(table)
    canonical_names = {
        alias_key(canonical.name): canonical.name
        for canonical in table.values()
        if canonical.name
    }
    seen: dict[str, str] = {}
    for entry in entries:
        target_id = entry.target_id
        if target_id not in table:
            if target_id in canonical_names:
                raise InputValidationError(
                    f"aliases validation: {category} target_id must be a "
                    f"canonical ID, not a name",
                    field="target_id",
                    actual=target_id,
                )
            raise InputValidationError(
                f"aliases validation: {category} target does not exist in "
                "clips.json canonical table",
                field="target_id",
                actual=target_id,
            )
        target_keys: set[str] = set()
        for alias in entry.aliases:
            key = alias_key(alias)
            if key in target_keys:
                raise InputValidationError(
                    f"aliases validation: {category} target {target_id!r} "
                    f"repeats alias {alias!r} (key={key!r})",
                    field="aliases",
                    actual=alias,
                )
            target_keys.add(key)
            previous = seen.get(key)
            if previous is not None and previous != target_id:
                raise InputValidationError(
                    f"aliases validation: {category} alias {alias!r} "
                    f"(key={key!r}) maps to both {previous!r} and "
                    f"{target_id!r}",
                    field="aliases",
                    actual=alias,
                )
            canonical_term = canonical_terms.get(key)
            if canonical_term is not None:
                raise InputValidationError(
                    f"aliases validation: {category} alias {alias!r} "
                    f"(key={key!r}) for target {target_id!r} conflicts with "
                    f"canonical term {canonical_term!r}",
                    field="aliases",
                    actual=alias,
                )
            seen[key] = target_id


def validate_aliases_targets(
    doc: AliasesDocument,
    clips_doc: ClipsDocument,
) -> None:
    """Validate targets and alias_key conflicts after canonical merge."""

    characters = build_canonical_characters(clips_doc)
    locations = build_canonical_locations(clips_doc)
    _validate_category(
        category="character",
        entries=doc.character_aliases,
        table=characters,  # type: ignore[arg-type]
    )
    _validate_category(
        category="location",
        entries=doc.location_aliases,
        table=locations,  # type: ignore[arg-type]
    )


def load_aliases_document(
    path: Path,
    clips_doc: ClipsDocument,
) -> AliasesDocument:
    """Load aliases.json with size, UTF-8-SIG, strict schema and target rules."""

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise InputValidationError(
            "cannot read aliases file",
            actual=path,
        ) from exc
    if size_bytes > MAX_ALIASES_FILE_BYTES:
        raise InputValidationError(
            f"aliases file exceeds {MAX_ALIASES_FILE_BYTES} bytes",
            actual=size_bytes,
        )
    text = read_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "invalid aliases.json",
            field=f"line {exc.lineno} col {exc.colno}",
            actual=exc.msg,
        ) from exc
    if not isinstance(data, dict):
        raise InputValidationError(
            "aliases.json must be a top-level object, "
            f"got {type(data).__name__}",
            actual=path,
        )
    try:
        doc = TypeAdapter(AliasesDocument).validate_python(data)
    except Exception as exc:  # pydantic ValidationError
        raise InputValidationError(
            f"invalid aliases.json schema: {exc}",
            actual=path,
        ) from exc
    validate_aliases_targets(doc, clips_doc)
    return doc


def character_alias_entries(
    doc: AliasesDocument,
    characters: dict[str, CanonicalCharacter],
) -> list[dict[str, str]]:
    """Stable character alias dictionary entries for the rule parser."""

    entries: list[dict[str, str]] = []
    for target_id in sorted(characters):
        canonical = characters[target_id]
        for entry in doc.character_aliases:
            if entry.target_id != target_id:
                continue
            for alias in sorted(entry.aliases, key=lambda item: (alias_key(item), item)):
                entries.append(
                    {
                        "kind": "character",
                        "key": alias_key(alias),
                        "display": alias,
                        "id": target_id,
                        "name": canonical.name,
                        "target_id": target_id,
                        "term_type": "alias",
                    }
                )
    return entries


def location_alias_entries(
    doc: AliasesDocument,
    locations: dict[str, CanonicalLocation],
) -> list[dict[str, str]]:
    """Stable location alias dictionary entries for the rule parser."""

    entries: list[dict[str, str]] = []
    for target_id in sorted(locations):
        canonical = locations[target_id]
        for entry in doc.location_aliases:
            if entry.target_id != target_id:
                continue
            for alias in sorted(entry.aliases, key=lambda item: (alias_key(item), item)):
                entries.append(
                    {
                        "kind": "location",
                        "key": alias_key(alias),
                        "display": alias,
                        "id": target_id,
                        "name": canonical.name,
                        "target_id": target_id,
                        "term_type": "alias",
                    }
                )
    return entries
