"""Rule-based script parser producing one ShotRequirement per paragraph."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, Decimal

from anime_remix.domain.models import (
    AliasesDocument,
    CharacterRef,
    ClipsDocument,
    ShotRequirement,
)
from anime_remix.errors import InputValidationError
from anime_remix.services.aliases import (
    build_canonical_characters,
    build_canonical_locations,
    character_alias_entries,
    location_alias_entries,
)
from anime_remix.services.input_loader import _normalize_name

DEFAULT_FRAMES = 72
MIN_FRAMES = 24
MAX_FRAMES = 192
DIALOGUE_CHARS_PER_SECOND = Decimal("4.5")
DIALOGUE_PADDING_SECONDS = Decimal("0.6")
MAX_PARAGRAPHS = 10
MIN_PARAGRAPHS = 3
MAX_PARAGRAPH_CODEPOINTS = 5000

_DIALOGUE_PAIRS = (
    ("\u201c", "\u201d"),  # ""
    ("\u300c", "\u300d"),  # 「」
    ("\u300e", "\u300f"),  # 『』
    ('"', '"'),
)
_ASCII_ID = re.compile(r"^[A-Za-z0-9_]+$")
_TERM_TYPE_RANK = {"id": 0, "name": 1, "alias": 2}


def split_paragraphs(text: str) -> list[str]:
    """Split on one or more blank lines, strip, and drop empty paragraphs."""

    chunks = re.split(r"\n[ \t]*\n+", text)
    paragraphs = [chunk.strip() for chunk in chunks]
    return [p for p in paragraphs if p]


def extract_dialogues(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Extract paired dialogue spans in original order; no nesting support."""

    spans: list[tuple[int, int]] = []
    parts: list[str] = []
    i = 0
    while i < len(text):
        best: tuple[int, str, str] | None = None
        for opener, closer in _DIALOGUE_PAIRS:
            pos = text.find(opener, i)
            if pos != -1 and (best is None or pos < best[0]):
                best = (pos, opener, closer)
        if best is None:
            break
        pos, opener, closer = best
        end = text.find(closer, pos + len(opener))
        if end == -1:
            i = pos + len(opener)
            continue
        spans.append((pos, end + len(closer)))
        parts.append(text[pos + len(opener) : end])
        i = end + len(closer)
    return "\n".join(parts), spans


def action_from_paragraph(text: str, dialogue_spans: list[tuple[int, int]]) -> str:
    """Remove dialogue spans, collapse whitespace, fall back to original."""

    if not dialogue_spans:
        return re.sub(r"\s+", " ", text).strip()
    kept: list[str] = []
    cursor = 0
    for start, end in dialogue_spans:
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    action = re.sub(r"\s+", " ", "".join(kept)).strip()
    if not action:
        action = re.sub(r"\s+", " ", text).strip()
    return action


def _boundary_pattern(display: str) -> str:
    if _ASCII_ID.fullmatch(display):
        return rf"(?<![A-Za-z0-9_]){re.escape(display)}(?![A-Za-z0-9_])"
    return re.escape(display)


def _term_rank(entry: dict[str, str]) -> int:
    return _TERM_TYPE_RANK.get(entry.get("term_type", ""), 1)


def _build_dictionary(
    doc: ClipsDocument,
    aliases: AliasesDocument | None = None,
) -> dict[str, list[dict[str, str]]]:
    characters: dict[str, dict[str, str]] = {}
    locations: dict[str, dict[str, str]] = {}
    for clip in doc.clips:
        for ref in clip.characters:
            if ref.id:
                characters.setdefault(
                    ref.id,
                    {
                        "kind": "character",
                        "key": ref.id,
                        "display": ref.id,
                        "id": ref.id,
                        "name": ref.name or "",
                        "target_id": ref.id,
                        "term_type": "id",
                    },
                )
            if ref.name:
                characters.setdefault(
                    _normalize_name(ref.name),
                    {
                        "kind": "character",
                        "key": _normalize_name(ref.name),
                        "display": ref.name,
                        "id": ref.id or "",
                        "name": ref.name,
                        "target_id": ref.id or "",
                        "term_type": "name",
                    },
                )
        if clip.location_id:
            locations.setdefault(
                clip.location_id,
                {
                    "kind": "location",
                    "key": clip.location_id,
                    "display": clip.location_id,
                    "id": clip.location_id,
                    "name": clip.location_name or "",
                    "target_id": clip.location_id,
                    "term_type": "id",
                },
            )
        if clip.location_name:
            locations.setdefault(
                _normalize_name(clip.location_name),
                {
                    "kind": "location",
                    "key": _normalize_name(clip.location_name),
                    "display": clip.location_name,
                    "id": clip.location_id or "",
                    "name": clip.location_name,
                    "target_id": clip.location_id or "",
                    "term_type": "name",
                },
            )
    if aliases is not None:
        for entry in character_alias_entries(
            aliases,
            build_canonical_characters(doc),
        ):
            characters.setdefault(
                f"alias:{entry['key']}:{entry['display']}",
                entry,
            )
        for entry in location_alias_entries(
            aliases,
            build_canonical_locations(doc),
        ):
            locations.setdefault(
                f"alias:{entry['key']}:{entry['display']}",
                entry,
            )
    character_entries = list(characters.values())
    location_entries = list(locations.values())
    character_entries.sort(
        key=lambda entry: (
            entry["target_id"],
            _term_rank(entry),
            entry["display"],
        )
    )
    location_entries.sort(
        key=lambda entry: (
            entry["target_id"],
            _term_rank(entry),
            entry["display"],
        )
    )
    return {
        "character": character_entries,
        "location": location_entries,
    }


def find_longest_non_overlapping(
    text: str,
    entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Greedy longest non-overlapping match sorted by text position."""

    matches: list[tuple[int, int, dict[str, str]]] = []
    for entry in entries:
        pattern = _boundary_pattern(entry["display"])
        for m in re.finditer(pattern, text):
            matches.append((m.start(), m.end(), entry))
    matches.sort(
        key=lambda t: (
            t[0],
            -(t[1] - t[0]),
            t[2].get("target_id") or t[2].get("key", ""),
            _term_rank(t[2]),
            t[2].get("display", ""),
        )
    )
    selected: list[tuple[int, int, dict[str, str]]] = []
    for start, end, entry in matches:
        if any(start < s_end and s_start < end for s_start, s_end, _ in selected):
            continue
        selected.append((start, end, entry))
    selected.sort(key=lambda t: t[0])
    return [entry for _, _, entry in selected]


def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the first hit per canonical target; order stays positional."""

    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for entry in entries:
        identity = (
            entry.get("target_id")
            or entry.get("id")
            or f"name:{_normalize_name(entry.get('name') or '')}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(entry)
    return result


def compute_target_frames(dialogue: str | None) -> int:
    """Compute integer target frames directly, without float seconds."""

    if not dialogue:
        return DEFAULT_FRAMES
    seconds = (
        Decimal(len(dialogue)) / DIALOGUE_CHARS_PER_SECOND
        + DIALOGUE_PADDING_SECONDS
    )
    dialogue_frames = int(
        (seconds * Decimal(24)).to_integral_value(rounding=ROUND_CEILING)
    )
    target = max(DEFAULT_FRAMES, dialogue_frames)
    return min(MAX_FRAMES, max(MIN_FRAMES, target))


def parse_script(
    text: str,
    clips: ClipsDocument,
    aliases: AliasesDocument | None = None,
) -> list[ShotRequirement]:
    """Parse a script into one ShotRequirement per non-empty paragraph."""

    paragraphs = split_paragraphs(text)
    if len(paragraphs) < MIN_PARAGRAPHS or len(paragraphs) > MAX_PARAGRAPHS:
        raise InputValidationError(
            f"script must have {MIN_PARAGRAPHS}..{MAX_PARAGRAPHS} paragraphs",
            actual=len(paragraphs),
        )
    for index, paragraph in enumerate(paragraphs, start=1):
        if len(paragraph) > MAX_PARAGRAPH_CODEPOINTS:
            raise InputValidationError(
                f"paragraph {index} exceeds {MAX_PARAGRAPH_CODEPOINTS} code points",
                actual=len(paragraph),
            )

    dictionary = _build_dictionary(clips, aliases)
    requirements: list[ShotRequirement] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        dialogue, spans = extract_dialogues(paragraph)
        action = action_from_paragraph(paragraph, spans)
        characters = [
            CharacterRef(
                id=entry["id"] or None,
                name=entry["name"] or None,
            )
            for entry in _dedupe_entries(
                find_longest_non_overlapping(
                    paragraph,
                    dictionary["character"],
                )
            )
        ]
        location_entries = _dedupe_entries(
            find_longest_non_overlapping(paragraph, dictionary["location"])
        )
        location_id = location_entries[0]["id"] or None if location_entries else None
        location_name = (
            location_entries[0]["name"] or None if location_entries else None
        )
        requirements.append(
            ShotRequirement(
                id=f"shot_{index:03d}",
                order=index,
                source_text=paragraph,
                characters=characters,
                location_id=location_id,
                location_name=location_name,
                action=action,
                target_frames=compute_target_frames(dialogue),
                dialogue=dialogue or None,
            )
        )
    return requirements
