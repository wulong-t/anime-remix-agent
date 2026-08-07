"""Bounded deterministic clip retrieval and scoring."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from anime_remix.domain.models import (
    ClipAsset,
    ProbedClip,
    ScoreBreakdown,
    ShotRequirement,
    quantize_score,
)
from anime_remix.errors import RetrievalError

SEQUENCE_MATCHER_MAX_CODEPOINTS = 256
NORMALIZE_MAX_CODEPOINTS = 2048

DEFAULT_WEIGHTS: dict[str, Decimal] = {
    "character": Decimal("0.25"),
    "location": Decimal("0.15"),
    "action": Decimal("0.45"),
    "duration": Decimal("0.15"),
}
TOTAL_GATE = Decimal("0.55")
CHARACTER_GATE = Decimal("0.50")
ACTION_GATE = Decimal("0.25")
TOP_K = 3
MIN_FREEZE_SOURCE_FRAMES = 24


def selection_strategy(reason_code: str) -> str:
    """Map a Selection reason_code to its renderer strategy name."""

    if reason_code in ("exact_length", "center_trim"):
        return "clip"
    if reason_code == "short_source_freeze":
        return "freeze_frame"
    if reason_code == "no_candidate":
        return "placeholder"
    raise RetrievalError(
        "unknown selection reason_code",
        field="reason_code",
        actual=reason_code,
    )


def normalize_for_match(text: str) -> str:
    """NFKC, lowercase, keep letters/digits only, cap at 2048 code points."""

    normalized: list[str] = []
    for char in unicodedata.normalize("NFKC", text).lower():
        if len(normalized) >= NORMALIZE_MAX_CODEPOINTS:
            break
        if unicodedata.category(char)[0] in ("L", "N"):
            normalized.append(char)
    return "".join(normalized)


def bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def bigram_jaccard(a: str, b: str) -> Decimal:
    if not a or not b:
        return Decimal("0.000000")
    left = bigrams(a)
    right = bigrams(b)
    if not left or not right:
        return Decimal("0.000000")
    return Decimal(len(left & right)) / Decimal(len(left | right))


def text_similarity(a: str, b: str) -> Decimal:
    """Bounded bigram/SequenceMatcher similarity with Decimal output."""

    a_norm = normalize_for_match(a)[:NORMALIZE_MAX_CODEPOINTS]
    b_norm = normalize_for_match(b)[:NORMALIZE_MAX_CODEPOINTS]
    if not a_norm or not b_norm:
        return Decimal("0.000000")
    jaccard = bigram_jaccard(a_norm, b_norm)
    if max(len(a_norm), len(b_norm)) <= SEQUENCE_MATCHER_MAX_CODEPOINTS:
        sequence = Decimal(
            str(SequenceMatcher(None, a_norm, b_norm, autojunk=False).ratio())
        )
    else:
        sequence = Decimal(0)
    return quantize_score(max(jaccard, sequence))


def _person_key(ref: Any) -> dict[str, str]:
    return {
        "id": ref.id or "",
        "name": normalize_for_match(ref.name or ""),
    }


def _person_matches(a: dict[str, str], b: dict[str, str]) -> bool:
    """Match identities without letting equal names merge distinct IDs.

    Fixed rule: when both sides have non-empty IDs, only identical IDs match;
    a name match is only considered when at least one side has no ID.
    """

    if a["id"] and b["id"]:
        return a["id"] == b["id"]
    if a["name"] and b["name"]:
        return a["name"] == b["name"]
    return False


def _max_person_matches(
    required: list[dict[str, str]],
    asset: list[dict[str, str]],
) -> int:
    if not required or not asset:
        return 0
    match_to_asset: dict[int, int] = {}

    def _augment(req_index: int, seen: set[int]) -> bool:
        for asset_index in range(len(asset)):
            if asset_index in seen or not _person_matches(
                required[req_index], asset[asset_index]
            ):
                continue
            seen.add(asset_index)
            if (
                asset_index not in match_to_asset
                or _augment(match_to_asset[asset_index], seen)
            ):
                match_to_asset[asset_index] = req_index
                return True
        return False

    matched = 0
    for req_index in range(len(required)):
        if _augment(req_index, set()):
            matched += 1
    return matched


def _character_score(
    requirement: ShotRequirement,
    asset_characters: list[dict[str, str]],
) -> Decimal | None:
    if not requirement.characters:
        return None
    required = [_person_key(ref) for ref in requirement.characters]
    matched = _max_person_matches(required, asset_characters)
    if matched == 0:
        return Decimal("0.000000")
    recall = Decimal(matched) / Decimal(len(required))
    precision = Decimal(matched) / Decimal(max(len(asset_characters), 1))
    f2 = (Decimal(5) * precision * recall) / (Decimal(4) * precision + recall)
    return quantize_score(f2)


def _location_score(
    requirement: ShotRequirement,
    clip: ClipAsset,
    clip_location_norm: str,
) -> Decimal | None:
    if not requirement.location_id and not requirement.location_name:
        return None
    if requirement.location_id and clip.location_id == requirement.location_id:
        return Decimal("1.000000")
    if requirement.location_name and clip.location_name:
        return _text_similarity_cached(
            normalize_for_match(requirement.location_name),
            bigrams(normalize_for_match(requirement.location_name)),
            clip_location_norm,
            bigrams(clip_location_norm),
        )
    return Decimal("0.000000")


def _action_score(
    req_index: _RequirementIndex,
    clip_index: _ClipIndex,
) -> Decimal:
    return quantize_score(
        max(
            _text_similarity_cached(
                req_index.action_norm,
                req_index.action_bigrams,
                clip_index.action_norm,
                clip_index.action_bigrams,
            ),
            Decimal("0.9")
            * _text_similarity_cached(
                req_index.action_norm,
                req_index.action_bigrams,
                clip_index.description_norm,
                clip_index.description_bigrams,
            ),
            Decimal("0.8")
            * _text_similarity_cached(
                req_index.source_text_norm,
                req_index.source_text_bigrams,
                clip_index.description_norm,
                clip_index.description_bigrams,
            ),
        )
    )


def _duration_score(nb_frames: int, target_frames: int) -> Decimal:
    if nb_frames >= target_frames:
        raw = Decimal(1) - min(
            Decimal(nb_frames - target_frames) / Decimal(max(target_frames, 1)),
            Decimal(1),
        )
    else:
        raw = Decimal(nb_frames) / Decimal(target_frames)
    return quantize_score(raw)


@dataclass
class _ClipIndex:
    asset: ClipAsset
    action_norm: str
    action_bigrams: set[str]
    description_norm: str
    description_bigrams: set[str]
    location_name_norm: str
    characters: list[dict[str, str]]


@dataclass
class _RequirementIndex:
    requirement: ShotRequirement
    action_norm: str
    action_bigrams: set[str]
    source_text_norm: str
    source_text_bigrams: set[str]
    location_name_norm: str
    characters: list[dict[str, str]]


def _make_clip_index(clip: ClipAsset) -> _ClipIndex:
    action_norm = normalize_for_match(clip.action)[:NORMALIZE_MAX_CODEPOINTS]
    description_norm = normalize_for_match(clip.description)[
        :NORMALIZE_MAX_CODEPOINTS
    ]
    return _ClipIndex(
        asset=clip,
        action_norm=action_norm,
        action_bigrams=bigrams(action_norm),
        description_norm=description_norm,
        description_bigrams=bigrams(description_norm),
        location_name_norm=normalize_for_match(clip.location_name or ""),
        characters=[_person_key(ref) for ref in clip.characters],
    )


def _make_requirement_index(req: ShotRequirement) -> _RequirementIndex:
    action_norm = normalize_for_match(req.action)[:NORMALIZE_MAX_CODEPOINTS]
    source_text_norm = normalize_for_match(req.source_text)[
        :NORMALIZE_MAX_CODEPOINTS
    ]
    return _RequirementIndex(
        requirement=req,
        action_norm=action_norm,
        action_bigrams=bigrams(action_norm),
        source_text_norm=source_text_norm,
        source_text_bigrams=bigrams(source_text_norm),
        location_name_norm=normalize_for_match(req.location_name or ""),
        characters=[_person_key(ref) for ref in req.characters],
    )


def _text_similarity_cached(
    a_norm: str,
    a_bigrams: set[str],
    b_norm: str,
    b_bigrams: set[str],
) -> Decimal:
    if not a_norm or not b_norm:
        return Decimal("0.000000")
    if not a_bigrams or not b_bigrams:
        return Decimal("0.000000")
    jaccard = Decimal(len(a_bigrams & b_bigrams)) / Decimal(
        len(a_bigrams | b_bigrams)
    )
    if max(len(a_norm), len(b_norm)) <= SEQUENCE_MATCHER_MAX_CODEPOINTS:
        sequence = Decimal(
            str(SequenceMatcher(None, a_norm, b_norm, autojunk=False).ratio())
        )
    else:
        sequence = Decimal(0)
    return quantize_score(max(jaccard, sequence))


def _active_weights(active: set[str]) -> dict[str, Decimal]:
    total = sum(DEFAULT_WEIGHTS[key] for key in active)
    return {
        key: quantize_score(DEFAULT_WEIGHTS[key] / total)
        for key in sorted(active)
    }


@dataclass
class ScoredCandidate:
    asset: ProbedClip
    rank: int
    score: ScoreBreakdown


@dataclass
class Selection:
    asset: ProbedClip | None
    rank: int | None
    reason_code: str
    source_in_frame: int
    source_frame_count: int
    score: ScoreBreakdown | None


def _score_requirement(
    req_index: _RequirementIndex,
    clip_index: _ClipIndex,
    probed: ProbedClip,
) -> ScoreBreakdown:
    req = req_index.requirement
    clip = clip_index.asset
    character = _character_score(req, clip_index.characters)
    location = _location_score(req, clip, clip_index.location_name_norm)
    action = _action_score(req_index, clip_index)
    duration = _duration_score(probed.nb_frames, req.target_frames)
    active: set[str] = set()
    components: dict[str, Decimal] = {}
    if character is not None:
        active.add("character")
        components["character"] = character
    if location is not None:
        active.add("location")
        components["location"] = location
    active.add("action")
    components["action"] = action
    active.add("duration")
    components["duration"] = duration
    weights = _active_weights(active)
    total = quantize_score(
        sum(weights[key] * components[key] for key in active)
    )
    return ScoreBreakdown(
        character=character,
        location=location,
        action=action,
        duration=duration,
        active_weights=weights,
        total=total,
    )


def _candidate_sort_key(
    candidate: ScoredCandidate,
) -> tuple[Decimal, Decimal, Decimal, str]:
    character = (
        candidate.score.character
        if candidate.score.character is not None
        else Decimal(-1)
    )
    return (
        -candidate.score.total,
        -character,
        -candidate.score.action,
        candidate.asset.asset.id,
    )


def retrieve(
    requirements: list[ShotRequirement],
    probed_clips: list[ProbedClip],
) -> tuple[dict[str, Selection], dict[str, Any]]:
    """Score all candidates, gate-scan, and return selections + audit document."""

    clip_indexes = [_make_clip_index(probed.asset) for probed in probed_clips]
    selections: dict[str, Selection] = {}
    audit_shots: list[dict[str, Any]] = []

    for requirement in requirements:
        req_index = _make_requirement_index(requirement)
        candidates: list[ScoredCandidate] = []
        for clip_index, probed in zip(clip_indexes, probed_clips):
            score = _score_requirement(req_index, clip_index, probed)
            candidates.append(ScoredCandidate(asset=probed, rank=0, score=score))
        candidates.sort(key=_candidate_sort_key)
        for rank, candidate in enumerate(candidates, start=1):
            candidate.rank = rank

        checked: list[dict[str, Any]] = []
        selected: Selection | None = None
        freeze_fallback: ScoredCandidate | None = None
        for candidate in candidates:
            gates: dict[str, bool | None] = {
                "total": candidate.score.total >= TOTAL_GATE,
                "character": None,
                "action": None,
            }
            if candidate.score.total < TOTAL_GATE:
                gates["action"] = False
                checked.append(
                    {
                        "rank": candidate.rank,
                        "asset_id": candidate.asset.asset.id,
                        "gates": gates,
                        "frame_gate": None,
                        "skip_reason": "total_threshold",
                    }
                )
                break
            char_ok = (
                not requirement.characters
                or (
                    candidate.score.character is not None
                    and candidate.score.character >= CHARACTER_GATE
                )
            )
            action_ok = candidate.score.action >= ACTION_GATE
            gates["character"] = char_ok
            gates["action"] = action_ok
            if not char_ok or not action_ok:
                checked.append(
                    {
                        "rank": candidate.rank,
                        "asset_id": candidate.asset.asset.id,
                        "gates": gates,
                        "frame_gate": None,
                        "skip_reason": (
                            "character" if not char_ok else "action"
                        ),
                    }
                )
                continue

            nb_frames = candidate.asset.nb_frames
            if nb_frames >= requirement.target_frames:
                # clip eligible: select immediately and stop.
                checked.append(
                    {
                        "rank": candidate.rank,
                        "asset_id": candidate.asset.asset.id,
                        "gates": gates,
                        "frame_gate": "clip_eligible",
                        "skip_reason": None,
                    }
                )
                if nb_frames == requirement.target_frames:
                    reason_code = "exact_length"
                    source_in_frame = 0
                else:
                    reason_code = "center_trim"
                    source_in_frame = (
                        nb_frames - requirement.target_frames
                    ) // 2
                selected = Selection(
                    asset=candidate.asset,
                    rank=candidate.rank,
                    reason_code=reason_code,
                    source_in_frame=source_in_frame,
                    source_frame_count=requirement.target_frames,
                    score=candidate.score,
                )
                break

            if nb_frames >= MIN_FREEZE_SOURCE_FRAMES:
                # freeze eligible: save the highest-ranked fallback only.
                checked.append(
                    {
                        "rank": candidate.rank,
                        "asset_id": candidate.asset.asset.id,
                        "gates": gates,
                        "frame_gate": "freeze_eligible",
                        "skip_reason": None,
                        "freeze_fallback": True,
                    }
                )
                if freeze_fallback is None:
                    freeze_fallback = candidate
                continue

            checked.append(
                {
                    "rank": candidate.rank,
                    "asset_id": candidate.asset.asset.id,
                    "gates": gates,
                    "frame_gate": "too_short",
                    "skip_reason": "too_short",
                }
            )

        if selected is None and freeze_fallback is not None:
            # No full clip anywhere: freeze the highest-ranked short source.
            selected = Selection(
                asset=freeze_fallback.asset,
                rank=freeze_fallback.rank,
                reason_code="short_source_freeze",
                source_in_frame=0,
                source_frame_count=freeze_fallback.asset.nb_frames,
                score=freeze_fallback.score,
            )
        if selected is None:
            selected = Selection(
                asset=None,
                rank=None,
                reason_code="no_candidate",
                source_in_frame=0,
                source_frame_count=0,
                score=None,
            )
        selections[requirement.id] = selected

        top3 = candidates[:TOP_K]
        audit_shots.append(
            {
                "shot_id": requirement.id,
                "total_candidates": len(candidates),
                "top_3": [
                    {
                        "rank": candidate.rank,
                        "asset_id": candidate.asset.asset.id,
                        "score": candidate.score.model_dump(mode="json"),
                    }
                    for candidate in top3
                ],
                "selected": {
                    "selected_asset_id": (
                        selected.asset.asset.id
                        if selected.asset is not None
                        else None
                    ),
                    "selected_global_rank": selected.rank,
                    "selected_strategy": selection_strategy(
                        selected.reason_code
                    ),
                    "reason_code": selected.reason_code,
                    "source_in_frame": selected.source_in_frame,
                    "source_frame_count": selected.source_frame_count,
                },
                "checked_gates": checked,
                "unique_skip_reasons": sorted(
                    {
                        entry["skip_reason"]
                        for entry in checked
                        if entry["skip_reason"]
                    }
                ),
            }
        )

    return selections, {
        "schema_version": "1.9",
        "shots": audit_shots,
    }
