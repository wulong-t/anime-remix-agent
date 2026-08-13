"""Shot-to-reference-image binding (post-review asset attachment).

The user first reviews and approves a shot plan, then attaches concrete
reference images per shot by editing a generated binding JSON.  A validated
binding is the source of the Reference Bundle consumed by the Keyframe
Planner and later by Qwen: it only references assets registered in the
``image_assets.json`` catalog.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object
from anime_remix.services.image_assets import (
    ImageAssetCatalog,
)
from anime_remix.services.script.shot_plan import ShotPlanDocument

_SCHEMA_VERSION = "shot-asset-binding-v1"
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

AssetType = Literal[
    "character",
    "background",
    "foreground",
    "prop",
    "style",
]


class ShotBindingEntry(BaseModel):
    """One shot with its attached reference assets."""

    model_config = _STRICT_CONFIG

    shot_id: str
    asset_id: str
    asset_type: AssetType
    note: str | None = None

    @field_validator("shot_id", "asset_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("shot_id/asset_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator("note", mode="before")
    @classmethod
    def _note(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("note must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("note must be non-empty when provided")
        return stripped


class ShotBindingTemplate(BaseModel):
    """One shot's editable template row with candidate asset choices."""

    model_config = _STRICT_CONFIG

    shot_id: str
    scene_id: str
    candidates: list[dict[str, str]]

    @field_validator("shot_id", "scene_id", mode="before")
    @classmethod
    def _ids(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("shot_id/scene_id must be strings")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid id {value!r}")
        return stripped

    @field_validator("candidates", mode="before")
    @classmethod
    def _candidates(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("candidates must be a list")
        cleaned: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("candidate entries must be objects")
            asset_id = item.get("asset_id")
            asset_type = item.get("asset_type")
            if not isinstance(asset_id, str) or not isinstance(asset_type, str):
                raise TypeError("candidate entries need asset_id and asset_type")
            if asset_type not in (
                "character",
                "background",
                "foreground",
                "prop",
                "style",
            ):
                raise ValueError(f"invalid candidate asset_type {asset_type!r}")
            cleaned.append(
                {"asset_id": asset_id.strip(), "asset_type": asset_type}
            )
        return cleaned


class ShotAssetBindingDocument(BaseModel):
    """Strict shot-asset binding document (user-edited)."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["shot-asset-binding-v1"] = _SCHEMA_VERSION
    bindings: list[ShotBindingEntry]

    @model_validator(mode="after")
    def _bindings(self) -> ShotAssetBindingDocument:
        if not self.bindings:
            raise ValueError("bindings must not be empty")
        seen: set[tuple[str, str]] = set()
        for entry in self.bindings:
            key = (entry.shot_id, entry.asset_id)
            if key in seen:
                raise ValueError(
                    f"duplicate binding for shot {entry.shot_id} / "
                    f"asset {entry.asset_id}"
                )
            seen.add(key)
        return self


class ShotBindingTemplateDocument(BaseModel):
    """Generated editable template: candidates per shot, bindings empty."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["shot-asset-binding-v1"] = _SCHEMA_VERSION
    shots: list[ShotBindingTemplate]


def parse_binding(data: object) -> ShotAssetBindingDocument:
    """Parse and strictly validate a shot-asset binding document."""

    try:
        return TypeAdapter(ShotAssetBindingDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(f"invalid shot asset binding: {exc}") from exc


def load_binding(path: str | object) -> ShotAssetBindingDocument:
    """Load a binding JSON file from disk."""

    return parse_binding(load_json_object(Path(str(path))))


def generate_binding_template(
    shot_plan: ShotPlanDocument,
    catalog: ImageAssetCatalog,
) -> ShotBindingTemplateDocument:
    """Build an editable template: candidate assets per shot, no bindings yet."""

    shots: list[ShotBindingTemplate] = []
    for shot in shot_plan.shots:
        subjects = shot.subjects
        candidates = [record for record in catalog]
        characters = [
            record
            for record in candidates
            if record.asset_type == "character"
            and any(
                subject.lower() in (record.subject_or_scene_id or "").lower()
                for subject in subjects
            )
        ]
        preferred = characters or [
            record
            for record in candidates
            if record.asset_type == "character"
        ]
        candidates_list = [
            {"asset_id": record.asset_id, "asset_type": record.asset_type}
            for record in preferred
        ]
        candidates_list.extend(
            {
                "asset_id": record.asset_id,
                "asset_type": record.asset_type,
            }
            for record in candidates
            if record.asset_id not in {c["asset_id"] for c in candidates_list}
            and record.asset_type in ("background", "foreground", "prop", "style")
        )
        shots.append(
            ShotBindingTemplate(
                shot_id=shot.shot_id,
                scene_id=shot.scene_id,
                candidates=candidates_list,
            )
        )
    return ShotBindingTemplateDocument(shots=shots)


def write_binding_template(
    template: ShotBindingTemplateDocument,
    out_dir: Path,
) -> Path:
    """Write the editable template JSON into out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shot_asset_binding.template.json"
    dump_json_atomic(
        path,
        template.model_dump(mode="json"),
        sort_keys=True,
    )
    return path


def validate_binding_against_plan(
    binding: ShotAssetBindingDocument,
    shot_plan: ShotPlanDocument,
    catalog: ImageAssetCatalog,
) -> dict[str, list[dict[str, str]]]:
    """Validate bindings and return per-shot Reference Bundles."""

    shot_ids = {shot.shot_id for shot in shot_plan.shots}
    catalog_by_id = {record.asset_id: record for record in catalog}

    unknown_shots = sorted({entry.shot_id for entry in binding.bindings} - shot_ids)
    if unknown_shots:
        raise InputValidationError(
            f"bindings reference unknown shots: {', '.join(unknown_shots)}"
        )

    bound_shots = {entry.shot_id for entry in binding.bindings}
    missing = sorted(shot_ids - bound_shots)
    if missing:
        raise InputValidationError(
            f"shots without any reference binding: {', '.join(missing)}"
        )

    bundles: dict[str, list[dict[str, str]]] = {}
    for shot in shot_plan.shots:
        entries = [
            entry for entry in binding.bindings if entry.shot_id == shot.shot_id
        ]
        if not entries:
            continue
        references: list[dict[str, str]] = []
        seen_assets: set[str] = set()
        for entry in entries:
            record = catalog_by_id.get(entry.asset_id)
            if record is None:
                raise InputValidationError(
                    f"shot {entry.shot_id} references unregistered asset "
                    f"{entry.asset_id!r}",
                    asset_id=entry.asset_id,
                )
            if record.asset_type != entry.asset_type:
                raise InputValidationError(
                    f"shot {entry.shot_id} references asset "
                    f"{entry.asset_id!r} with wrong type "
                    f"({entry.asset_type!r} vs registered {record.asset_type!r})",
                    asset_id=entry.asset_id,
                )
            if entry.asset_id in seen_assets:
                raise InputValidationError(
                    f"shot {entry.shot_id} repeats asset {entry.asset_id!r}",
                    asset_id=entry.asset_id,
                )
            seen_assets.add(entry.asset_id)
            references.append(
                {
                    "asset_id": entry.asset_id,
                    "asset_type": entry.asset_type,
                    "path": str(record.resolved_path),
                    "note": entry.note or "",
                }
            )
        bundles[shot.shot_id] = references
    return bundles


def write_reference_bundles(
    bundles: dict[str, list[dict[str, str]]],
    out_dir: Path,
) -> Path:
    """Write one reference_bundle.json per shot into out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for shot_id, references in sorted(bundles.items()):
        dump_json_atomic(
            out_dir / f"{shot_id}.reference_bundle.json",
            {
                "schema_version": "reference-bundle-v1",
                "shot_id": shot_id,
                "references": references,
            },
            sort_keys=True,
        )
    return out_dir


def _norm(text: str) -> str:
    """NFKC-normalize, casefold and strip for matching."""

    return unicodedata.normalize("NFKC", text).strip().lower()


def _match_score(subject: str, record) -> int:
    """Score how well an asset's metadata matches a subject/prop term."""

    needle = _norm(subject)
    if not needle:
        return 0
    id_text = _norm(record.asset_id)
    scene_text = _norm(record.subject_or_scene_id or "")
    notes_text = _norm(record.quality_notes or "")
    if needle in scene_text:
        return 3
    if needle in id_text:
        return 2
    if needle in notes_text:
        return 1
    return 0


_TIER_ORDER = ("canonical", "derived", "approved_generated", "generated_candidate")
_TIER_RANK = {tier: index for index, tier in enumerate(_TIER_ORDER)}


def _tier_rank(record) -> int:
    """Deterministic trust rank; unknown tiers sort after every known tier."""

    return _TIER_RANK.get(record.source_tier, len(_TIER_ORDER))


def _high_tier_candidates(records: list, used: set[str]) -> list:
    """Trusted candidates only: canonical > derived > approved_generated."""

    pool = [
        record
        for record in records
        if record.asset_id not in used
        and _tier_rank(record) < _TIER_RANK["generated_candidate"]
    ]
    pool.sort(key=lambda record: (_tier_rank(record), record.asset_id))
    return pool


def _generated_candidates(records: list, used: set[str]) -> list:
    """Last-resort candidates; never part of the default matching pool."""

    pool = [
        record
        for record in records
        if record.asset_id not in used
        and record.source_tier == "generated_candidate"
    ]
    pool.sort(key=lambda record: record.asset_id)
    return pool


def _confidence(score: int, *, tie: bool) -> tuple[str, str]:
    """Map a match score to (confidence, entry decision)."""

    if score >= 2 and not tie:
        return "high", "auto"
    if score == 1 or tie:
        return "low", "needs_review"
    return "unresolved", "needs_review"


def _pick_match(
    records: list,
    used: set[str],
    term: str,
    *,
    fallback: bool,
) -> tuple[object | None, int, str, str]:
    """Pick the best record for a textual term.

    High-tier candidates are matched first (score desc, tier asc,
    asset_id asc).  ``generated_candidate`` assets never participate in
    that pool and are only used as a last resort when no high-tier
    candidate exists.  Returns ``(record, score, reason, confidence)``;
    record is None when nothing is eligible.
    """

    high = _high_tier_candidates(records, used)
    if high:
        scored = sorted(
            (
                (_match_score(term, record), _tier_rank(record), record.asset_id, record)
                for record in high
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        best_score, _, _, best = scored[0]
        if best_score >= 1:
            tie = len(scored) >= 2 and scored[1][0] == best_score
            confidence, _ = _confidence(best_score, tie=tie)
            return best, best_score, f"text match for {term}", confidence
        if not fallback:
            return None, 0, f"no text match for {term}", "unresolved"
        return best, 0, f"fallback for {term} (no text match)", "unresolved"
    last_resort = _generated_candidates(records, used)
    if last_resort:
        scored = sorted(
            (
                (_match_score(term, record), record.asset_id, record)
                for record in last_resort
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best = scored[0][2]
        best_score = scored[0][0]
        confidence = "low" if best_score >= 1 else "unresolved"
        return (
            best,
            best_score,
            f"last-resort generated candidate for {term}",
            confidence,
        )
    return None, 0, f"no candidate for {term}", "unresolved"


def _pick_style(
    records: list,
    used: set[str],
) -> tuple[object | None, int, str, str]:
    """Pick a global style anchor, or ask for review on ambiguity."""

    high = _high_tier_candidates(records, used)
    if len(high) == 1:
        return high[0], 3, "global style anchor (single candidate)", "high"
    if len(high) > 1:
        return (
            high[0],
            1,
            "multiple style candidates; first picked for review",
            "low",
        )
    last_resort = _generated_candidates(records, used)
    if len(last_resort) >= 1:
        return (
            last_resort[0],
            1,
            "last-resort generated style candidate",
            "low",
        )
    return None, 0, "no style candidate", "unresolved"


def _shot_decision(
    entries: list[dict[str, str | int]],
    gaps: list[dict[str, str]],
) -> str:
    """One shot decision: auto | needs_review | unresolved."""

    if any(gap["severity"] == "missing" for gap in gaps):
        return "unresolved"
    if any(gap["severity"] == "unmatched" for gap in gaps):
        return "needs_review"
    if any(entry["confidence"] != "high" for entry in entries):
        return "needs_review"
    return "auto"


def auto_bind(
    shot_plan: ShotPlanDocument,
    catalog: ImageAssetCatalog,
) -> tuple[ShotAssetBindingDocument, dict[str, dict[str, object]]]:
    """Automatically bind assets per shot using shot semantics + metadata.

    Matching is textual and deterministic (NFKC + casefold) over
    ``asset_id`` / ``subject_or_scene_id`` / ``quality_notes``.  The
    trusted tier (canonical > derived > approved_generated) forms the
    default candidate pool; ``generated_candidate`` assets are only used
    as a last resort and always flagged for review.  The returned report
    explains every choice (tier / score / reason / confidence) and gives
    each shot a decision: ``auto``, ``needs_review`` or ``unresolved``.
    """

    by_type: dict[str, list] = {
        "character": [],
        "background": [],
        "foreground": [],
        "prop": [],
        "style": [],
    }
    for record in catalog:
        by_type[record.asset_type].append(record)

    bindings: list[ShotBindingEntry] = []
    report: dict[str, dict[str, object]] = {}
    for shot in shot_plan.shots:
        shot_report: list[dict[str, str | int]] = []
        gaps: list[dict[str, str]] = []
        used: set[str] = set()

        for subject in shot.subjects:
            best, score, reason, confidence = _pick_match(
                by_type["character"],
                used,
                subject,
                fallback=True,
            )
            if best is None:
                gaps.append(
                    {
                        "severity": "missing",
                        "reason": f"no character candidate for {subject}",
                    }
                )
                continue
            used.add(best.asset_id)
            bindings.append(
                ShotBindingEntry(
                    shot_id=shot.shot_id,
                    asset_id=best.asset_id,
                    asset_type=best.asset_type,
                    note=f"auto character for {subject}",
                )
            )
            shot_report.append(
                {
                    "asset_id": best.asset_id,
                    "asset_type": best.asset_type,
                    "tier": best.source_tier,
                    "matched_for": subject,
                    "score": score,
                    "reason": reason,
                    "confidence": confidence,
                }
            )

        best, score, reason, confidence = _pick_match(
            by_type["background"],
            used,
            shot.setting,
            fallback=True,
        )
        if best is None:
            gaps.append(
                {
                    "severity": (
                        "missing" if not by_type["background"] else "unmatched"
                    ),
                    "reason": f"no background for setting {shot.setting}",
                }
            )
        else:
            used.add(best.asset_id)
            bindings.append(
                ShotBindingEntry(
                    shot_id=shot.shot_id,
                    asset_id=best.asset_id,
                    asset_type=best.asset_type,
                    note="auto background for setting",
                )
            )
            shot_report.append(
                {
                    "asset_id": best.asset_id,
                    "asset_type": best.asset_type,
                    "tier": best.source_tier,
                    "matched_for": shot.setting,
                    "score": score,
                    "reason": reason,
                    "confidence": confidence,
                }
            )

        for prop in shot.props:
            best, score, reason, confidence = _pick_match(
                by_type["prop"],
                used,
                prop,
                fallback=False,
            )
            if best is None:
                gaps.append(
                    {
                        "severity": (
                            "missing" if not by_type["prop"] else "unmatched"
                        ),
                        "reason": f"no prop candidate for {prop}",
                    }
                )
                continue
            used.add(best.asset_id)
            bindings.append(
                ShotBindingEntry(
                    shot_id=shot.shot_id,
                    asset_id=best.asset_id,
                    asset_type=best.asset_type,
                    note=f"auto prop for {prop}",
                )
            )
            shot_report.append(
                {
                    "asset_id": best.asset_id,
                    "asset_type": best.asset_type,
                    "tier": best.source_tier,
                    "matched_for": prop,
                    "score": score,
                    "reason": reason,
                    "confidence": confidence,
                }
            )

        if by_type["style"]:
            style, score, reason, confidence = _pick_style(
                by_type["style"],
                used,
            )
            if style is None:
                gaps.append(
                    {
                        "severity": "unmatched",
                        "reason": reason,
                    }
                )
            else:
                used.add(style.asset_id)
                bindings.append(
                    ShotBindingEntry(
                        shot_id=shot.shot_id,
                        asset_id=style.asset_id,
                        asset_type=style.asset_type,
                        note="auto style anchor",
                    )
                )
                shot_report.append(
                    {
                        "asset_id": style.asset_id,
                        "asset_type": style.asset_type,
                        "tier": style.source_tier,
                        "matched_for": "style",
                        "score": score,
                        "reason": reason,
                        "confidence": confidence,
                    }
                )

        report[shot.shot_id] = {
            "decision": _shot_decision(shot_report, gaps),
            "entries": shot_report,
        }
    return ShotAssetBindingDocument(bindings=bindings), report
