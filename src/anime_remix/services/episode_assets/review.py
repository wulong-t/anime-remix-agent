"""Human review workflow: CSV worksheet + validated catalog corrections."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object
from anime_remix.services.image_assets import (
    ImageAssetsDocument,
    load_image_assets,
)

_REVIEW_COLUMNS = [
    "asset_id",
    "path",
    "asset_type",
    "subject_or_scene_id",
    "reference_roles",
    "view_angle",
    "pose",
    "expression",
    "outfit",
    "quality_notes",
    "decision",
    "corrected_asset_type",
    "corrected_subject",
    "corrected_roles",
    "review_notes",
]

_ASSET_TYPES = {"character", "background", "foreground", "prop", "style"}
_ROLES = {
    "identity_reference",
    "pose_reference",
    "expression_reference",
    "outfit_reference",
    "scene_reference",
    "prop_reference",
    "style_reference",
}
_DECISIONS = {"keep", "revise", "reject"}


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _split_roles(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in re.split(r"[;；]", value)
        if item.strip()
    ]


def write_review_sheet(*, catalog: Path, output: Path) -> Path:
    """Write a UTF-8 CSV worksheet with one row per catalog asset."""

    data = load_json_object(catalog)
    try:
        document = TypeAdapter(ImageAssetsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid image_assets.json catalog: {exc}",
            actual=catalog,
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REVIEW_COLUMNS)
        writer.writeheader()
        for entry in document.assets:
            writer.writerow(
                {
                    "asset_id": entry.asset_id,
                    "path": entry.path,
                    "asset_type": entry.asset_type,
                    "subject_or_scene_id": entry.subject_or_scene_id or "",
                    "reference_roles": ";".join(entry.reference_roles),
                    "view_angle": entry.view_angle or "",
                    "pose": entry.pose or "",
                    "expression": entry.expression or "",
                    "outfit": entry.outfit or "",
                    "quality_notes": entry.quality_notes or "",
                    "decision": "keep",
                    "corrected_asset_type": "",
                    "corrected_subject": "",
                    "corrected_roles": "",
                    "review_notes": "",
                }
            )
    return output


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class ReviewApplyResult:
    catalog_path: Path
    review_record_path: Path
    kept_count: int
    revised_count: int
    rejected_count: int


def apply_review(
    *,
    catalog: Path,
    worksheet: Path,
    output_catalog: Path,
    review_record: Path,
) -> ReviewApplyResult:
    """Apply reviewed decisions to the catalog and record them."""

    data = load_json_object(catalog)
    try:
        document = TypeAdapter(ImageAssetsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid image_assets.json catalog: {exc}",
            actual=catalog,
        ) from exc
    entries = {entry.asset_id: entry for entry in document.assets}
    rows = _read_rows(worksheet)
    reviews: dict[str, dict] = {}
    corrected: list[dict] = []
    kept = revised = rejected = 0
    for row in rows:
        asset_id = _optional(row.get("asset_id"))
        if not asset_id:
            raise InputValidationError("worksheet row is missing asset_id")
        if asset_id not in entries:
            raise InputValidationError(
                f"worksheet references unknown asset_id: {asset_id}",
                actual=asset_id,
            )
        decision = _optional(row.get("decision")) or "keep"
        if decision not in _DECISIONS:
            raise InputValidationError(
                f"invalid decision {decision!r} for {asset_id}",
                actual=decision,
            )
        notes = _optional(row.get("review_notes"))
        source = entries[asset_id]
        payload = source.model_dump(mode="json")
        corrections: dict = {}
        effective = decision
        if decision == "revise":
            corrected_type = _optional(row.get("corrected_asset_type"))
            corrected_subject = _optional(row.get("corrected_subject"))
            corrected_roles = _split_roles(row.get("corrected_roles"))
            info_type = _optional(row.get("asset_type"))
            info_subject = _optional(row.get("subject_or_scene_id"))
            info_roles = _split_roles(row.get("reference_roles"))
            asset_type = corrected_type or (
                info_type if info_type != source.asset_type else None
            )
            subject = corrected_subject or (
                info_subject
                if info_subject != (source.subject_or_scene_id or "")
                else None
            )
            roles = corrected_roles or (
                info_roles
                if set(info_roles) != set(source.reference_roles)
                else []
            )
            if asset_type is not None and asset_type not in _ASSET_TYPES:
                raise InputValidationError(
                    f"invalid corrected_asset_type {asset_type!r} for {asset_id}",
                    actual=asset_type,
                )
            unknown_roles = sorted(set(roles) - _ROLES)
            if unknown_roles:
                raise InputValidationError(
                    f"invalid corrected_roles for {asset_id}: {unknown_roles}",
                    actual=unknown_roles,
                )
            if asset_type is not None:
                payload["asset_type"] = asset_type
                corrections["asset_type"] = asset_type
            if subject is not None:
                payload["subject_or_scene_id"] = subject
                corrections["subject_or_scene_id"] = subject
            if roles:
                payload["reference_roles"] = roles
                corrections["reference_roles"] = roles
            if notes:
                existing = payload.get("quality_notes")
                payload["quality_notes"] = "; ".join(
                    item for item in (existing, notes) if item
                )
                corrections["quality_notes"] = notes
            if not corrections:
                effective = "keep"
        if effective == "keep":
            kept += 1
        elif effective == "revise":
            revised += 1
        else:
            rejected += 1
        if effective != "reject":
            corrected.append(payload)
        reviews[asset_id] = {
            "decision": decision,
            "effective": effective,
            "corrections": corrections,
            "review_notes": notes,
        }
    try:
        reviewed_document = TypeAdapter(ImageAssetsDocument).validate_python(
            {"schema_version": "image-assets-v1", "assets": corrected}
        )
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"reviewed catalog is invalid: {exc}",
            actual=corrected,
        ) from exc
    dump_json_atomic(
        output_catalog,
        reviewed_document.model_dump(mode="json"),
        sort_keys=True,
    )
    load_image_assets(output_catalog)
    dump_json_atomic(
        review_record,
        {
            "schema_version": "episode-assets-review-v1",
            "source_catalog": catalog.name,
            "output_catalog": output_catalog.name,
            "reviews": reviews,
        },
        sort_keys=True,
    )
    return ReviewApplyResult(
        catalog_path=output_catalog,
        review_record_path=review_record,
        kept_count=kept,
        revised_count=revised,
        rejected_count=rejected,
    )
