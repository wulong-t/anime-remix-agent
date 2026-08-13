"""Write extracted frames into a strict ``image_assets.json`` catalog."""

from __future__ import annotations

from pathlib import Path

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic
from anime_remix.services.episode_assets.classifier import FrameClassification
from anime_remix.services.image_assets import load_image_assets


def _slug(value: str, *, limit: int = 48) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip()
    ).strip("_")
    cleaned = re_sub_underscores(cleaned)
    if not cleaned:
        cleaned = "episode"
    return cleaned[:limit]


def re_sub_underscores(value: str) -> str:
    out = value
    while "__" in out:
        out = out.replace("__", "_")
    return out


def asset_id_for(frame_index: int, *, title: str | None) -> str:
    base = _slug(title) if title else "episode"
    return f"{base}_f{frame_index:04d}"[:64]


def build_catalog_entry(
    *,
    asset_id: str,
    rel_path: str,
    frame_sha256: str,
    timestamp_seconds: float,
    video_name: str,
    classification: FrameClassification | None,
    rights_status: str,
) -> dict:
    optional: dict = {}
    if classification is not None:
        for key in ("view_angle", "pose", "expression", "outfit", "quality_notes"):
            value = getattr(classification, key)
            if value is not None:
                optional[key] = value
    return {
        "asset_id": asset_id,
        "path": rel_path,
        "asset_type": (
            classification.asset_type if classification is not None else "background"
        ),
        "rights_status": rights_status,
        "subject_or_scene_id": (
            classification.subject_or_scene_id
            if classification is not None
            else f"frame at t={timestamp_seconds:.3f}s"
        ),
        **optional,
        "source_tier": "derived",
        "reference_roles": (
            classification.effective_reference_roles
            if classification is not None
            else ["scene_reference"]
        ),
        "provenance": {
            "sha256": frame_sha256,
            "parent_asset_id": None,
            "parent_sha256": None,
            "source_path": video_name,
            "note": f"extracted from episode frame at t={timestamp_seconds:.3f}s",
        },
        "analysis_status": "analyzed" if classification is not None else "pending",
    }


def build_crop_entry(
    *,
    asset_id: str,
    rel_path: str,
    crop_sha256: str,
    parent_asset_id: str,
    parent_sha256: str,
    timestamp_seconds: float,
    video_name: str,
    label: str,
    frame_subject: str,
    rights_status: str,
) -> dict:
    asset_type = "character" if label in {"face", "character"} else "background"
    roles = (
        ["identity_reference", "expression_reference"]
        if label == "face"
        else (["identity_reference"] if asset_type == "character" else ["scene_reference"])
    )
    return {
        "asset_id": asset_id,
        "path": rel_path,
        "asset_type": asset_type,
        "rights_status": rights_status,
        "subject_or_scene_id": f"{frame_subject} ({label} crop)",
        "source_tier": "derived",
        "reference_roles": roles,
        "provenance": {
            "sha256": crop_sha256,
            "parent_asset_id": parent_asset_id,
            "parent_sha256": parent_sha256,
            "source_path": video_name,
            "note": (
                f"{label} crop from episode frame at t={timestamp_seconds:.3f}s"
            ),
        },
        "analysis_status": "analyzed",
    }


def write_episode_catalog(
    *,
    output_dir: Path,
    video_name: str,
    rights_status: str,
    entries: list[dict],
) -> Path:
    catalog_path = output_dir / "image_assets.json"
    if catalog_path.exists():
        raise InputValidationError(
            f"output catalog already exists: {catalog_path}; choose a new --output dir",
            actual=str(catalog_path),
        )
    dump_json_atomic(
        catalog_path,
        {"schema_version": "image-assets-v1", "assets": entries},
        sort_keys=True,
    )
    load_image_assets(catalog_path)
    return catalog_path
