"""ASSET-BOOTSTRAP-MVP tests: assets CLI + tier-aware bind-auto.

All media is synthetic PNG bytes built with struct + zlib inside
``tmp_path``; no real media, network, remote or LLM is involved.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.json_io import sha256_file
from anime_remix.services.image_assets import load_image_assets
from anime_remix.services.script.binding import auto_bind
from anime_remix.services.script.shot_plan import parse_shot_plan

runner = CliRunner()

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def make_png(width: int, height: int, *, seed: int = 0) -> bytes:
    """Minimal valid PNG bytes; seed varies bytes to change the SHA-256."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([seed & 0xFF]) * (width * 3)
    scanlines = row * height
    return (
        _PNG_MAGIC
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _images(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    return images


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_id": "asset_x",
        "path": "images/asset_x.png",
        "asset_type": "character",
        "rights_status": "user-owned",
        "subject_or_scene_id": None,
        "view_angle": None,
        "pose": None,
        "expression": None,
        "outfit": None,
        "time_of_day": None,
        "quality_notes": None,
    }
    base.update(overrides)
    return base


def _write_catalog(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "image_assets.json"
    path.write_text(
        json.dumps(
            {"schema_version": "image-assets-v1", "assets": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _shot_plan_data(
    *,
    subjects: list[str],
    setting: str,
    props: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "shot-plan-v1",
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_01",
                "order": 1,
                "narrative_purpose": "opening",
                "duration_seconds": 4.0,
                "shot_scale": "wide",
                "composition": "centered",
                "camera_position": "front",
                "camera_motion": "fixed",
                "subjects": subjects,
                "setting": setting,
                "props": props or [],
                "start_state": "standing",
                "action_beats": [
                    {"time_seconds": 0.0, "description": "start"},
                    {"time_seconds": 2.0, "description": "move"},
                ],
                "end_state": "leaning",
                "emotion_arc": "calm",
                "dialogue": None,
                "continuity_in": None,
                "continuity_out": None,
            }
        ],
    }


def test_register_canonical_bootstrap_and_load(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "lin_xia.png").write_bytes(make_png(32, 48, seed=1))
    (images / "rooftop.png").write_bytes(make_png(64, 36, seed=2))
    catalog_path = tmp_path / "image_assets.json"

    result = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "lin_xia.png"),
            "--paths",
            str(images / "rooftop.png"),
            "--type",
            "character",
            "--roles",
            "identity_reference",
            "--roles",
            "pose_reference",
            "--note",
            "smoke fixture",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "lin_xia" in result.output
    assert "rooftop" in result.output

    catalog = load_image_assets(catalog_path)
    record = catalog.get("lin_xia")
    assert record is not None
    assert record.asset_type == "character"
    assert record.source_tier == "canonical"
    assert record.analysis_status == "pending"
    assert record.reference_roles == ("identity_reference", "pose_reference")
    assert record.quality_notes == "smoke fixture"
    assert record.subject_or_scene_id is None
    assert record.path == "images/lin_xia.png"
    assert (record.width, record.height) == (32, 48)
    provenance = record.provenance
    assert provenance is not None
    assert provenance["sha256"] == sha256_file(images / "lin_xia.png")
    assert provenance["source_path"] == str((images / "lin_xia.png").resolve())
    assert provenance["parent_asset_id"] is None


def test_register_dedup_by_sha256(tmp_path: Path) -> None:
    images = _images(tmp_path)
    image = images / "same.png"
    image.write_bytes(make_png(16, 16, seed=7))
    catalog_path = tmp_path / "image_assets.json"
    args = [
        "assets",
        "register",
        "--catalog",
        str(catalog_path),
        "--paths",
        str(image),
        "--type",
        "character",
    ]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "duplicate skipped" in second.output
    assert "same" in second.output

    catalog = load_image_assets(catalog_path)
    assert len(catalog) == 1
    assert catalog.ids == ("same",)


def test_register_id_sanitize_and_collision_suffix(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "my char!.png").write_bytes(make_png(8, 8, seed=1))
    (images / "my_char.png").write_bytes(make_png(8, 8, seed=2))
    catalog_path = tmp_path / "image_assets.json"

    result = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "my char!.png"),
            "--paths",
            str(images / "my_char.png"),
            "--type",
            "prop",
        ],
    )
    assert result.exit_code == 0, result.output
    catalog = load_image_assets(catalog_path)
    assert catalog.ids == ("my_char", "my_char-2")


def test_register_dir_top_level_only_no_recursion(tmp_path: Path) -> None:
    images = _images(tmp_path)
    sub = images / "sub"
    sub.mkdir()
    (images / "a.png").write_bytes(make_png(8, 8, seed=1))
    (images / "note.txt").write_text("skip me")
    (sub / "b.png").write_bytes(make_png(8, 8, seed=2))
    catalog_path = tmp_path / "image_assets.json"

    result = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--dir",
            str(images),
            "--type",
            "background",
        ],
    )
    assert result.exit_code == 0, result.output
    catalog = load_image_assets(catalog_path)
    assert catalog.ids == ("a",)


def test_register_rejects_paths_outside_catalog_dir(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.png").write_bytes(make_png(8, 8, seed=1))
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    catalog_path = manifest_dir / "image_assets.json"

    result = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(outside / "x.png"),
            "--type",
            "style",
        ],
    )
    assert result.exit_code == 2
    assert "must live inside the catalog directory" in result.output
    assert not catalog_path.exists()


def test_register_rejects_invalid_role(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "x.png").write_bytes(make_png(8, 8, seed=1))
    catalog_path = tmp_path / "image_assets.json"

    result = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "x.png"),
            "--type",
            "character",
            "--roles",
            "bogus_role",
        ],
    )
    assert result.exit_code == 2
    assert "invalid --roles value" in result.output
    assert not catalog_path.exists()


def test_register_candidate_and_promote(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "base.png").write_bytes(make_png(16, 16, seed=1))
    (images / "cand.png").write_bytes(make_png(16, 16, seed=2))
    catalog_path = tmp_path / "image_assets.json"

    registered = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "base.png"),
            "--type",
            "character",
        ],
    )
    assert registered.exit_code == 0, registered.output

    candidate = runner.invoke(
        app,
        [
            "assets",
            "register-candidate",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "cand.png"),
            "--generated-from",
            "base",
        ],
    )
    assert candidate.exit_code == 0, candidate.output
    assert "generated from base" in candidate.output

    catalog = load_image_assets(catalog_path)
    record = catalog.get("cand")
    assert record is not None
    assert record.source_tier == "generated_candidate"
    assert record.asset_type == "character"  # inherited from parent
    provenance = record.provenance
    assert provenance is not None
    assert provenance["parent_asset_id"] == "base"
    assert provenance["parent_sha256"] == sha256_file(images / "base.png")
    assert provenance["sha256"] == sha256_file(images / "cand.png")

    promoted = runner.invoke(
        app,
        [
            "assets",
            "promote",
            "--catalog",
            str(catalog_path),
            "--asset-id",
            "cand",
        ],
    )
    assert promoted.exit_code == 0, promoted.output
    catalog = load_image_assets(catalog_path)
    record = catalog.get("cand")
    assert record is not None
    assert record.source_tier == "approved_generated"
    assert record.provenance is not None
    assert record.provenance["note"] == "promoted"


def test_promote_rejects_non_candidate(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "base.png").write_bytes(make_png(8, 8, seed=1))
    catalog_path = tmp_path / "image_assets.json"
    registered = runner.invoke(
        app,
        [
            "assets",
            "register",
            "--catalog",
            str(catalog_path),
            "--paths",
            str(images / "base.png"),
            "--type",
            "character",
        ],
    )
    assert registered.exit_code == 0, registered.output

    result = runner.invoke(
        app,
        [
            "assets",
            "promote",
            "--catalog",
            str(catalog_path),
            "--asset-id",
            "base",
        ],
    )
    assert result.exit_code == 2
    assert "only generated_candidate assets can be promoted" in result.output


def test_old_manifest_backward_compatible_defaults(tmp_path: Path) -> None:
    images = _images(tmp_path)
    (images / "old.png").write_bytes(make_png(8, 8, seed=1))
    manifest = _write_catalog(
        tmp_path,
        [
            _entry(
                asset_id="old",
                path="images/old.png",
                asset_type="background",
            )
        ],
    )

    catalog = load_image_assets(manifest)
    record = catalog.get("old")
    assert record is not None
    assert record.source_tier == "canonical"
    assert record.reference_roles == ()
    assert record.provenance is None
    assert record.analysis_status == "pending"


def _binding_catalog(
    tmp_path: Path,
    entries: list[dict[str, Any]],
) -> tuple[Path, Any]:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for entry in entries:
        image = tmp_path / entry["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(make_png(16, 16, seed=hash(entry["asset_id"]) & 0xFF))
    manifest = _write_catalog(tmp_path, entries)
    return manifest, parse_shot_plan(
        json.loads(
            json.dumps(
                _shot_plan_data(subjects=["hero"], setting="city"),
                ensure_ascii=False,
            )
        )
    )


def test_auto_bind_tier_priority_prefers_canonical_on_tie(
    tmp_path: Path,
) -> None:
    manifest, plan = _binding_catalog(
        tmp_path,
        [
            _entry(
                asset_id="char_a",
                path="images/char_a.png",
                subject_or_scene_id="char_hero",
                source_tier="canonical",
            ),
            _entry(
                asset_id="char_b",
                path="images/char_b.png",
                subject_or_scene_id="char_hero",
                source_tier="approved_generated",
            ),
            _entry(
                asset_id="bg_city",
                path="images/bg_city.png",
                asset_type="background",
                subject_or_scene_id="bg_city",
            ),
            _entry(
                asset_id="style_a",
                path="images/style_a.png",
                asset_type="style",
                subject_or_scene_id="style_a",
            ),
        ],
    )
    catalog = load_image_assets(manifest)

    draft, report = auto_bind(plan, catalog)

    characters = [e for e in draft.bindings if e.asset_type == "character"]
    assert characters and characters[0].asset_id == "char_a"
    entries = report["shot_001"]["entries"]
    character_entry = next(
        e for e in entries if e["asset_type"] == "character"
    )
    assert character_entry["tier"] == "canonical"
    assert character_entry["confidence"] == "low"  # tied best score
    assert report["shot_001"]["decision"] == "needs_review"


def test_auto_bind_candidate_never_overshadows_canonical(
    tmp_path: Path,
) -> None:
    manifest, plan = _binding_catalog(
        tmp_path,
        [
            _entry(
                asset_id="char_a",
                path="images/char_a.png",
                subject_or_scene_id="char_hero",
                source_tier="canonical",
            ),
            _entry(
                asset_id="cand",
                path="images/cand.png",
                subject_or_scene_id="char_hero",
                source_tier="generated_candidate",
            ),
            _entry(
                asset_id="bg_city",
                path="images/bg_city.png",
                asset_type="background",
                subject_or_scene_id="bg_city",
            ),
            _entry(
                asset_id="style_a",
                path="images/style_a.png",
                asset_type="style",
                subject_or_scene_id="style_a",
            ),
        ],
    )
    catalog = load_image_assets(manifest)

    draft, report = auto_bind(plan, catalog)

    bound_ids = {e.asset_id for e in draft.bindings}
    assert "char_a" in bound_ids
    assert "cand" not in bound_ids
    entries = report["shot_001"]["entries"]
    character_entry = next(
        e for e in entries if e["asset_type"] == "character"
    )
    assert character_entry["confidence"] == "high"
    assert report["shot_001"]["decision"] == "auto"


def test_auto_bind_generated_candidate_only_as_last_resort(
    tmp_path: Path,
) -> None:
    manifest, plan = _binding_catalog(
        tmp_path,
        [
            _entry(
                asset_id="cand",
                path="images/cand.png",
                subject_or_scene_id="char_hero",
                source_tier="generated_candidate",
            ),
            _entry(
                asset_id="bg_city",
                path="images/bg_city.png",
                asset_type="background",
                subject_or_scene_id="bg_city",
            ),
            _entry(
                asset_id="style_a",
                path="images/style_a.png",
                asset_type="style",
                subject_or_scene_id="style_a",
            ),
        ],
    )
    catalog = load_image_assets(manifest)

    draft, report = auto_bind(plan, catalog)

    assert any(e.asset_id == "cand" for e in draft.bindings)
    entries = report["shot_001"]["entries"]
    character_entry = next(
        e for e in entries if e["asset_type"] == "character"
    )
    assert character_entry["tier"] == "generated_candidate"
    assert character_entry["confidence"] == "low"
    assert report["shot_001"]["decision"] == "needs_review"


def test_auto_bind_low_score_and_unresolved_escalation(
    tmp_path: Path,
) -> None:
    manifest, plan = _binding_catalog(
        tmp_path,
        [
            _entry(
                asset_id="char_a",
                path="images/char_a.png",
                subject_or_scene_id="char_hero",
            ),
            _entry(
                asset_id="bg_city",
                path="images/bg_city.png",
                asset_type="background",
                subject_or_scene_id="bg_city",
            ),
            _entry(
                asset_id="style_a",
                path="images/style_a.png",
                asset_type="style",
                subject_or_scene_id="style_a",
            ),
            _entry(
                asset_id="prop_other",
                path="images/prop_other.png",
                asset_type="prop",
                subject_or_scene_id="prop_other",
            ),
        ],
    )
    plan = parse_shot_plan(
        json.loads(
            json.dumps(
                _shot_plan_data(
                    subjects=["hero"], setting="nowhere", props=["soda"]
                ),
                ensure_ascii=False,
            )
        )
    )
    catalog = load_image_assets(manifest)

    draft, report = auto_bind(plan, catalog)

    # A prop asset exists but "soda" matches no registered metadata ->
    # prop stays unbound and the shot needs review.
    prop_bound = [e for e in draft.bindings if e.asset_type == "prop"]
    assert not prop_bound
    # Background fallback has no text match -> unresolved confidence.
    entries = report["shot_001"]["entries"]
    background_entry = next(
        e for e in entries if e["asset_type"] == "background"
    )
    assert background_entry["score"] == 0
    assert background_entry["confidence"] == "unresolved"
    assert report["shot_001"]["decision"] == "needs_review"


def test_auto_bind_unresolved_when_asset_type_missing(tmp_path: Path) -> None:
    manifest, plan = _binding_catalog(
        tmp_path,
        [
            _entry(
                asset_id="bg_city",
                path="images/bg_city.png",
                asset_type="background",
                subject_or_scene_id="bg_city",
            ),
            _entry(
                asset_id="style_a",
                path="images/style_a.png",
                asset_type="style",
                subject_or_scene_id="style_a",
            ),
        ],
    )
    catalog = load_image_assets(manifest)

    _, report = auto_bind(plan, catalog)

    assert report["shot_001"]["decision"] == "unresolved"
