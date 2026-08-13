"""Shot-asset binding tests: template, validation and reference bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import load_image_assets
from anime_remix.services.script.binding import (
    auto_bind,
    generate_binding_template,
    parse_binding,
    validate_binding_against_plan,
)
from anime_remix.services.script.shot_plan import parse_shot_plan

runner = CliRunner()

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png(width: int, height: int) -> bytes:
    import struct
    import zlib

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        _PNG_MAGIC
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _make_catalog(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    (images / "lin_xia.png").write_bytes(_png(32, 48))
    (images / "su_hang.png").write_bytes(_png(32, 48))
    (images / "rooftop.png").write_bytes(_png(64, 36))
    (images / "soda.png").write_bytes(_png(16, 16))
    manifest = tmp_path / "image_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "image-assets-v1",
                "assets": [
                    {
                        "asset_id": "char_lin_xia",
                        "path": "images/lin_xia.png",
                        "asset_type": "character",
                        "subject_or_scene_id": "char_lin_xia",
                        "rights_status": "user-owned",
                    },
                    {
                        "asset_id": "char_su_hang",
                        "path": "images/su_hang.png",
                        "asset_type": "character",
                        "subject_or_scene_id": "char_su_hang",
                        "rights_status": "user-owned",
                    },
                    {
                        "asset_id": "bg_rooftop",
                        "path": "images/rooftop.png",
                        "asset_type": "background",
                        "subject_or_scene_id": "loc_rooftop",
                        "rights_status": "user-owned",
                    },
                    {
                        "asset_id": "prop_soda",
                        "path": "images/soda.png",
                        "asset_type": "prop",
                        "subject_or_scene_id": "prop_soda",
                        "quality_notes": "合成道具参考：汽水罐",
                        "rights_status": "user-owned",
                    },
                    {
                        "asset_id": "style_watercolor",
                        "path": "images/soda.png",
                        "asset_type": "style",
                        "subject_or_scene_id": "style_watercolor",
                        "rights_status": "user-owned",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _shot_plan(tmp_path: Path) -> Path:
    path = tmp_path / "shot_plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "shot-plan-v1",
                "shots": [
                    {
                        "shot_id": "shot_001",
                        "scene_id": "scene_01",
                        "order": 1,
                        "narrative_purpose": "建立角色与环境",
                        "duration_seconds": 4.0,
                        "shot_scale": "wide",
                        "composition": "角色居中",
                        "camera_position": "正面平视",
                        "camera_motion": "fixed",
                        "subjects": ["林夏"],
                        "setting": "黄昏的学校天台",
                        "props": ["汽水"],
                        "start_state": "站在天台入口",
                        "action_beats": [
                            {"time_seconds": 0.0, "description": "推门"},
                            {"time_seconds": 2.0, "description": "停下"},
                        ],
                        "end_state": "倚在栏杆上",
                        "emotion_arc": "平静",
                        "dialogue": None,
                        "continuity_in": None,
                        "continuity_out": "望向远处",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _binding_data(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "shot-asset-binding-v1",
        "bindings": [
            {
                "shot_id": "shot_001",
                "asset_id": "char_lin_xia",
                "asset_type": "character",
                "note": "正面参考",
            },
            {
                "shot_id": "shot_001",
                "asset_id": "bg_rooftop",
                "asset_type": "background",
            },
        ],
    }
    base.update(overrides)
    return base


def test_template_lists_candidates_per_shot(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))

    template = generate_binding_template(plan, catalog)

    assert len(template.shots) == 1
    shot = template.shots[0]
    assert shot.shot_id == "shot_001"
    ids = [c["asset_id"] for c in shot.candidates]
    assert "char_lin_xia" in ids
    assert "char_su_hang" in ids
    assert "bg_rooftop" in ids
    assert "prop_soda" in ids


def test_valid_binding_produces_reference_bundle(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))
    binding = parse_binding(_binding_data())

    bundles = validate_binding_against_plan(binding, plan, catalog)

    refs = bundles["shot_001"]
    assert [r["asset_id"] for r in refs] == ["char_lin_xia", "bg_rooftop"]
    assert refs[0]["note"] == "正面参考"
    assert refs[1]["path"].endswith("images\\rooftop.png") or refs[1][
        "path"
    ].endswith("images/rooftop.png")


def test_binding_rejects_missing_shot(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))
    binding = parse_binding(
        _binding_data(
            bindings=[
                {
                    "shot_id": "shot_999",
                    "asset_id": "char_lin_xia",
                    "asset_type": "character",
                }
            ]
        )
    )

    with pytest.raises(InputValidationError) as exc_info:
        validate_binding_against_plan(binding, plan, catalog)
    assert "shot_999" in str(exc_info.value)


def test_binding_rejects_unregistered_asset(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))
    binding = parse_binding(
        _binding_data(
            bindings=[
                {
                    "shot_id": "shot_001",
                    "asset_id": "not_registered",
                    "asset_type": "character",
                }
            ]
        )
    )

    with pytest.raises(InputValidationError) as exc_info:
        validate_binding_against_plan(binding, plan, catalog)
    assert "not_registered" in str(exc_info.value)


def test_binding_rejects_wrong_type(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))
    binding = parse_binding(
        _binding_data(
            bindings=[
                {
                    "shot_id": "shot_001",
                    "asset_id": "char_lin_xia",
                    "asset_type": "background",
                }
            ]
        )
    )

    with pytest.raises(InputValidationError):
        validate_binding_against_plan(binding, plan, catalog)


def test_cli_bind_template_and_validate(tmp_path: Path) -> None:
    manifest = _make_catalog(tmp_path)
    shot_plan_path = _shot_plan(tmp_path)
    out_dir = tmp_path / "bindings"
    bundles_dir = tmp_path / "bundles"

    result = runner.invoke(
        app,
        [
            "director",
            "bind-template",
            "--shot-plan",
            str(shot_plan_path),
            "--image-assets",
            str(manifest),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    template_path = out_dir / "shot_asset_binding.template.json"
    assert template_path.exists()

    binding_path = out_dir / "shot_asset_binding.json"
    binding_path.write_text(
        json.dumps(_binding_data(), ensure_ascii=False),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "director",
            "bind-validate",
            "--shot-plan",
            str(shot_plan_path),
            "--image-assets",
            str(manifest),
            "--binding",
            str(binding_path),
            "--output",
            str(bundles_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (bundles_dir / "shot_001.reference_bundle.json").exists()


def test_auto_bind_matches_subject_setting_prop_and_style(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))

    draft, report = auto_bind(plan, catalog)

    entries = draft.bindings
    assert entries, "auto-bind produced no bindings"
    ids = [(e.asset_id, e.asset_type) for e in entries]
    assert ("char_lin_xia", "character") in ids
    assert ("bg_rooftop", "background") in ids
    assert ("prop_soda", "prop") in ids
    assert ("style_watercolor", "style") in ids
    assert report["shot_001"], "report must explain each shot's choices"


def test_auto_bind_draft_validates(tmp_path: Path) -> None:
    catalog = load_image_assets(_make_catalog(tmp_path))
    plan = parse_shot_plan(json.loads(_shot_plan(tmp_path).read_text(encoding="utf-8")))
    draft, _ = auto_bind(plan, catalog)

    bundles = validate_binding_against_plan(draft, plan, catalog)

    assert "shot_001" in bundles
    assert any(r["asset_type"] == "character" for r in bundles["shot_001"])
    assert any(r["asset_type"] == "background" for r in bundles["shot_001"])


def test_cli_bind_auto_writes_draft(tmp_path: Path) -> None:
    manifest = _make_catalog(tmp_path)
    shot_plan_path = _shot_plan(tmp_path)
    out_dir = tmp_path / "auto"

    result = runner.invoke(
        app,
        [
            "director",
            "bind-auto",
            "--shot-plan",
            str(shot_plan_path),
            "--image-assets",
            str(manifest),
            "--output",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    draft_path = out_dir / "shot_asset_binding.auto.json"
    assert draft_path.exists()
    assert "char_lin_xia" in result.output
    assert "bg_rooftop" in result.output
