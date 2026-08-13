"""I1 image asset manifest contract tests (synthetic bytes only)."""

from __future__ import annotations

import inspect
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from anime_remix.errors import InputValidationError, UnsafePathError
from anime_remix.services import image_assets as image_assets_module
from anime_remix.services.image_assets import load_image_assets

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_png(width: int, height: int) -> bytes:
    """Minimal PNG bytes; pixel data only generated for small sizes."""

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    if 0 < width * height <= 4096:
        scanlines = b"".join(
            b"\x00" + b"\x00\x00\x00" * width for _ in range(height)
        )
        idat = zlib.compress(scanlines)
    else:
        idat = zlib.compress(b"")
    return (
        _PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def make_jpeg(width: int, height: int) -> bytes:
    """Minimal JPEG bytes with a SOF0 segment for the given dimensions."""

    payload = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    sof = b"\xff\xc0" + struct.pack(">H", len(payload) + 2) + payload
    return b"\xff\xd8" + sof + b"\xff\xd9"


def asset(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "asset_id": "char_lin_xia",
        "path": "assets/lin_xia.png",
        "asset_type": "character",
        "subject_or_scene_id": "char_lin_xia",
        "view_angle": "front",
        "pose": "standing",
        "expression": "calm",
        "outfit": "school_uniform",
        "time_of_day": "dusk",
        "quality_notes": "reference only",
        "rights_status": "user-owned",
    }
    base.update(overrides)
    return base


def write_manifest(
    tmp_path: Path,
    assets: list[dict[str, Any]],
    *,
    name: str = "image_assets.json",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {"schema_version": "image-assets-v1", "assets": assets},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_loads_five_types_in_declared_order(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "lin_xia.png").write_bytes(make_png(320, 480))
    (images / "rooftop.jpg").write_bytes(make_jpeg(640, 360))
    (images / "rain.jpg").write_bytes(make_jpeg(100, 100))
    (images / "sword.png").write_bytes(make_png(64, 64))
    (images / "watercolor.png").write_bytes(make_png(48, 48))
    manifest = write_manifest(
        tmp_path,
        [
            asset(asset_id="char_lin_xia", path="images/lin_xia.png"),
            asset(
                asset_id="bg_rooftop",
                path="images/rooftop.jpg",
                asset_type="background",
                subject_or_scene_id="loc_rooftop",
                view_angle=None,
                pose=None,
                expression=None,
                outfit=None,
                quality_notes=None,
            ),
            asset(
                asset_id="fg_rain",
                path="images/rain.jpg",
                asset_type="foreground",
                subject_or_scene_id="fx_rain",
            ),
            asset(
                asset_id="prop_sword",
                path="images/sword.png",
                asset_type="prop",
            ),
            asset(
                asset_id="style_watercolor",
                path="images/watercolor.png",
                asset_type="style",
            ),
        ],
    )

    catalog = load_image_assets(manifest)

    assert catalog.ids == (
        "char_lin_xia",
        "bg_rooftop",
        "fg_rain",
        "prop_sword",
        "style_watercolor",
    )
    assert len(catalog) == 5
    first = catalog.get("char_lin_xia")
    assert first is not None
    assert first.asset_type == "character"
    assert first.resolved_path == (tmp_path / "images/lin_xia.png").resolve()
    assert first.format == "png"
    assert (first.width, first.height) == (320, 480)
    assert first.subject_or_scene_id == "char_lin_xia"
    assert first.pose == "standing"
    assert first.rights_status == "user-owned"
    background = catalog.get("bg_rooftop")
    assert background is not None
    assert background.format == "jpeg"
    assert (background.width, background.height) == (640, 360)
    assert background.view_angle is None
    assert catalog.get("missing_id") is None
    assert [record.asset_id for record in catalog] == list(catalog.ids)


def test_deterministic_repeatable_load(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    (tmp_path / "b.jpg").write_bytes(make_jpeg(32, 32))
    manifest = write_manifest(
        tmp_path,
        [
            asset(asset_id="a", path="a.png", asset_type="style"),
            asset(asset_id="b", path="b.jpg", asset_type="background"),
        ],
    )

    first = load_image_assets(manifest)
    second = load_image_assets(manifest)

    assert first.records == second.records
    assert first.ids == second.ids


def test_optional_metadata_may_be_null(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = write_manifest(
        tmp_path,
        [
            asset(
                path="a.png",
                subject_or_scene_id=None,
                view_angle=None,
                pose=None,
                expression=None,
                outfit=None,
                time_of_day=None,
                quality_notes=None,
            )
        ],
    )

    record = load_image_assets(manifest).get("char_lin_xia")

    assert record is not None
    assert record.view_angle is None
    assert record.time_of_day is None
    assert record.quality_notes is None
    assert record.rights_status == "user-owned"


def test_duplicate_asset_id_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = write_manifest(
        tmp_path,
        [asset(path="a.png"), asset(path="a.png")],
    )

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_empty_assets_rejected(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_schema_version_drift_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = tmp_path / "image_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "image-assets-v2",
                "assets": [asset(path="a.png")],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_extra_fields_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    top = tmp_path / "top.json"
    top.write_text(
        json.dumps(
            {
                "schema_version": "image-assets-v1",
                "assets": [asset(path="a.png")],
                "extra": 1,
            }
        ),
        encoding="utf-8",
    )
    entry = tmp_path / "entry.json"
    entry.write_text(
        json.dumps(
            {
                "schema_version": "image-assets-v1",
                "assets": [asset(path="a.png", extra_field=1)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError):
        load_image_assets(top)
    with pytest.raises(InputValidationError):
        load_image_assets(entry)


def test_empty_rights_status_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))

    for rights in ("", "   "):
        manifest = write_manifest(
            tmp_path,
            [asset(path="a.png", rights_status=rights)],
        )
        with pytest.raises(InputValidationError):
            load_image_assets(manifest)


def test_whitespace_only_optional_metadata_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = write_manifest(tmp_path, [asset(path="a.png", pose="  ")])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_non_string_strict_field_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = write_manifest(tmp_path, [asset(asset_id=123, path="a.png")])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_unknown_asset_type_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(make_png(16, 16))
    manifest = write_manifest(tmp_path, [asset(asset_type="video")])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_missing_file_rejected(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [asset(path="missing.png")])

    with pytest.raises(UnsafePathError):
        load_image_assets(manifest)


@pytest.mark.parametrize(
    "bad_path",
    [
        "C:/x/a.png",
        "C:\\x\\a.png",
        "/abs/a.png",
        "\\\\server\\share\\a.png",
        "https://example.com/a.png",
        "http://example.com/a.png",
        "file:///c:/a.png",
        "../outside.png",
        "assets/../../outside.png",
        "assets/..",
    ],
)
def test_unsafe_paths_rejected(tmp_path: Path, bad_path: str) -> None:
    manifest = write_manifest(tmp_path, [asset(path=bad_path)])

    with pytest.raises(UnsafePathError):
        load_image_assets(manifest)


def test_whitespace_only_path_rejected(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, [asset(path="   ")])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


@pytest.mark.parametrize("path_name", ["a.gif", "a.webp", "a.bmp", "a"])
def test_unsupported_extension_rejected(tmp_path: Path, path_name: str) -> None:
    (tmp_path / path_name).write_bytes(b"whatever")
    manifest = write_manifest(tmp_path, [asset(path=path_name)])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_signature_extension_mismatch_rejected(tmp_path: Path) -> None:
    (tmp_path / "png_but_jpg.jpg").write_bytes(make_png(16, 16))
    (tmp_path / "jpg_but_png.png").write_bytes(make_jpeg(16, 16))

    for bad_path in ("png_but_jpg.jpg", "jpg_but_png.png"):
        manifest = write_manifest(tmp_path, [asset(path=bad_path)])
        with pytest.raises(InputValidationError):
            load_image_assets(manifest)


@pytest.mark.parametrize(
    ("path_name", "payload"),
    [
        ("a.png", b"not an image at all"),
        ("a.png", _PNG_MAGIC),
        ("a.png", _PNG_MAGIC + b"\x00" * 16),
        ("a.jpg", b"\xff\xd8\xff\xd9"),
        ("a.jpg", b"\xff\xd8\xff\x00\xc0\x00"),
        ("a.jpg", b"\xff\xd8"),
        ("a.jpg", b"\xff\xd8\xff\xc0\x00\x02"),
    ],
)
def test_malformed_files_rejected(
    tmp_path: Path, path_name: str, payload: bytes
) -> None:
    (tmp_path / path_name).write_bytes(payload)
    manifest = write_manifest(tmp_path, [asset(path=path_name)])

    with pytest.raises(InputValidationError):
        load_image_assets(manifest)


def test_invalid_dimensions_rejected(tmp_path: Path) -> None:
    (tmp_path / "zero_w.png").write_bytes(make_png(0, 100))
    (tmp_path / "zero_h.jpg").write_bytes(make_jpeg(100, 0))
    (tmp_path / "huge.png").write_bytes(make_png(20000, 10))
    (tmp_path / "huge.jpg").write_bytes(make_jpeg(10, 20000))

    for bad_path in ("zero_w.png", "zero_h.jpg", "huge.png", "huge.jpg"):
        manifest = write_manifest(tmp_path, [asset(path=bad_path)])
        with pytest.raises(InputValidationError):
            load_image_assets(manifest)


def test_same_path_multiple_ids_allowed(tmp_path: Path) -> None:
    (tmp_path / "shared.png").write_bytes(make_png(20, 20))
    manifest = write_manifest(
        tmp_path,
        [
            asset(asset_id="char_a", path="shared.png"),
            asset(
                asset_id="char_b",
                path="shared.png",
                rights_status="licensed",
            ),
        ],
    )

    catalog = load_image_assets(manifest)
    first = catalog.get("char_a")
    second = catalog.get("char_b")

    assert first is not None and second is not None
    assert first.resolved_path == second.resolved_path
    assert first.rights_status == "user-owned"
    assert second.rights_status == "licensed"


def test_symlink_outside_rejected(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(make_png(10, 10))
    link = images / "link.png"
    try:
        os.symlink(outside, link)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "cannot create symlinks on this Windows host: required "
                "privilege not held (WinError 1314)"
            )
        pytest.skip(f"cannot create symlinks: {exc}")
    except NotImplementedError:
        pytest.skip("symlinks are not supported on this platform")
    manifest = write_manifest(tmp_path, [asset(path="images/link.png")])

    with pytest.raises(UnsafePathError):
        load_image_assets(manifest)


def test_source_hygiene_no_discovery_and_exact_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(make_png(12, 12))
    (images / "b.jpg").write_bytes(make_jpeg(12, 12))
    manifest = write_manifest(
        tmp_path,
        [
            asset(asset_id="a", path="images/a.png"),
            asset(asset_id="b", path="images/b.jpg"),
        ],
    )

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("directory enumeration is forbidden in I1 loader")

    monkeypatch.setattr(os, "scandir", forbid)
    monkeypatch.setattr(os, "listdir", forbid)
    monkeypatch.setattr(os, "walk", forbid)

    opened: list[Path] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    catalog = load_image_assets(manifest)

    assert len(catalog) == 2
    opened_resolved = {str(path.resolve()) for path in opened}
    expected = {
        str(manifest.resolve()),
        str((images / "a.png").resolve()),
        str((images / "b.jpg").resolve()),
    }
    assert opened_resolved == expected


def test_module_has_no_forbidden_access_patterns() -> None:
    source = inspect.getsource(image_assets_module)
    forbidden = (
        "os.environ",
        "os.scandir",
        "os.listdir",
        "os.walk",
        "subprocess",
        "socket",
        "hashlib",
        "requests",
        "PIL",
    )
    for token in forbidden:
        assert token not in source, f"module must not reference {token}"
