from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from anime_remix.domain.models import ClipsDocument
from anime_remix.errors import InputValidationError, UnsafePathError
from anime_remix.services.input_loader import (
    canonicalize_character_refs,
    validate_clip_path,
    validate_timeline_source_path,
)


def _doc(tmp_path: Path, name: str = "outside.mp4") -> ClipsDocument:
    return TypeAdapter(ClipsDocument).validate_python(
        {
            "schema_version": "1.9",
            "clips": [
                {
                    "id": "clip_001",
                    "path": "clips/clip_001.mp4",
                    "characters": [{"id": "char_a", "name": "林夏"}],
                    "location_id": "loc_a",
                    "location_name": "学校",
                    "action": "站立",
                    "description": "林夏站着。",
                }
            ],
        }
    )


def test_clip_path_escape_rejected(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, "../outside.mp4", clip_id="clip_001")
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, str(outside), clip_id="clip_001")
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, "https://example.com/a.mp4", clip_id="clip_001")
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, "file:///c:/a.mp4", clip_id="clip_001")


def test_clip_path_missing_file_rejected(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, "clips/missing.mp4", clip_id="clip_001")


def test_clip_path_valid(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    media = clips / "clip_001.mp4"
    media.write_bytes(b"data")
    resolved = validate_clip_path(tmp_path, "clips/clip_001.mp4", clip_id="clip_001")
    assert resolved == media.resolve()


def test_timeline_source_path_allows_parent(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"data")
    timeline_dir = tmp_path / "runs" / "demo-001"
    timeline_dir.mkdir(parents=True)
    resolved = validate_timeline_source_path(timeline_dir, "../../source.mp4")
    assert resolved == source.resolve()


def test_timeline_source_path_rejects_absolute_and_url(tmp_path: Path) -> None:
    with pytest.raises(UnsafePathError):
        validate_timeline_source_path(tmp_path, str(tmp_path / "a.mp4"))
    with pytest.raises(UnsafePathError):
        validate_timeline_source_path(tmp_path, "http://example.com/a.mp4")


def test_canonicalize_character_refs() -> None:
    doc = TypeAdapter(ClipsDocument).validate_python(
        {
            "schema_version": "1.9",
            "clips": [
                {
                    "id": "clip_001",
                    "path": "a.mp4",
                    "characters": [
                        {"name": "林夏"},
                        {"id": "char_a", "name": "林夏"},
                        {"id": "char_b"},
                        {"id": "char_b", "name": "陆辰"},
                    ],
                    "action": "x",
                    "description": "y",
                }
            ],
        }
    )
    canonical = canonicalize_character_refs(doc)
    refs = canonical.clips[0].characters
    assert refs[0].id == "char_a" and refs[0].name == "林夏"
    assert refs[1].id == "char_b" and refs[1].name == "陆辰"
    assert len(refs) == 2


def test_canonicalize_conflict_rejected() -> None:
    doc = TypeAdapter(ClipsDocument).validate_python(
        {
            "schema_version": "1.9",
            "clips": [
                {
                    "id": "clip_001",
                    "path": "a.mp4",
                    "characters": [
                        {"id": "char_a", "name": "林夏"},
                        {"id": "char_a", "name": "林下"},
                    ],
                    "action": "x",
                    "description": "y",
                }
            ],
        }
    )
    with pytest.raises(InputValidationError):
        canonicalize_character_refs(doc)


def test_same_name_multiple_ids_rejected() -> None:
    doc = TypeAdapter(ClipsDocument).validate_python(
        {
            "schema_version": "1.9",
            "clips": [
                {
                    "id": "clip_001",
                    "path": "a.mp4",
                    "characters": [
                        {"id": "char_a", "name": "林夏"},
                        {"id": "char_b", "name": "林夏"},
                    ],
                    "action": "x",
                    "description": "y",
                }
            ],
        }
    )
    with pytest.raises(InputValidationError):
        canonicalize_character_refs(doc)


def test_schema_versions_rejected() -> None:
    for version in ("1.8", "2.0"):
        with pytest.raises(ValidationError):
            TypeAdapter(ClipsDocument).validate_python(
                {
                    "schema_version": version,
                    "clips": [],
                }
            )


def test_symlink_outside_rejected(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"data")
    link = clips / "link.mp4"
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
    with pytest.raises(UnsafePathError):
        validate_clip_path(tmp_path, "clips/link.mp4", clip_id="clip_001")
