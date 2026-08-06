"""Path safety: symlink / junction / reparse-point escape coverage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from anime_remix.domain.models import RenderProfile, Timeline
from anime_remix.errors import OutputValidationError, UnsafePathError
from anime_remix.services.input_loader import validate_clip_path
from anime_remix.workflows.render_workflow import _ensure_output_safety


def _minimal_timeline() -> Timeline:
    return Timeline.model_construct(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[],
    )


def _make_junction(link: Path, target: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("junction test is Windows-only")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not link.exists():
        pytest.skip(
            "cannot create junction "
            f"(cmd exit={result.returncode}): "
            f"{result.stdout.strip()} {result.stderr.strip()}"
        )


def test_output_symlink_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.mp4"
    target.write_bytes(b"x")
    link = tmp_path / "out.mp4"
    try:
        os.symlink(target, link)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "cannot create symlinks on this Windows host: required "
                "privilege not held (WinError 1314)"
            )
        pytest.skip(f"cannot create symlinks: {exc}")
    except NotImplementedError:
        pytest.skip("symlinks are not supported on this platform")
    with pytest.raises(OutputValidationError):
        _ensure_output_safety(
            tmp_path / "timeline.json",
            link,
            _minimal_timeline(),
            {},
            allow_managed_output=False,
        )


def test_junction_clip_path_escape_rejected(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    outside = tmp_path_factory.mktemp("junction-outside")
    (outside / "out.mp4").write_bytes(b"data")
    junction = clips / "junction"
    _make_junction(junction, outside)
    with pytest.raises(UnsafePathError):
        validate_clip_path(
            tmp_path,
            "clips/junction/out.mp4",
            clip_id="clip_001",
        )


def test_junction_output_inside_managed_run_rejected(tmp_path: Path) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".anime-remix-run").write_text("{}", encoding="utf-8")
    link = tmp_path / "link"
    _make_junction(link, managed)
    with pytest.raises(OutputValidationError):
        _ensure_output_safety(
            tmp_path / "timeline.json",
            link / "out.mp4",
            _minimal_timeline(),
            {},
            allow_managed_output=False,
        )


def test_junction_output_cannot_alias_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source_file = source_dir / "out.mp4"
    source_file.write_bytes(b"source")
    link = tmp_path / "link"
    _make_junction(link, source_dir)
    with pytest.raises(OutputValidationError):
        _ensure_output_safety(
            tmp_path / "timeline.json",
            link / "out.mp4",
            _minimal_timeline(),
            {"shot_001": source_file},
            allow_managed_output=False,
        )
