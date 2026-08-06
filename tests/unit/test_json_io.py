from __future__ import annotations

from pathlib import Path

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic, load_json_object


def test_top_level_array_rejected(tmp_path: Path) -> None:
    path = tmp_path / "clips.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_json_object(path)


def test_invalid_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_json_object(path)


def test_atomic_dump_stable_and_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    data = {"b": 1, "a": [1, 2], "text": "中文"}
    dump_json_atomic(path, data)
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert "\"b\": 1" in content
    assert "中文" in content
    assert load_json_object(path) == data
    assert sorted(tmp_path.iterdir()) == [path]


def test_nan_dump_rejected(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    with pytest.raises(InputValidationError):
        dump_json_atomic(path, {"value": float("nan")})

