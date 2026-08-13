"""I2 review-loop tests: preview, artifacts and CLI validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.services.script.review import (
    build_review_preview,
    validate_shot_plan_file,
    write_review_artifacts,
)
from anime_remix.services.script.shot_plan import parse_shot_plan

runner = CliRunner()


def _shot(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "建立角色与环境",
        "duration_seconds": 4.0,
        "shot_scale": "wide",
        "composition": "角色居中，背景为天台全景",
        "camera_position": "正面平视",
        "camera_motion": "fixed",
        "subjects": ["林夏"],
        "setting": "黄昏的学校天台",
        "props": ["书包"],
        "start_state": "林夏站在天台入口",
        "action_beats": [
            {"time_seconds": 0.0, "description": "推门走进天台"},
            {"time_seconds": 2.0, "description": "停在栏杆前"},
        ],
        "end_state": "林夏倚在栏杆上",
        "emotion_arc": "平静到放松",
        "dialogue": "“终于安静了。”",
        "continuity_in": "从走廊走上天台",
        "continuity_out": "望向远处城市",
    }
    base.update(overrides)
    return base


def _document() -> dict[str, Any]:
    return {
        "schema_version": "shot-plan-v1",
        "shots": [_shot()],
    }


def test_preview_contains_key_facts() -> None:
    document = parse_shot_plan(_document())
    preview = build_review_preview(document)
    assert "shot_001" in preview
    assert "建立角色与环境" in preview
    assert "4s" in preview
    assert "推门走进天台" in preview
    assert "“终于安静了。”" in preview
    assert "continuity out: 望向远处城市" in preview


def test_write_review_artifacts_and_validate(tmp_path: Path) -> None:
    document = parse_shot_plan(_document())
    review_dir = tmp_path / "review"

    path = write_review_artifacts(
        review_dir,
        document,
        run_manifest={"status": "needs_review"},
    )

    assert path == review_dir / "shot_plan.json"
    assert (review_dir / "shot_plan.review.md").exists()
    assert (review_dir / "run_manifest.json").exists()
    validated = validate_shot_plan_file(path)
    assert validated.shots[0].shot_id == "shot_001"


def test_validate_rejects_edited_invalid_plan(tmp_path: Path) -> None:
    path = tmp_path / "shot_plan.json"
    data = _document()
    data["shots"][0]["duration_seconds"] = -1
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = runner.invoke(
        app,
        ["director", "validate", "--shot-plan", str(path)],
    )

    assert result.exit_code == 2
    assert "invalid shot_plan.json" in result.output


def test_validate_accepts_edited_plan(tmp_path: Path) -> None:
    path = tmp_path / "shot_plan.json"
    data = _document()
    data["shots"][0]["duration_seconds"] = 5.5
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    result = runner.invoke(
        app,
        ["director", "validate", "--shot-plan", str(path)],
    )

    assert result.exit_code == 0
    assert "ok: shot plan valid" in result.output
    assert "5.5s" in result.output
