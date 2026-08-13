"""I3 keyframe_plan schema and Keyframe Planner harness tests (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.script import planner
from anime_remix.services.script.keyframe_plan import parse_keyframe_plan
from anime_remix.services.script.shot_plan import parse_shot_plan


def _shot_document(
    duration: float = 4.0,
    *,
    complexity: str = "simple",
) -> dict[str, Any]:
    beats = [{"time_seconds": 0.0, "description": "起始动作"}]
    if complexity == "complex":
        beats += [
            {"time_seconds": 1.0, "description": "转身"},
            {"time_seconds": 2.0, "description": "拔剑"},
            {"time_seconds": 3.0, "description": "斩击"},
        ]
    else:
        beats.append(
            {"time_seconds": duration, "description": "结束动作"}
        )
    return {
        "schema_version": "shot-plan-v1",
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_01",
                "order": 1,
                "narrative_purpose": "测试镜头",
                "duration_seconds": duration,
                "shot_scale": "wide",
                "composition": "角色居中",
                "camera_position": "正面平视",
                "camera_motion": "fixed",
                "subjects": ["林夏"],
                "setting": "黄昏的学校天台",
                "props": ["剑"] if complexity == "complex" else ["书包"],
                "start_state": "站立",
                "action_beats": beats,
                "end_state": "收势",
                "emotion_arc": "平静到专注",
                "dialogue": None,
                "continuity_in": "入场",
                "continuity_out": "收势",
            }
        ],
    }


def _assets() -> list[dict[str, str]]:
    return [
        {"asset_id": "char_lin_xia", "asset_type": "character"},
        {"asset_id": "bg_rooftop", "asset_type": "background"},
        {"asset_id": "prop_sword", "asset_type": "prop"},
    ]


def _keyframe(
    *,
    order: int,
    time_seconds: float,
    position: float,
    shot_duration: float = 4.0,
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "keyframe_id": f"kf_{order:03d}",
        "shot_id": "shot_001",
        "order": order,
        "time_seconds": time_seconds,
        "position": position,
        "visual_description": f"第 {order} 帧画面",
        "subject_pose": "站立",
        "expression": "平静",
        "gaze": "前方",
        "composition": "居中",
        "camera": "固定正面",
        "background_state": "黄昏天台",
        "foreground_state": "无",
        "prop_state": "书包在脚边",
        "required_assets": [
            {
                "asset_id": "char_lin_xia",
                "asset_type": "character",
                "locked_attributes": ["身份", "服装"],
            },
            {
                "asset_id": "bg_rooftop",
                "asset_type": "background",
                "locked_attributes": ["场景", "光线"],
            },
        ],
        "motion_from_previous": "相对上一帧的运动",
    }
    base.update(overrides)
    if order == 1:
        base["motion_from_previous"] = "起始状态"
    return base


def _plan(
    *,
    duration: float = 4.0,
    count: int = 2,
) -> dict[str, Any]:
    times = [duration * i / (count - 1) for i in range(count)]
    keyframes = [
        _keyframe(
            order=i + 1,
            time_seconds=times[i],
            position=times[i] / duration,
            shot_duration=duration,
        )
        for i in range(count)
    ]
    return {
        "schema_version": "keyframe-plan-v1",
        "shot_id": "shot_001",
        "shot_duration_seconds": duration,
        "keyframes": keyframes,
    }


def test_valid_simple_plan_parses() -> None:
    parsed = parse_keyframe_plan(_plan(duration=4.0, count=2))
    assert parsed.schema_version == "keyframe-plan-v1"
    assert parsed.shot_duration_seconds == 4.0
    assert [kf.time_seconds for kf in parsed.keyframes] == [0.0, 4.0]
    assert parsed.keyframes[0].required_assets[0].asset_id == "char_lin_xia"


def test_complex_plan_with_many_frames_parses() -> None:
    parsed = parse_keyframe_plan(_plan(duration=4.0, count=6))
    assert len(parsed.keyframes) == 6
    positions = [kf.position for kf in parsed.keyframes]
    assert positions[0] == 0 and positions[-1] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version="keyframe-plan-v2"),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=1, time_seconds=0.0, position=0.0, required_assets=[])),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=1, time_seconds=0.0, position=0.0, shot_id="other_shot")),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=1, time_seconds=0.0, position=0.0, keyframe_id="kf_002")),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=2, time_seconds=0.0, position=0.0)),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=1, time_seconds=0.0, position=0.5)),
        lambda d: d["keyframes"].__setitem__(1, _keyframe(order=2, time_seconds=3.9, position=0.975)),
        lambda d: d["keyframes"].__setitem__(1, _keyframe(order=2, time_seconds=4.0, position=0.975)),
        lambda d: d["keyframes"].__setitem__(1, _keyframe(order=2, time_seconds=4.0, position=1.0, visual_description="  ")),
        lambda d: d["keyframes"].__setitem__(0, _keyframe(order=1, time_seconds=0.0, position=0.0, required_assets=[{"asset_id": "char_lin_xia", "asset_type": "video", "locked_attributes": ["x"]}])),
    ],
)
def test_invalid_plans_rejected(mutate: Any) -> None:
    data = _plan(duration=4.0, count=2)
    mutate(data)
    with pytest.raises(InputValidationError):
        parse_keyframe_plan(data)


def test_build_asset_summary_is_text_only() -> None:
    summary = planner.build_asset_summary(_assets())
    assert "char_lin_xia (character)" in summary
    assert "bg_rooftop (background)" in summary
    assert "prop_sword (prop)" in summary


def test_run_planner_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_doc = parse_shot_plan(_shot_document(duration=4.0))
    shot = shot_doc.shots[0]
    calls: list[bytes] = []

    class FakeCompleted:
        def __init__(self, stdout: bytes, stderr: bytes) -> None:
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command: list[str], *, input: bytes, **kwargs: Any) -> FakeCompleted:
        calls.append(input)
        if len(calls) == 1:
            return FakeCompleted(
                b"not json",
                b"model: deepseek-v4-flash\nsession id: s1\ntokens used\n100",
            )
        payload = json.dumps(_plan(duration=4.0, count=3), ensure_ascii=False)
        return FakeCompleted(
            payload.encode("utf-8"),
            b"model: deepseek-v4-flash\n"
            b"session id: 019fea40-0000-0000-0000-000000000001\n"
            b"tokens used\n250",
        )

    monkeypatch.setattr(planner.subprocess, "run", fake_run)

    result = planner.run_planner_for_shot(
        shot,
        assets=_assets(),
        out_dir=tmp_path / "run",
    )

    assert len(calls) == 2
    assert result.tries == 2
    assert result.document.shot_id == "shot_001"
    assert len(result.document.keyframes) == 3
    assert (tmp_path / "run" / "run_manifest.json").exists()
    assert (tmp_path / "run" / "keyframe_plan.json").exists()
    assert (tmp_path / "run" / "attempt_01_raw_output.txt").exists()


def test_run_planner_fails_after_max_tries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_doc = parse_shot_plan(_shot_document(duration=4.0))
    shot = shot_doc.shots[0]

    class FakeCompleted:
        stdout = b"still not json"
        stderr = b"model: deepseek-v4-flash"

    monkeypatch.setattr(
        planner.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted(),
    )

    with pytest.raises(InputValidationError):
        planner.run_planner_for_shot(
            shot,
            assets=_assets(),
            out_dir=tmp_path / "run",
        )
    assert (tmp_path / "run" / "attempt_03_raw_output.txt").exists()


def test_run_planner_rejects_unknown_asset_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_doc = parse_shot_plan(_shot_document(duration=4.0))
    shot = shot_doc.shots[0]
    data = _plan(duration=4.0, count=2)
    data["keyframes"][0]["required_assets"] = [
        {
            "asset_id": "not_in_summary",
            "asset_type": "character",
            "locked_attributes": ["身份"],
        }
    ]

    class FakeCompleted:
        stdout = json.dumps(data, ensure_ascii=False).encode("utf-8")
        stderr = b"model: deepseek-v4-flash"

    monkeypatch.setattr(
        planner.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted(),
    )

    with pytest.raises(InputValidationError) as exc_info:
        planner.run_planner_for_shot(
            shot,
            assets=_assets(),
            out_dir=tmp_path / "run",
        )
    assert "not_in_summary" in str(exc_info.value)
