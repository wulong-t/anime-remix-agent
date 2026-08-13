"""I2 shot_plan.json schema and Director harness tests (no LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.script import director
from anime_remix.services.script.shot_plan import parse_shot_plan


def shot(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "建立角色与环境",
        "duration_seconds": 4.0,
        "shot_scale": "wide",
        "composition": "角色居中，背景为天台全景",
        "camera_position": "正面平视，距角色约 6 米",
        "camera_motion": "fixed",
        "subjects": ["林夏"],
        "setting": "黄昏的学校天台",
        "props": ["书包"],
        "start_state": "林夏站在天台入口，逆光",
        "action_beats": [
            {"time_seconds": 0.0, "description": "林夏推开门走进天台"},
            {"time_seconds": 2.0, "description": "她停在栏杆前放下书包"},
        ],
        "end_state": "林夏倚在栏杆上，望向远处",
        "emotion_arc": "平静到放松",
        "dialogue": "“终于安静了。”",
        "continuity_in": "林夏从走廊走上天台",
        "continuity_out": "林夏倚在栏杆上望向城市",
    }
    base.update(overrides)
    return base


def document(shots: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "shot-plan-v1", "shots": shots}


def test_valid_two_shot_document_parses() -> None:
    data = document(
        [
            shot(),
            shot(
                shot_id="shot_002",
                order=2,
                narrative_purpose="展现情绪转变",
                duration_seconds=3.0,
                shot_scale="close_up",
                composition="面部特写，背景虚化",
                camera_position="正面近景",
                camera_motion="fixed",
                subjects=["林夏"],
                setting="黄昏的学校天台",
                props=[],
                start_state="林夏望向远处",
                action_beats=[
                    {"time_seconds": 0.0, "description": "林夏眼神变化"},
                    {"time_seconds": 1.5, "description": "她攥紧栏杆"},
                ],
                end_state="林夏眼眶泛红",
                emotion_arc="放松到伤感",
                dialogue="“爸爸，我做到了。”",
                continuity_in="林夏倚在栏杆上望向城市",
                continuity_out="镜头停留在她泛红的眼睛",
            ),
        ]
    )

    parsed = parse_shot_plan(data)

    assert parsed.schema_version == "shot-plan-v1"
    assert [s.shot_id for s in parsed.shots] == ["shot_001", "shot_002"]
    assert parsed.shots[1].duration_seconds == 3.0
    assert parsed.shots[1].props == []
    assert parsed.shots[1].dialogue == "“爸爸，我做到了。”"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version="shot-plan-v2"),
        lambda d: d.__setitem__("extra", 1),
        lambda d: d["shots"].append(shot(shot_id="shot_001", order=3)),
        lambda d: d["shots"].__setitem__(0, shot(order=2)),
        lambda d: d["shots"].__setitem__(0, shot(duration_seconds=0)),
        lambda d: d["shots"].__setitem__(0, shot(duration_seconds=9999)),
        lambda d: d["shots"].__setitem__(0, shot(shot_scale="extreme")),
        lambda d: d["shots"].__setitem__(0, shot(start_state="  ")),
        lambda d: d["shots"].__setitem__(0, shot(action_beats=[])),
        lambda d: d["shots"].__setitem__(0, shot(action_beats=[{"time_seconds": 1.0, "description": "x"}])),
        lambda d: d["shots"].__setitem__(0, shot(action_beats=[{"time_seconds": 0.0, "description": "x"}, {"time_seconds": -1, "description": "y"}])),
        lambda d: d["shots"].__setitem__(0, shot(action_beats=[{"time_seconds": 0.0, "description": "x"}, {"time_seconds": 5.0, "description": "y"}])),
        lambda d: d["shots"].__setitem__(0, shot(dialogue="   ")),
        lambda d: d["shots"].__setitem__(0, shot(extra_field=1)),
        lambda d: d["shots"].__setitem__(0, shot(continuity_in=123)),
    ],
)
def test_invalid_documents_rejected(mutate: Any) -> None:
    data = document([shot()])
    mutate(data)
    with pytest.raises(InputValidationError):
        parse_shot_plan(data)


def test_extract_json_handles_fenced_noise() -> None:
    payload = json.dumps(document([shot()]), ensure_ascii=False)
    parsed = director._extract_json(f"```json\n{payload}\n```")
    assert parse_shot_plan(parsed).shots[0].shot_id == "shot_001"


def test_scene_only_shot_allows_empty_subjects() -> None:
    parsed = parse_shot_plan(
        document(
            [
                shot(
                    subjects=[],
                    props=[],
                    start_state="The empty rooftop is still in the sunset.",
                )
            ]
        )
    )

    assert parsed.shots[0].subjects == []


def test_extract_json_rejects_garbage() -> None:
    with pytest.raises(InputValidationError):
        director._extract_json("definitely not json")


def test_extract_model_identity_and_tokens() -> None:
    stderr = """
workdir: D:\\
model: deepseek-v4-flash
provider: deepseek
approval: never
reasoning effort: high
session id: 019fea34-1cb2-7cc0-bf42-e3cea9217cca
"""
    assert "deepseek-v4-flash" in director._extract_model_identity(stderr)
    assert "high" in director._extract_model_identity(stderr)
    assert (
        director._extract_session_id(stderr)
        == "019fea34-1cb2-7cc0-bf42-e3cea9217cca"
    )
    assert director._extract_tokens("tokens used\n7,527") == 7527
    assert director._extract_tokens("nothing") is None


def test_run_director_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.md"
    script.write_text("第一幕：林夏在天台看城市。", encoding="utf-8")
    calls: list[bytes] = []

    class FakeCompleted:
        def __init__(self, stdout: bytes, stderr: bytes) -> None:
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command: list[str], *, input: bytes, **kwargs: Any) -> FakeCompleted:
        calls.append(input)
        if len(calls) == 1:
            return FakeCompleted(
                b"not json at all",
                b"model: deepseek-v4-flash\nsession id: s1\ntokens used\n100",
            )
        payload = json.dumps(document([shot()]), ensure_ascii=False)
        return FakeCompleted(
            payload.encode("utf-8"),
            b"model: deepseek-v4-flash\n"
            b"session id: 019fea34-1cb2-7cc0-bf42-e3cea9217cca\n"
            b"tokens used\n200",
        )

    monkeypatch.setattr(director.subprocess, "run", fake_run)
    out_dir = tmp_path / "run"

    result = director.run_director_for_script(script, out_dir=out_dir)

    assert len(calls) == 2
    assert result.tries == 2
    assert result.session_id == "019fea34-1cb2-7cc0-bf42-e3cea9217cca"
    assert result.tokens_used == 200
    assert "shot_001" in result.document.model_dump()["shots"][0]["shot_id"]
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "shot_plan.json").exists()
    assert (out_dir / "attempt_01_raw_output.txt").exists()
    assert (out_dir / "attempt_02_raw_output.txt").exists()


def test_run_director_fails_after_max_tries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.md"
    script.write_text("短剧本。", encoding="utf-8")

    class FakeCompleted:
        stdout = b"still not json"
        stderr = b"model: deepseek-v4-flash"

    monkeypatch.setattr(
        director.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted(),
    )

    with pytest.raises(InputValidationError):
        director.run_director_for_script(script, out_dir=tmp_path / "run")
    assert (tmp_path / "run" / "attempt_03_raw_output.txt").exists()
