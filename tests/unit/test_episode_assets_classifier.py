"""Unit tests for episode frame classification parsing and stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_remix.errors import InputValidationError
from anime_remix.services.episode_assets.classifier import (
    DashScopeEpisodeClassifier,
    FrameClassification,
    StubEpisodeClassifier,
    parse_classification,
)


def test_parse_classification_accepts_markdown_wrapped_json() -> None:
    text = (
        "```json\n"
        '{"asset_type": "character", '
        '"subject_or_scene_id": "Mira", '
        '"reference_roles": ["identity_reference", "outfit_reference"], '
        '"view_angle": "three-quarter front", "pose": "standing", '
        '"expression": "calm", "outfit": "black coat", '
        '"quality_notes": null, "character_names": ["Mira"]}\n'
        "```"
    )
    parsed = parse_classification(text)
    assert parsed.asset_type == "character"
    assert parsed.subject_or_scene_id == "Mira"
    assert parsed.effective_reference_roles == [
        "identity_reference",
        "outfit_reference",
    ]


def test_parse_classification_rejects_bad_json() -> None:
    with pytest.raises(InputValidationError, match="JSON"):
        parse_classification("not json at all")


def test_parse_classification_rejects_unknown_asset_type() -> None:
    text = (
        '{"asset_type": "logo", "subject_or_scene_id": "x", '
        '"reference_roles": []}'
    )
    with pytest.raises(InputValidationError, match="frame classification"):
        parse_classification(text)


def test_parse_classification_normalizes_scene_and_chinese_roles() -> None:
    text = (
        '{"asset_type": "scene", "subject_or_scene_id": "办公室", '
        '"reference_roles": ["场景参考", "道具参考"], '
        '"view_angle": "高角度", "quality_notes": "清晰"}'
    )
    parsed = parse_classification(text)
    assert parsed.asset_type == "background"
    assert parsed.reference_roles == ["scene_reference", "prop_reference"]


def test_stub_classifier_returns_deterministic_label() -> None:
    stub = StubEpisodeClassifier()
    result = stub.classify(Path("C:/frame.png"))
    assert result.asset_type == "background"
    assert result.reference_roles == ["scene_reference"]
    assert "stub classifier" in result.quality_notes


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.request_id = "fake-request"

        class _Content:
            def __init__(self, text: str) -> None:
                self.content = [{"text": text}]

        class _Choice:
            def __init__(self, text: str) -> None:
                self.message = _Content(text)

        class _Output:
            def __init__(self, text: str) -> None:
                self.choices = [_Choice(text)]

        self.output = _Output(text)


def test_dashscope_classifier_uses_injected_call_and_parses_label() -> None:
    payload = (
        '{"asset_type": "background", '
        '"subject_or_scene_id": "observatory", '
        '"reference_roles": ["scene_reference"], '
        '"view_angle": null, "pose": null, "expression": null, '
        '"outfit": null, "quality_notes": "clean establishing shot", '
        '"character_names": []}'
    )
    captured: list[dict] = []

    def fake_call(**kwargs):
        captured.append(kwargs)
        return _FakeResponse(payload)

    classifier = DashScopeEpisodeClassifier(call_fn=fake_call)
    result = classifier.classify(Path("C:/frame.png"))
    assert isinstance(result, FrameClassification)
    assert result.asset_type == "background"
    assert result.subject_or_scene_id == "observatory"
    assert classifier.last_request_id == "fake-request"
    assert captured[0]["model"] == "qwen-vl-max"
