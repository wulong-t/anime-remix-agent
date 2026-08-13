"""Unit tests for episode frame cropping."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from anime_remix.errors import InputValidationError
from anime_remix.services.episode_assets.cropper import (
    BoundingBox,
    DashScopeCropper,
    StubCropper,
    apply_crop,
    parse_crop_regions,
)


def _png(colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (200, 100), colour).save(output, format="PNG")
    return output.getvalue()


def test_parse_crop_regions_accepts_markdown_json() -> None:
    text = (
        "```json\n"
        '{"boxes": [{"x0": 0.2, "y0": 0.3, "x1": 0.5, "y1": 0.7, '
        '"label": "face"}]}\n'
        "```"
    )
    parsed = parse_crop_regions(text)
    assert len(parsed.boxes) == 1
    assert parsed.boxes[0].label == "face"
    assert parsed.boxes[0].x0 == 0.2


def test_parse_crop_regions_rejects_empty_and_bad_json() -> None:
    with pytest.raises(InputValidationError, match="no JSON"):
        parse_crop_regions("nothing")
    with pytest.raises(InputValidationError, match="JSON"):
        parse_crop_regions('{"boxes": [broken]}')


def test_parse_crop_regions_repairs_bare_number_boxes() -> None:
    text = '{"boxes": [{"x0": 0.14, 0.17, 0.26, 0.77}]}'
    parsed = parse_crop_regions(text)
    assert len(parsed.boxes) == 1
    assert parsed.boxes[0].x0 == 0.14
    assert parsed.boxes[0].y0 == 0.17
    assert parsed.boxes[0].x1 == 0.26
    assert parsed.boxes[0].y1 == 0.77


def test_stub_cropper_returns_deterministic_center_box() -> None:
    stub = StubCropper()
    result = stub.crop_regions(Path("C:/frame.png"))
    assert len(result.boxes) == 1
    assert result.boxes[0].label == "character"


def test_apply_crop_respects_padding_and_bounds(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(_png())
    out_path = tmp_path / "crop.png"
    box = BoundingBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75, label="face")
    width, height = apply_crop(image_path, box, out_path, pad=0.05)
    assert (width, height) == (120, 60)
    with Image.open(out_path) as image:
        assert image.size == (120, 60)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.request_id = "fake-crop"

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


def test_dashscope_cropper_uses_injected_call() -> None:
    payload = (
        '{"boxes": [{"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.5, '
        '"label": "character"}]}'
    )
    captured: list[dict] = []

    def fake_call(**kwargs):
        captured.append(kwargs)
        return _FakeResponse(payload)

    cropper = DashScopeCropper(call_fn=fake_call)
    result = cropper.crop_regions(Path("C:/frame.png"))
    assert len(result.boxes) == 1
    assert cropper.last_request_id == "fake-crop"
    assert captured[0]["model"] == "qwen-vl-max"
