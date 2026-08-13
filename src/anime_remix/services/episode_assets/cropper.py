"""Face/region cropping for extracted episode frames (optional stage)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from anime_remix.errors import InputValidationError

_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
_BARE_BOX_PATTERN = re.compile(
    r'\{"x0":\s*([0-9]+(?:\.[0-9]+)?),\s*([0-9]+(?:\.[0-9]+)?),\s*'
    r"([0-9]+(?:\.[0-9]+)?),\s*([0-9]+(?:\.[0-9]+)?)\}"
)


def _coordinate(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within 0..1")
    return number


class BoundingBox(BaseModel):
    model_config = _STRICT_CONFIG

    x0: float
    y0: float
    x1: float
    y1: float
    label: str = "region"

    @field_validator("x0", "y0", "x1", "y1", mode="before")
    @classmethod
    def _coords(cls, value: object, info) -> object:
        return _coordinate(value, info.field_name)

    @field_validator("label", mode="before")
    @classmethod
    def _label(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("label must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("box must have positive width and height")
        return self


class CropRegions(BaseModel):
    model_config = _STRICT_CONFIG

    boxes: list[BoundingBox]


def parse_crop_regions(text: str) -> CropRegions:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise InputValidationError("crop response contains no JSON object")
    raw = cleaned[start : end + 1]
    repaired = _BARE_BOX_PATTERN.sub(
        lambda match: (
            f'{{"x0": {match.group(1)}, "y0": {match.group(2)}, '
            f'"x1": {match.group(3)}, "y1": {match.group(4)}}}'
        ),
        raw,
    )
    try:
        payload = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"invalid crop JSON: {exc}",
            actual=cleaned[start : end + 1][:300],
        ) from exc
    try:
        return TypeAdapter(CropRegions).validate_python(payload)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid crop regions: {exc}",
            actual=payload,
        ) from exc


class EpisodeCropper(Protocol):
    def crop_regions(self, image_path: Path) -> CropRegions: ...


class StubCropper:
    """Deterministic center crop used by tests and offline dry runs."""

    def __init__(
        self,
        *,
        box: tuple[float, float, float, float] = (0.25, 0.20, 0.75, 0.80),
        label: str = "character",
    ) -> None:
        self.box = BoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3], label=label)
        self.last_request_id = "stub-crop"

    def crop_regions(self, image_path: Path) -> CropRegions:
        return CropRegions(boxes=[self.box])


class DashScopeCropper:
    """Locate faces/characters with DashScope qwen-vl (paid)."""

    provider = "dashscope"

    def __init__(
        self,
        *,
        model: str = "qwen-vl-max",
        call_fn=None,
    ) -> None:
        self.model = model
        self._call_fn = call_fn
        self.last_request_id: str | None = None

    def crop_regions(self, image_path: Path) -> CropRegions:
        from dashscope import MultiModalConversation

        prompt = (
            "You are extracting reference assets from one anime episode frame. "
            "Return ONLY a JSON object with a 'boxes' array. For every visible "
            "character or face, add {\"x0\", \"y0\", \"x1\", \"y1\"} normalized "
            "0..1 bounding-box coordinates and a label ('face' or 'character'). "
            "Use exactly this shape per box: "
            "{\"x0\": 0.14, \"y0\": 0.17, \"x1\": 0.26, \"y1\": 0.77, "
            "\"label\": \"character\"}. Every coordinate key must be quoted "
            "and each box must contain all four named coordinates plus the label. "
            "Use tight boxes around the head/face for 'face' and the full visible "
            "character for 'character'. If nothing is visible, return {\"boxes\": []}."
            "No markdown, no extra text."
        )
        call = (
            self._call_fn
            if self._call_fn is not None
            else MultiModalConversation.call
        )
        response = call(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": image_path.resolve().as_uri()},
                        {"text": prompt},
                    ],
                }
            ],
        )
        self.last_request_id = str(getattr(response, "request_id", "") or "")
        content = response.output.choices[0].message.content
        text = (
            "".join(item.get("text", "") for item in content)
            if isinstance(content, list)
            else str(content)
        )
        return parse_crop_regions(text)


def apply_crop(
    image_path: Path,
    box: BoundingBox,
    out_path: Path,
    *,
    pad: float = 0.05,
) -> tuple[int, int]:
    """Crop one normalized box with a small padding margin."""

    with Image.open(image_path) as image:
        width, height = image.size
        x0 = max(0, int((box.x0 - pad) * width))
        y0 = max(0, int((box.y0 - pad) * height))
        x1 = min(width, int((box.x1 + pad) * width))
        y1 = min(height, int((box.y1 + pad) * height))
        if x1 <= x0 or y1 <= y0:
            raise InputValidationError("crop box collapses to empty after padding")
        cropped = image.crop((x0, y0, x1, y1))
        cropped.save(out_path, format="PNG")
        return cropped.size
