"""Vision-assisted and stub classification of extracted episode frames."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import ReferenceRole

_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

AssetType = Literal["character", "background", "foreground", "prop", "style"]

_ASSET_TYPE_ALIASES: dict[str, str] = {
    "scene": "background",
    "背景": "background",
    "场景": "background",
    "人物": "character",
    "角色": "character",
    "道具": "prop",
    "前景": "foreground",
    "风格": "style",
}
_ROLE_ALIASES: dict[str, str] = {
    "identity": "identity_reference",
    "identity_reference": "identity_reference",
    "pose": "pose_reference",
    "pose_reference": "pose_reference",
    "expression": "expression_reference",
    "expression_reference": "expression_reference",
    "outfit": "outfit_reference",
    "outfit_reference": "outfit_reference",
    "scene": "scene_reference",
    "scene_reference": "scene_reference",
    "prop": "prop_reference",
    "prop_reference": "prop_reference",
    "style": "style_reference",
    "style_reference": "style_reference",
    "人物参考": "identity_reference",
    "身份参考": "identity_reference",
    "姿态参考": "pose_reference",
    "表情参考": "expression_reference",
    "服装参考": "outfit_reference",
    "场景参考": "scene_reference",
    "道具参考": "prop_reference",
    "风格参考": "style_reference",
}

_ROLE_BY_ASSET_TYPE: dict[str, tuple[ReferenceRole, ...]] = {
    "character": ("identity_reference",),
    "background": ("scene_reference",),
    "foreground": (),
    "prop": ("prop_reference",),
    "style": ("style_reference",),
}


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be non-empty")
    return cleaned


class FrameClassification(BaseModel):
    """Structured label for one extracted frame."""

    model_config = _STRICT_CONFIG

    asset_type: AssetType
    subject_or_scene_id: str
    reference_roles: list[ReferenceRole]
    view_angle: str | None = None
    pose: str | None = None
    expression: str | None = None
    outfit: str | None = None
    quality_notes: str | None = None
    character_names: list[str] = []

    @field_validator("subject_or_scene_id", mode="before")
    @classmethod
    def _subject(cls, value: object) -> object:
        return _clean_text(value, "subject_or_scene_id")

    @field_validator("reference_roles", mode="before")
    @classmethod
    def _roles(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("reference_roles must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or item not in ReferenceRole.__args__:
                raise ValueError(f"invalid reference_role {item!r}")
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    @field_validator("character_names", mode="before")
    @classmethod
    def _names(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("character_names must be a list")
        cleaned: list[str] = []
        for item in value:
            name = _clean_text(item, "character_names item")
            if name not in cleaned:
                cleaned.append(name)
        return cleaned

    @field_validator(
        "view_angle",
        "pose",
        "expression",
        "outfit",
        "quality_notes",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        return _clean_text(value, "optional metadata")

    @property
    def effective_reference_roles(self) -> list[ReferenceRole]:
        roles = list(_ROLE_BY_ASSET_TYPE.get(self.asset_type, ()))
        for role in self.reference_roles:
            if role not in roles:
                roles.append(role)
        return roles


def parse_classification(text: str) -> FrameClassification:
    """Parse a FrameClassification JSON object out of model output."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise InputValidationError("classification response contains no JSON object")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"invalid classification JSON: {exc}",
            actual=cleaned[start : end + 1][:300],
        ) from exc
    if isinstance(payload, dict):
        asset_type = payload.get("asset_type")
        if isinstance(asset_type, str):
            payload["asset_type"] = _ASSET_TYPE_ALIASES.get(
                asset_type.strip().casefold(),
                asset_type,
            )
        roles = payload.get("reference_roles")
        if isinstance(roles, list):
            normalized_roles: list[str] = []
            for role in roles:
                mapped = _ROLE_ALIASES.get(
                    str(role).strip().casefold(),
                    str(role).strip(),
                )
                if mapped in ReferenceRole.__args__:
                    normalized_roles.append(mapped)
            payload["reference_roles"] = normalized_roles
    try:
        return TypeAdapter(FrameClassification).validate_python(payload)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid frame classification: {exc}",
            actual=payload,
        ) from exc


class EpisodeClassifier(Protocol):
    def classify(self, image_path: Path) -> FrameClassification: ...


class StubEpisodeClassifier:
    """Deterministic offline classifier for tests and dry runs."""

    def __init__(
        self,
        *,
        asset_type: AssetType = "background",
        subject: str = "stub classification",
    ) -> None:
        self.asset_type = asset_type
        self.subject = subject
        self.last_request_id = "stub-classify"

    def classify(self, image_path: Path) -> FrameClassification:
        return FrameClassification(
            asset_type=self.asset_type,
            subject_or_scene_id=self.subject,
            reference_roles=list(_ROLE_BY_ASSET_TYPE.get(self.asset_type, ())),
            quality_notes=(
                "stub classifier; rerun with --executor dashscope for real labels"
            ),
        )


class DashScopeEpisodeClassifier:
    """Classify one frame with DashScope qwen-vl (paid, human-reviewable)."""

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

    def classify(self, image_path: Path) -> FrameClassification:
        from dashscope import MultiModalConversation

        prompt = (
            "You are extracting reference assets from one anime episode frame. "
            "Return ONLY a JSON object with exactly these keys:\n"
            "asset_type (one of: character, background, foreground, prop, style),\n"
            "subject_or_scene_id (the character name or scene description, in Chinese if the source is Chinese),\n"
            "reference_roles (array from: identity_reference, pose_reference, expression_reference, "
            "outfit_reference, scene_reference, prop_reference, style_reference),\n"
            "view_angle, pose, expression, outfit (strings or null),\n"
            "quality_notes (strings or null; note blur, motion, cropping or low reference value),\n"
            "character_names (array of visible character names).\n"
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
        return parse_classification(text)
