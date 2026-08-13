"""Renderer Adapter boundary (Phase 3: frozen adapter + DashScope adapter).

Round 7 freeze: RenderIntent is model-agnostic; the Adapter compiles it into
a model-specific ModelRenderRequest.  Two adapters are implemented here:

- ``QwenImageEditAdapter``: Qwen-Image-Edit-2511 on existing remote paths
  (WHO/HOW visual slots + textualized scene per the I4..I7 evidence).
  Frozen; do not change.
- ``QwenImage30Adapter``: qwen-image-3.0 (DashScope standard, non-pro),
  identity-safe single-WHO/text-HOW by default with an explicit pose-only
  visual-HOW mode, a direct 1-3 reference scene-synthesis operation, plus
  textual scene semantics and explicit DashScope generation parameters.

The stub executor keeps the offline pipeline honest.
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from anime_remix.errors import EnvironmentCapabilityError, InputValidationError
from anime_remix.services.execution.imaging import image_to_png_bytes


def _conditions_with_role(
    conditions: list[dict], roles: set[str]
) -> list[dict]:
    return [c for c in conditions if c["role"] in roles]


_SEED_MAX = 2**31 - 1
_SIZE_PATTERN = re.compile(r"^[1-9][0-9]*\*[1-9][0-9]*$")
_MIN_SIZE_PIXELS = 512 * 512
_MAX_SIZE_PIXELS = 2048 * 2048
_MAX_ASPECT_NUM = 8
_NEGATIVE_PROMPT = (
    "text, caption, logo, watermark, signature, label, "
    "blurry, low quality, jpeg artifacts, "
    "deformed hands, extra fingers, fused fingers, missing limbs, "
    "bad anatomy, distorted face, extra eyes, cropped head"
)


def _readable_label(value: str) -> str:
    """Turn internal identifiers into plain prompt text."""

    return value.replace("_", " ").replace(".", " ").strip()


def _optional_prompt_value(
    values: dict,
    key: str,
    *,
    fallback: str,
) -> str:
    value = values.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _qwen30_character_prompt(
    *,
    intent: dict,
    keyframe_state: dict,
    scene_description: str,
    visual_how: bool,
    continuity_reference: bool,
    context_reference_role: str | None,
) -> str:
    """Compile a reference-first prompt containing only uncovered changes."""

    pose = _optional_prompt_value(
        keyframe_state,
        "subject_pose",
        fallback=str(intent["subject_pose"]),
    )
    expression = _optional_prompt_value(
        keyframe_state,
        "expression",
        fallback="preserve the HOW expression",
    )
    gaze = _optional_prompt_value(
        keyframe_state,
        "gaze",
        fallback="preserve the HOW gaze and eye state",
    )
    motion = _optional_prompt_value(
        keyframe_state,
        "motion_from_previous",
        fallback="establish the target endpoint state",
    )

    locks: list[str] = []
    raw_locks = keyframe_state.get("locked_attributes", [])
    if isinstance(raw_locks, list):
        for value in raw_locks:
            if isinstance(value, str):
                readable = _readable_label(value)
                if readable and readable not in locks:
                    locks.append(readable)
    character_locks = keyframe_state.get("character_locks")
    if isinstance(character_locks, dict):
        if character_locks.get("identity") is True and "identity" not in locks:
            locks.append("identity")
        if (
            character_locks.get("hairstyle") is True
            and "hairstyle" not in locks
        ):
            locks.append("hairstyle")
        costume = character_locks.get("costume_variant")
        if isinstance(costume, str) and costume.strip():
            costume_lock = f"costume ({_readable_label(costume)})"
            locks = [value for value in locks if value != "costume"]
            if costume_lock not in locks:
                locks.append(costume_lock)
    lock_phrases: list[str] = []
    for lock in locks:
        if lock == "identity":
            phrase = "identity and facial features"
        elif lock == "hairstyle":
            phrase = "hairstyle and hair color"
        elif lock == "costume" or lock.startswith("costume ("):
            phrase = "clothing and accessories"
        else:
            phrase = lock
        if phrase not in lock_phrases:
            lock_phrases.append(phrase)
    locked_text = (
        ", ".join(lock_phrases)
        if lock_phrases
        else "identity and appearance"
    )

    fallback_values = {
        "composition": _optional_prompt_value(
            keyframe_state,
            "composition",
            fallback=f"{intent['shot_scale']} shot",
        ),
        "camera": _readable_label(
            _optional_prompt_value(
                keyframe_state,
                "camera",
                fallback=str(intent["camera_view"]),
            )
        ),
        "background_state": _optional_prompt_value(
            keyframe_state,
            "background_state",
            fallback=scene_description,
        ),
        "foreground_state": _optional_prompt_value(
            keyframe_state,
            "foreground_state",
            fallback="no additional foreground constraint",
        ),
        "prop_state": _optional_prompt_value(
            keyframe_state,
            "prop_state",
            fallback="no additional prop constraint",
        ),
    }
    fallback_labels = {
        "composition": "Composition",
        "camera": "Camera",
        "background_state": "Background",
        "foreground_state": "Foreground",
        "prop_state": "Props",
    }
    policy = keyframe_state.get("prompt_policy")
    requested_fallbacks: object = None
    if isinstance(policy, dict):
        requested_fallbacks = policy.get("text_fallback_fields")
    if requested_fallbacks is None:
        context_covers_frame = context_reference_role in {
            "scene",
            "source_frame",
        }
        fallback_fields = (
            []
            if continuity_reference or context_covers_frame
            else list(fallback_values)
        )
    else:
        if not isinstance(requested_fallbacks, list) or any(
            not isinstance(field, str) for field in requested_fallbacks
        ):
            raise InputValidationError(
                "prompt_policy.text_fallback_fields must be a string list"
            )
        unknown = sorted(set(requested_fallbacks) - set(fallback_values))
        if unknown:
            raise InputValidationError(
                f"unknown text fallback fields: {unknown}"
            )
        fallback_fields = list(dict.fromkeys(requested_fallbacks))

    if visual_how:
        reference_instructions = (
            "Image 1 is the WHO identity and appearance reference. "
            "Image 2 is a pose-only HOW control reference. Use Image 2 only "
            "for the subject's exact body pose, head angle, gaze and eye "
            "state, facial expression, arm and hand positions, and finger "
            "arrangement. Do not copy any identity, face, hairstyle, hair "
            "accessories, clothing, texture, color, or background from "
            "Image 2. "
        )
    elif continuity_reference:
        reference_instructions = (
            "Image 1 is the canonical WHO identity and appearance reference. "
            "Image 2 is the previous approved keyframe from the same shot. "
            "Use Image 2 pixels as the authority for every visible attribute "
            "that is not explicitly changed below, including scene, camera, "
            "composition, lighting, palette, linework, texture, and props. "
            "If Image 1 and Image 2 differ in identity or appearance, Image "
            "1 is authoritative. "
        )
    elif context_reference_role in {"scene", "source_frame"}:
        reference_instructions = (
            "Image 1 is the canonical WHO identity and appearance reference. "
            "Image 2 is the full-frame visual reference. Use Image 2 pixels "
            "as the authority for scene, composition, camera, lighting, "
            "palette, linework, texture, visual style, and visible props. "
            "Use Image 1 as the authority for character identity and "
            "appearance. "
        )
    elif context_reference_role == "style":
        reference_instructions = (
            "Image 1 is the canonical WHO identity and appearance reference. "
            "Image 2 is the visual-style reference. Match its palette, "
            "linework, texture, shading, and rendering treatment without "
            "copying its character identity. "
        )
    else:
        reference_instructions = (
            "Image 1 is the only visual character reference. Transform the "
            "same character from Image 1 into the explicit pose and state "
            "described below. Do not replace or redesign that character's "
            "identity, face, hairstyle, hair color, clothing, or accessories. "
        )
    parts = [
        (
            "Create one final endpoint keyframe by editing the supplied "
            "references."
        ),
        reference_instructions.strip(),
        f"Preserve from Image 1 exactly as shown: {locked_text}.",
        (
            "Do not redesign, restyle, or verbally reinterpret attributes "
            "already covered by a selected reference."
        ),
        f"Required motion or state change: {motion}.",
        f"Target pose: {pose}.",
        f"Target expression: {expression}.",
        f"Target gaze and eye state: {gaze}.",
    ]
    if fallback_fields:
        fallback_text = "; ".join(
            f"{fallback_labels[field]}: {fallback_values[field]}"
            for field in fallback_fields
        )
        parts.append(
            "No selected reference covers the following target facts, so "
            f"apply only these textual fallbacks: {fallback_text}."
        )
    parts.extend(
        [
            (
                "Keep every other visible attribute unchanged from the "
                "selected references."
            ),
            (
                "Do not add text, captions, logos, signatures, labels, or "
                "watermarks."
            ),
        ]
    )
    return " ".join(parts)


def _qwen30_first_frame_prompt(
    *,
    intent: dict,
    conditions: list[dict],
) -> str:
    """Compile one staged first-frame instruction without visual restatement."""

    stage_operation = intent.get("stage_operation")
    instruction = intent.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise InputValidationError(
            "first_frame_fusion requires a non-empty stage instruction"
        )
    attributes = intent.get("reference_attributes", [])
    if not isinstance(attributes, list) or any(
        not isinstance(item, str) or not item.strip() for item in attributes
    ):
        raise InputValidationError(
            "first_frame_fusion reference_attributes must be a string list"
        )
    text_fallbacks = intent.get("text_fallbacks", {})
    if not isinstance(text_fallbacks, dict) or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in text_fallbacks.items()
    ):
        raise InputValidationError(
            "first_frame_fusion text_fallbacks must be a string mapping"
        )
    roles = [condition.get("role") for condition in conditions]
    if stage_operation == "synthesize_base":
        if len(conditions) > 2:
            raise InputValidationError(
                "synthesize_base accepts at most two visual references"
            )
        if not conditions:
            reference_instruction = (
                "No visual reference exists for this base canvas. Use only "
                "the explicitly listed fallback facts; do not invent named "
                "characters, props, text, logos, or additional design motifs."
            )
        else:
            descriptions = []
            for index, role in enumerate(roles, start=1):
                authority = {
                    "scene": "scene identity, spatial structure and lighting",
                    "style": "palette, linework, shading and texture",
                    "source_frame": "the complete visible frame",
                }.get(str(role), "only the referenced visual attributes")
                descriptions.append(
                    f"Image {index} is authoritative for {authority}."
                )
            reference_instruction = " ".join(descriptions)
    elif stage_operation == "fuse_component":
        if len(conditions) != 2 or roles[0] != "source_frame":
            raise InputValidationError(
                "fuse_component requires current canvas in slot 1 and one "
                "component reference in slot 2"
            )
        attribute_text = ", ".join(attributes) or "referenced appearance"
        reference_instruction = (
            "Image 1 is the current approved frame canvas and is "
            "authoritative for every unchanged pixel and visual fact. Image "
            f"2 is authoritative only for this component's {attribute_text}. "
            "Do not copy any Image 2 fact outside that explicit authority; "
            "in particular, do not import its unrelated background, camera, "
            "lighting, palette, linework, or texture."
        )
        if "final-frame canvas placement" in attributes:
            reference_instruction += (
                " Image 2's approved component placement, facing direction, "
                "functional prop axis and external-attachment geometry are hard "
                "constraints. Do not mirror, rotate, translate, rescale or "
                "recompose that atomic group; integrate it at the recorded anchors."
            )
    elif stage_operation == "synthesize_component":
        if not 1 <= len(conditions) <= 2:
            raise InputValidationError(
                "synthesize_component requires one or two visual references"
            )
        if not any(role in {"identity", "prop"} for role in roles):
            raise InputValidationError(
                "synthesize_component requires WHO or prop visual authority"
            )
        descriptions = []
        for index, role in enumerate(roles, start=1):
            authority = {
                "identity": (
                    "the named character's identity, face, hair, body proportions, "
                    "canonical clothing and native visual style"
                ),
                "pose": (
                    "action, pose and expression geometry only; ignore its identity, "
                    "clothing, background and visual style"
                ),
                "prop": "the named prop's identity, shape, material and native style",
            }.get(str(role), "only its explicitly assigned component function")
            descriptions.append(f"Image {index} is authoritative for {authority}.")
        reference_instruction = " ".join(descriptions) + (
            " Render only the planned subject or interacting subject group as one "
            "coherent production-quality component plate. Use a plain unobtrusive "
            "background and do not import any reference background, labels or layout."
        )
    elif stage_operation == "apply_text_delta":
        if len(conditions) != 1 or roles[0] != "source_frame":
            raise InputValidationError(
                "apply_text_delta requires only the current canvas"
            )
        reference_instruction = (
            "Image 1 is the current approved frame canvas. Preserve "
            "every visible fact except the explicitly requested delta."
        )
    else:
        raise InputValidationError(
            f"unsupported first-frame stage operation {stage_operation!r}"
        )

    opening = (
        "Construct one reference-grounded component plate for later frame assembly."
        if stage_operation == "synthesize_component"
        else "Construct one reference-grounded anchor frame for this shot."
    )
    parts = [
        opening,
        reference_instruction,
        f"Required operation or spatial/action delta: {instruction.strip()}",
    ]
    if text_fallbacks:
        fallback_text = "; ".join(
            f"{_readable_label(key)}: {value.strip()}"
            for key, value in text_fallbacks.items()
        )
        parts.append(
            "No selected image covers these facts; use only these minimal "
            f"text fallbacks: {fallback_text}."
        )
    parts.extend(
        [
            (
                "Do not verbally reinterpret or redesign attributes already "
                "covered by a selected image."
            ),
            "Do not add text, captions, logos, signatures, labels, or watermarks.",
        ]
    )
    return " ".join(parts)


class Adapter(Protocol):
    adapter_id: str
    model_id: str
    revision: str

    def compile(
        self,
        *,
        operation: str,
        intent: dict,
        keyframe_state: dict,
        scene_description: str,
        conditions: list[dict],
    ) -> dict:
        ...


class QwenImageEditAdapter:
    """Qwen-Image-Edit-2511 adapter (WHO + HOW visual slots, scene as text)."""

    adapter_id = "qwen-edit-2511-adapter-v1"
    model_id = "Qwen/Qwen-Image-Edit-2511"
    revision = "main"

    def __init__(self, *, seed: int = 0, steps: int = 40) -> None:
        self._seed = seed
        self._steps = steps

    def compile(
        self,
        *,
        operation: str,
        intent: dict,
        keyframe_state: dict,
        scene_description: str,
        conditions: list[dict],
    ) -> dict:
        if operation == "character_synthesis":
            identity = _conditions_with_role(conditions, {"identity"})
            how = _conditions_with_role(conditions, {"pose", "expression"})
            if not identity:
                raise InputValidationError(
                    "character_synthesis requires an identity condition"
                )
            if not how:
                raise InputValidationError(
                    "character_synthesis requires a pose/expression condition"
                )
            slots = [
                {"slot": 1, "condition_ref": identity[0]["condition_id"]},
                {"slot": 2, "condition_ref": how[0]["condition_id"]},
            ]
            prompt = (
                "Keep the character identity from image 1 (WHO) and adopt "
                "the pose and expression from image 2 (HOW). "
                f"Scene: {scene_description}. "
                f"Pose: {intent['subject_pose']}. "
                f"Expression: {keyframe_state['expression']}. "
                f"Shot scale: {intent['shot_scale']}, "
                f"camera: {intent['camera_view']}. "
                "Do not add text, logos, or watermarks."
            )
        elif operation == "local_inpaint":
            slots = []
            prompt = (
                "Repair only the masked region: fix seams, edges, contact "
                "areas and shadows around the composited character. "
                "Preserve everything outside the mask exactly, including "
                "identity, costume, hair, scene and composition. "
                "Do not add text, logos, or watermarks."
            )
        else:
            raise InputValidationError(
                f"unsupported adapter operation {operation!r}"
            )
        return {
            "adapter_id": self.adapter_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "conditions": slots,
            "prompt": prompt,
            "parameters": {
                "seed": self._seed,
                "num_inference_steps": self._steps,
                "true_cfg_scale": 4.0,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            },
        }


class QwenImage30Adapter:
    """qwen-image-3.0 reference-first endpoint and fusion adapter.

    Image 1 is one canonical identity reference.  A first endpoint may use
    one full-frame scene/source/style reference as Image 2; a last endpoint
    uses the previous approved first frame as Image 2.  Prompt text describes
    action/state changes and only those visual fields not covered by a
    selected reference.  It never restates a full visual description.

    Explicit visual-HOW mode is retained for the standalone experiment and
    is restricted to identity-stripped pose controls.  It is forbidden in the
    first/last runner because that runner reserves Image 2 for visual context
    or previous-frame continuity.  The staged operations never exceed two
    primary references.

    ``direct_scene_synthesis`` is the reference-direct endpoint: it accepts
    1-3 reference images and composes the whole frame in a single model call.
    By default it requires exactly one identity reference plus up to two
    prop/scene/style references; with ``intent["scene_only"] = True`` it
    requires zero identity references (pure scene establishment).  The caller
    supplies the complete ultra-detailed instruction (per-element state,
    placement, pose, expression, gaze and scene facts) in
    ``intent["instruction"]``; the adapter maps the references to slots and
    freezes model and generation parameters.  This is the default endpoint
    for "generate the elements into the frame" requests: no component
    plate is produced and nothing is composited afterwards.

    ``first_frame_fusion`` is the preferred first-frame workflow.  It accepts
    zero-to-two inputs for a base stage and then current-canvas + component
    reference for each sequential fusion stage.  Covered visual facts remain
    in images; text is limited to action/spatial deltas and uncovered facts.

    ``revision`` is ``provider-managed-alias`` because DashScope does not pin
    an immutable model snapshot.  ``seed`` is 0..2147483647 inclusive and
    ``size`` must be a ``W*H`` string within the qwen-image-3.0 edit
    constraints (total pixels 512*512..2048*2048, aspect 1:8..8:1).
    """

    adapter_id = "qwen-image-30-adapter-v6-reference-first"
    model_id = "qwen-image-3.0"
    revision = "provider-managed-alias"
    supports_previous_keyframe = True
    supports_first_frame_fusion = True

    def __init__(
        self,
        *,
        seed: int = 0,
        size: str = "1280*720",
        visual_how: bool = False,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InputValidationError(
                f"invalid seed {seed!r}: must be an integer"
            )
        if not 0 <= seed <= _SEED_MAX:
            raise InputValidationError(
                f"invalid seed {seed}: must be in 0..{_SEED_MAX}"
            )
        self._seed = seed
        self._size = self._normalize_size(size)
        if not isinstance(visual_how, bool):
            raise InputValidationError("visual_how must be a boolean")
        self._visual_how = visual_how
        self.uses_visual_how = visual_how
        self.adapter_id = (
            "qwen-image-30-adapter-v4-visual-how"
            if visual_how
            else "qwen-image-30-adapter-v6-reference-first"
        )

    @staticmethod
    def _endpoint_conditions(
        conditions: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        identity = _conditions_with_role(conditions, {"identity"})
        how = _conditions_with_role(conditions, {"pose", "expression"})
        continuity = [
            condition
            for condition in conditions
            if condition["role"] == "source_frame"
            and condition["condition_id"] == "cond_previous_keyframe"
        ]
        context = [
            condition
            for condition in conditions
            if condition["role"] in {"scene", "style", "source_frame"}
            and condition not in continuity
        ]
        return identity, how, continuity, context

    def shot_runner_static_condition_ids(
        self,
        conditions: list[dict],
        *,
        endpoint_role: str,
    ) -> tuple[str, ...]:
        """Return exact static assets the first/last runner may read."""

        identity, _how, _continuity, context = self._endpoint_conditions(
            conditions
        )
        if len(identity) != 1:
            raise InputValidationError(
                "first/last reference-first execution requires exactly one "
                "canonical identity condition"
            )
        if endpoint_role == "first":
            if len(context) > 1:
                raise InputValidationError(
                    "only one full-frame scene/style reference can accompany "
                    "the WHO image; fuse or stage additional references"
                )
            selected = [identity[0]]
            if context:
                selected.append(context[0])
            return tuple(item["condition_id"] for item in selected)
        if endpoint_role == "last":
            return (identity[0]["condition_id"],)
        raise InputValidationError(
            f"unsupported endpoint_role {endpoint_role!r}"
        )

    @staticmethod
    def _normalize_size(size: str) -> str:
        if not isinstance(size, str):
            raise InputValidationError(
                f"invalid size {size!r}: expected a 'W*H' string like "
                "'1280*720'"
            )
        normalized = size.strip()
        if not _SIZE_PATTERN.match(normalized):
            raise InputValidationError(
                f"invalid size {normalized!r}: expected 'W*H' with positive "
                "dimensions like '1280*720'"
            )
        width, height = (int(part) for part in normalized.split("*"))
        if not _MIN_SIZE_PIXELS <= width * height <= _MAX_SIZE_PIXELS:
            raise InputValidationError(
                f"invalid size {normalized!r}: total pixels must be within "
                f"{_MIN_SIZE_PIXELS}..{_MAX_SIZE_PIXELS}"
            )
        if not (
            width * _MAX_ASPECT_NUM >= height
            and height * _MAX_ASPECT_NUM >= width
        ):
            raise InputValidationError(
                f"invalid size {normalized!r}: aspect ratio must be within "
                "1:8 and 8:1"
            )
        return normalized

    def compile(
        self,
        *,
        operation: str,
        intent: dict,
        keyframe_state: dict,
        scene_description: str,
        conditions: list[dict],
    ) -> dict:
        if operation == "first_frame_fusion":
            if self._visual_how:
                raise InputValidationError(
                    "visual-HOW mode cannot execute staged first-frame fusion"
                )
            if len(conditions) > 2:
                raise InputValidationError(
                    "first_frame_fusion is limited to two primary visual "
                    "references per stage"
                )
            slots = [
                {"slot": index, "condition_ref": item["condition_id"]}
                for index, item in enumerate(conditions, start=1)
            ]
            prompt = _qwen30_first_frame_prompt(
                intent=intent,
                conditions=conditions,
            )
            stage_operation = intent.get("stage_operation")
            role_suffix = (
                "-" + "-".join(str(item.get("role")) for item in conditions)
                if stage_operation == "synthesize_component"
                else ""
            )
            adapter_id = (
                "qwen-image-30-adapter-v7-first-frame-"
                f"{stage_operation}{role_suffix}"
            )
        elif operation == "character_synthesis":
            identity, how, continuity, context = self._endpoint_conditions(
                conditions
            )
            if not identity:
                raise InputValidationError(
                    "character_synthesis requires an identity condition"
                )
            if self._visual_how and not how:
                raise InputValidationError(
                    "visual-HOW character_synthesis requires a "
                    "pose/expression condition"
                )
            if self._visual_how and continuity:
                raise InputValidationError(
                    "visual-HOW and previous-keyframe continuity cannot be "
                    "combined: qwen-image-3.0 is limited to two primary "
                    "visual references"
                )
            if self._visual_how and context:
                raise InputValidationError(
                    "visual-HOW and scene/style reference cannot be combined: "
                    "qwen-image-3.0 is limited to two primary visual "
                    "references"
                )
            if not continuity and len(context) > 1:
                raise InputValidationError(
                    "only one full-frame scene/style reference can accompany "
                    "the WHO image; fuse or stage additional references"
                )
            slots = [
                {"slot": 1, "condition_ref": identity[0]["condition_id"]}
            ]
            if self._visual_how:
                slots.append(
                    {"slot": 2, "condition_ref": how[0]["condition_id"]}
                )
            elif continuity:
                slots.append(
                    {
                        "slot": 2,
                        "condition_ref": continuity[0]["condition_id"],
                    }
                )
            elif context:
                slots.append(
                    {"slot": 2, "condition_ref": context[0]["condition_id"]}
                )
            prompt = _qwen30_character_prompt(
                intent=intent,
                keyframe_state=keyframe_state,
                scene_description=scene_description,
                visual_how=self._visual_how,
                continuity_reference=bool(continuity),
                context_reference_role=(
                    str(context[0]["role"])
                    if context and not continuity
                    else None
                ),
            )
            adapter_id = (
                "qwen-image-30-adapter-v6-continuity-action-delta"
                if continuity
                else (
                    "qwen-image-30-adapter-v6-reference-first-context"
                    if context
                    else self.adapter_id
                )
            )
        elif operation == "direct_scene_synthesis":
            if not 1 <= len(conditions) <= 3:
                raise InputValidationError(
                    "direct_scene_synthesis accepts 1-3 reference images; "
                    f"got {len(conditions)}"
                )
            identity = _conditions_with_role(conditions, {"identity"})
            scene_only = intent.get("scene_only") is True
            if scene_only:
                if identity:
                    raise InputValidationError(
                        "scene_only direct_scene_synthesis must not include "
                        "an identity reference"
                    )
            elif len(identity) != 1:
                raise InputValidationError(
                    "direct_scene_synthesis requires exactly one identity "
                    "reference unless intent['scene_only'] is true"
                )
            instruction = intent.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise InputValidationError(
                    "direct_scene_synthesis requires intent['instruction'] "
                    "with the complete ultra-detailed frame instruction"
                )
            slots = [
                {"slot": index, "condition_ref": item["condition_id"]}
                for index, item in enumerate(conditions, start=1)
            ]
            prompt = instruction.strip()
            role_suffix = "-".join(
                str(item.get("role")) for item in conditions
            )
            adapter_id = (
                "qwen-image-30-adapter-v7-direct-frame-" + role_suffix
            )
        elif operation == "local_inpaint":
            slots = []
            prompt = (
                "Repair only the masked region: fix seams, edges, contact "
                "areas and shadows around the composited character. "
                "Preserve everything outside the mask exactly, including "
                "identity, costume, hair, scene and composition. "
                "Do not add text, logos, or watermarks."
            )
            adapter_id = self.adapter_id
        else:
            raise InputValidationError(
                f"unsupported adapter operation {operation!r}"
            )
        return {
            "adapter_id": adapter_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "conditions": slots,
            "prompt": prompt,
            "parameters": {
                "seed": self._seed,
                "n": 1,
                "size": self._size,
                "prompt_extend": False,
                "prompt_extend_mode": "direct",
                "watermark": False,
                "negative_prompt": _NEGATIVE_PROMPT,
            },
        }


# Compatibility alias: the standard (non-pro) model replaced the previous pro
# default.  Historical run scripts and manifests used the old name; keep it
# importable while all new code uses ``QwenImage30Adapter``.
QwenImage30ProAdapter = QwenImage30Adapter


class Executor(Protocol):
    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        ...


class StubImageExecutor:
    """Deterministic synthetic executor used by the offline smoke test."""

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        if operation == "first_frame_fusion":
            if inputs:
                return next(iter(inputs.values()))
            return self._synthetic_character()
        if operation == "character_synthesis":
            return self._synthetic_character()
        if operation == "local_inpaint":
            return self._patch_mask(inputs)
        raise InputValidationError(
            f"stub executor does not support operation {operation!r}"
        )

    @staticmethod
    def _synthetic_character() -> bytes:
        image = Image.new("RGB", (256, 256), (0, 180, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (72, 48, 184, 208), radius=24, fill=(220, 60, 60)
        )
        return image_to_png_bytes(image)

    @staticmethod
    def _patch_mask(inputs: dict[str, bytes]) -> bytes:
        composite = Image.open(BytesIO(inputs["composite_image"])).convert(
            "RGB"
        )
        mask = Image.open(BytesIO(inputs["inpaint_mask"])).convert("L")
        box = mask.getbbox()
        if box is None:
            return image_to_png_bytes(composite)
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        radius = max(8, (x1 - x0) // 8)
        draw = ImageDraw.Draw(composite)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(255, 255, 255),
        )
        return image_to_png_bytes(composite)


class RemoteQwenExecutor:
    """Prepares the exact remote request manifest; GPU execution is remote.

    Phase 3 local boundary: the manifest (inputs + hashes + frozen prompt and
    sampling parameters) is written under ``<run_dir>/requests``, then
    execution raises until the user enables the remote server
    (AGENTS.md section 7: notify the user only after local prep is ready).
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.requests_dir = Path(run_dir) / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self._counter += 1
        manifest = {
            "operation": operation,
            "model_id": request_payload["model_id"],
            "revision": request_payload["revision"],
            "conditions": request_payload["conditions"],
            "prompt": request_payload["prompt"],
            "parameters": request_payload["parameters"],
            "inputs": {
                name: {
                    "bytes": len(data),
                }
                for name, data in inputs.items()
            },
        }
        path = self.requests_dir / f"request_{self._counter:03d}.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise EnvironmentCapabilityError(
            "remote Qwen execution requires the GPU server; prepared "
            f"manifest at {path}"
        )
