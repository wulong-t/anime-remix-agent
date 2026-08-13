"""Local LoRA-stack adapter and executor (Phase 1, checklist task 4).

The LoRA stack (Illustrious-XL base + style LoRA + character LoRA +
ControlNet pose + inpainting) is a local inference backend.  This module
implements the same two contracts the Composer already consumes:

- ``LocalLoraStackAdapter``: compiles a frozen request payload exactly like
  the Qwen adapters, with LoRA weight references, ControlNet pose source and
  sampling parameters recorded in ``parameters``.
- ``LocalLoraExecutor``: executes the payload and records the same
  request/manifest metadata shape as ``DashScopeQwenExecutor``.

Backends:

- ``backend="stub"``: deterministic synthetic PNG for offline contract and
  regression tests; metadata ``status="stubbed"`` so nobody mistakes it for
  real output.
- ``backend="local"``: requires an injected ``backend_fn`` seam; without it
  the executor raises ``EnvironmentCapabilityError`` until the real inference
  stack exists (trained LoRA weights + local GPU pipeline).

Contract invariants (implementation checklist): no changes to
``first-frame-plan-v1`` or Timeline 1.9; same request/manifest/SHA256
recording path as the DashScope executor; existing Composer tests keep their
semantics.  Text-delta edits stay on the Qwen path, so this adapter refuses
``apply_text_delta``.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from anime_remix.errors import EnvironmentCapabilityError, InputValidationError
from anime_remix.services.execution.imaging import image_to_png_bytes

_ADAPTER_PREFIX = "local-lora-stack-v1"
_MODEL_ID = "illustrious-xl"
_DEFAULT_REVISION = "illustrious-xl-v2.0-pending"
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
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_CONTROLNET_TYPES = {"openpose", "sketch"}


def _conditions_with_role(
    conditions: list[dict], roles: set[str]
) -> list[dict]:
    return [c for c in conditions if c["role"] in roles]


def _validate_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(
            f"invalid seed {value!r}: must be an integer"
        )
    if not 0 <= value <= _SEED_MAX:
        raise InputValidationError(
            f"invalid seed {value}: must be in 0..{_SEED_MAX}"
        )
    return value


def _validate_steps(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputValidationError(
            f"invalid steps {value!r}: must be a positive integer"
        )
    return value


def _validate_size(value: object) -> str:
    if not isinstance(value, str):
        raise InputValidationError(
            f"invalid size {value!r}: expected a 'W*H' string like "
            "'1280*720'"
        )
    normalized = value.strip()
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


def _validate_lora_config(value: object, label: str) -> dict:
    if value is None or (isinstance(value, dict) and not value):
        return {}
    if not isinstance(value, dict):
        raise InputValidationError(
            f"{label} must be a dict with path/sha256/trigger, "
            f"got {type(value).__name__}"
        )
    path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path, str) or not path:
        raise InputValidationError(f"{label} requires a non-empty path")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise InputValidationError(
            f"{label} requires a 64-char sha256, got {sha256!r}"
        )
    trigger = value.get("trigger")
    if trigger is not None and (
        not isinstance(trigger, str) or not trigger.strip()
    ):
        raise InputValidationError(f"{label} trigger must be a non-empty string")
    return dict(value)


def _parse_size_pixels(size: str) -> tuple[int, int]:
    width, height = (int(part) for part in size.split("*"))
    return width, height


def _prompt_sentence(text: str) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned.removesuffix(".")


class LocalLoraStackAdapter:
    """Reference-first adapter for the local LoRA stack.

    Identity, style and pose are carried by the trained LoRA weights and the
    ControlNet condition; the compiled prompt only expresses action/state
    changes, facts not covered by a selected reference, and the no-text
    constraint (same Reference-First discipline as the Qwen adapters).
    """

    adapter_id = _ADAPTER_PREFIX
    model_id = _MODEL_ID
    revision = _DEFAULT_REVISION

    def __init__(
        self,
        *,
        seed: int = 0,
        steps: int = 30,
        size: str = "1280*720",
        cfg_scale: float = 6.0,
        base_model_revision: str = _DEFAULT_REVISION,
        style_lora: dict | None = None,
        character_lora: dict | None = None,
        controlnet_type: str = "openpose",
        negative_prompt: str = _NEGATIVE_PROMPT,
    ) -> None:
        self._seed = _validate_seed(seed)
        self._steps = _validate_steps(steps)
        self._size = _validate_size(size)
        if (
            isinstance(cfg_scale, bool)
            or not isinstance(cfg_scale, (int, float))
            or not 0 < float(cfg_scale) <= 30
        ):
            raise InputValidationError(
                f"invalid cfg_scale {cfg_scale!r}: must be in (0, 30]"
            )
        self._cfg_scale = float(cfg_scale)
        if not isinstance(base_model_revision, str) or not base_model_revision:
            raise InputValidationError(
                "base_model_revision must be a non-empty string"
            )
        self.revision = base_model_revision
        self._style_lora = _validate_lora_config(style_lora, "style_lora")
        self._character_lora = _validate_lora_config(
            character_lora, "character_lora"
        )
        if controlnet_type not in _CONTROLNET_TYPES:
            raise InputValidationError(
                f"invalid controlnet_type {controlnet_type!r}: expected "
                f"one of {sorted(_CONTROLNET_TYPES)}"
            )
        self._controlnet_type = controlnet_type
        if not isinstance(negative_prompt, str):
            raise InputValidationError("negative_prompt must be a string")
        self._negative_prompt = negative_prompt

    def _base_parameters(
        self, *, intent: dict, inpaint: bool = False
    ) -> dict:
        controlnet: dict = {"type": self._controlnet_type}
        if not inpaint:
            pose_source = intent.get("control_pose_path")
            if pose_source is not None:
                if not isinstance(pose_source, str) or not pose_source:
                    raise InputValidationError(
                        "control_pose_path must be a non-empty string"
                    )
                controlnet["pose_source"] = pose_source
        return {
            "seed": self._seed,
            "steps": self._steps,
            "cfg_scale": self._cfg_scale,
            "size": self._size,
            "negative_prompt": self._negative_prompt,
            "base_model_revision": self.revision,
            "style_lora": self._style_lora,
            "character_lora": self._character_lora,
            "controlnet": controlnet,
        }

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
            return self._compile_character_synthesis(
                intent=intent,
                keyframe_state=keyframe_state,
                scene_description=scene_description,
                conditions=conditions,
            )
        if operation == "first_frame_fusion":
            return self._compile_first_frame_fusion(
                intent=intent,
                keyframe_state=keyframe_state,
                conditions=conditions,
            )
        if operation == "local_inpaint":
            return {
                "adapter_id": f"{_ADAPTER_PREFIX}-local_inpaint",
                "model_id": self.model_id,
                "revision": self.revision,
                "conditions": [],
                "prompt": (
                    "Repair only the masked region: fix seams, edges, "
                    "contact areas and shadows around the composited "
                    "character. Preserve everything outside the mask "
                    "exactly, including identity, costume, hair, scene "
                    "and composition. Do not add text, logos, or "
                    "watermarks."
                ),
                "parameters": self._base_parameters(
                    intent=intent, inpaint=True
                ),
            }
        if operation == "direct_scene_synthesis":
            raise InputValidationError(
                "the LoRA stack does not serve scene-only direct "
                "synthesis; scene establishment stays on the approved "
                "first-frame plan or style-LoRA text synthesis"
            )
        raise InputValidationError(
            f"unsupported adapter operation {operation!r}"
        )

    def _compile_character_synthesis(
        self,
        *,
        intent: dict,
        keyframe_state: dict,
        scene_description: str,
        conditions: list[dict],
    ) -> dict:
        identity = _conditions_with_role(conditions, {"identity"})
        how = _conditions_with_role(conditions, {"pose", "expression"})
        context = _conditions_with_role(
            conditions, {"scene", "style", "source_frame"}
        )
        if len(identity) > 1:
            raise InputValidationError(
                "character_synthesis accepts at most one identity condition"
            )
        if len(how) > 1:
            raise InputValidationError(
                "character_synthesis accepts at most one pose/expression "
                "condition"
            )
        if len(context) > 1:
            raise InputValidationError(
                "character_synthesis accepts at most one scene/style "
                "condition; other WHERE facts must be textualized"
            )
        slots = []
        for index, item in enumerate(identity + how + context, start=1):
            slots.append({"slot": index, "condition_ref": item["condition_id"]})

        instruction = intent.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            prompt = instruction.strip()
        else:
            parts: list[str] = []
            if scene_description:
                parts.append(f"Scene: {_prompt_sentence(scene_description)}")
            pose = intent.get("subject_pose")
            if pose:
                parts.append(f"Pose: {_prompt_sentence(pose)}")
            expression = keyframe_state.get("expression")
            if expression:
                parts.append(f"Expression: {_prompt_sentence(expression)}")
            action = intent.get("action_description") or intent.get("action")
            if action:
                parts.append(f"Action: {_prompt_sentence(action)}")
            parts.append(
                "Do not add text, captions, logos, signatures, labels, or "
                "watermarks."
            )
            prompt = " ".join(parts)
        parameters = self._base_parameters(intent=intent)
        if how:
            parameters["controlnet"]["pose_condition_ref"] = how[0][
                "condition_id"
            ]
        return {
            "adapter_id": f"{_ADAPTER_PREFIX}-character_synthesis",
            "model_id": self.model_id,
            "revision": self.revision,
            "conditions": slots,
            "prompt": prompt,
            "parameters": parameters,
        }

    def _compile_first_frame_fusion(
        self,
        *,
        intent: dict,
        keyframe_state: dict,
        conditions: list[dict],
    ) -> dict:
        if len(conditions) > 2:
            raise InputValidationError(
                "LoRA first_frame_fusion accepts at most two condition "
                f"slots, got {len(conditions)}"
            )
        stage_operation = intent.get("stage_operation")
        if stage_operation == "adopt_anchor":
            raise InputValidationError(
                "adopt_anchor is a deterministic stage and requires no "
                "model call"
            )
        if stage_operation == "apply_text_delta":
            raise InputValidationError(
                "apply_text_delta stays on the Qwen path; the LoRA stack "
                "does not serve text-delta edits"
            )
        slots = [
            {"slot": index, "condition_ref": item["condition_id"]}
            for index, item in enumerate(conditions, start=1)
        ]
        instruction = intent.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            prompt = instruction.strip()
        else:
            parts: list[str] = []
            scene = intent.get("scene_description")
            if scene:
                parts.append(f"Scene: {_prompt_sentence(scene)}")
            action = intent.get("action_description") or intent.get("action")
            if action:
                parts.append(f"Action: {_prompt_sentence(action)}")
            expression = keyframe_state.get("expression")
            if expression:
                parts.append(f"Expression: {_prompt_sentence(expression)}")
            parts.append(
                "Do not add text, captions, logos, signatures, labels, or "
                "watermarks."
            )
            prompt = " ".join(parts)
        parameters = self._base_parameters(intent=intent)
        pose_refs = [
            item["condition_id"]
            for item in conditions
            if item.get("role") in {"pose", "expression"}
        ]
        if pose_refs:
            parameters["controlnet"]["pose_condition_ref"] = pose_refs[0]
        adapter_id = f"{_ADAPTER_PREFIX}-first_frame_fusion"
        if stage_operation:
            adapter_id = f"{adapter_id}-{stage_operation}"
        return {
            "adapter_id": adapter_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "conditions": slots,
            "prompt": prompt,
            "parameters": parameters,
        }


class LocalLoraExecutor:
    """Execute a frozen LoRA-stack request; record like the Qwen executors."""

    provider = "local"

    def __init__(
        self,
        *,
        backend: str = "stub",
        backend_fn: Callable[[dict, str, dict[str, bytes]], bytes] | None = None,
    ) -> None:
        if backend not in {"stub", "local"}:
            raise InputValidationError(
                f"invalid backend {backend!r}: expected 'stub' or 'local'"
            )
        self.backend = backend
        self._backend_fn = backend_fn
        self.last_metadata: dict | None = None
        self.last_request_summary: dict | None = None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.last_metadata = None
        self.last_request_summary = None
        validated = self._validate_request(request_payload, operation, inputs)
        self.last_request_summary = validated["summary"]
        started = time.monotonic()
        if self.backend == "stub":
            output = self._stub_output(validated, inputs)
            status = "stubbed"
        else:
            self._verify_lora_weights(validated["parameters"])
            if self._backend_fn is None:
                raise EnvironmentCapabilityError(
                    "local LoRA inference requires trained LoRA weights and "
                    "the local inference stack (Phase 1 execution after user "
                    "authorization); pass backend_fn for the test seam"
                )
            output = self._backend_fn(
                request_payload, operation, inputs
            )
            if not isinstance(output, bytes) or not output.startswith(
                _PNG_MAGIC
            ):
                raise InputValidationError(
                    "local LoRA backend_fn must return PNG bytes"
                )
            status = "succeeded"
        self.last_metadata = {
            "provider": self.provider,
            "model": validated["model_id"],
            "request_id": None,
            "status": status,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 1),
            "backend": self.backend,
            "usage": {
                "input_image_count": len(inputs),
                "output_image_count": 1,
                "style_lora_sha256": validated["parameters"]
                .get("style_lora", {})
                .get("sha256"),
                "character_lora_sha256": validated["parameters"]
                .get("character_lora", {})
                .get("sha256"),
                "output_sha256": hashlib.sha256(output).hexdigest(),
            },
        }
        return output

    def _validate_request(
        self,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> dict:
        if not isinstance(request_payload, dict):
            raise InputValidationError("request_payload must be a dict")
        if not isinstance(inputs, dict):
            raise InputValidationError("inputs must be a dict")
        adapter_id = request_payload.get("adapter_id")
        if not isinstance(adapter_id, str) or not adapter_id.startswith(
            _ADAPTER_PREFIX
        ):
            raise InputValidationError(
                "LocalLoraExecutor only serves adapter ids prefixed "
                f"{_ADAPTER_PREFIX!r}; refused adapter_id {adapter_id!r}"
            )
        model_id = request_payload.get("model_id")
        if model_id != _MODEL_ID:
            raise InputValidationError(
                "LocalLoraExecutor only serves "
                f"{_MODEL_ID!r}; refused request for model "
                f"{model_id!r} instead of silently changing it"
            )
        revision = request_payload.get("revision")
        if not isinstance(revision, str) or not revision:
            raise InputValidationError(
                "LoRA request requires a non-empty base model revision"
            )
        conditions = request_payload.get("conditions")
        if not isinstance(conditions, list) or not 0 <= len(conditions) <= 3:
            count = (
                len(conditions) if isinstance(conditions, list) else "invalid"
            )
            raise InputValidationError(
                "LoRA request accepts 0-3 condition slots, "
                f"got {count}"
            )
        prompt = request_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InputValidationError(
                "LoRA request requires a non-empty prompt"
            )
        parameters = request_payload.get("parameters")
        if not isinstance(parameters, dict):
            raise InputValidationError(
                "LoRA request parameters must be a dict"
            )
        required_parameters = {
            "seed",
            "steps",
            "size",
            "negative_prompt",
            "style_lora",
            "character_lora",
            "controlnet",
            "base_model_revision",
        }
        missing = sorted(required_parameters - set(parameters))
        if missing:
            raise InputValidationError(
                "LoRA request parameters missing required "
                f"field(s): {', '.join(missing)}"
            )
        _validate_seed(parameters["seed"])
        _validate_steps(parameters["steps"])
        size = _validate_size(parameters["size"])
        if not isinstance(parameters["negative_prompt"], str):
            raise InputValidationError(
                "negative_prompt must be a string"
            )
        _validate_lora_config(parameters["style_lora"], "style_lora")
        _validate_lora_config(parameters["character_lora"], "character_lora")
        controlnet = parameters["controlnet"]
        if not isinstance(controlnet, dict):
            raise InputValidationError(
                "controlnet parameters must be a dict"
            )
        if controlnet.get("type") not in _CONTROLNET_TYPES:
            raise InputValidationError(
                "controlnet type must be one of "
                f"{sorted(_CONTROLNET_TYPES)}"
            )
        for condition in conditions:
            condition_ref = condition.get("condition_ref")
            if not isinstance(condition_ref, str) or not condition_ref:
                raise InputValidationError(
                    "invalid condition slot: condition_ref must be a "
                    "non-empty string"
                )
            data = inputs.get(condition_ref)
            if not isinstance(data, bytes) or not data:
                raise InputValidationError(
                    "missing input image for condition_ref "
                    f"{condition_ref!r}"
                )
        if operation == "local_inpaint":
            for required in ("composite_image", "inpaint_mask"):
                data = inputs.get(required)
                if not isinstance(data, bytes) or not data:
                    raise InputValidationError(
                        f"local_inpaint requires input {required!r}"
                    )
        input_hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in inputs.items()
        }
        return {
            "model_id": model_id,
            "revision": revision,
            "size": size,
            "parameters": parameters,
            "summary": {
                "adapter_id": adapter_id,
                "model_id": model_id,
                "revision": revision,
                "operation": operation,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "input_count": len(inputs),
                "input_sha256s": input_hashes,
                "parameters": parameters,
            },
        }

    def _verify_lora_weights(self, parameters: dict) -> None:
        for label in ("style_lora", "character_lora"):
            config = parameters.get(label) or {}
            if not config:
                continue
            path = Path(config["path"])
            if not path.exists():
                raise EnvironmentCapabilityError(
                    f"{label} weight file not found: {path}"
                )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != config["sha256"]:
                raise EnvironmentCapabilityError(
                    f"{label} weight SHA256 mismatch for {path}: "
                    f"file={digest[:16]} expected={config['sha256'][:16]}"
                )

    def _stub_output(
        self, validated: dict, inputs: dict[str, bytes]
    ) -> bytes:
        width, height = _parse_size_pixels(validated["size"])
        seed = validated["parameters"]["seed"]
        background = (
            (seed * 7 + 41) % 256,
            (seed * 13 + 97) % 256,
            (seed * 17 + 173) % 256,
        )
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        inset = (seed * 11 + 3) % 256
        draw.rounded_rectangle(
            (width // 8, height // 8, width * 7 // 8, height * 7 // 8),
            radius=max(8, min(width, height) // 16),
            fill=(inset, inset, (inset * 2) % 256),
        )
        if "composite_image" in inputs:
            try:
                with Image.open(BytesIO(inputs["composite_image"])) as source:
                    base = source.convert("RGB").resize((width, height))
            except Exception:  # noqa: BLE001 - stub should be robust
                base = image
            patch = ImageDraw.Draw(base)
            radius = max(8, min(width, height) // 16)
            patch.ellipse(
                (
                    width // 2 - radius,
                    height // 2 - radius,
                    width // 2 + radius,
                    height // 2 + radius,
                ),
                fill=(seed % 256, (seed * 3) % 256, 255),
            )
            image = base
        return image_to_png_bytes(image)


class LocalLoraExecutorProtocol(Protocol):
    """Duck-typed executor view used by the Composer."""

    provider: str
    last_metadata: dict | None
    last_request_summary: dict | None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        ...
