"""Unit tests for the local LoRA inference backend (dependency-free parts).

The heavy diffusers/torch code path is only exercised on the GPU host (see
the remote bootstrap self-test); these tests cover constructor validation,
prompt assembly, control-image resolution and error translation that must
work without the GPU stack installed.
"""

from __future__ import annotations

import pytest

from anime_remix.errors import EnvironmentCapabilityError, InputValidationError
from anime_remix.services.execution.local_lora_inference import (
    LocalLoraInferenceBackend,
)


def _payload(**overrides: object) -> dict:
    payload = {
        "adapter_id": "local-lora-stack-v1-character_synthesis",
        "model_id": "illustrious-xl",
        "revision": "test",
        "conditions": [{"slot": 1, "condition_ref": "cond_identity_test_character"}],
        "prompt": "office scene, friendly chat",
        "parameters": {
            "seed": 0,
            "steps": 30,
            "cfg_scale": 6.0,
            "size": "1024*1024",
            "negative_prompt": "text, watermark",
            "style_lora": {
                "path": "models/style.safetensors",
                "sha256": "a" * 64,
                "trigger": "synthetic anime style",
            },
            "character_lora": {
                "path": "models/test_character.safetensors",
                "sha256": "b" * 64,
                "trigger": "test character",
            },
            "controlnet": {
                "type": "openpose",
                "pose_condition_ref": "cond_pose",
            },
            "base_model_revision": "test",
        },
    }
    payload.update(overrides)
    return payload


def _backend(tmp_path) -> LocalLoraInferenceBackend:
    base = tmp_path / "base"
    base.mkdir()
    return LocalLoraInferenceBackend(base_model_path=base)


def test_backend_requires_existing_base_model(tmp_path) -> None:
    with pytest.raises(EnvironmentCapabilityError, match="base model"):
        LocalLoraInferenceBackend(base_model_path=tmp_path / "missing")


def test_backend_validates_scales(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(InputValidationError, match="conditioning_scale"):
        LocalLoraInferenceBackend(
            base_model_path=base, controlnet_conditioning_scale=5
        )
    with pytest.raises(InputValidationError, match="img2img_strength"):
        LocalLoraInferenceBackend(
            base_model_path=base, img2img_strength=0
        )


def test_backend_prompt_includes_triggers(tmp_path) -> None:
    backend = _backend(tmp_path)
    parameters = _payload()["parameters"]
    prompt = backend._prompt_with_triggers("office scene", parameters)
    assert prompt.startswith("synthetic anime style, test character, office scene")
    empty = backend._prompt_with_triggers("plain", {"style_lora": {}, "character_lora": {}})
    assert empty == "plain"


def test_backend_missing_deps_raise(tmp_path) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("torch installed; missing-deps path not applicable here")
    backend = _backend(tmp_path)
    with pytest.raises(EnvironmentCapabilityError, match="torch"):
        backend(
            request_payload=_payload(),
            operation="character_synthesis",
            inputs={"cond_identity_test_character": b"x", "cond_pose": b"y"},
        )


def test_backend_control_image_bytes_resolution(tmp_path) -> None:
    backend = _backend(tmp_path)
    params = _payload()["parameters"]["controlnet"]
    pose_bytes = b"pose-png"
    inputs = {"cond_pose": pose_bytes}
    assert backend._control_image_bytes(params, inputs) == pose_bytes
    params_no_ref = {"type": "openpose", "pose_source": "missing.png"}
    assert backend._control_image_bytes(params_no_ref, inputs) is None
    source = tmp_path / "pose.png"
    source.write_bytes(b"precomputed-pose")
    params_file = {"type": "openpose", "pose_source": str(source)}
    assert backend._control_image_bytes(params_file, inputs) == b"precomputed-pose"


def test_backend_pil_rejects_garbage(tmp_path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(InputValidationError, match="cannot decode"):
        backend._pil(b"not-an-image", "probe")


def test_backend_init_image_uses_first_condition(tmp_path) -> None:
    backend = _backend(tmp_path)
    payload = _payload()
    canvas = b"canvas-png"
    inputs = {"cond_identity_test_character": canvas}
    assert backend._init_image("character_synthesis", payload, inputs) is None
    fusion = dict(payload)
    fusion["conditions"] = [
        {"slot": 1, "condition_ref": "cond_identity_test_character"}
    ]
    with pytest.raises(InputValidationError, match="cannot decode"):
        backend._init_image("first_frame_fusion", fusion, inputs)
