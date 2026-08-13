"""Unit tests for the local LoRA-stack adapter and executor.

These tests never touch a GPU or the network: the LoRA weights are synthetic
temp files and the local backend is injected.  The frozen request shape,
parameter validation, SHA256 recording and error translation are verified
against the same contract the Qwen adapters/executors implement.
"""

from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from anime_remix.errors import EnvironmentCapabilityError, InputValidationError
from anime_remix.services.execution.imaging import image_to_png_bytes
from anime_remix.services.execution.local_lora_executor import (
    LocalLoraExecutor,
    LocalLoraStackAdapter,
)


def _png_bytes() -> bytes:
    return image_to_png_bytes(Image.new("RGB", (64, 64), (255, 0, 0)))


def _identity_condition() -> dict:
    return {
        "role": "identity",
        "condition_id": "cond_identity_test_character",
    }


def _pose_condition() -> dict:
    return {
        "role": "pose",
        "condition_id": "cond_pose_reference",
    }


def _adapter(**kwargs) -> LocalLoraStackAdapter:
    return LocalLoraStackAdapter(**kwargs)


def test_adapter_compiles_character_synthesis_payload() -> None:
    adapter = _adapter(
        seed=7,
        steps=30,
        size="1280*720",
        style_lora={
            "path": "models/lora/style.safetensors",
            "sha256": "a" * 64,
            "trigger": "synthetic anime style",
        },
        character_lora={
            "path": "models/lora/test_character.safetensors",
            "sha256": "b" * 64,
            "trigger": "test character",
        },
        controlnet_type="openpose",
    )
    payload = adapter.compile(
        operation="character_synthesis",
        intent={"subject_pose": "standing, holding a folder", "control_pose_path": "runs/pose/pose.png"},
        keyframe_state={"expression": "smiling gently"},
        scene_description="bright warm office, afternoon sunlight",
        conditions=[_identity_condition(), _pose_condition()],
    )
    assert payload["adapter_id"] == "local-lora-stack-v1-character_synthesis"
    assert payload["model_id"] == "illustrious-xl"
    assert payload["revision"] == "illustrious-xl-v2.0-pending"
    assert payload["conditions"] == [
        {"slot": 1, "condition_ref": "cond_identity_test_character"},
        {"slot": 2, "condition_ref": "cond_pose_reference"},
    ]
    assert "Scene: bright warm office" in payload["prompt"]
    assert "Expression: smiling gently" in payload["prompt"]
    assert "watermarks" in payload["prompt"]
    parameters = payload["parameters"]
    assert parameters["seed"] == 7
    assert parameters["steps"] == 30
    assert parameters["size"] == "1280*720"
    assert parameters["controlnet"] == {
        "type": "openpose",
        "pose_source": "runs/pose/pose.png",
        "pose_condition_ref": "cond_pose_reference",
    }
    assert parameters["style_lora"]["trigger"] == "synthetic anime style"
    assert parameters["character_lora"]["trigger"] == "test character"


def test_adapter_uses_full_instruction_prompt() -> None:
    payload = _adapter().compile(
        operation="character_synthesis",
        intent={
            "instruction": (
                "test character, synthetic anime, office, close-up, masterpiece"
            )
        },
        keyframe_state={},
        scene_description="ignored by instruction",
        conditions=[_identity_condition()],
    )
    assert (
        payload["prompt"]
        == "test character, synthetic anime, office, close-up, masterpiece"
    )


def test_adapter_rejects_unknown_operation() -> None:
    with pytest.raises(InputValidationError):
        _adapter().compile(
            operation="not_an_operation",
            intent={},
            keyframe_state={},
            scene_description="",
            conditions=[],
        )


def test_adapter_refuses_text_delta_stage() -> None:
    with pytest.raises(InputValidationError, match="apply_text_delta"):
        _adapter().compile(
            operation="first_frame_fusion",
            intent={"stage_operation": "apply_text_delta"},
            keyframe_state={},
            scene_description="",
            conditions=[],
        )


def test_adapter_refuses_scene_only_direct_synthesis() -> None:
    with pytest.raises(InputValidationError, match="direct"):
        _adapter().compile(
            operation="direct_scene_synthesis",
            intent={"scene_only": True},
            keyframe_state={},
            scene_description="",
            conditions=[],
        )


def test_adapter_validates_seed_size_and_cfg() -> None:
    with pytest.raises(InputValidationError):
        _adapter(seed=-1)
    with pytest.raises(InputValidationError):
        _adapter(size="100*100")
    with pytest.raises(InputValidationError):
        _adapter(cfg_scale=100)
    with pytest.raises(InputValidationError):
        _adapter(controlnet_type="depth")


def test_stub_executor_is_deterministic_and_records_metadata() -> None:
    executor = LocalLoraExecutor(backend="stub")
    payload = _adapter(seed=11).compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    inputs = {"cond_identity_test_character": _png_bytes()}
    first = executor.execute(
        request_payload=payload, operation="character_synthesis", inputs=inputs
    )
    second = executor.execute(
        request_payload=payload, operation="character_synthesis", inputs=inputs
    )
    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    metadata = executor.last_metadata
    assert metadata is not None
    assert metadata["status"] == "stubbed"
    assert metadata["backend"] == "stub"
    assert metadata["provider"] == "local"
    assert metadata["usage"]["output_sha256"] == hashlib.sha256(first).hexdigest()
    summary = executor.last_request_summary
    assert summary is not None
    assert summary["input_sha256s"]["cond_identity_test_character"] == hashlib.sha256(
        inputs["cond_identity_test_character"]
    ).hexdigest()
    assert summary["prompt_sha256"]


def test_stub_executor_rejects_missing_condition_input() -> None:
    executor = LocalLoraExecutor(backend="stub")
    payload = _adapter().compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition(), _pose_condition()],
    )
    with pytest.raises(InputValidationError, match="cond_pose_reference"):
        executor.execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs={"cond_identity_test_character": _png_bytes()},
        )


def test_executor_refuses_foreign_model() -> None:
    payload = _adapter().compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    payload["model_id"] = "qwen-image-3.0"
    with pytest.raises(InputValidationError, match="illustrious-xl"):
        LocalLoraExecutor(backend="stub").execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs={"cond_identity_test_character": _png_bytes()},
        )


def test_executor_requires_all_parameters() -> None:
    payload = _adapter().compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    del payload["parameters"]["controlnet"]
    with pytest.raises(InputValidationError, match="controlnet"):
        LocalLoraExecutor(backend="stub").execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs={"cond_identity_test_character": _png_bytes()},
        )


def test_local_backend_requires_seam_or_weights(tmp_path) -> None:
    weight = tmp_path / "test_character.safetensors"
    weight.write_bytes(b"fake-lora-weights")
    payload = _adapter(
        character_lora={
            "path": str(weight),
            "sha256": hashlib.sha256(b"fake-lora-weights").hexdigest(),
        }
    ).compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    inputs = {"cond_identity_test_character": _png_bytes()}
    with pytest.raises(EnvironmentCapabilityError, match="backend_fn"):
        LocalLoraExecutor(backend="local").execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs=inputs,
        )


def test_local_backend_verifies_weights_and_succeeds(tmp_path) -> None:
    weight = tmp_path / "style.safetensors"
    weight_bytes = b"fake-style-weights"
    weight.write_bytes(weight_bytes)
    adapter = _adapter(
        style_lora={
            "path": str(weight),
            "sha256": hashlib.sha256(weight_bytes).hexdigest(),
        }
    )
    payload = adapter.compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    inputs = {"cond_identity_test_character": _png_bytes()}

    def backend_fn(request_payload: dict, operation: str, inputs: dict[str, bytes]) -> bytes:
        return image_to_png_bytes(
            Image.new("RGB", (1280, 720), (10, 20, 30))
        )

    executor = LocalLoraExecutor(backend="local", backend_fn=backend_fn)
    output = executor.execute(
        request_payload=payload,
        operation="character_synthesis",
        inputs=inputs,
    )
    assert output.startswith(b"\x89PNG\r\n\x1a\n")
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "succeeded"
    assert executor.last_metadata["backend"] == "local"

    payload["parameters"]["style_lora"]["sha256"] = "0" * 64
    with pytest.raises(EnvironmentCapabilityError, match="SHA256 mismatch"):
        executor.execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs=inputs,
        )


def test_local_backend_fn_must_return_png(tmp_path) -> None:
    payload = _adapter().compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    executor = LocalLoraExecutor(
        backend="local",
        backend_fn=lambda request_payload, operation, inputs: b"not-png",
    )
    with pytest.raises(InputValidationError, match="PNG"):
        executor.execute(
            request_payload=payload,
            operation="character_synthesis",
            inputs={"cond_identity_test_character": _png_bytes()},
        )


def test_stub_local_inpaint_requires_composite_and_mask() -> None:
    executor = LocalLoraExecutor(backend="stub")
    payload = _adapter().compile(
        operation="local_inpaint",
        intent={},
        keyframe_state={},
        scene_description="",
        conditions=[],
    )
    with pytest.raises(InputValidationError, match="inpaint_mask"):
        executor.execute(
            request_payload=payload,
            operation="local_inpaint",
            inputs={"composite_image": _png_bytes()},
        )
    output = executor.execute(
        request_payload=payload,
        operation="local_inpaint",
        inputs={
            "composite_image": _png_bytes(),
            "inpaint_mask": _png_bytes(),
        },
    )
    assert output.startswith(b"\x89PNG\r\n\x1a\n")
    assert executor.last_metadata["status"] == "stubbed"


def test_compile_roundtrip_through_executor_records_request_hash() -> None:
    adapter = _adapter(seed=3)
    payload = adapter.compile(
        operation="character_synthesis",
        intent={},
        keyframe_state={},
        scene_description="office",
        conditions=[_identity_condition()],
    )
    payload["prompt"] = "Frozen prompt"
    inputs = {"cond_identity_test_character": _png_bytes()}
    executor = LocalLoraExecutor(backend="stub")
    executor.execute(
        request_payload=payload,
        operation="character_synthesis",
        inputs=inputs,
    )
    summary = executor.last_request_summary
    assert summary is not None
    assert summary["adapter_id"] == payload["adapter_id"]
    assert summary["parameters"] == payload["parameters"]
    assert summary["operation"] == "character_synthesis"
