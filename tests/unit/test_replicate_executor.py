"""Unit tests for the Replicate-hosted Qwen executor boundary.

These tests never touch the network: a fake client is injected and the
executor's request mapping, secret boundary and error translation are
verified against the frozen ModelRenderRequest contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anime_remix.errors import (
    EnvironmentCapabilityError,
    InputValidationError,
    RenderError,
)
from anime_remix.services.execution.replicate_executor import (
    REPLICATE_MODEL_ID,
    REPLICATE_TOKEN_ENV,
    ReplicateQwenExecutor,
)


class _FakeOutput:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakePrediction:
    def __init__(
        self,
        prediction_id: str,
        *,
        status: str = "succeeded",
        output: object = None,
        error: str | None = None,
        version: str = "ver-fake",
    ) -> None:
        self.id = prediction_id
        self.status = status
        self.output = output
        self.error = error
        self.version = version

    def reload(self) -> None:
        return None


class _FakePredictions:
    def __init__(self, factory: object) -> None:
        self._factory = factory
        self.calls: list[dict] = []

    def create(self, *, model: str, input: dict) -> object:
        self.calls.append({"model": model, "input": _capture(input)})
        factory = self._factory
        if callable(factory):
            return factory()
        raise factory


def _capture(value: object) -> object:
    """Deep-copy an input value, reading local temp files while they exist."""

    if isinstance(value, dict):
        return {key: _capture(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_capture(item) for item in value]
    if isinstance(value, Path):
        return value.read_bytes()
    return value


class _FakeClient:
    def __init__(self, factory: object) -> None:
        self.predictions = _FakePredictions(factory)


def _compiled_request(seed: int = 7) -> dict:
    return {
        "adapter_id": "qwen-edit-2511-adapter-v1",
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "revision": "main",
        "conditions": [
            {"slot": 1, "condition_ref": "cond_001"},
            {"slot": 2, "condition_ref": "cond_002"},
        ],
        "prompt": "Keep identity from image 1; adopt pose from image 2.",
        "parameters": {
            "seed": seed,
            "num_inference_steps": 40,
            "true_cfg_scale": 4.0,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        },
    }


def _inputs() -> dict[str, bytes]:
    return {
        "cond_001": b"who-identity-bytes",
        "cond_002": b"how-pose-bytes",
    }


def _executor(
    client: object,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    token: str = "r8_fake_secret_token",
) -> ReplicateQwenExecutor:
    if monkeypatch is not None:
        monkeypatch.setenv(REPLICATE_TOKEN_ENV, token)
    return ReplicateQwenExecutor(client=client)


def test_request_mapping_preserves_slot_order_and_filters_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        lambda: _FakePrediction("p_001", output=_FakeOutput(b"out"))
    )
    executor = _executor(client, monkeypatch=monkeypatch)

    result = executor.execute(
        request_payload=_compiled_request(seed=7),
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == b"out"
    assert len(client.predictions.calls) == 1
    call = client.predictions.calls[0]
    assert call["model"] == REPLICATE_MODEL_ID
    sent = call["input"]
    assert sent["prompt"] == _compiled_request()["prompt"]
    assert sent["seed"] == 7
    assert sent["output_format"] == "png"
    assert "num_inference_steps" not in sent
    assert "true_cfg_scale" not in sent
    assert "guidance_scale" not in sent
    assert "num_images_per_prompt" not in sent
    assert isinstance(sent["image"], list) and len(sent["image"]) == 2


def test_who_how_image_order_matches_compiled_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        lambda: _FakePrediction(
            "p_002", output=_FakeOutput(b"rendered-character")
        )
    )
    executor = _executor(client, monkeypatch=monkeypatch)
    identity = b"who-identity-bytes"
    pose = b"how-pose-bytes"

    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs={"cond_001": identity, "cond_002": pose},
    )

    assert result == b"rendered-character"
    sent = client.predictions.calls[0]["input"]
    assert sent["image"] == [identity, pose]


def test_missing_token_raises_sanitized_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REPLICATE_TOKEN_ENV, raising=False)
    executor = ReplicateQwenExecutor()
    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    message = str(exc_info.value)
    assert REPLICATE_TOKEN_ENV in message
    assert "never logged or stored" in message
    assert "r8_" not in message


def test_provider_failure_becomes_render_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(
        lambda: _FakePrediction(
            "p_003",
            status="failed",
            error="model exploded: cuda oom",
        )
    )
    executor = _executor(client, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "p_003" in str(exc_info.value)
    assert "model exploded" in str(exc_info.value)
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "failed"
    assert executor.last_metadata["prediction_id"] == "p_003"


def test_provider_request_exception_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(RuntimeError("network unreachable"))
    executor = _executor(client, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "network unreachable" in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)


def test_secret_never_enters_metadata_or_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "r8_super_secret_value"
    client = _FakeClient(
        lambda: _FakePrediction("p_004", output=_FakeOutput(b"ok"))
    )
    executor = _executor(client, monkeypatch=monkeypatch, token=token)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert token not in json.dumps(executor.last_metadata)
    for name, value in vars(executor).items():
        assert token not in repr(value), f"token leaked via {name}"


def test_missing_condition_input_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(lambda: _FakePrediction("p_005"))
    executor = _executor(client, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs={"cond_001": b"only-identity"},
        )


def test_local_inpaint_is_refused_with_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient(lambda: _FakePrediction("p_006"))
    executor = _executor(client, monkeypatch=monkeypatch)
    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="local_inpaint",
            inputs={"composite_image": b"x", "inpaint_mask": b"y"},
        )
    assert "F-013" in str(exc_info.value)
    assert "mask" in str(exc_info.value)
