"""Unit tests for the DashScope qwen-image-3.0 executor boundary.

These tests never touch the network: the SDK call and image download are
injected, and the executor's request shape, secret boundary, parameter
validation, URL policy and error translation are verified against the frozen
ModelRenderRequest contract.  The default transport is additionally verified
to route through the official ``dashscope.MultiModalConversation.call``.
"""

from __future__ import annotations

import base64
import builtins
import hashlib
import json
import logging
import os
import traceback
import types
from pathlib import Path
from urllib.parse import unquote, urlparse

import dashscope
import pytest

from anime_remix.errors import (
    EnvironmentCapabilityError,
    InputValidationError,
    RenderError,
)
from anime_remix.services.execution import dashscope_executor as executor_module
from anime_remix.services.execution.dashscope_executor import (
    _PNG_MAGIC,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_HTTP_API_URL,
    DASHSCOPE_MODEL_ID,
    DashScopeQwenExecutor,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"who-identity-payload"
_JPEG = b"\xff\xd8\xff" + b"how-pose-payload"
_TOKEN = "sk-fake_secret_token"
_OUT = _PNG_MAGIC + b"generated-png-bytes"
_DEFAULT_USAGE = {
    "output_width": 1280,
    "output_height": 720,
    "input_image_count": 2,
    "input_image_type": "image",
    "output_image_count": 1,
    "output_image_type": "image",
}


class _FakeSdkResponse(dict):
    """Mirrors the SDK's dict-subclass response with attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _sdk_response(
    *,
    status_code: int = 200,
    request_id: str = "req_abc",
    code: str = "",
    message: str = "",
    output: dict | None = None,
    usage: dict | None = None,
) -> _FakeSdkResponse:
    if output is None:
        output = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "image": (
                                    "https://dashscope.aliyuncs.com/"
                                    "result.png"
                                )
                            }
                        ]
                    }
                }
            ]
        }
    return _FakeSdkResponse(
        status_code=status_code,
        request_id=request_id,
        code=code,
        message=message,
        output=output,
        usage=usage if usage is not None else _DEFAULT_USAGE,
    )


class _FakeSdk:
    def __init__(
        self,
        *,
        response: _FakeSdkResponse | None = None,
        call_error: Exception | None = None,
        download: bytes = _OUT,
        download_error: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else _sdk_response()
        self.call_error = call_error
        self.download_bytes = download
        self.download_error = download_error
        self.calls: list[tuple[dict, str | None]] = []
        self.downloads: list[str] = []

    def call(self, request: dict, api_key: str | None):
        self.calls.append((request, api_key))
        if self.call_error is not None:
            raise self.call_error
        return self.response

    def download(self, url: str) -> bytes:
        self.downloads.append(url)
        if self.download_error is not None:
            raise self.download_error
        return self.download_bytes


def _compiled_request(
    *,
    seed: int = 7,
    size: str = "1280*720",
    model_id: str = DASHSCOPE_MODEL_ID,
    adapter_id: str = "qwen-image-30-adapter-v6-reference-first",
    prompt: str = "Keep identity from image 1; adopt pose from image 2.",
    **parameter_overrides: object,
) -> dict:
    parameters: dict[str, object] = {
        "seed": seed,
        "n": 1,
        "size": size,
        "prompt_extend": False,
        "prompt_extend_mode": "direct",
        "watermark": False,
        "negative_prompt": "text, logo, watermark, bad anatomy",
    }
    parameters.update(parameter_overrides)
    return {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "revision": "provider-managed-alias",
        "conditions": [
            {"slot": 1, "condition_ref": "cond_001"},
            {"slot": 2, "condition_ref": "cond_002"},
        ],
        "prompt": prompt,
        "parameters": parameters,
    }


def _inputs() -> dict[str, bytes]:
    return {"cond_001": _PNG, "cond_002": _JPEG}


def _executor(
    sdk: _FakeSdk,
    *,
    monkeypatch: pytest.MonkeyPatch,
    token: str = _TOKEN,
) -> DashScopeQwenExecutor:
    monkeypatch.setenv(DASHSCOPE_API_KEY, token)
    return DashScopeQwenExecutor(
        call_fn=sdk.call,
        download_fn=sdk.download,
    )


def _sent_request(sdk: _FakeSdk) -> tuple[dict, str | None]:
    assert len(sdk.calls) == 1
    request, api_key = sdk.calls[0]
    assert api_key == _TOKEN
    return request, api_key


def test_request_shape_order_and_base64_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    compiled = _compiled_request(
        seed=7,
        num_inference_steps=40,  # unsupported field must be dropped
    )

    result = executor.execute(
        request_payload=compiled,
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == _OUT
    request, _ = _sent_request(sdk)
    assert request["model"] == DASHSCOPE_MODEL_ID
    messages = request["input"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [next(iter(item)) for item in content] == ["image", "image", "text"]
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[1]["image"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(content[0]["image"].split(",", 1)[1]) == _PNG
    assert base64.b64decode(content[1]["image"].split(",", 1)[1]) == _JPEG
    assert content[2] == {"text": compiled["prompt"]}
    assert request["parameters"] == {
        "seed": 7,
        "n": 1,
        "size": "1280*720",
        "prompt_extend": False,
        "prompt_extend_mode": "direct",
        "watermark": False,
        "negative_prompt": "text, logo, watermark, bad anatomy",
    }
    assert "num_inference_steps" not in request["parameters"]
    assert "num_inference_steps" not in request


def test_who_how_image_order_follows_compiled_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    # Inputs dict order is reversed; the compiled slot order must win.
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs={"cond_002": _JPEG, "cond_001": _PNG},
    )
    request, _ = _sent_request(sdk)
    content = request["input"]["messages"][0]["content"]
    assert base64.b64decode(content[0]["image"].split(",", 1)[1]) == _PNG
    assert base64.b64decode(content[1]["image"].split(",", 1)[1]) == _JPEG


def test_default_transport_uses_official_multi_modal_conversation_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def fake_call(**kwargs):
        local_paths = []
        local_bytes = []
        for item in kwargs["messages"][0]["content"]:
            image_uri = item.get("image")
            if not image_uri:
                continue
            parsed = urlparse(image_uri)
            raw_path = unquote(parsed.path)
            if os.name == "nt" and raw_path.startswith("/"):
                raw_path = raw_path[1:]
            local_path = Path(raw_path)
            local_paths.append(local_path)
            local_bytes.append(local_path.read_bytes())
        captured.append(
            {
                "kwargs": kwargs,
                "base_http_api_url": dashscope.base_http_api_url,
                "local_paths": local_paths,
                "local_bytes": local_bytes,
            }
        )
        return _sdk_response()

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", fake_call)
    monkeypatch.setattr(
        dashscope,
        "base_http_api_url",
        "https://unpinned.example.invalid/",
    )
    monkeypatch.setenv(DASHSCOPE_API_KEY, _TOKEN)
    executor = DashScopeQwenExecutor(download_fn=lambda url: _OUT)
    compiled = _compiled_request(seed=7)

    result = executor.execute(
        request_payload=compiled,
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == _OUT
    assert len(captured) == 1
    captured_call = captured[0]
    kwargs = captured_call["kwargs"]
    assert kwargs["api_key"] == _TOKEN
    assert kwargs["model"] == DASHSCOPE_MODEL_ID
    # Qwen-Image 3.0's official SDK example omits ``stream``.  SDK 1.26.6
    # otherwise serializes explicit ``stream=False`` into model parameters.
    assert "stream" not in kwargs
    assert kwargs["request_timeout"] == 300
    assert (
        captured_call["base_http_api_url"]
        == DASHSCOPE_BASE_HTTP_API_URL
        == "https://dashscope.aliyuncs.com/api/v1"
    )
    content = kwargs["messages"][0]["content"]
    assert kwargs["messages"][0]["role"] == "user"
    assert content[-1] == {"text": compiled["prompt"]}
    assert [item["image"].startswith("file://") for item in content[:-1]] == [
        True,
        True,
    ]
    assert captured_call["local_bytes"] == [_PNG, _JPEG]
    assert all(not path.exists() for path in captured_call["local_paths"])
    for key, value in compiled["parameters"].items():
        assert kwargs[key] == value, f"SDK kwarg {key!r} mismatch"
    assert "parameters" not in kwargs
    assert "input" not in kwargs
    assert executor.last_request_summary is not None
    assert (
        executor.last_request_summary["input_transport"]
        == "dashscope_sdk_temporary_oss"
    )


def test_official_sdk_uploads_temporary_references_before_model_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dashscope.client.base_api import BaseApi
    from dashscope.utils.oss_utils import OssUtils

    uploads: list[dict] = []
    model_calls: list[dict] = []

    def fake_upload(
        *,
        model,
        file_path,
        api_key=None,
        upload_certificate=None,
        **kwargs,
    ):
        path = Path(file_path)
        uploads.append(
            {
                "model": model,
                "bytes": path.read_bytes(),
                "api_key": api_key,
                "certificate": upload_certificate,
            }
        )
        certificate = upload_certificate or {"fixture": "certificate"}
        return f"oss://temporary/{path.name}", certificate

    def fake_base_call(cls, **kwargs):
        model_calls.append(kwargs)
        response = _sdk_response()
        response["output"]["choices"][0]["message"]["role"] = "assistant"
        response["headers"] = {}
        return response

    monkeypatch.setattr(OssUtils, "upload", fake_upload)
    monkeypatch.setattr(BaseApi, "call", classmethod(fake_base_call))
    monkeypatch.setenv(DASHSCOPE_API_KEY, _TOKEN)
    executor = DashScopeQwenExecutor(download_fn=lambda url: _OUT)

    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == _OUT
    assert [entry["bytes"] for entry in uploads] == [_PNG, _JPEG]
    assert uploads[0]["certificate"] is None
    assert uploads[1]["certificate"] == {"fixture": "certificate"}
    assert len(model_calls) == 1
    model_call = model_calls[0]
    content = model_call["input"]["messages"][0]["content"]
    assert [item["image"] for item in content[:-1]] == [
        "oss://temporary/reference_1.png",
        "oss://temporary/reference_2.jpg",
    ]
    assert model_call["headers"]["X-DashScope-OssResourceResolve"] == "enable"
    assert "stream" not in model_call


def test_default_call_forces_dashscope_logger_to_info_or_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_call(**kwargs):
        logger = logging.getLogger("dashscope")
        captured["level"] = logger.level
        captured["effective_level"] = logger.getEffectiveLevel()
        return _sdk_response()

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", fake_call)
    monkeypatch.setenv(DASHSCOPE_API_KEY, _TOKEN)
    logger = logging.getLogger("dashscope")
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    executor = DashScopeQwenExecutor(download_fn=lambda url: _OUT)

    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == _OUT
    assert captured["level"] >= logging.INFO
    assert captured["effective_level"] >= logging.INFO
    assert logging.getLogger("dashscope").getEffectiveLevel() >= logging.INFO


def test_missing_key_raises_sanitized_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DASHSCOPE_API_KEY, raising=False)

    def _explode(*args, **kwargs):
        raise AssertionError("SDK must not be called without a key")

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", _explode)
    executor = DashScopeQwenExecutor()
    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    message = str(exc_info.value)
    assert DASHSCOPE_API_KEY in message
    assert "never logged or stored" in message
    assert _TOKEN not in message


def test_non_standard_key_is_rejected_before_network_or_media_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DASHSCOPE_API_KEY, "compatible-access-credential")

    def _explode(*args, **kwargs):
        raise AssertionError("SDK must not be called with a non-sk key")

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", _explode)
    executor = DashScopeQwenExecutor()

    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )

    message = str(exc_info.value)
    assert "sk-ws-" in message
    assert "compatible-access-credential" not in message
    assert executor.last_metadata is None


def test_sk_ws_key_with_punctuation_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "sk-ws-example.segment+/=_value"
    monkeypatch.setenv(DASHSCOPE_API_KEY, token)
    called = False

    def _reach_sdk(*args, **kwargs):
        nonlocal called
        called = True
        raise RuntimeError("transport reached")

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", _reach_sdk)
    executor = DashScopeQwenExecutor()
    with pytest.raises(RenderError, match="transport reached"):
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert called is True


def test_sanitize_redacts_sk_ws_key_with_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "sk-ws-example.segment+/=_value"
    sdk = _FakeSdk(call_error=RuntimeError(f"boom {token}"))
    executor = _executor(sdk, monkeypatch=monkeypatch)

    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )

    assert token not in str(exc_info.value)
    assert "***" in str(exc_info.value)


def test_injected_call_fn_does_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DASHSCOPE_API_KEY, raising=False)
    sdk = _FakeSdk()
    executor = DashScopeQwenExecutor(
        call_fn=sdk.call,
        download_fn=sdk.download,
    )
    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert result == _OUT
    assert len(sdk.calls) == 1
    assert sdk.calls[0][1] is None


def _block_dashscope_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _no_dashscope(name: str, *args, **kwargs):
        if name == "dashscope" or name.startswith("dashscope."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_dashscope)


def test_sdk_availability_resolved_before_key_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_dashscope_import(monkeypatch)

    class _ExplodingEnv(dict):
        def get(self, key, default=None):
            if key == DASHSCOPE_API_KEY:
                raise AssertionError(
                    "DASHSCOPE_API_KEY must not be read before SDK "
                    "availability is resolved"
                )
            return dict.get(self, key, default)

    monkeypatch.setattr(
        executor_module,
        "os",
        types.SimpleNamespace(environ=_ExplodingEnv()),
    )
    executor = DashScopeQwenExecutor()

    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )

    message = str(exc_info.value)
    assert "DashScope SDK is not available" in message
    assert "dashscope" in message.lower()
    assert executor.last_request_summary is not None
    serialized = json.dumps(executor.last_request_summary)
    assert _TOKEN not in serialized
    assert "C:" not in serialized


def test_injected_call_fn_remains_keyless_when_sdk_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_dashscope_import(monkeypatch)
    monkeypatch.delenv(DASHSCOPE_API_KEY, raising=False)
    sdk = _FakeSdk()
    executor = DashScopeQwenExecutor(
        call_fn=sdk.call,
        download_fn=sdk.download,
    )

    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )

    assert result == _OUT
    assert len(sdk.calls) == 1
    assert sdk.calls[0][1] is None


def test_last_request_summary_is_safe_and_matches_compiled_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    compiled = _compiled_request(seed=7)

    executor.execute(
        request_payload=compiled,
        operation="character_synthesis",
        inputs=_inputs(),
    )

    request, _ = sdk.calls[0]
    summary = executor.last_request_summary
    assert summary is not None
    assert summary["model"] == request["model"] == DASHSCOPE_MODEL_ID
    content = request["input"]["messages"][0]["content"]
    images = [item for item in content if "image" in item]
    texts = [item for item in content if "text" in item]
    assert summary["roles"] == ["WHO", "HOW"]
    assert [entry["role"] for entry in summary["inputs"]] == ["WHO", "HOW"]
    for entry, item in zip(summary["inputs"], images, strict=True):
        base64_data = item["image"].partition(",")[2]
        raw = base64.b64decode(base64_data)
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["base64_chars"] == len(base64_data)
    assert summary["prompt_sha256"] == hashlib.sha256(
        texts[0]["text"].encode("utf-8")
    ).hexdigest()
    assert summary["parameters"] == request["parameters"]
    assert summary["parameters"] == compiled["parameters"]

    serialized = json.dumps(summary)
    assert _TOKEN not in serialized
    assert base64.b64encode(_PNG).decode("ascii") not in serialized
    assert base64.b64encode(_JPEG).decode("ascii") not in serialized
    assert "Keep identity from image 1" not in serialized
    assert "cond_001" not in serialized
    assert "cond_002" not in serialized
    assert "C:" not in serialized


@pytest.mark.parametrize(
    ("adapter_id", "expected_roles"),
    [
        (
            "qwen-image-30-adapter-v6-reference-first-context",
            ["WHO", "CONTEXT"],
        ),
        (
            "qwen-image-30-adapter-v6-continuity-action-delta",
            ["WHO", "PREVIOUS"],
        ),
        (
            "qwen-image-30-adapter-v7-first-frame-synthesize_component-identity-pose",
            ["WHO", "HOW"],
        ),
        (
            "qwen-image-30-adapter-v7-first-frame-synthesize_component-identity-prop",
            ["WHO", "PROP"],
        ),
    ],
)
def test_request_summary_names_reference_first_roles(
    monkeypatch: pytest.MonkeyPatch,
    adapter_id: str,
    expected_roles: list[str],
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(adapter_id=adapter_id),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert executor.last_request_summary is not None
    assert executor.last_request_summary["roles"] == expected_roles


def test_request_summary_names_direct_frame_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    request = _compiled_request(
        adapter_id=(
            "qwen-image-30-adapter-v7-direct-frame-identity-prop-prop"
        )
    )
    request["conditions"] = [
        {"slot": 1, "condition_ref": "cond_001"},
        {"slot": 2, "condition_ref": "cond_002"},
        {"slot": 3, "condition_ref": "cond_003"},
    ]
    executor.execute(
        request_payload=request,
        operation="direct_scene_synthesis",
        inputs={"cond_001": _PNG, "cond_002": _JPEG, "cond_003": _PNG},
    )
    assert executor.last_request_summary is not None
    assert executor.last_request_summary["roles"] == ["WHO", "PROP", "PROP"]


def test_wrong_model_is_rejected_without_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(model_id="qwen-image-edit-2511"),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert DASHSCOPE_MODEL_ID in str(exc_info.value)
    assert sdk.calls == []


def test_missing_condition_input_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs={"cond_001": _PNG},
        )
    assert sdk.calls == []


@pytest.mark.parametrize("condition_count", [4])
def test_condition_count_outside_0_to_3_rejected(
    monkeypatch: pytest.MonkeyPatch,
    condition_count: int,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    request = _compiled_request()
    request["conditions"] = [
        {"slot": i + 1, "condition_ref": f"cond_{i:03d}"}
        for i in range(condition_count)
    ]
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=request,
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert sdk.calls == []


def test_first_frame_text_only_base_accepts_zero_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    request = _compiled_request(
        adapter_id=(
            "qwen-image-30-adapter-v7-first-frame-synthesize_base"
        ),
        prompt="Generate only the uncovered scene base facts.",
    )
    request["conditions"] = []

    result = executor.execute(
        request_payload=request,
        operation="first_frame_fusion",
        inputs={},
    )

    assert result == _OUT
    sent, _ = _sent_request(sdk)
    assert sent["input"]["messages"][0]["content"] == [
        {"text": request["prompt"]}
    ]
    assert executor.last_request_summary["roles"] == []


def test_local_inpaint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(EnvironmentCapabilityError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="local_inpaint",
            inputs={"composite_image": _PNG, "inpaint_mask": b"x"},
        )
    message = str(exc_info.value)
    assert "no independent mask input" in message
    assert "mask" in message
    assert sdk.calls == []


def test_unsupported_operation_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=_compiled_request(),
            operation="video_synthesis",
            inputs=_inputs(),
        )
    assert sdk.calls == []


def test_http_failure_is_render_error_no_retry_no_body_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(
        response=_sdk_response(
            status_code=400,
            request_id="req_http_fail",
            code="InvalidParameter",
            message="sk-abc123 data:image/png;base64,AAAA",
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    message = str(exc_info.value)
    assert "HTTP status 400" in message
    assert "sk-abc123" not in message
    assert "data:image/png;base64,AAAA" not in message
    assert "req_http_fail" not in message  # raw response never echoed
    assert len(sdk.calls) == 1  # no automatic retry
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "failed"
    assert executor.last_metadata["request_id"] == "req_http_fail"
    assert executor.last_metadata["http_status"] == 400
    assert executor.last_metadata["provider_code"] == "InvalidParameter"
    assert "sk-abc123" not in executor.last_metadata["provider_message"]
    assert "data:image/png;base64,AAAA" not in (
        executor.last_metadata["provider_message"]
    )


def test_call_exception_wrapped_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(
        call_error=RuntimeError(
            "boom sk-abc123 data:image/png;base64,AAAA"
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    message = str(exc_info.value)
    assert "boom" in message
    assert "RuntimeError" in message
    assert "***" in message
    assert "sk-abc123" not in message
    assert "data:image/png;base64,AAAA" not in message
    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert "sk-abc123" not in formatted
    assert "data:image/png;base64,AAAA" not in formatted
    assert len(sdk.calls) == 1
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "failed"


def test_unusable_response_is_render_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(
        response=_FakeSdkResponse(request_id="req_no_status")
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "unusable response" in str(exc_info.value)
    assert sdk.downloads == []
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "failed"


def test_response_without_image_url_is_render_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(
        response=_sdk_response(
            output={
                "choices": [
                    {"message": {"content": [{"text": "no image"}]}}
                ]
            }
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "no usable image URL" in str(exc_info.value)
    assert sdk.downloads == []


def test_download_failure_wrapped_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(download_error=RuntimeError("download sk-abc123"))
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    message = str(exc_info.value)
    assert "download failed" in message
    assert "***" in message
    assert "sk-abc123" not in message
    formatted = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert "sk-abc123" not in formatted
    assert "data:image/png;base64,AAAA" not in formatted


def test_empty_download_is_render_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(download=b"")
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "no usable bytes" in str(exc_info.value)


@pytest.mark.parametrize(
    ("image_url", "allowed", "fragment"),
    [
        ("https://dashscope.aliyuncs.com/result.png", True, ""),
        (
            "https://dashscope-result-oss-cn-hangzhou.aliyuncs.com/x.png",
            True,
            "",
        ),
        ("http://dashscope.aliyuncs.com/result.png", False, "must use HTTPS"),
        ("https://evil.example.com/x.png", False, "not an allowed"),
    ],
)
def test_output_url_https_and_host_policy(
    monkeypatch: pytest.MonkeyPatch,
    image_url: str,
    allowed: bool,
    fragment: str,
) -> None:
    sdk = _FakeSdk(
        response=_sdk_response(
            output={
                "choices": [
                    {"message": {"content": [{"image": image_url}]}}
                ]
            }
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    if allowed:
        result = executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
        assert result == _OUT
        assert sdk.downloads == [image_url]
    else:
        with pytest.raises(RenderError) as exc_info:
            executor.execute(
                request_payload=_compiled_request(),
                operation="character_synthesis",
                inputs=_inputs(),
            )
        assert fragment in str(exc_info.value)
        assert image_url not in str(exc_info.value)
        assert sdk.downloads == []


def test_success_downloads_first_image_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_url = "https://dashscope.aliyuncs.com/first.png"
    second_url = "https://dashscope.aliyuncs.com/second.png"
    sdk = _FakeSdk(
        response=_sdk_response(
            output={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"image": first_url},
                                {"image": second_url},
                            ]
                        }
                    }
                ]
            }
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert result == _OUT
    assert sdk.downloads == [first_url]
    assert len(sdk.calls) == 1


def test_metadata_is_sanitized_and_never_contains_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = "https://dashscope.aliyuncs.com/result.png"
    usage = dict(_DEFAULT_USAGE)
    usage["unexpected_field"] = "sk-zzz"
    sdk = _FakeSdk(
        response=_sdk_response(
            request_id="req_secret_check",
            output={
                "choices": [{"message": {"content": [{"image": image_url}]}}]
            },
            usage=usage,
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    metadata = executor.last_metadata
    assert metadata is not None
    assert metadata["provider"] == "dashscope"
    assert metadata["model"] == DASHSCOPE_MODEL_ID
    assert metadata["request_id"] == "req_secret_check"
    assert metadata["status"] == "succeeded"
    assert isinstance(metadata["duration_ms"], float)
    assert set(metadata["usage"]) == {
        "output_width",
        "output_height",
        "input_image_count",
        "input_image_type",
        "output_image_count",
        "output_image_type",
    }
    serialized = json.dumps(metadata)
    assert _TOKEN not in serialized
    assert base64.b64encode(_PNG).decode("ascii") not in serialized
    assert base64.b64encode(_JPEG).decode("ascii") not in serialized
    assert image_url not in serialized
    assert "Keep identity from image 1" not in serialized
    assert "sk-zzz" not in serialized


def test_token_only_in_call_and_never_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    request, api_key = sdk.calls[0]
    assert api_key == _TOKEN
    assert _TOKEN not in json.dumps(request)
    for name, value in vars(executor).items():
        assert _TOKEN not in repr(value), f"token leaked via {name}"
    assert _TOKEN not in json.dumps(executor.last_metadata)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"seed": "0"}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**31}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"n": 0}, "n"),
        ({"n": 7}, "n"),
        ({"size": "1280x720"}, "size"),
        ({"size": ""}, "size"),
        ({"prompt_extend": "false"}, "prompt_extend"),
        ({"prompt_extend_mode": "auto"}, "prompt_extend_mode"),
        ({"watermark": 1}, "watermark"),
        ({"negative_prompt": "  "}, "negative_prompt"),
    ],
)
def test_invalid_parameter_values_rejected(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    fragment: str,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(**overrides),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert fragment in str(exc_info.value)
    assert sdk.calls == []


def test_missing_required_parameter_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    request = _compiled_request()
    del request["parameters"]["seed"]
    with pytest.raises(InputValidationError) as exc_info:
        executor.execute(
            request_payload=request,
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "seed" in str(exc_info.value)
    assert sdk.calls == []


def test_unsupported_image_magic_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs={"cond_001": b"not-an-image", "cond_002": _JPEG},
        )
    assert "PNG or JPEG" in str(exc_info.value)
    assert sdk.calls == []


def test_missing_prompt_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=_compiled_request(prompt="   "),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert sdk.calls == []


def test_download_wrong_format_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(download=_JPEG + b"not-png")
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert "not a PNG" in str(exc_info.value)
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "failed"
    assert executor.last_metadata["request_id"] == "req_abc"


@pytest.mark.parametrize(
    ("request_id", "expected"),
    [
        (
            "9b1b5c34-2d0e-4e4f-a1b2-1234567890ab",
            "9b1b5c34-2d0e-4e4f-a1b2-1234567890ab",
        ),
        ("req_abc", "req_abc"),
        ("sk-abc123", None),
        ("data:image/png;base64,AAAA", None),
        ("x" * 200, None),
    ],
)
def test_request_id_sanitized_before_metadata(
    monkeypatch: pytest.MonkeyPatch,
    request_id: str,
    expected: str | None,
) -> None:
    sdk = _FakeSdk(response=_sdk_response(request_id=request_id))
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert executor.last_metadata is not None
    assert executor.last_metadata["request_id"] == expected
    serialized = json.dumps(executor.last_metadata)
    assert "sk-abc123" not in serialized
    assert "data:image/png;base64,AAAA" not in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_width", -1),
        ("output_width", "1280"),
        ("output_height", 1280.0),
        ("input_image_count", -2),
        ("input_image_type", "sk-abc123"),
        ("output_image_type", "data:image/png;base64,AAAA"),
        ("output_image_type", "x" * 33),
    ],
)
def test_usage_adversarial_values_discarded(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    usage = dict(_DEFAULT_USAGE)
    usage[field] = value
    sdk = _FakeSdk(response=_sdk_response(usage=usage))
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert executor.last_metadata is not None
    assert field not in executor.last_metadata["usage"]
    serialized = json.dumps(executor.last_metadata)
    assert "sk-abc123" not in serialized
    assert "data:image/png;base64,AAAA" not in serialized


def test_url_validation_failure_sets_failed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_url = "https://evil.example.com/x.png"
    sdk = _FakeSdk(
        response=_sdk_response(
            request_id="req_bad_host",
            output={
                "choices": [{"message": {"content": [{"image": image_url}]}}]
            },
        )
    )
    executor = _executor(sdk, monkeypatch=monkeypatch)
    with pytest.raises(RenderError):
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    metadata = executor.last_metadata
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert metadata["request_id"] == "req_bad_host"
    assert image_url not in json.dumps(metadata)


def test_execute_resets_last_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs=_inputs(),
    )
    assert executor.last_metadata is not None
    assert executor.last_metadata["status"] == "succeeded"
    assert executor.last_request_summary is not None
    with pytest.raises(InputValidationError):
        executor.execute(
            request_payload=_compiled_request(model_id="qwen-image-edit-2511"),
            operation="character_synthesis",
            inputs=_inputs(),
        )
    assert executor.last_metadata is None
    assert executor.last_request_summary is None


def test_input_image_over_10_mib_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    oversized = _PNG_MAGIC + b"x" * (10 * 1024 * 1024 - len(_PNG_MAGIC) + 1)
    with pytest.raises(InputValidationError) as exc_info:
        executor.execute(
            request_payload=_compiled_request(),
            operation="character_synthesis",
            inputs={"cond_001": oversized, "cond_002": _JPEG},
        )
    assert "10 MiB" in str(exc_info.value)
    assert sdk.calls == []


def test_input_image_exactly_10_mib_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    boundary = _PNG_MAGIC + b"x" * (10 * 1024 * 1024 - len(_PNG_MAGIC))
    result = executor.execute(
        request_payload=_compiled_request(),
        operation="character_synthesis",
        inputs={"cond_001": boundary, "cond_002": _JPEG},
    )
    assert result == _OUT


@pytest.mark.parametrize(
    ("size", "allowed"),
    [
        ("512*512", True),
        ("2048*2048", True),
        ("512*4096", True),
        ("4096*512", True),
        ("1280*720", True),
        ("512*511", False),
        ("2048*2049", False),
        ("512*4097", False),
        ("4097*512", False),
        ("256*1023", False),
        (1024, False),
    ],
)
def test_executor_size_constraints(
    monkeypatch: pytest.MonkeyPatch,
    size: str | int,
    allowed: bool,
) -> None:
    sdk = _FakeSdk()
    executor = _executor(sdk, monkeypatch=monkeypatch)
    if allowed:
        result = executor.execute(
            request_payload=_compiled_request(size=size),
            operation="character_synthesis",
            inputs=_inputs(),
        )
        assert result == _OUT
    else:
        with pytest.raises(InputValidationError):
            executor.execute(
                request_payload=_compiled_request(size=size),
                operation="character_synthesis",
                inputs=_inputs(),
            )
        assert sdk.calls == []
