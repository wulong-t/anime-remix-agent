"""DashScope-hosted ``qwen-image-3.0`` executor (Phase 3 Real, local).

Round 11 freeze (2026-08-11): the executor is the only layer that touches a
real model.  It consumes an already-compiled ``ModelRenderRequest``
(``request_payload``) and returns output image bytes; it never compiles
prompts, selects references, routes failures or writes ledger facts.

API facts (frozen 2026-08-11):

- Default transport: the official Python DashScope SDK
  ``MultiModalConversation.call``, which POSTs to
  ``https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation``
  with ``Authorization: Bearer`` built from the api key.
- The API key is read only at call time from ``DASHSCOPE_API_KEY``; never
  stored, logged or serialized.  The default transport requires it; an
  injected ``call_fn`` (offline seam) does not.
- The default call pins ``dashscope.base_http_api_url`` to
  ``https://dashscope.aliyuncs.com/api/v1`` and passes
  ``request_timeout=300``.  It deliberately omits the SDK ``stream`` kwarg:
  SDK 1.26.6 serializes an explicit ``stream=False`` into the model's
  ``parameters`` object, while the Qwen-Image 3.0 API does not define that
  parameter and the official SDK example omits it.  The ``dashscope`` logger
  is raised to INFO so SDK DEBUG logging cannot emit the full Base64 payload.
  SDK availability is resolved before ``DASHSCOPE_API_KEY`` is read; an
  unavailable SDK raises a clear ``EnvironmentCapabilityError`` while an
  injected ``call_fn`` stays keyless.
- ``model``: ``qwen-image-3.0`` (standard, non-pro); this executor refuses any
  other model rather than silently changing a request.
  rather than silently changing a request.
- Reference assets are PNG/JPEG only and each input must be at most 10 MiB
  (project reference-asset boundary).
- ``input.messages`` is exactly one user message; the internal compiled
  ``content`` has 0-3 image objects (``data:<mime>;base64,<data>``) followed
  by exactly one text object.  Immediately around a real call, the executor
  materializes those validated bytes as private temporary ``file://``
  references.  The official SDK uploads them to temporary OSS, substitutes
  short ``oss://`` references and adds its resource-resolution header.  This
  avoids a multi-megabyte JSON POST without exposing original local paths;
  the temporary local copies are removed after the call returns.
- Supported parameters: ``n`` (1..6), ``negative_prompt``, ``size`` (a
  ``W*H`` string; total pixels 512*512..2048*2048, aspect 1:8..8:1),
  ``prompt_extend``, ``prompt_extend_mode`` (editing must be ``direct``),
  ``watermark``, ``seed`` (0..2147483647 inclusive).  Only these are sent,
  flattened as SDK call kwargs (the SDK maps them to request ``parameters``).
- The output image URL is ``output.choices[0].message.content[*].image``,
  a PNG that expires in 24h and must be downloaded immediately.
- The model rate limit is 1 RPM; there is no automatic retry.  The executor
  calls ``MultiModalConversation.call`` exactly once, producing exactly one
  model-generation POST.  Before that POST, the SDK obtains one reusable
  temporary-upload certificate and uploads each reference once; those
  storage operations are data transfer, not additional model invocations.

The API has no independent mask input, so ``local_inpaint`` is refused with
an ``EnvironmentCapabilityError`` instead of attempting an unconstrained
edit.

Transport seam: ``call_fn`` and ``download_fn`` are injectable so unit tests
never touch the network.  ``call_fn`` receives the compiled SDK request dict
and the api key, and returns an SDK-style response object.  The default
implementation is the official ``MultiModalConversation.call``.

Secret boundary: the API key is read from ``DASHSCOPE_API_KEY`` only at call
time, is never stored on this object, and never appears in ``last_metadata``
or any ledger/manifest payload.  Errors are sanitized so ``sk-*`` secrets and
``data:image/...;base64`` payloads cannot leak, and raw response bodies are
never echoed.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from anime_remix.errors import (
    EnvironmentCapabilityError,
    InputValidationError,
    RenderError,
)

DASHSCOPE_API_KEY = "DASHSCOPE_API_KEY"
DASHSCOPE_MODEL_ID = "qwen-image-3.0"
DASHSCOPE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "aigc/multimodal-generation/generation"
)
DASHSCOPE_BASE_HTTP_API_URL = "https://dashscope.aliyuncs.com/api/v1"

_REQUEST_TIMEOUT_S = 300
_DOWNLOAD_TIMEOUT_S = 60.0
_SEED_MAX = 2**31 - 1
_MAX_INPUT_BYTES = 10 * 1024 * 1024
_DASHSCOPE_LOGGER_NAME = "dashscope"
_INPUT_ROLES = ("WHO", "HOW", "WHERE")
_SIZE_PATTERN = re.compile(r"^[1-9][0-9]*\*[1-9][0-9]*$")
_MIN_SIZE_PIXELS = 512 * 512
_MAX_SIZE_PIXELS = 2048 * 2048
_MAX_ASPECT_NUM = 8
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_SK_PATTERN = re.compile(r"sk-[^\s'\"<>]+")
_DATA_URL_PATTERN = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+"
)
_DASHSCOPE_KEY_PATTERN = re.compile(r"^sk-\S+$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_REQUEST_ID_LENGTH = 128
_PROVIDER_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_PROVIDER_MESSAGE_LENGTH = 512
_USAGE_TEXT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,31}$")
_USAGE_FIELDS = (
    "output_width",
    "output_height",
    "input_image_count",
    "input_image_type",
    "output_image_count",
    "output_image_type",
)
_USAGE_NUMERIC_FIELDS = frozenset(
    ("output_width", "output_height", "input_image_count", "output_image_count")
)
_USAGE_TEXT_FIELDS = frozenset(("input_image_type", "output_image_type"))

SdkCallFn = Callable[[dict[str, Any], str], Any]
DownloadFn = Callable[[str], bytes]


def _default_call(request: dict[str, Any], api_key: str) -> Any:
    """Call the official DashScope SDK exactly once (no automatic retry).

    ``request`` is the compiled frozen request dict (``model``, ``input``,
    ``parameters``).  Parameters are flattened into SDK call kwargs, which
    the SDK maps to the HTTP request ``parameters`` field.  The endpoint is
    pinned explicitly, the SDK ``stream`` kwarg is omitted so it cannot leak
    into the model parameters, the SDK logger is forced to INFO-or-higher
    before the request, and the timeout is frozen at 300s.
    """

    import dashscope
    from dashscope import MultiModalConversation

    _force_dashscope_log_level()
    dashscope.base_http_api_url = DASHSCOPE_BASE_HTTP_API_URL
    with _sdk_temporary_file_messages(request) as messages:
        return MultiModalConversation.call(
            api_key=api_key,
            model=request["model"],
            messages=messages,
            request_timeout=_REQUEST_TIMEOUT_S,
            **request["parameters"],
        )


@contextmanager
def _sdk_temporary_file_messages(
    request: dict[str, Any],
) -> Iterator[list[dict[str, Any]]]:
    """Materialize compiled data URLs for the SDK's temporary OSS upload."""

    messages = copy.deepcopy(request["input"]["messages"])
    with TemporaryDirectory(prefix="anime-remix-dashscope-") as directory:
        reference_index = 0
        for message in messages:
            for item in message.get("content", []):
                data_url = item.get("image")
                if not isinstance(data_url, str):
                    continue
                metadata, separator, encoded = data_url.partition(",")
                if separator != "," or not metadata.endswith(";base64"):
                    continue
                if metadata == "data:image/png;base64":
                    suffix = ".png"
                elif metadata == "data:image/jpeg;base64":
                    suffix = ".jpg"
                else:  # Defensive: compilation accepts only PNG/JPEG.
                    raise InputValidationError(
                        "unsupported compiled DashScope image MIME type"
                    )
                reference_index += 1
                path = Path(directory) / f"reference_{reference_index}{suffix}"
                path.write_bytes(base64.b64decode(encoded, validate=True))
                item["image"] = path.resolve().as_uri()
        yield messages


def _force_dashscope_log_level() -> None:
    """Raise the DashScope SDK logger to INFO so DEBUG cannot leak Base64.

    DashScope 1.26.6 logs the full request body - including the Base64
    image data - at DEBUG through the ``dashscope`` logger (see
    ``dashscope.api_entities.http_request``).  Forcing INFO-or-higher before
    every default call keeps the compiled request out of log handlers.
    """

    logging.getLogger(_DASHSCOPE_LOGGER_NAME).setLevel(logging.INFO)


def _default_sdk_available() -> bool:
    """Return whether the official DashScope SDK import resolves."""

    try:
        import dashscope  # noqa: F401
        from dashscope import MultiModalConversation  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def _request_summary(
    request: dict[str, Any],
    *,
    input_roles: tuple[str, ...] = _INPUT_ROLES,
) -> dict[str, Any]:
    """Build a safe manifest summary from the compiled SDK request.

    The summary contains the model id, role-ordered input digests and
    sizes, the prompt SHA-256 and the frozen parameters.  It never contains
    file paths, the raw prompt, raw Base64 payloads or API keys.
    """

    messages = request["input"]["messages"]
    content = messages[0]["content"] if messages else []
    images = [item for item in content if "image" in item]
    texts = [item for item in content if "text" in item]
    prompt = texts[0]["text"] if texts else ""
    inputs_summary: list[dict[str, Any]] = []
    for index, item in enumerate(images):
        data_url = item["image"]
        _, _, base64_data = data_url.partition(",")
        try:
            raw = base64.b64decode(base64_data)
        except (TypeError, ValueError):
            raw = b""
        inputs_summary.append(
            {
                "role": (
                    input_roles[index]
                    if index < len(input_roles)
                    else f"SLOT{index + 1}"
                ),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "base64_chars": len(base64_data),
            }
        )
    return {
        "model": request["model"],
        "roles": [entry["role"] for entry in inputs_summary],
        "inputs": inputs_summary,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "parameters": dict(request["parameters"]),
    }


def _compiled_input_roles(request_payload: dict, count: int) -> tuple[str, ...]:
    adapter_id = request_payload.get("adapter_id")
    if isinstance(adapter_id, str) and adapter_id.startswith(
        "qwen-image-30-adapter-v7-first-frame-"
    ):
        if "synthesize_component" in adapter_id:
            role_suffix = adapter_id.split("synthesize_component-", maxsplit=1)[-1]
            component_roles = {
                "identity": "WHO",
                "pose": "HOW",
                "prop": "PROP",
            }
            return tuple(
                component_roles.get(item, "COMPONENT")
                for item in role_suffix.split("-")[:count]
            )
        if adapter_id.endswith("fuse_component"):
            return ("CANVAS", "COMPONENT")[:count]
        if adapter_id.endswith("apply_text_delta"):
            return ("CANVAS",)[:count]
        return ("SCENE", "STYLE")[:count]
    if isinstance(adapter_id, str) and adapter_id.startswith(
        "qwen-image-30-adapter-v7-direct-frame-"
    ):
        role_suffix = adapter_id[
            len("qwen-image-30-adapter-v7-direct-frame-") :
        ]
        role_map = {
            "identity": "WHO",
            "pose": "HOW",
            "expression": "HOW",
            "prop": "PROP",
            "scene": "SCENE",
            "background": "SCENE",
            "style": "STYLE",
            "source_frame": "CANVAS",
        }
        roles = [
            role_map.get(item, "COMPONENT")
            for item in role_suffix.split("-")[:count]
        ]
        while len(roles) < count:
            roles.append(f"SLOT{len(roles) + 1}")
        return tuple(roles)
    if count == 1:
        return ("WHO",)
    if adapter_id == "qwen-image-30-adapter-v6-reference-first-context":
        return ("WHO", "CONTEXT")
    if adapter_id == "qwen-image-30-adapter-v6-continuity-action-delta":
        return ("WHO", "PREVIOUS")
    return _INPUT_ROLES


def _default_download(url: str) -> bytes:
    """Download the generated image bytes immediately (URL pre-validated)."""

    with urllib_request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        return response.read()


def _sanitize(value: object) -> str:
    """Redact ``sk-*`` secrets and Base64 data URLs from a message."""

    text = str(value)
    text = _SK_PATTERN.sub("***", text)
    text = _DATA_URL_PATTERN.sub("data:image/***;base64,***", text)
    return text


def _detect_image_mime(data: bytes) -> str:
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    raise InputValidationError(
        "unsupported input image: expected PNG or JPEG magic bytes"
    )


def _sanitize_request_id(value: object) -> str | None:
    """Return a safe request id or ``None`` for untrusted provider input."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_REQUEST_ID_LENGTH:
        return None
    if not _REQUEST_ID_PATTERN.fullmatch(text):
        return None
    if _SK_PATTERN.search(text) or "data:image/" in text.lower():
        return None
    return text


def _response_get(response: object, key: str) -> Any:
    """Read a field from an SDK-style response (dict or attribute access)."""

    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _extract_status_code(response: object) -> int | None:
    value = _response_get(response, "status_code")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _extract_request_id(response: object) -> str | None:
    return _sanitize_request_id(_response_get(response, "request_id"))


def _sanitize_provider_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not _PROVIDER_CODE_PATTERN.fullmatch(text):
        return None
    return text


def _sanitize_provider_message(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(_sanitize(value).split())
    if not text:
        return None
    return text[:_MAX_PROVIDER_MESSAGE_LENGTH]


def _extract_first_image_url(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    choices = output.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image = item.get("image")
            if isinstance(image, str) and image.strip():
                return image.strip()
    return None


def _sanitize_usage_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 32:
        return None
    if not _USAGE_TEXT_PATTERN.fullmatch(text):
        return None
    if _SK_PATTERN.search(text) or "data:image/" in text.lower():
        return None
    return text


def _sanitize_usage(payload: object) -> dict[str, Any]:
    output = _response_get(payload, "output")
    usage = output.get("usage") if isinstance(output, dict) else None
    if not isinstance(usage, dict):
        usage = _response_get(payload, "usage")
    if not isinstance(usage, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key in _USAGE_FIELDS:
        if key not in usage:
            continue
        value = usage[key]
        if key in _USAGE_NUMERIC_FIELDS:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _SEED_MAX
            ):
                continue
            sanitized[key] = value
        elif key in _USAGE_TEXT_FIELDS:
            text = _sanitize_usage_text(value)
            if text is not None:
                sanitized[key] = text
    return sanitized


def _validate_output_url(url: str) -> None:
    try:
        parsed = urllib_parse.urlparse(url)
        host = parsed.hostname
    except ValueError:
        raise RenderError("dashscope returned an invalid image URL") from None
    if parsed.scheme != "https" or not host:
        raise RenderError("dashscope image URL must use HTTPS")
    if host != "aliyuncs.com" and not host.endswith(".aliyuncs.com"):
        raise RenderError(
            "dashscope image URL host is not an allowed aliyuncs.com host"
        )


def _validate_seed(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(
            f"invalid parameter {key!r}: expected an integer"
        )
    if not 0 <= value <= _SEED_MAX:
        raise InputValidationError(
            f"invalid parameter {key!r}: expected an integer in 0..{_SEED_MAX}"
        )
    return value


def _validate_n(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(
            f"invalid parameter {key!r}: expected an integer"
        )
    if not 1 <= value <= 6:
        raise InputValidationError(
            f"invalid parameter {key!r}: expected an integer in 1..6"
        )
    return value


def _validate_size(key: str, value: object) -> str:
    if not isinstance(value, str):
        raise InputValidationError(
            f"invalid parameter {key!r}: expected a string like '1280*720'"
        )
    normalized = value.strip()
    if not _SIZE_PATTERN.match(normalized):
        raise InputValidationError(
            f"invalid parameter {key!r}: expected 'W*H' like '1280*720'"
        )
    width, height = (int(part) for part in normalized.split("*"))
    if not _MIN_SIZE_PIXELS <= width * height <= _MAX_SIZE_PIXELS:
        raise InputValidationError(
            f"invalid parameter {key!r}: total pixels must be within "
            f"{_MIN_SIZE_PIXELS}..{_MAX_SIZE_PIXELS}"
        )
    if not (
        width * _MAX_ASPECT_NUM >= height
        and height * _MAX_ASPECT_NUM >= width
    ):
        raise InputValidationError(
            f"invalid parameter {key!r}: aspect ratio must be within "
            "1:8 and 8:1"
        )
    return normalized


def _validate_bool(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InputValidationError(
            f"invalid parameter {key!r}: expected a boolean"
        )
    return value


def _validate_prompt_extend_mode(key: str, value: object) -> str:
    if value != "direct":
        raise InputValidationError(
            f"invalid parameter {key!r}: qwen-image-3.0 editing "
            "requires prompt_extend_mode='direct'"
        )
    return "direct"


def _validate_negative_prompt(key: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(
            f"invalid parameter {key!r}: expected a non-empty string"
        )
    return value.strip()


_PARAMETER_VALIDATORS: dict[str, Callable[[str, object], Any]] = {
    "seed": _validate_seed,
    "n": _validate_n,
    "size": _validate_size,
    "prompt_extend": _validate_bool,
    "prompt_extend_mode": _validate_prompt_extend_mode,
    "watermark": _validate_bool,
    "negative_prompt": _validate_negative_prompt,
}


class DashScopeQwenExecutor:
    """Execute a frozen ModelRenderRequest on DashScope's qwen-image-3.0."""

    provider = "dashscope"

    def __init__(
        self,
        *,
        call_fn: SdkCallFn | None = None,
        download_fn: DownloadFn | None = None,
    ) -> None:
        self._call_fn = call_fn if call_fn is not None else _default_call
        # Only the default SDK transport needs the real API key; injected
        # callables are the offline/dry-run seam and may run keyless.
        self._uses_default_call = call_fn is None
        self._download_fn = (
            download_fn if download_fn is not None else _default_download
        )
        self.last_metadata: dict[str, Any] | None = None
        self.last_request_summary: dict[str, Any] | None = None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.last_metadata = None
        self.last_request_summary = None
        if operation in {
            "character_synthesis",
            "first_frame_fusion",
            "direct_scene_synthesis",
        }:
            return self._execute_character_synthesis(request_payload, inputs)
        if operation == "local_inpaint":
            raise EnvironmentCapabilityError(
                "DashScope qwen-image-3.0 has no independent mask input; "
                "the frozen local_inpaint contract (mask inside mutable / "
                "outside protected) cannot be represented through this API. "
                "Refusing an unconstrained editing request."
            )
        raise InputValidationError(
            f"DashScopeQwenExecutor does not support operation {operation!r}"
        )

    def _execute_character_synthesis(
        self,
        request_payload: dict,
        inputs: dict[str, bytes],
    ) -> bytes:
        request = self._compile_request(request_payload, inputs)
        self.last_request_summary = _request_summary(
            request,
            input_roles=_compiled_input_roles(
                request_payload,
                len(request_payload["conditions"]),
            ),
        )
        self.last_request_summary["input_transport"] = (
            "dashscope_sdk_temporary_oss"
            if self._uses_default_call
            else "injected_transport_data_url"
        )
        if self._uses_default_call and not _default_sdk_available():
            raise EnvironmentCapabilityError(
                "DashScope SDK is not available: the default transport "
                "requires the project dependency 'dashscope>=1.26.6,<2'; "
                "install it before running DashScope execution"
            )
        token = os.environ.get(DASHSCOPE_API_KEY)
        if not token and self._uses_default_call:
            raise EnvironmentCapabilityError(
                f"missing {DASHSCOPE_API_KEY}; set it before running "
                "DashScope execution (the key is never logged or stored)"
            )
        if (
            self._uses_default_call
            and token is not None
            and not _DASHSCOPE_KEY_PATTERN.fullmatch(token)
        ):
            raise EnvironmentCapabilityError(
                f"invalid {DASHSCOPE_API_KEY} credential type: the native "
                "DashScope multimodal API requires an sk-, sk-ws-, or "
                "sk-sp- API key without whitespace; "
                "a compatible-mode or plan access credential cannot be "
                "used here (the supplied value was not logged or stored)"
            )
        started = time.monotonic()
        try:
            response = self._call_fn(request, token)
        except Exception as exc:  # noqa: BLE001 - transport seam may raise anything
            self.last_metadata = self._failure_metadata(None, started)
            raise RenderError(
                "dashscope request failed: "
                f"{type(exc).__name__}: {_sanitize(exc)}"
            ) from None
        request_id = _extract_request_id(response)
        status = _extract_status_code(response)
        if status is None:
            self.last_metadata = self._failure_metadata(
                request_id,
                started,
                response=response,
            )
            raise RenderError("dashscope returned an unusable response")
        if status != 200:
            self.last_metadata = self._failure_metadata(
                request_id,
                started,
                response=response,
                http_status=status,
            )
            raise RenderError(
                f"dashscope request failed with HTTP status {status}"
            )
        image_url = _extract_first_image_url(response)
        if image_url is None:
            self.last_metadata = self._failure_metadata(request_id, started)
            raise RenderError(
                "dashscope response contains no usable image URL"
            )
        try:
            _validate_output_url(image_url)
        except RenderError:
            self.last_metadata = self._failure_metadata(request_id, started)
            raise
        try:
            image_bytes = self._download_fn(image_url)
        except Exception as exc:  # noqa: BLE001 - transport seam may raise anything
            self.last_metadata = self._failure_metadata(request_id, started)
            raise RenderError(
                "dashscope image download failed: "
                f"{type(exc).__name__}: {_sanitize(exc)}"
            ) from None
        if not isinstance(image_bytes, bytes) or not image_bytes:
            self.last_metadata = self._failure_metadata(request_id, started)
            raise RenderError(
                "dashscope image download returned no usable bytes"
            )
        if not image_bytes.startswith(_PNG_MAGIC):
            self.last_metadata = self._failure_metadata(request_id, started)
            raise RenderError(
                "dashscope image download is not a PNG; expected the "
                "API's PNG output"
            )
        self.last_metadata = {
            "provider": self.provider,
            "model": DASHSCOPE_MODEL_ID,
            "request_id": request_id,
            "status": "succeeded",
            "duration_ms": round((time.monotonic() - started) * 1000.0, 1),
            "usage": _sanitize_usage(response),
        }
        return image_bytes

    def _compile_request(
        self,
        request_payload: dict,
        inputs: dict[str, bytes],
    ) -> dict[str, Any]:
        if not isinstance(request_payload, dict):
            raise InputValidationError("request_payload must be a dict")
        if not isinstance(inputs, dict):
            raise InputValidationError("inputs must be a dict")
        model_id = request_payload.get("model_id")
        if model_id != DASHSCOPE_MODEL_ID:
            raise InputValidationError(
                "DashScope executor only serves "
                f"{DASHSCOPE_MODEL_ID!r}; refused request for model "
                f"{model_id!r} instead of silently changing it"
            )
        conditions = request_payload.get("conditions")
        if not isinstance(conditions, list) or not 0 <= len(conditions) <= 3:
            count = (
                len(conditions) if isinstance(conditions, list) else "invalid"
            )
            raise InputValidationError(
                "DashScope request accepts 0-3 condition slots, "
                f"got {count}"
            )
        prompt = request_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InputValidationError(
                "DashScope request requires a non-empty prompt"
            )
        parameters = request_payload.get("parameters")
        if not isinstance(parameters, dict):
            raise InputValidationError(
                "DashScope request parameters must be a dict"
            )
        api_parameters = {
            key: validator(key, parameters[key])
            for key, validator in _PARAMETER_VALIDATORS.items()
            if key in parameters
        }
        if len(api_parameters) != len(_PARAMETER_VALIDATORS):
            missing = sorted(set(_PARAMETER_VALIDATORS) - set(api_parameters))
            raise InputValidationError(
                "DashScope request parameters missing required "
                f"field(s): {', '.join(missing)}"
            )

        content: list[dict[str, str]] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                raise InputValidationError(
                    "invalid condition slot: expected a dict"
                )
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
            if len(data) > _MAX_INPUT_BYTES:
                raise InputValidationError(
                    f"input image for condition_ref {condition_ref!r} "
                    "exceeds the 10 MiB PNG/JPEG reference asset boundary"
                )
            mime = _detect_image_mime(data)
            data_url = (
                f"data:{mime};base64,"
                f"{base64.b64encode(data).decode('ascii')}"
            )
            content.append({"image": data_url})
        content.append({"text": prompt})
        return {
            "model": model_id,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": api_parameters,
        }

    def _failure_metadata(
        self,
        request_id: str | None,
        started: float,
        *,
        response: object | None = None,
        http_status: int | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "model": DASHSCOPE_MODEL_ID,
            "request_id": request_id,
            "status": "failed",
            "duration_ms": round((time.monotonic() - started) * 1000.0, 1),
        }
        if http_status is not None:
            metadata["http_status"] = http_status
        if response is not None:
            provider_code = _sanitize_provider_code(
                _response_get(response, "code")
            )
            provider_message = _sanitize_provider_message(
                _response_get(response, "message")
            )
            if provider_code is not None:
                metadata["provider_code"] = provider_code
            if provider_message is not None:
                metadata["provider_message"] = provider_message
        return metadata
