"""Replicate-hosted ``qwen/qwen-image-edit-2511`` executor (Phase 3 Real).

Round 10 freeze (2026-08-11): the executor is the only layer that touches a
real model.  It consumes an already-compiled ``ModelRenderRequest``
(``request_payload``) and returns output image bytes; it never compiles
prompts, selects references, routes failures or writes ledger facts.

API facts (verified 2026-08-11 against the public Replicate schema):

- model: ``qwen/qwen-image-edit-2511``, version
  ``a0670a7f47d5975347c105b6ce71456c4377d511993975988127dee03ca6c729``
- inputs: ``image`` (required list of 1..3 jpeg/png/gif/webp), ``prompt``
  (required), ``seed``, ``go_fast``, ``aspect_ratio``, ``output_format``,
  ``output_quality``, ``disable_safety_checker``.
- The hosted wrapper does NOT expose ``num_inference_steps`` /
  ``true_cfg_scale`` / ``guidance_scale`` (compiled by the frozen adapter)
  and does NOT accept an independent inpaint mask.  Unsupported sampling
  fields are dropped in the executor mapping; ``local_inpaint`` is refused
  (finding F-013) because the frozen mask contract (inside mutable / outside
  protected) cannot be expressed through this hosted API.

Secret boundary: the API token is read from ``REPLICATE_API_TOKEN`` only at
call time, is never stored on this object, and never appears in
``last_metadata`` or any ledger/manifest payload.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from anime_remix.errors import (
    EnvironmentCapabilityError,
    InputValidationError,
    RenderError,
)

REPLICATE_MODEL_ID = "qwen/qwen-image-edit-2511"
REPLICATE_MODEL_VERSION = (
    "a0670a7f47d5975347c105b6ce71456c4377d511993975988127dee03ca6c729"
)
REPLICATE_TOKEN_ENV = "REPLICATE_API_TOKEN"

# Fields the hosted wrapper exposes.  Sampling knobs compiled by the frozen
# adapter (steps / cfg scales) are intentionally dropped here: this is the
# executor-layer mapping, the adapter contract is unchanged.
_SUPPORTED_PARAMETER_FIELDS = (
    "seed",
    "go_fast",
    "aspect_ratio",
    "output_quality",
    "disable_safety_checker",
)
_DEFAULT_OUTPUT_FORMAT = "png"
_DEFAULT_POLL_INTERVAL_S = 2.0
_DEFAULT_WAIT_TIMEOUT_S = 600.0
_TOKEN_PATTERN = re.compile(r"r8_[A-Za-z0-9_]+")


def _sanitize(value: object) -> str:
    """Redact anything that looks like a Replicate token from a message."""

    return _TOKEN_PATTERN.sub("***", str(value))


class ReplicateQwenExecutor:
    """Execute a frozen ModelRenderRequest on Replicate's hosted Qwen model."""

    provider = "replicate"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = REPLICATE_MODEL_ID,
        version: str = REPLICATE_MODEL_VERSION,
        temp_dir: str | Path | None = None,
        wait_timeout_seconds: float = _DEFAULT_WAIT_TIMEOUT_S,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self.model = model
        self.version = version
        self.wait_timeout_seconds = wait_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._client = client
        self._temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.last_metadata: dict[str, Any] | None = None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        if operation == "character_synthesis":
            return self._execute_character_synthesis(request_payload, inputs)
        if operation == "local_inpaint":
            raise EnvironmentCapabilityError(
                "Replicate-hosted qwen/qwen-image-edit-2511 does not expose "
                "an independent mask input; the frozen local_inpaint contract "
                "(mask inside mutable / outside protected) cannot be expressed "
                "through this hosted API. Refusing an unconstrained edit. "
                "See implementation finding F-013."
            )
        raise InputValidationError(
            f"ReplicateQwenExecutor does not support operation {operation!r}"
        )

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        token = os.environ.get(REPLICATE_TOKEN_ENV)
        if not token:
            raise EnvironmentCapabilityError(
                f"missing {REPLICATE_TOKEN_ENV}; set it before running "
                "Replicate execution (the token is never logged or stored)"
            )
        from replicate import Client  # local import keeps unit tests light

        return Client(api_token=token)

    def _execute_character_synthesis(
        self,
        request_payload: dict,
        inputs: dict[str, bytes],
    ) -> bytes:
        conditions = request_payload.get("conditions", [])
        if not conditions:
            raise InputValidationError(
                "character_synthesis request has no condition slots"
            )
        image_inputs: list[bytes] = []
        for condition in conditions:
            condition_ref = condition.get("condition_ref")
            if not isinstance(condition_ref, str) or not condition_ref:
                raise InputValidationError(
                    "invalid condition slot: condition_ref must be a "
                    "non-empty string"
                )
            data = inputs.get(condition_ref)
            if data is None:
                raise InputValidationError(
                    "missing input image for condition_ref "
                    f"{condition_ref!r}"
                )
            image_inputs.append(data)
        prompt = request_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise InputValidationError(
                "character_synthesis request requires a prompt"
            )
        parameters = request_payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise InputValidationError("request parameters must be a dict")

        replicate_input: dict[str, Any] = {"image": [], "prompt": prompt}
        for key in _SUPPORTED_PARAMETER_FIELDS:
            if key in parameters and parameters[key] is not None:
                replicate_input[key] = parameters[key]
        if "seed" not in replicate_input:
            replicate_input["seed"] = 0
        replicate_input["output_format"] = _DEFAULT_OUTPUT_FORMAT

        with tempfile.TemporaryDirectory(
            prefix="replicate-qwen-", dir=self._temp_dir
        ) as tmpdir:
            image_paths: list[Path] = []
            for index, data in enumerate(image_inputs):
                path = Path(tmpdir) / f"reference_{index}.png"
                path.write_bytes(data)
                image_paths.append(path)
            replicate_input["image"] = image_paths
            return self._run_prediction(replicate_input)

    def _run_prediction(self, replicate_input: dict[str, Any]) -> bytes:
        client = self._client_or_raise()
        started = time.monotonic()
        try:
            prediction = client.predictions.create(
                model=self.model, input=replicate_input
            )
        except EnvironmentCapabilityError:
            raise
        except Exception as exc:
            raise RenderError(
                "replicate prediction request failed: "
                f"{type(exc).__name__}: {_sanitize(exc)}"
            ) from exc

        prediction_id = getattr(prediction, "id", None)
        status = self._wait_for_terminal(prediction)
        duration_ms = (time.monotonic() - started) * 1000.0
        self.last_metadata = {
            "provider": self.provider,
            "model": self.model,
            "version": getattr(prediction, "version", None) or self.version,
            "prediction_id": prediction_id,
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }
        if status != "succeeded":
            error = getattr(prediction, "error", None)
            detail = f": {_sanitize(error)}" if error else ""
            raise RenderError(
                f"replicate prediction {prediction_id} failed with status "
                f"{status!r}{detail}"
            )
        output = getattr(prediction, "output", None)
        data = self._read_output_bytes(output)
        if not isinstance(data, bytes) or not data:
            raise RenderError(
                f"replicate prediction {prediction_id} returned no usable "
                "image bytes"
            )
        return data

    def _wait_for_terminal(self, prediction: Any) -> str | None:
        status = getattr(prediction, "status", None)
        deadline = time.monotonic() + self.wait_timeout_seconds
        while status not in ("succeeded", "failed", "canceled"):
            if time.monotonic() >= deadline:
                raise RenderError(
                    "replicate prediction "
                    f"{getattr(prediction, 'id', None)} timed out after "
                    f"{self.wait_timeout_seconds:g}s"
                )
            time.sleep(self.poll_interval_seconds)
            reload = getattr(prediction, "reload", None)
            if callable(reload):
                reload()
            status = getattr(prediction, "status", None)
            if status is None:
                break
        return status

    @staticmethod
    def _read_output_bytes(output: Any) -> bytes | None:
        """Read the first usable image from a prediction output."""

        if isinstance(output, (list, tuple)):
            for item in output:
                data = ReplicateQwenExecutor._read_output_bytes(item)
                if data:
                    return data
            return None
        read = getattr(output, "read", None)
        if callable(read):
            data = read()
            if isinstance(data, bytes) and data:
                return data
            return None
        if isinstance(output, str):
            import httpx

            response = httpx.get(output, timeout=60.0)
            response.raise_for_status()
            return response.content
        return None
