"""Run one frozen Vidu Q2 Pro start/end-frame request contract.

The default path is deliberately offline.  ``--dry-run`` validates the
contract and both local anchors, compiles the provider request summary, and
writes a manifest without reading credentials or touching the network.

Real execution requires all three independent gates:

* ``--execute-paid`` on the command line;
* a contract whose ``authorization.status`` is ``granted``;
* ``DASHSCOPE_API_KEY`` and ``DASHSCOPE_WORKSPACE_ID`` in the environment.

A live run uploads exactly two images to DashScope temporary OSS, submits at
most one Vidu task, polls that same task, downloads one MP4, and stops.  There
is no task retry.  If the process stops after submission, running the same
run directory resumes polling the recorded task instead of submitting again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from PIL import Image, UnidentifiedImageError

MODEL_ID = "vidu/viduq2-pro_start-end2video"
REGION = "cn-beijing"
API_KEY_ENV = "DASHSCOPE_API_KEY"
WORKSPACE_ID_ENV = "DASHSCOPE_WORKSPACE_ID"
UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
TASK_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
QUERY_PATH_PREFIX = "/api/v1/tasks/"
UNIT_PRICE_CNY_PER_SECOND = 0.15625
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_BYTES = 250 * 1024 * 1024
EXPECTED_540P_PIXELS = 960 * 540
EXPECTED_ASPECT_RATIO = 16 / 9
POLL_INTERVAL_SECONDS = 15
MAX_POLL_QUERIES = 48
_KEY_RE = re.compile(r"sk-(?:ws-|sp-)?[^\s]+\Z")
_WORKSPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}
_ACTIVE_STATUSES = {"PENDING", "RUNNING"}
_ALLOWED_UPLOAD_SUFFIXES = (".aliyuncs.com",)
_ALLOWED_OUTPUT_SUFFIXES = (
    ".aliyuncs.com",
    ".alicdn.com",
    ".amazonaws.com.cn",
    ".amazonaws.com",
)


class ContractError(ValueError):
    """The frozen local experiment contract is invalid or has drifted."""


class EnvironmentError(RuntimeError):
    """Required local execution capability is unavailable."""


class ProviderError(RuntimeError):
    """DashScope or Vidu returned an unusable response."""


@dataclass(frozen=True)
class InputProbe:
    role: str
    relative_path: str
    absolute_path: Path
    sha256: str
    bytes: int
    image_format: str
    width: int
    height: int

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("absolute_path")
        data["path"] = data.pop("relative_path")
        return data


class Transport(Protocol):
    """Small injected seam used by offline tests and the real HTTP path."""

    def upload(self, *, path: Path, model: str, api_key: str) -> str: ...

    def submit(
        self,
        *,
        workspace_id: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...

    def query(
        self,
        *,
        workspace_id: str,
        api_key: str,
        task_id: str,
    ) -> dict[str, Any]: ...

    def download(self, *, url: str) -> bytes: ...


def _sanitize(value: object) -> str:
    text = str(value)
    text = re.sub(r"sk-(?:ws-|sp-)?[^\s'\"<>]+", "***", text)
    text = re.sub(r"oss://[^\s'\"<>]+", "oss://***", text)
    text = re.sub(r"https://[^\s'\"<>]+\?[^\s'\"<>]+", "https://***/***", text)
    return " ".join(text.split())[:1000]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract: {_sanitize(exc)}") from None
    if not isinstance(data, dict):
        raise ContractError("contract root must be a JSON object")
    return data


def _require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ContractError(f"contract field {key!r} must be an object")
    return value


def _require_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"contract field {key!r} must be an integer")
    return value


def _resolve_inside(project_root: Path, relative_path: object) -> tuple[str, Path]:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ContractError("input path must be a non-empty relative path")
    candidate_text = relative_path.replace("\\", "/")
    candidate = Path(candidate_text)
    if candidate.is_absolute():
        raise ContractError("input path must be relative to the project root")
    try:
        root = project_root.resolve(strict=True)
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ContractError(
            f"input path is missing or escapes the project root: {candidate_text}"
        ) from None
    if not resolved.is_file():
        raise ContractError(f"input path is not a file: {candidate_text}")
    return candidate_text, resolved


def _probe_input(project_root: Path, entry: dict[str, Any]) -> InputProbe:
    role = entry.get("role")
    if role not in {"start_frame", "end_frame"}:
        raise ContractError("input role must be start_frame or end_frame")
    relative_path, path = _resolve_inside(project_root, entry.get("path"))
    expected_sha256 = entry.get("sha256")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ContractError(f"{role} sha256 must be 64 lowercase hex characters")
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ContractError(f"{role} must be within 1..{MAX_INPUT_BYTES} bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ContractError(
            f"{role} SHA256 drift: expected {expected_sha256}, got {digest}"
        )
    try:
        with Image.open(path) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ContractError(
            f"{role} is not a decodable image: {_sanitize(exc)}"
        ) from None
    if image_format not in {"PNG", "JPEG", "WEBP"}:
        raise ContractError(f"{role} format {image_format!r} is unsupported by Vidu")
    if width <= 0 or height <= 0 or not 0.25 <= width / height <= 4.0:
        raise ContractError(f"{role} aspect ratio must be within 1:4..4:1")
    return InputProbe(
        role=role,
        relative_path=relative_path,
        absolute_path=path,
        sha256=digest,
        bytes=size,
        image_format=image_format,
        width=width,
        height=height,
    )


def validate_contract(
    contract: dict[str, Any], project_root: Path
) -> tuple[list[InputProbe], dict[str, Any]]:
    """Validate all frozen fields and exact local media before any network use."""

    if contract.get("schema_version") != "i6-vidu-start-end-request-v1":
        raise ContractError("unsupported Vidu contract schema_version")
    if contract.get("stage", "I6") not in {"I6", "I7"}:
        raise ContractError("Vidu contract stage must be I6 or I7")
    if contract.get("model") != MODEL_ID:
        raise ContractError(f"model must remain frozen to {MODEL_ID!r}")
    if contract.get("region") != "华北2（北京）":
        raise ContractError("region must remain 华北2（北京）")
    prompt = contract.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
        raise ContractError("prompt must contain 1..5000 characters")
    inputs = contract.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ContractError("Vidu start/end contract requires exactly two inputs")
    if not all(isinstance(entry, dict) for entry in inputs):
        raise ContractError("each input must be an object")
    probes = [_probe_input(project_root, entry) for entry in inputs]
    if [probe.role for probe in probes] != ["start_frame", "end_frame"]:
        raise ContractError("input order must be start_frame followed by end_frame")
    pixel_ratio = (probes[0].width * probes[0].height) / (
        probes[1].width * probes[1].height
    )
    if not 0.8 <= pixel_ratio <= 1.25:
        raise ContractError("start/end total-pixel ratio must be within 0.8..1.25")

    parameters = _require_dict(contract, "parameters")
    if parameters.get("resolution") != "540P":
        raise ContractError("this frozen Vidu request must use resolution 540P")
    duration = _require_int(parameters, "duration")
    if not 1 <= duration <= 8:
        raise ContractError("duration must be within the conservative Q2 range 1..8")
    seed = _require_int(parameters, "seed")
    if not 0 <= seed <= 2_147_483_647:
        raise ContractError("seed must be within 0..2147483647")
    if parameters.get("watermark") is not False:
        raise ContractError("watermark must remain false")
    if "audio" in parameters:
        raise ContractError("Vidu Q2 Pro start/end does not support audio parameter")

    limits = _require_dict(contract, "limits")
    required_limits = {
        "maximum_tasks": 1,
        "maximum_outputs": 1,
        "automatic_retry": False,
        "content_retry": False,
    }
    for key, expected in required_limits.items():
        if limits.get(key) != expected:
            raise ContractError(f"limit {key!r} must remain {expected!r}")

    price = _require_dict(contract, "listed_price_cny")
    expected_max = round(duration * UNIT_PRICE_CNY_PER_SECOND, 6)
    if price.get("unit_price_per_second") != UNIT_PRICE_CNY_PER_SECOND:
        raise ContractError(
            "listed 540P unit price has drifted from the frozen contract"
        )
    if price.get("maximum_for_one_successful_task") != expected_max:
        raise ContractError("listed maximum cost does not match duration x unit price")

    request_summary = {
        "model": MODEL_ID,
        "region": REGION,
        "input_count": 2,
        "input_order": ["start_frame", "end_frame"],
        "input_transport": "dashscope_temporary_oss_48h",
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "parameters": {
            "resolution": "540P",
            "duration": duration,
            "seed": seed,
            "watermark": False,
        },
        "maximum_tasks": 1,
        "maximum_outputs": 1,
        "automatic_retry": False,
        "listed_maximum_cny": expected_max,
    }
    return probes, request_summary


def build_payload(
    contract: dict[str, Any], start_url: str, end_url: str
) -> dict[str, Any]:
    if not start_url.startswith(("oss://", "https://", "http://")):
        raise ProviderError("start-frame upload did not return a usable URL")
    if not end_url.startswith(("oss://", "https://", "http://")):
        raise ProviderError("end-frame upload did not return a usable URL")
    parameters = contract["parameters"]
    return {
        "model": MODEL_ID,
        "input": {
            "media": [
                {"type": "image", "url": start_url},
                {"type": "image", "url": end_url},
            ],
            "prompt": contract["prompt"],
        },
        "parameters": {
            "resolution": parameters["resolution"],
            "duration": parameters["duration"],
            "watermark": parameters["watermark"],
            "seed": parameters["seed"],
        },
    }


def _validated_https_url(url: object, suffixes: tuple[str, ...], label: str) -> str:
    if not isinstance(url, str):
        raise ProviderError(f"{label} URL is missing")
    try:
        parsed = urllib_parse.urlparse(url)
    except ValueError:
        raise ProviderError(f"{label} URL is invalid") from None
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise ProviderError(f"{label} URL must use HTTPS")
    if not any(host == suffix[1:] or host.endswith(suffix) for suffix in suffixes):
        raise ProviderError(f"{label} URL host is outside the provider allowlist")
    return url


def _json_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            response_body = response.read(2 * 1024 * 1024 + 1)
            if len(response_body) > 2 * 1024 * 1024:
                raise ProviderError("provider JSON response exceeds 2 MiB")
    except urllib_error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ProviderError(f"provider HTTP {exc.code}: {_sanitize(detail)}") from None
    except (OSError, urllib_error.URLError) as exc:
        raise ProviderError(f"provider transport failed: {_sanitize(exc)}") from None
    try:
        decoded = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderError("provider returned invalid JSON") from None
    if not isinstance(decoded, dict):
        raise ProviderError("provider JSON root must be an object")
    return decoded


def _multipart_body(fields: dict[str, str], path: Path) -> tuple[bytes, str]:
    boundary = f"----anime-remix-{secrets.token_hex(16)}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (f'Content-Disposition: form-data; name="{name}"\r\n\r\n').encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


class DefaultTransport:
    """Strict standard-library transport for the documented Beijing APIs."""

    def upload(self, *, path: Path, model: str, api_key: str) -> str:
        policy_url = f"{UPLOAD_POLICY_URL}?{urllib_parse.urlencode({'action': 'getPolicy', 'model': model})}"
        policy_response = _json_request(
            method="GET",
            url=policy_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        policy = policy_response.get("data")
        if not isinstance(policy, dict):
            raise ProviderError("upload policy response has no data object")
        required = {
            "upload_host",
            "upload_dir",
            "oss_access_key_id",
            "signature",
            "policy",
            "x_oss_object_acl",
            "x_oss_forbid_overwrite",
        }
        if not required.issubset(policy) or not all(
            isinstance(policy[key], str) and policy[key] for key in required
        ):
            raise ProviderError("upload policy is missing required string fields")
        upload_host = _validated_https_url(
            policy["upload_host"], _ALLOWED_UPLOAD_SUFFIXES, "upload"
        )
        object_key = f"{policy['upload_dir'].rstrip('/')}/{path.name}"
        fields = {
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "Signature": policy["signature"],
            "policy": policy["policy"],
            "x-oss-object-acl": policy["x_oss_object_acl"],
            "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
            "key": object_key,
            "success_action_status": "200",
        }
        body, boundary = _multipart_body(fields, path)
        request = urllib_request.Request(
            upload_host,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=120.0) as response:
                response.read(4096)
                if response.status != 200:
                    raise ProviderError(
                        f"temporary OSS upload returned {response.status}"
                    )
        except urllib_error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ProviderError(
                f"temporary OSS upload HTTP {exc.code}: {_sanitize(detail)}"
            ) from None
        except (OSError, urllib_error.URLError) as exc:
            raise ProviderError(
                f"temporary OSS upload failed: {_sanitize(exc)}"
            ) from None
        return f"oss://{object_key}"

    @staticmethod
    def _base_url(workspace_id: str) -> str:
        return f"https://{workspace_id}.{REGION}.maas.aliyuncs.com"

    def submit(
        self,
        *,
        workspace_id: str,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return _json_request(
            method="POST",
            url=f"{self._base_url(workspace_id)}{TASK_PATH}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            payload=payload,
        )

    def query(
        self,
        *,
        workspace_id: str,
        api_key: str,
        task_id: str,
    ) -> dict[str, Any]:
        return _json_request(
            method="GET",
            url=(
                f"{self._base_url(workspace_id)}{QUERY_PATH_PREFIX}"
                f"{urllib_parse.quote(task_id, safe='')}"
            ),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def download(self, *, url: str) -> bytes:
        validated = _validated_https_url(url, _ALLOWED_OUTPUT_SUFFIXES, "output video")
        try:
            with urllib_request.urlopen(validated, timeout=120.0) as response:
                _validated_https_url(
                    response.geturl(), _ALLOWED_OUTPUT_SUFFIXES, "redirected output"
                )
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_OUTPUT_BYTES:
                        raise ProviderError("output video exceeds 250 MiB safety cap")
                    chunks.append(chunk)
        except urllib_error.HTTPError as exc:
            raise ProviderError(f"output download HTTP {exc.code}") from None
        except (OSError, urllib_error.URLError) as exc:
            raise ProviderError(f"output download failed: {_sanitize(exc)}") from None
        data = b"".join(chunks)
        if len(data) < 12 or data[4:8] != b"ftyp":
            raise ProviderError("downloaded output is not an MP4 file")
        return data


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ProviderError(f"provider returned an invalid {label}")
    return value


def _output(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("output")
    if not isinstance(value, dict):
        raise ProviderError("provider response has no output object")
    return value


def _task_status(response: dict[str, Any]) -> str:
    value = _output(response).get("task_status")
    if value not in _ACTIVE_STATUSES | _TERMINAL_STATUSES:
        raise ProviderError("provider returned an unknown task status")
    return value


def _safe_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    allowed = {
        "duration",
        "size",
        "output_video_duration",
        "fps",
        "video_count",
        "audio",
        "SR",
    }
    safe: dict[str, Any] = {}
    for key in allowed:
        value = usage.get(key)
        if isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes):
            safe[key] = value
    return safe


def probe_mp4(path: Path, expected_duration: int) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise EnvironmentError("ffprobe is required to validate the downloaded MP4")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise ProviderError(f"ffprobe rejected output: {_sanitize(result.stderr)}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ProviderError("ffprobe returned invalid JSON") from None
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ProviderError("ffprobe response has no streams")
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or audio:
        raise ProviderError("Vidu Q2 output must contain one video stream and no audio")
    stream = video[0]
    if stream.get("codec_name") != "h264":
        raise ProviderError("Vidu Q2 output video codec must be H.264")
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(
            stream.get("duration") or payload.get("format", {}).get("duration")
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ProviderError(
            "ffprobe output lacks usable media dimensions or timing"
        ) from None
    if abs(fps - 24.0) > 0.01:
        raise ProviderError(f"Vidu Q2 output must be 24 fps, got {fps}")
    if width <= 0 or height <= 0:
        raise ProviderError("Vidu Q2 output has invalid dimensions")
    pixels = width * height
    if not 0.9 * EXPECTED_540P_PIXELS <= pixels <= 1.1 * EXPECTED_540P_PIXELS:
        raise ProviderError(
            "Vidu Q2 output does not match the frozen 540P pixel class: "
            f"got {width}x{height}"
        )
    if abs(width / height - EXPECTED_ASPECT_RATIO) > 0.03:
        raise ProviderError(
            "Vidu Q2 output did not preserve the frozen 16:9 endpoint aspect ratio"
        )
    if abs(duration - expected_duration) > 0.35:
        raise ProviderError(
            f"Vidu Q2 output duration drifted: expected {expected_duration}s, got {duration}s"
        )
    return {
        "container": "mp4",
        "codec": "h264",
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "video_streams": 1,
        "audio_streams": 0,
    }


def _initial_manifest(
    contract: dict[str, Any],
    probes: list[InputProbe],
    request_summary: dict[str, Any],
    outcome: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": "i6-vidu-run-manifest-v1",
        "run_id": contract.get("run_id"),
        "stage": contract.get("stage", "I6"),
        "outcome": outcome,
        "detail": detail,
        "provider": "Alibaba Cloud Model Studio / Vidu",
        "model": MODEL_ID,
        "inputs": [probe.safe_dict() for probe in probes],
        "request_summary": request_summary,
        "execution": {
            "upload_count": 0,
            "task_submit_count": 0,
            "query_count": 0,
            "output_count": 0,
            "automatic_retries": 0,
            "task_id": None,
            "request_id": None,
        },
    }


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read_json(path)
    if value.get("schema_version") != "i6-vidu-provider-state-v1":
        raise ContractError("provider state has an unsupported schema")
    return value


def run_experiment(
    *,
    contract_path: Path,
    project_root: Path,
    run_dir: Path,
    dry_run: bool,
    execute_paid: bool,
    transport: Transport | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    probe_fn: Callable[[Path, int], dict[str, Any]] = probe_mp4,
) -> dict[str, Any]:
    if dry_run == execute_paid:
        raise ContractError("choose exactly one of dry_run or execute_paid")
    contract = _read_json(contract_path)
    probes, request_summary = validate_contract(contract, project_root)
    project_root = project_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(project_root)
    except ValueError:
        raise ContractError("run_dir must remain inside the project root") from None
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run-manifest.json"
    state_path = run_dir / "provider-state.json"
    output_path = run_dir / "raw.mp4"

    if dry_run:
        manifest = _initial_manifest(
            contract,
            probes,
            request_summary,
            "dry_run",
            "local preflight passed; no credentials read, no network, no upload, no cost",
        )
        _atomic_json(manifest_path, manifest)
        return manifest

    authorization = _require_dict(contract, "authorization")
    if authorization.get("status") != "granted":
        raise ContractError(
            "live execution refused: contract authorization.status is not granted"
        )
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key or not _KEY_RE.fullmatch(api_key):
        raise EnvironmentError(
            f"{API_KEY_ENV} is missing or invalid; its value was not logged"
        )
    workspace_id = os.environ.get(WORKSPACE_ID_ENV)
    if not workspace_id or not _WORKSPACE_RE.fullmatch(workspace_id):
        raise EnvironmentError(
            f"{WORKSPACE_ID_ENV} is missing or invalid; set the Beijing business-space ID locally"
        )
    active_transport = transport or DefaultTransport()
    state = _load_state(state_path)
    if state is None:
        state = {
            "schema_version": "i6-vidu-provider-state-v1",
            "state": "execution_started",
            "upload_count": 0,
            "task_submit_count": 0,
            "query_count": 0,
            "output_count": 0,
            "task_id": None,
            "request_id": None,
        }
        _atomic_json(state_path, state)
    elif state.get("state") == "succeeded" and output_path.exists():
        return _read_json(manifest_path)
    elif state.get("state") in {"failed", "blocked_before_submit"}:
        raise ContractError(
            "this run is closed by the no-retry rule; a new task requires a new explicit contract"
        )

    task_id = state.get("task_id")
    if task_id is None:
        if state.get("upload_count", 0) != 0 or state.get("task_submit_count", 0) != 0:
            state["state"] = "blocked_before_submit"
            _atomic_json(state_path, state)
            raise ContractError(
                "an interrupted pre-submit attempt cannot be repeated automatically"
            )
        try:
            uploaded = [
                active_transport.upload(
                    path=probe.absolute_path,
                    model=MODEL_ID,
                    api_key=api_key,
                )
                for probe in probes
            ]
            state["upload_count"] = 2
            _atomic_json(state_path, state)
            response = active_transport.submit(
                workspace_id=workspace_id,
                api_key=api_key,
                payload=build_payload(contract, uploaded[0], uploaded[1]),
            )
            task_id = _safe_id(_output(response).get("task_id"), "task_id")
            request_id_value = response.get("request_id")
            request_id = (
                _safe_id(request_id_value, "request_id")
                if request_id_value is not None
                else None
            )
            state.update(
                {
                    "state": "submitted",
                    "task_submit_count": 1,
                    "task_id": task_id,
                    "request_id": request_id,
                }
            )
            _atomic_json(state_path, state)
        except Exception:
            if state.get("task_id") is None:
                state["state"] = "blocked_before_submit"
                _atomic_json(state_path, state)
            raise
    else:
        task_id = _safe_id(task_id, "recorded task_id")

    terminal_response: dict[str, Any] | None = None
    for query_index in range(MAX_POLL_QUERIES):
        response = active_transport.query(
            workspace_id=workspace_id,
            api_key=api_key,
            task_id=task_id,
        )
        state["query_count"] = int(state.get("query_count", 0)) + 1
        status = _task_status(response)
        state["state"] = status.lower()
        _atomic_json(state_path, state)
        if status in _TERMINAL_STATUSES:
            terminal_response = response
            break
        if query_index + 1 < MAX_POLL_QUERIES:
            sleep_fn(POLL_INTERVAL_SECONDS)
    if terminal_response is None:
        raise ProviderError(
            "poll window ended; rerun the same run directory to resume this task without resubmission"
        )
    status = _task_status(terminal_response)
    if status != "SUCCEEDED":
        state["state"] = "failed"
        _atomic_json(state_path, state)
        output = _output(terminal_response)
        raise ProviderError(
            f"Vidu task ended as {status}: {_sanitize(output.get('message', 'no message'))}"
        )
    output = _output(terminal_response)
    video_url = output.get("video_url")
    video_bytes = active_transport.download(url=video_url)
    if len(video_bytes) < 12 or video_bytes[4:8] != b"ftyp":
        raise ProviderError("downloaded output is not an MP4 file")
    temporary_output = run_dir / ".raw.mp4.tmp"
    temporary_output.write_bytes(video_bytes)
    media_probe = probe_fn(temporary_output, contract["parameters"]["duration"])
    os.replace(temporary_output, output_path)
    output_sha256 = hashlib.sha256(video_bytes).hexdigest()
    state.update({"state": "succeeded", "output_count": 1})
    _atomic_json(state_path, state)

    manifest = _initial_manifest(
        contract,
        probes,
        request_summary,
        "success",
        "one Vidu task succeeded; raw MP4 downloaded and validated; stopped for human review",
    )
    manifest["execution"] = {
        "upload_count": state["upload_count"],
        "task_submit_count": state["task_submit_count"],
        "query_count": state["query_count"],
        "output_count": 1,
        "automatic_retries": 0,
        "task_id": state.get("task_id"),
        "request_id": state.get("request_id"),
        "provider_usage": _safe_usage(terminal_response),
        "output": {
            "path": "raw.mp4",
            "bytes": len(video_bytes),
            "sha256": output_sha256,
            "media_probe": media_probe,
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _write_failure_manifest(
    *,
    run_dir: Path,
    project_root: Path,
    contract_path: Path,
    exc: Exception,
) -> None:
    """Best-effort safe failure evidence for the CLI real-mode path."""

    try:
        root = project_root.resolve(strict=True)
        destination = run_dir.resolve()
        destination.relative_to(root)
        destination.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return
    detail_source = str(exc)
    try:
        contract = _read_json(contract_path)
        prompt = contract.get("prompt")
        if isinstance(prompt, str) and prompt:
            detail_source = detail_source.replace(prompt, "***")
        run_id = contract.get("run_id")
    except ContractError:
        run_id = None
    safe_state: dict[str, Any] = {}
    state_path = destination / "provider-state.json"
    try:
        state = _load_state(state_path)
    except ContractError:
        state = None
    if state is not None:
        for key in (
            "state",
            "upload_count",
            "task_submit_count",
            "query_count",
            "output_count",
            "task_id",
            "request_id",
        ):
            value = state.get(key)
            if isinstance(value, (str, int)) or value is None:
                safe_state[key] = value
    manifest = {
        "schema_version": "i6-vidu-run-manifest-v1",
        "run_id": run_id,
        "stage": contract.get("stage", "I6"),
        "outcome": "error",
        "detail": _sanitize(detail_source),
        "error_type": type(exc).__name__,
        "provider": "Alibaba Cloud Model Studio / Vidu",
        "model": MODEL_ID,
        "execution": safe_state,
    }
    _atomic_json(destination / "run-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("request_contract.json"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    try:
        manifest = run_experiment(
            contract_path=args.contract.resolve(),
            project_root=project_root,
            run_dir=args.run_dir,
            dry_run=args.dry_run,
            execute_paid=args.execute_paid,
        )
    except (ContractError, EnvironmentError, ProviderError) as exc:
        if args.execute_paid:
            _write_failure_manifest(
                run_dir=args.run_dir,
                project_root=project_root,
                contract_path=args.contract.resolve(),
                exc=exc,
            )
        print(f"VIDU STOPPED: {_sanitize(exc)}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"VIDU {manifest['outcome'].upper()}: {args.run_dir / 'run-manifest.json'}"
    )


if __name__ == "__main__":
    main()
