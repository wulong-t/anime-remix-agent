#!/usr/bin/env python
"""G1-MK1-R-PREP-L: deterministic single-sample remote sampling support.

Subcommands:
  run -- revalidate the exact G1-MK1 package and the strict
        ``g1-mk1-runner-config-v1`` runner configuration, run exactly one
        frozen AniSora BF16 command, validate the single raw sample, and
        atomically publish either complete success evidence (raw_shot.mp4 +
        strict ``g1-mk1-sampling-receipt-v1`` + valid-sample marker) or
        complete failure evidence.

Boundary: this implementation and its tests never read repository real media,
never call a model or use the network, and never emit keys or environment
variable values in outputs. All paths are consumed from the explicit runner
configuration; nothing is discovered recursively.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.errors import AnimeRemixError
from experiments.manual_keyframe_mvp import (
    manual_keyframe_mvp as harness,
)

RUNNER_CONFIG_SCHEMA = "g1-mk1-runner-config-v1"
PREFLIGHT_SCHEMA = "g1-mk1-remote-preflight-v1"
RESULT_SCHEMA = "g1-mk1-remote-sample-result-v1"
VALID_SAMPLE_SCHEMA = "g1-mk1-valid-sample-v1"
RAW_RECOVERY_SCHEMA = "g1-mk1-raw-recovery-v1"

RUNNER_CONFIG_FIELDS = {
    "schema_version",
    "python_executable",
    "anisora_workdir",
    "bf16_runner_script",
    "checkpoint_dir",
    "checkpoint_files",
    "ffmpeg",
    "ffprobe",
    "nvidia_smi",
    "cgroup_memory_current",
    "cgroup_memory_peak",
    "cgroup_memory_events",
}
RUNNER_CONFIG_EXECUTABLE_FIELDS = (
    "python_executable",
    "ffmpeg",
    "ffprobe",
    "nvidia_smi",
)
RUNNER_CONFIG_ABSOLUTE_PATH_FIELDS = (
    "anisora_workdir",
    "bf16_runner_script",
    "checkpoint_dir",
    "cgroup_memory_current",
    "cgroup_memory_peak",
    "cgroup_memory_events",
)
CHECKPOINT_RECORD_FIELDS = {"relative_path", "size_bytes"}

FROZEN_GUIDE_POSITIONS = [0, 1]

_PYTHON_CUDA_PROBE = (
    "import importlib.util, json, platform, sys\n"
    "spec = importlib.util.find_spec('torch')\n"
    "torch_available = spec is not None\n"
    "cuda_available = False\n"
    "cuda_version = None\n"
    "device_count = 0\n"
    "if torch_available:\n"
    "    import torch\n"
    "    cuda_available = bool(torch.cuda.is_available())\n"
    "    cuda_version = torch.version.cuda\n"
    "    device_count = int(torch.cuda.device_count())\n"
    "print(json.dumps({'python_version': sys.version.split()[0], "
    "'platform': platform.platform(), 'torch_available': torch_available, "
    "'cuda_available': cuda_available, 'cuda_version': cuda_version, "
    "'device_count': device_count}))"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _frozen_argv(
    config: dict[str, Any],
    runner_output_dir: Path,
    runtime_input_path: Path,
    sample_steps: int,
) -> list[str]:
    """Construct the exact frozen AniSora command.

    Only paths and the packaged-contract ``sample_steps`` vary; every other
    parameter stays byte-for-byte frozen. ``sample_steps`` always comes from
    the validated packaged contract, never from a CLI override or a hardcoded
    default.
    """

    return [
        str(config["python_executable"]),
        str(config["bf16_runner_script"]),
        "--task",
        "i2v-14B",
        "--size",
        "1280*720",
        "--ckpt_dir",
        str(config["checkpoint_dir"]),
        "--image",
        str(runner_output_dir),
        "--prompt",
        str(runtime_input_path),
        "--base_seed",
        "4096",
        "--frame_num",
        "81",
        "--sample_steps",
        str(sample_steps),
        "--sample_shift",
        "5",
        "--sample_guide_scale",
        "5",
        "--offload_model",
        "True",
    ]


def _validate_runner_config(data: Any) -> dict[str, Any]:
    """Strictly validate the ``g1-mk1-runner-config-v1`` object."""

    data = harness._require_object(data, "runner_config", "input_contract")
    problems: list[str] = []
    missing = sorted(RUNNER_CONFIG_FIELDS - set(data))
    unknown = sorted(set(data) - RUNNER_CONFIG_FIELDS)
    if missing:
        problems.append(f"runner_config missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"runner_config unknown fields: {', '.join(unknown)}")
    if data.get("schema_version") != RUNNER_CONFIG_SCHEMA:
        problems.append(
            f"runner_config schema_version must be {RUNNER_CONFIG_SCHEMA}"
        )
    for field in RUNNER_CONFIG_FIELDS - {"schema_version", "checkpoint_files"}:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(
                f"runner_config {field} must be a non-empty string"
            )
        elif harness.CONTROL_RE.search(value):
            problems.append(
                f"runner_config {field} must not contain control characters"
            )
    for field in RUNNER_CONFIG_ABSOLUTE_PATH_FIELDS:
        value = data.get(field)
        if (
            isinstance(value, str)
            and value.strip()
            and not Path(value).is_absolute()
        ):
            problems.append(
                f"runner_config {field} must be an absolute path"
            )
    for field in RUNNER_CONFIG_EXECUTABLE_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if not path.is_absolute() and (
                "/" in value or "\\" in value
            ):
                problems.append(
                    f"runner_config {field} must be an absolute path or a "
                    "bare command name"
                )
    checkpoint_files = data.get("checkpoint_files")
    if not isinstance(checkpoint_files, list) or not checkpoint_files:
        problems.append(
            "runner_config checkpoint_files must be a non-empty list"
        )
    else:
        for index, record in enumerate(checkpoint_files):
            label = f"checkpoint_files[{index}]"
            if not isinstance(record, dict):
                problems.append(f"{label} must be an object")
                continue
            record_missing = sorted(CHECKPOINT_RECORD_FIELDS - set(record))
            record_unknown = sorted(set(record) - CHECKPOINT_RECORD_FIELDS)
            if record_missing:
                problems.append(
                    f"{label} missing fields: {', '.join(record_missing)}"
                )
            if record_unknown:
                problems.append(
                    f"{label} unknown fields: {', '.join(record_unknown)}"
                )
            relative = record.get("relative_path")
            if not isinstance(relative, str):
                problems.append(f"{label}.relative_path must be a string")
            else:
                try:
                    harness._validate_canonical_relative(
                        relative, f"{label}.relative_path", "input_contract"
                    )
                except harness.HarnessError as exc:
                    problems.append(str(exc))
            size = record.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                problems.append(
                    f"{label}.size_bytes must be a positive integer"
                )
    if problems:
        raise harness.HarnessError("input_contract", "; ".join(problems))
    return data


def _config_evidence(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "python_executable": config["python_executable"],
        "anisora_workdir": config["anisora_workdir"],
        "bf16_runner_script": config["bf16_runner_script"],
        "checkpoint_dir": config["checkpoint_dir"],
        "checkpoint_files": [dict(record) for record in config["checkpoint_files"]],
        "ffmpeg": config["ffmpeg"],
        "ffprobe": config["ffprobe"],
        "nvidia_smi": config["nvidia_smi"],
        "cgroup_memory_current": config["cgroup_memory_current"],
        "cgroup_memory_peak": config["cgroup_memory_peak"],
        "cgroup_memory_events": config["cgroup_memory_events"],
    }


def validate_package(package: Path) -> dict[str, Any]:
    """Reuse the frozen G1-MK1 harness to validate the exact package.

    Revalidates the package manifest, all nine members, request, inspection,
    approval, sampling contract, guide hashes, ``first_formal_gate.active``
    and the ``anisora_input.txt`` binding. The harness itself is not
    modified.
    """

    package = Path(package)
    harness._reject_link_or_reparse(package, "package root")
    package_root = package.resolve()
    manifest_bytes = harness._capture_bytes(
        package_root / "package_manifest.json", "evidence_incomplete"
    )
    manifest = harness._validate_package_manifest(
        harness._load_json_bytes(
            manifest_bytes, "package_manifest", "evidence_incomplete"
        ),
        package_root,
    )
    package_manifest_sha256 = harness._sha256_bytes(manifest_bytes)
    members = harness._capture_package_members(package_root)
    harness._validate_package_manifest_bindings(manifest, members)

    request_bytes = members["request.json"]
    info = harness._validate_request_bytes(
        request_bytes, package_root / "request.json"
    )
    if info["request_id"] != manifest["request_id"]:
        raise harness.HarnessError(
            "evidence_incomplete",
            "package_manifest request_id does not match packaged request",
        )
    if info["request_sha256"] != manifest["request_sha256"]:
        raise harness.HarnessError(
            "evidence_incomplete",
            "package_manifest request_sha256 does not match packaged request",
        )
    harness._verify_guide_snapshot_hashes(
        info, members["inputs/k0.png"], members["inputs/k_end.png"]
    )
    if (
        info["start_sha256"] != manifest["start_sha256"]
        or info["end_sha256"] != manifest["end_sha256"]
    ):
        raise harness.HarnessError(
            "evidence_incomplete",
            "package_manifest guide hashes do not match packaged request",
        )

    inspection_bytes = members["inspection.json"]
    inspection_data = harness._load_json_bytes(
        inspection_bytes, "inspection", "approval_blocked"
    )
    harness._validate_inspection(
        inspection_data,
        info,
        members["inputs/k0.png"],
        members["inputs/k_end.png"],
        "request.json",
    )
    gate_active = inspection_data["first_formal_gate"]["active"]
    gate_problems = inspection_data["first_formal_gate"]["problems"]
    if gate_active is not True:
        raise harness.HarnessError(
            "approval_blocked",
            "first_formal_gate.active must be true before sampling",
        )

    harness._validate_approval(
        harness._load_json_bytes(
            members["approval.json"], "approval", "approval_blocked"
        ),
        info["request_id"],
        info["request_sha256"],
        info["start_sha256"],
        info["end_sha256"],
    )

    contract_bytes = members["sampling_contract.json"]
    contract = harness._validate_sampling_contract(
        harness._load_json_bytes(
            contract_bytes, "sampling_contract", "evidence_incomplete"
        ),
        info,
        gate_active,
        gate_problems,
    )
    sampling_contract_sha256 = harness._sha256_bytes(contract_bytes)
    harness._validate_anisora_input_bytes(contract, members["anisora_input.txt"])

    return {
        "package_root": package_root,
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "package_manifest_sha256": package_manifest_sha256,
        "members": members,
        "info": info,
        "inspection_data": inspection_data,
        "contract": contract,
        "contract_bytes": contract_bytes,
        "sampling_contract_sha256": sampling_contract_sha256,
    }


def _resolve_executable(raw: str, what: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        if not path.exists() or not path.is_file():
            raise harness.HarnessError(
                "remote_environment", f"{what} does not exist: {raw}"
            )
        return path
    resolved = shutil.which(raw)
    if resolved is None:
        raise harness.HarnessError(
            "remote_environment", f"{what} not found on PATH: {raw}"
        )
    return Path(resolved)


def _run_probe(
    args: list[str], what: str, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise harness.HarnessError(
            "remote_environment", f"{what}: executable not found: {exc}"
        ) from exc
    except OSError as exc:
        raise harness.HarnessError(
            "remote_environment", f"{what}: failed to start: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise harness.HarnessError(
            "remote_environment", f"{what} timed out after {timeout}s"
        ) from exc


def _read_text(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise harness.HarnessError(
            "remote_environment", f"cannot read {what}: {path}: {exc}"
        ) from exc


def _probe_python_cuda(python_path: Path) -> dict[str, Any]:
    completed = _run_probe(
        [str(python_path), "-c", _PYTHON_CUDA_PROBE], "python/cuda probe"
    )
    if completed.returncode != 0:
        raise harness.HarnessError(
            "remote_environment",
            "python/cuda probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-1000:],
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise harness.HarnessError(
            "remote_environment", "python/cuda probe returned no JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise harness.HarnessError(
            "remote_environment", "python/cuda probe returned a non-object"
        )
    return payload


def _probe_version(exe: Path, name: str) -> str:
    completed = _run_probe([str(exe), "-version"], f"{name} -version")
    if completed.returncode != 0:
        raise harness.HarnessError(
            "remote_environment",
            f"{name} -version failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-1000:],
        )
    return completed.stdout.splitlines()[0].strip() if completed.stdout else ""


def _probe_nvidia_smi(exe: Path) -> list[str]:
    completed = _run_probe(
        [
            str(exe),
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "nvidia-smi probe",
    )
    if completed.returncode != 0:
        raise harness.HarnessError(
            "remote_environment",
            "nvidia-smi probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[-1000:],
        )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _judge_python_cuda(payload: Any) -> None:
    """Hard-gate the python/CUDA probe payload before any runner invocation.

    A missing/malformed field or a false/empty condition is a
    ``remote_environment`` preflight failure; ``bool`` is never accepted as
    the device-count integer.
    """

    problems: list[str] = []
    if not isinstance(payload, dict):
        problems.append("python/cuda probe payload must be an object")
    else:
        if payload.get("torch_available") is not True:
            problems.append(
                "python/cuda probe torch_available must be exactly true"
            )
        if payload.get("cuda_available") is not True:
            problems.append(
                "python/cuda probe cuda_available must be exactly true"
            )
        device_count = payload.get("device_count")
        if (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or device_count < 1
        ):
            problems.append(
                "python/cuda probe device_count must be an integer >= 1"
            )
        cuda_version = payload.get("cuda_version")
        if not isinstance(cuda_version, str) or not cuda_version:
            problems.append(
                "python/cuda probe cuda_version must be a non-empty string"
            )
    if problems:
        raise harness.HarnessError(
            "remote_environment", "; ".join(problems)
        )


def _judge_nvidia_smi(rows: list[str]) -> None:
    """Require at least one non-empty nvidia-smi GPU row (R1)."""

    if not rows or any(not line.strip() for line in rows):
        raise harness.HarnessError(
            "remote_environment",
            "nvidia-smi probe returned no GPU rows",
        )


def _preflight(
    package_info: dict[str, Any],
    config: dict[str, Any],
    argv: list[str],
) -> dict[str, Any]:
    """Verify exact configured files, executables, sizes and probes.

    Any failure raises ``HarnessError`` with layer ``remote_environment`` and
    the runner is never invoked.
    """

    evidence: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "request_id": package_info["info"]["request_id"],
        "package_manifest_sha256": package_info["package_manifest_sha256"],
        "sampling_contract_sha256": package_info["sampling_contract_sha256"],
        "first_formal_gate": {"active": True},
        "runner_config": _config_evidence(config),
        "runner": {"cwd": config["anisora_workdir"], "argv": argv},
        "probes": {},
        "passed": True,
        "failures": [],
        "created_at": _utc_now(),
    }
    problems: list[str] = []
    resolved: dict[str, Path] = {}
    for field in RUNNER_CONFIG_EXECUTABLE_FIELDS:
        try:
            resolved[field] = _resolve_executable(
                config[field], field
            )
        except harness.HarnessError as exc:
            problems.append(str(exc))
    workdir = Path(config["anisora_workdir"])
    if not workdir.is_dir():
        problems.append(f"anisora_workdir is not a directory: {workdir}")
    runner_script = Path(config["bf16_runner_script"])
    if not runner_script.is_file():
        problems.append(f"bf16_runner_script is not a file: {runner_script}")
    checkpoint_dir = Path(config["checkpoint_dir"])
    if harness._is_link_or_reparse(checkpoint_dir):
        problems.append(
            "checkpoint_dir must not be a symlink/reparse point"
        )
    elif not checkpoint_dir.is_dir():
        problems.append(f"checkpoint_dir is not a directory: {checkpoint_dir}")
    checkpoint_evidence: list[dict[str, Any]] = []
    for record in config["checkpoint_files"]:
        relative = record["relative_path"]
        if harness._is_link_or_reparse(checkpoint_dir):
            continue
        try:
            harness._reject_symlink_components(
                checkpoint_dir,
                relative,
                f"checkpoint file {relative}",
                layer="remote_environment",
            )
        except harness.HarnessError as exc:
            problems.append(
                f"checkpoint file {relative}: {exc}"
            )
            continue
        try:
            resolved_checkpoint = (checkpoint_dir / relative).resolve(
                strict=True
            )
        except OSError as exc:
            problems.append(
                f"checkpoint file is not a regular file: {relative}: {exc}"
            )
            continue
        if not harness._is_within(
            resolved_checkpoint, checkpoint_dir.resolve()
        ):
            problems.append(
                f"checkpoint file escapes checkpoint_dir: {relative}"
            )
            continue
        if not resolved_checkpoint.is_file():
            problems.append(
                f"checkpoint file is not a regular file: {relative}"
            )
            continue
        try:
            actual_size = resolved_checkpoint.stat().st_size
        except OSError as exc:
            problems.append(f"cannot stat checkpoint file {relative}: {exc}")
            continue
        if actual_size != record["size_bytes"]:
            problems.append(
                f"checkpoint size mismatch for {relative}: "
                f"actual={actual_size} expected={record['size_bytes']}"
            )
        checkpoint_evidence.append(
            {"relative_path": relative, "size_bytes": actual_size}
        )
    for field in (
        "cgroup_memory_current",
        "cgroup_memory_peak",
        "cgroup_memory_events",
    ):
        path = Path(config[field])
        if not path.is_file():
            problems.append(f"{field} is not a file: {path}")
    if problems:
        evidence["passed"] = False
        evidence["failures"] = problems
        raise harness.HarnessError(
            "remote_environment", "; ".join(problems)
        )

    probes: dict[str, Any] = {}
    python_cuda_payload = _probe_python_cuda(
        resolved["python_executable"]
    )
    _judge_python_cuda(python_cuda_payload)
    probes["python_cuda"] = python_cuda_payload
    probes["ffmpeg_version"] = _probe_version(resolved["ffmpeg"], "ffmpeg")
    probes["ffprobe_version"] = _probe_version(resolved["ffprobe"], "ffprobe")
    nvidia_rows = _probe_nvidia_smi(resolved["nvidia_smi"])
    _judge_nvidia_smi(nvidia_rows)
    probes["nvidia_smi"] = nvidia_rows
    probes["cgroup"] = {
        "memory_current": _read_text(
            Path(config["cgroup_memory_current"]), "cgroup_memory_current"
        ).strip(),
        "memory_peak": _read_text(
            Path(config["cgroup_memory_peak"]), "cgroup_memory_peak"
        ).strip(),
        "memory_events": _read_text(
            Path(config["cgroup_memory_events"]), "cgroup_memory_events"
        ).strip(),
    }
    probes["checkpoint_files"] = checkpoint_evidence
    evidence["probes"] = probes
    return evidence


def _gpu_monitor(
    nvidia_smi: Path, path: Path, stop: threading.Event
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        str(nvidia_smi),
                        "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    handle.write(f"{time.time()},{completed.stdout.strip()}\n")
                    handle.flush()
            except Exception:  # noqa: BLE001, S110 - sampling best-effort
                pass
            stop.wait(1)


def _memory_monitor(
    cgroup_current: Path, path: Path, stop: threading.Event
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            try:
                text = cgroup_current.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
                if text:
                    handle.write(f"{time.time()},{text}\n")
                    handle.flush()
            except Exception:  # noqa: BLE001, S110 - sampling best-effort
                pass
            stop.wait(1)


def _resource_evidence_best_effort(
    config: dict[str, Any], staging: Path
) -> dict[str, Any]:
    """Best-effort resource evidence for the failure path (R8).

    Never raises; unavailable resources are recorded as explicit bounded
    errors so the permitted failure evidence can still be atomically
    published while preserving the original failure.
    """

    summary: dict[str, Any] = {
        "gpu_samples": "gpu_samples.csv",
        "memory_samples": "memory_samples.csv",
        "unavailable": [],
    }
    events_text = ""
    for field in (
        "cgroup_memory_current",
        "cgroup_memory_peak",
        "cgroup_memory_events",
    ):
        try:
            value = _read_text(Path(config[field]), field).strip()
        except harness.HarnessError as exc:
            summary["unavailable"].append(
                {"field": field, "error": str(exc)}
            )
            continue
        if field == "cgroup_memory_events":
            events_text = value
        else:
            summary[field] = value
    parsed: dict[str, int] = {}
    for line in events_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in (
            "oom",
            "oom_kill",
            "oom_group_kill",
        ):
            try:
                parsed[parts[0]] = int(parts[1])
            except ValueError:
                pass
    try:
        events_path = staging / "memory_events.txt"
        events_path.write_text(events_text + "\n", encoding="utf-8")
        summary["memory_events_path"] = "memory_events.txt"
    except OSError as exc:
        summary["unavailable"].append(
            {"field": "memory_events.txt", "error": str(exc)}
        )
    if parsed:
        summary["memory_events"] = parsed
    return summary


def _sampling_receipt(
    package_info: dict[str, Any], raw_sha256: str
) -> dict[str, Any]:
    info = package_info["info"]
    sample_steps = package_info["contract"]["frozen_parameters"][
        "sample_steps"
    ]
    receipt = dict(harness.FROZEN_SAMPLING, sample_steps=sample_steps)
    receipt.update(
        {
            "schema_version": harness.RECEIPT_SCHEMA,
            "request_id": info["request_id"],
            "request_sha256": info["request_sha256"],
            "package_manifest_sha256": package_info[
                "package_manifest_sha256"
            ],
            "sampling_contract_sha256": package_info[
                "sampling_contract_sha256"
            ],
            "start_sha256": info["start_sha256"],
            "end_sha256": info["end_sha256"],
            "raw_sha256": raw_sha256,
            "status": "success",
        }
    )
    return receipt


def _runtime_input_line(package_info: dict[str, Any]) -> str:
    prompt = package_info["contract"]["resolved_prompt"]
    k0 = package_info["package_root"] / "inputs" / "k0.png"
    k_end = package_info["package_root"] / "inputs" / "k_end.png"
    positions = ",".join(str(p) for p in FROZEN_GUIDE_POSITIONS)
    return f"{prompt}@@{k0},{k_end}&&{positions}"


def _recovery_paths(output: Path) -> tuple[Path, Path]:
    """Deterministic sibling recovery paths for an output path."""

    output = Path(output)
    return (
        output.with_name(output.name + ".raw-recovery.mp4"),
        output.with_name(output.name + ".raw-recovery.json"),
    )


def _copy_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically create ``path`` with the exact ``data`` bytes.

    Never overwrites: the target is created via a same-directory temp file
    plus ``os.replace`` after an existence check. Any failure cleans only the
    temp file and raises ``evidence_incomplete``.
    """

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise harness.HarnessError(
                "evidence_incomplete",
                f"raw recovery path already exists: {path}",
            )
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise harness.HarnessError(
            "evidence_incomplete",
            f"cannot persist raw recovery {path}: {exc}",
        ) from exc
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _recovery_manifest_dict(
    package_info: dict[str, Any],
    raw_sha256: str,
    size_bytes: int,
    sample_steps: int,
    validation_status: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": RAW_RECOVERY_SCHEMA,
        "request_id": package_info["info"]["request_id"],
        "request_sha256": package_info["info"]["request_sha256"],
        "package_manifest_sha256": package_info["package_manifest_sha256"],
        "sampling_contract_sha256": package_info[
            "sampling_contract_sha256"
        ],
        "start_sha256": package_info["info"]["start_sha256"],
        "end_sha256": package_info["info"]["end_sha256"],
        "raw_sha256": raw_sha256,
        "size_bytes": size_bytes,
        "sample_steps": sample_steps,
        "validation_status": validation_status,
        "created_at": _utc_now(),
    }
    if validation_status == "valid":
        manifest["validated_at"] = _utc_now()
    return manifest


def _persist_raw_recovery(
    package_info: dict[str, Any],
    raw_bytes: bytes,
    raw_sha256: str,
    sample_steps: int,
    recovery_mp4: Path,
    recovery_manifest: Path,
) -> None:
    """Persist the exact raw bytes plus an unverified recovery manifest.

    Runs immediately after the runner produces the exact regular non-link
    ``0.mp4`` and before exit-code/probe/decode validation can fail, so later
    validation or orchestration failures cannot delete the model result.
    """

    _copy_bytes_atomic(recovery_mp4, raw_bytes)
    try:
        harness.dump_json_atomic(
            recovery_manifest,
            _recovery_manifest_dict(
                package_info,
                raw_sha256,
                len(raw_bytes),
                sample_steps,
                "unverified",
            ),
        )
    except Exception as exc:
        raise harness.HarnessError(
            "evidence_incomplete",
            f"cannot write raw recovery manifest: {exc}",
        ) from exc


def _mark_raw_recovery_valid(
    recovery_manifest: Path,
    expected_raw_sha256: str,
    expected_sample_steps: int,
) -> None:
    """Atomically promote the recovery manifest to ``valid``."""

    try:
        data = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise harness.HarnessError(
            "evidence_incomplete",
            f"cannot read raw recovery manifest: {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise harness.HarnessError(
            "evidence_incomplete",
            "raw recovery manifest must be a JSON object",
        )
    if data.get("raw_sha256") != expected_raw_sha256:
        raise harness.HarnessError(
            "evidence_incomplete",
            "raw recovery manifest raw_sha256 drifted from the validated raw",
        )
    if data.get("sample_steps") != expected_sample_steps:
        raise harness.HarnessError(
            "evidence_incomplete",
            "raw recovery manifest sample_steps drifted from the packaged "
            "contract",
        )
    data["validation_status"] = "valid"
    data["validated_at"] = _utc_now()
    harness.dump_json_atomic(recovery_manifest, data)


def _publish_failure(
    staging: Path,
    output: Path,
    package_info: dict[str, Any],
    config: dict[str, Any],
    argv: list[str] | None,
    invoked: bool,
    exit_code: int | None,
    runtime_seconds: float | None,
    exc: harness.HarnessError,
    runtime_input_bytes: bytes | None = None,
) -> None:
    try:
        preflight_path = staging / "preflight.json"
        if preflight_path.is_file():
            preflight = json.loads(
                preflight_path.read_text(encoding="utf-8")
            )
            preflight["passed"] = False
            preflight["failures"] = [str(exc)]
        else:
            preflight = {
                "schema_version": PREFLIGHT_SCHEMA,
                "request_id": package_info["info"]["request_id"],
                "package_manifest_sha256": package_info[
                    "package_manifest_sha256"
                ],
                "sampling_contract_sha256": package_info[
                    "sampling_contract_sha256"
                ],
                "first_formal_gate": {"active": True},
                "runner_config": _config_evidence(config),
                "runner": {
                    "cwd": config["anisora_workdir"],
                    "argv": argv,
                },
                "probes": {},
                "passed": False,
                "failures": [str(exc)],
                "created_at": _utc_now(),
            }
        harness.dump_json_atomic(preflight_path, preflight)
        status = "preflight_failed" if not invoked else "sampling_technical"
        result = {
            "schema_version": RESULT_SCHEMA,
            "request_id": package_info["info"]["request_id"],
            "status": status,
            "runner": {
                "invoked": invoked,
                "exit_code": exit_code,
                "runtime_seconds": runtime_seconds,
                "cwd": config["anisora_workdir"],
                "argv": argv,
            },
            "resources": (
                _resource_evidence_best_effort(config, staging)
                if invoked
                else {}
            ),
            "failure": {"layer": exc.layer, "message": str(exc)},
            "created_at": _utc_now(),
        }
        harness.dump_json_atomic(staging / "result.json", result)
        if runtime_input_bytes is not None:
            harness._write_staged_bytes(
                staging / "runtime_input.txt", runtime_input_bytes
            )
        harness._publish_dir(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def cmd_run(
    package: Path,
    runner_config: Path,
    output: Path,
) -> Path:
    """Validate the package/config, run one sample and publish evidence."""

    output = Path(output)
    recovery_mp4, recovery_manifest = _recovery_paths(output)
    if output.exists():
        raise harness.HarnessError(
            "input_contract", f"run output already exists: {output}"
        )
    for recovery_path in (recovery_mp4, recovery_manifest):
        if recovery_path.exists() or recovery_path.is_symlink():
            raise harness.HarnessError(
                "input_contract",
                f"raw recovery path already exists: {recovery_path}",
            )
    package_info = validate_package(package)
    sample_steps = package_info["contract"]["frozen_parameters"][
        "sample_steps"
    ]
    config = _validate_runner_config(
        harness._load_json_bytes(
            harness._capture_bytes(runner_config, "input_contract"),
            "runner_config",
            "input_contract",
        )
    )

    parent = output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(parent))
    )
    argv: list[str] | None = None
    invoked = False
    exit_code: int | None = None
    runtime_seconds: float | None = None
    runtime_input_bytes: bytes | None = None
    try:
        runner_output = staging / "runner-output"
        runner_output.mkdir(parents=True, exist_ok=False)
        runtime_input = runner_output / "anisora_input.txt"
        runtime_input_bytes = (
            _runtime_input_line(package_info) + "\n"
        ).encode("utf-8")
        runtime_input_sha256 = harness._sha256_bytes(runtime_input_bytes)
        runtime_input.write_bytes(runtime_input_bytes)
        argv = _frozen_argv(
            config, runner_output, runtime_input, sample_steps
        )
        preflight = _preflight(package_info, config, argv)
        harness.dump_json_atomic(staging / "preflight.json", preflight)

        log_path = staging / "runner.log"
        gpu_path = staging / "gpu_samples.csv"
        memory_path = staging / "memory_samples.csv"
        stop = threading.Event()
        threads = [
            threading.Thread(
                target=_gpu_monitor,
                args=(Path(config["nvidia_smi"]), gpu_path, stop),
                daemon=True,
            ),
            threading.Thread(
                target=_memory_monitor,
                args=(
                    Path(config["cgroup_memory_current"]),
                    memory_path,
                    stop,
                ),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        started = time.monotonic()
        try:
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    argv,
                    cwd=config["anisora_workdir"],
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=5)
        invoked = True
        exit_code = completed.returncode
        runtime_seconds = round(time.monotonic() - started, 3)
        resources = _resource_evidence_best_effort(config, staging)

        # R5a: probe the immediate runner-output dir for the exact regular
        # non-link 0.mp4 without raising, so recovery can be persisted before
        # any later validation (R6 / exit code / probe / decode) can fail.
        produced_names: list[str] = []
        for entry in runner_output.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".mp4":
                produced_names.append(entry.name)
        produced = runner_output / "0.mp4"
        exact_regular_0mp4 = (
            produced_names == ["0.mp4"]
            and not produced.is_symlink()
            and produced.is_file()
        )
        if exact_regular_0mp4:
            raw_bytes = produced.read_bytes()
            raw_sha256 = harness._sha256_bytes(raw_bytes)
            # G1-MK2-L: persist the exact raw recovery artifact immediately,
            # before exit-code/probe/decode validation can fail, so the model
            # result survives later validation or orchestration failures.
            _persist_raw_recovery(
                package_info,
                raw_bytes,
                raw_sha256,
                sample_steps,
                recovery_mp4,
                recovery_manifest,
            )

        # R6: the passed runtime input must still be the captured bytes.
        if runtime_input.is_symlink() or not runtime_input.is_file():
            raise harness.HarnessError(
                "evidence_incomplete",
                "runtime input file is missing or not a regular file",
            )
        if (
            harness._capture_bytes(runtime_input, "evidence_incomplete")
            != runtime_input_bytes
        ):
            raise harness.HarnessError(
                "evidence_incomplete",
                "runtime input file was mutated during sampling",
            )

        # R5: a non-zero runner exit code always fails, even with 0.mp4.
        if exit_code != 0:
            raise harness.HarnessError(
                "sampling_technical",
                f"runner failed with exit code {exit_code}",
            )

        # R5b: enforce the exact output shape (narrow runtime exception).
        if produced_names != ["0.mp4"]:
            raise harness.HarnessError(
                "sampling_technical",
                "runner output must contain exactly one MP4 named 0.mp4, "
                f"got {produced_names}",
            )
        if produced.is_symlink() or not produced.is_file():
            raise harness.HarnessError(
                "sampling_technical",
                "runner produced no regular 0.mp4",
            )
        raw_bytes = produced.read_bytes()
        raw_sha256 = harness._sha256_bytes(raw_bytes)

        toolkit = FFmpegToolkit(
            ffmpeg=str(Path(config["ffmpeg"])),
            ffprobe=str(Path(config["ffprobe"])),
        )
        try:
            raw_summary = harness._preflight_raw(
                toolkit, produced, raw_sha256
            )
            harness._decode_check_raw(toolkit, produced)
        except (harness.HarnessError, AnimeRemixError) as exc:
            raise harness.HarnessError(
                "sampling_technical", f"invalid raw sample: {exc}"
            ) from exc

        # Every formal raw validation passed: promote the recovery to valid.
        _mark_raw_recovery_valid(
            recovery_manifest, raw_sha256, sample_steps
        )

        receipt = _sampling_receipt(package_info, raw_sha256)
        harness._validate_receipt(
            receipt,
            package_info["info"]["request_id"],
            package_info["info"]["request_sha256"],
            package_info["package_manifest_sha256"],
            package_info["sampling_contract_sha256"],
            package_info["info"]["start_sha256"],
            package_info["info"]["end_sha256"],
            sample_steps,
        )
        harness._write_staged_bytes(staging / "raw_shot.mp4", raw_bytes)
        if harness.sha256_file(staging / "raw_shot.mp4") != raw_sha256:
            raise harness.HarnessError(
                "evidence_incomplete",
                "TOCTOU: raw changed while copying into the output",
            )
        harness.dump_json_atomic(
            staging / "sampling_receipt.json", receipt
        )
        receipt_sha256 = harness.sha256_file(
            staging / "sampling_receipt.json"
        )
        valid = {
            "schema_version": VALID_SAMPLE_SCHEMA,
            "request_id": package_info["info"]["request_id"],
            "package_manifest_sha256": package_info[
                "package_manifest_sha256"
            ],
            "raw_sha256": raw_sha256,
            "guide_positions": list(FROZEN_GUIDE_POSITIONS),
            "raw_shot_mp4": {
                "path": "raw_shot.mp4",
                "sha256": raw_sha256,
                "size_bytes": len(raw_bytes),
            },
            "sampling_receipt": {
                "path": "sampling_receipt.json",
                "sha256": receipt_sha256,
            },
            "runner_invocations": 1,
            "created_at": _utc_now(),
        }
        harness.dump_json_atomic(
            staging / "valid_sample_complete.json", valid
        )
        harness._write_staged_bytes(
            staging / "runtime_input.txt", runtime_input_bytes
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "request_id": package_info["info"]["request_id"],
            "status": "success",
            "runner": {
                "invoked": invoked,
                "exit_code": exit_code,
                "runtime_seconds": runtime_seconds,
                "cwd": config["anisora_workdir"],
                "argv": argv,
            },
            "raw": {
                "path": "raw_shot.mp4",
                "sha256": raw_sha256,
                "size_bytes": len(raw_bytes),
                "probe": raw_summary,
            },
            "runtime_input": {
                "path": "runtime_input.txt",
                "sha256": runtime_input_sha256,
            },
            "resources": resources,
            "failure": None,
            "created_at": _utc_now(),
        }
        harness.dump_json_atomic(staging / "result.json", result)
        harness._publish_dir(staging, output)
    except harness.HarnessError as exc:
        _publish_failure(
            staging,
            output,
            package_info,
            config,
            argv,
            invoked,
            exit_code,
            runtime_seconds,
            exc,
            runtime_input_bytes,
        )
        raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote_sample",
        description="G1-MK1-R deterministic single-sample remote runner",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="validate package/config and run exactly one sample"
    )
    run_parser.add_argument("--package", required=True, type=Path)
    run_parser.add_argument("--runner-config", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            cmd_run(args.package, args.runner_config, args.output)
        else:
            raise harness.HarnessError(
                "input_contract", f"unknown command: {args.command}"
            )
    except harness.HarnessError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
