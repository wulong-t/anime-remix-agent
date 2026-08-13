"""G1-MK1-R-PREP-L tests: remote_sample + qa_evidence + fake runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import types
import zlib
from pathlib import Path

import pytest

from experiments.manual_keyframe_mvp import manual_keyframe_mvp as harness
from experiments.manual_keyframe_mvp import qa_evidence, remote_sample


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _solid_png_bytes(
    width: int, height: int, color: tuple[int, int, int]
) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(color) * width
    raw = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def _make_inputs(
    root: Path,
    *,
    k0_color: tuple[int, int, int] = (255, 255, 255),
    k_end_color: tuple[int, int, int] = (255, 255, 255),
) -> dict[str, object]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    k0 = inputs / "k0.png"
    k_end = inputs / "k_end.png"
    k0.write_bytes(_solid_png_bytes(1280, 720, k0_color))
    k_end.write_bytes(_solid_png_bytes(1280, 720, k_end_color))
    return {
        "k0": k0,
        "k_end": k_end,
        "k0_sha": _sha256_file(k0),
        "k_end_sha": _sha256_file(k_end),
    }


def _base_request(info: dict[str, object], **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": harness.REQUEST_SCHEMA,
        "request_id": "g1mk1-synthetic-001",
        "start_keyframe": "inputs/k0.png",
        "end_keyframe": "inputs/k_end.png",
        "start_provenance": "inputs/k0.provenance.json",
        "end_provenance": "inputs/k_end.provenance.json",
        "start_sha256": info["k0_sha"],
        "end_sha256": info["k_end_sha"],
        "subject_description": "An original 2D cel-animation woman",
        "scene_description": "A quiet observatory control room at dusk",
        "action": "turn head slightly to the right",
        "start_state": "near-frontal calm",
        "end_state": "three-quarter calm",
        "emotion": "calm",
        "shot_scale": "medium",
        "camera": "fixed",
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
    }
    data.update(overrides)
    return data


def _provenance_asset_for(name: str) -> str:
    return "inputs/k0.png" if name.startswith("k0.") else "inputs/k_end.png"


def _write_provenance(root: Path, name: str, image: Path) -> Path:
    path = root / "inputs" / name
    path.write_text(
        json.dumps(
            {
                "asset": _provenance_asset_for(name),
                "sha256": _sha256_file(image),
                "creation_method": "human-edited synthetic test",
                "external_inputs": [],
                "named_references": {
                    "artists": [],
                    "studios": [],
                    "series": [],
                    "characters": [],
                },
                "rights_basis": "original/authorized synthetic test",
                "public_demo_allowed": False,
                "notes": "synthetic test asset",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _make_request_root(
    tmp_path: Path,
    *,
    k0_color: tuple[int, int, int] = (255, 255, 255),
    k_end_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "request-root"
    info = _make_inputs(root, k0_color=k0_color, k_end_color=k_end_color)
    _write_provenance(root, "k0.provenance.json", info["k0"])  # type: ignore[arg-type]
    _write_provenance(root, "k_end.provenance.json", info["k_end"])  # type: ignore[arg-type]
    request = _base_request(info)
    (root / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    return root, request


def _run_inspect(request_root: Path, tmp_path: Path) -> Path:
    output = tmp_path / "inspection"
    harness.cmd_inspect(request_root / "request.json", output)
    return output


def _approve(pending: Path) -> Path:
    data = json.loads(pending.read_text(encoding="utf-8"))
    for key in data["rights"]:
        data["rights"][key] = True
    for key in (
        "identity",
        "endpoint_pose",
        "body_camera_background",
        "style",
        "artifact",
    ):
        data["visual_review"][key] = "pass"
    data["visual_review"]["accept_borderline"] = False
    data["visual_review"]["overall"] = "approved"
    data["approved_at"] = "2026-08-09T12:00:00+08:00"
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return pending


def _make_package(
    tmp_path: Path,
    _monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "package",
    k0_color: tuple[int, int, int] = (255, 255, 255),
    k_end_color: tuple[int, int, int] = (0, 0, 0),
    sample_steps: int = 40,
) -> Path:
    request_root, _ = _make_request_root(
        tmp_path, k0_color=k0_color, k_end_color=k_end_color
    )
    # The first formal gate is run-scoped: the synthetic request already
    # satisfies the exact path/SHA/image contract, so no frozen K0 constants
    # (removed from the harness) are needed to activate it.
    inspection = _run_inspect(request_root, tmp_path)
    approval = _approve(inspection / "approval.json")
    return harness.cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / name,
        sample_steps,
    )


def _staging_leftovers(parent: Path) -> list[Path]:
    return [p for p in parent.iterdir() if ".staging-" in p.name]


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    if os.name == "nt":
        path = tmp_path / "fake_nvidia_smi.cmd"
        path.write_text(
            "@echo off\r\n"
            "echo 0, NVIDIA GeForce RTX 4090, 24564, 1234, 100\r\n",
            encoding="ascii",
        )
    else:
        path = tmp_path / "fake_nvidia_smi"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "print('0, NVIDIA GeForce RTX 4090, 24564, 1234, 100')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
    return path


def _default_runner_config(tmp_path: Path) -> dict[str, object]:
    workdir = tmp_path / "anisora-workdir"
    workdir.mkdir(exist_ok=True)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = checkpoint_dir / "model-00001.safetensors"
    checkpoint.write_bytes(b"\x00" * 4096)
    runner = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "fake_anisora_runner.py"
    )
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir(exist_ok=True)
    (cgroup / "memory.current").write_text("1048576\n", encoding="utf-8")
    (cgroup / "memory.peak").write_text("2097152\n", encoding="utf-8")
    (cgroup / "memory.events").write_text(
        "oom 0\noom_kill 0\n", encoding="utf-8"
    )
    return {
        "schema_version": "g1-mk1-runner-config-v1",
        "python_executable": sys.executable,
        "anisora_workdir": str(workdir),
        "bf16_runner_script": str(runner),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": [
            {
                "relative_path": "model-00001.safetensors",
                "size_bytes": 4096,
            }
        ],
        "ffmpeg": shutil.which("ffmpeg") or "ffmpeg",
        "ffprobe": shutil.which("ffprobe") or "ffprobe",
        "nvidia_smi": str(_fake_nvidia_smi(tmp_path)),
        "cgroup_memory_current": str(cgroup / "memory.current"),
        "cgroup_memory_peak": str(cgroup / "memory.peak"),
        "cgroup_memory_events": str(cgroup / "memory.events"),
    }


def _write_runner_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "runner-config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _mock_passing_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock a passing CUDA/GPU preflight (R1).

    The local development Python reports no torch/CUDA, so successful
    remote-sample tests must explicitly mock a passing probe.
    """

    monkeypatch.setattr(
        remote_sample,
        "_probe_python_cuda",
        lambda python_path: {
            "python_version": "3.13.5",
            "platform": "synthetic",
            "torch_available": True,
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 1,
        },
    )
    monkeypatch.setattr(
        remote_sample,
        "_probe_nvidia_smi",
        lambda exe: ["0, NVIDIA GeForce RTX 4090, 24564, 1234, 100"],
    )


def _gradient_png_bytes(width: int, height: int) -> bytes:
    """Deterministic non-uniform RGB PNG (R2 byte-for-byte canvas test)."""

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    denominator = width + height
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(
                (
                    (x * 255) // width,
                    (y * 255) // height,
                    ((x + y) * 255) // denominator,
                )
            )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _rgba_png_bytes(width: int, height: int) -> bytes:
    """Deterministic RGBA PNG with varying alpha (R2 RGBA case)."""

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(
                (
                    (x * 255) // width,
                    (y * 255) // height,
                    ((width - x) * 255) // width,
                    255 - (x % 64),
                )
            )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def _make_raw_mp4(path: Path) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x704:r=16",
            "-frames:v",
            "81",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _make_generated_mp4(
    path: Path,
    first_color: str = "white",
    last_color: str = "black",
    *,
    conformant: bool = True,
) -> None:
    """Synthesize a 121-frame 1280x720 clip for the finalized contract.

    With ``conformant=True`` the clip carries the full finalized media
    metadata (H.264 high/3.1, yuv420p, 1/48000 timescale, BT.709 limited,
    chroma-left). With ``conformant=False`` those tags are omitted so tests
    can exercise the R7 rejection path.
    """

    segment_dir = path.parent / (path.stem + "-seg")
    segment_dir.mkdir(exist_ok=True)
    first = segment_dir / "first.mp4"
    last = segment_dir / "last.mp4"
    try:
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={first_color}:s=1280x720:r=24",
                "-frames:v",
                "120",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(first),
            ],
            check=True,
        )
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={last_color}:s=1280x720:r=24",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(last),
            ],
            check=True,
        )
        command = [
            _ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(first),
            "-i",
            str(last),
            "-filter_complex",
            (
                "[0:v][1:v]concat=n=2:v=1:a=0,"
                "setparams=range=limited:color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709:field_mode=prog"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ]
        if conformant:
            command.extend(
                [
                    "-profile:v",
                    "high",
                    "-level:v",
                    "3.1",
                    "-video_track_timescale",
                    "48000",
                    "-chroma_sample_location",
                    "left",
                    "-map_metadata",
                    "-1",
                ]
            )
        command.append(str(path))
        subprocess.run(command, check=True)
    finally:
        shutil.rmtree(segment_dir, ignore_errors=True)


def _receipt_for(package: Path, raw_sha: str) -> dict[str, object]:
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (package / "request.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (package / "sampling_contract.json").read_text(encoding="utf-8")
    )
    receipt = dict(
        harness.FROZEN_SAMPLING,
        sample_steps=contract["frozen_parameters"]["sample_steps"],
    )
    receipt.update(
        {
            "schema_version": harness.RECEIPT_SCHEMA,
            "request_id": request["request_id"],
            "request_sha256": manifest["request_sha256"],
            "package_manifest_sha256": _sha256_file(
                package / "package_manifest.json"
            ),
            "sampling_contract_sha256": _sha256_file(
                package / "sampling_contract.json"
            ),
            "start_sha256": manifest["start_sha256"],
            "end_sha256": manifest["end_sha256"],
            "raw_sha256": raw_sha,
            "status": "success",
        }
    )
    return receipt


def _build_finalized_run(
    tmp_path: Path,
    package: Path,
    raw_path: Path,
    generated_path: Path,
) -> Path:
    run_dir = tmp_path / "finalized-run"
    run_dir.mkdir()
    raw_sha = _sha256_file(raw_path)
    receipt = _receipt_for(package, raw_sha)
    receipt_bytes = json.dumps(receipt, indent=2).encode("utf-8")
    request = json.loads(
        (package / "request.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (package / "sampling_contract.json").read_text(encoding="utf-8")
    )
    shutil.copyfile(
        package / "package_manifest.json",
        run_dir / "package_manifest.json",
    )
    (run_dir / "remote_receipt.json").write_bytes(receipt_bytes)
    shutil.copyfile(raw_path, run_dir / "raw_shot.mp4")
    run_clip = run_dir / "generated_clip.mp4"
    shutil.copyfile(generated_path, run_clip)
    (run_dir / "timeline.json").write_text(
        json.dumps(harness._timeline_dict(request, run_clip), indent=2),
        encoding="utf-8",
    )
    (run_dir / "clips.json").write_text(
        json.dumps(harness._clips_document_dict(request), indent=2),
        encoding="utf-8",
    )
    generation_manifest = {
        "schema_version": harness.GENERATION_MANIFEST_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": manifest["request_sha256"],
        "approval_sha256": _sha256_file(package / "approval.json"),
        "package": {
            "path": "package_manifest.json",
            "sha256": _sha256_file(package / "package_manifest.json"),
        },
        "remote_receipt": {
            "path": "remote_receipt.json",
            "sha256": _sha256_bytes(receipt_bytes),
        },
        "raw": {
            "path": "raw_shot.mp4",
            "sha256": raw_sha,
            "probe": {},
        },
        "normalized": {
            "path": "generated_clip.mp4",
            "sha256": _sha256_file(generated_path),
            "probe": {},
        },
        "timeline": {
            "path": "timeline.json",
            "sha256": _sha256_file(run_dir / "timeline.json"),
        },
        "clips": {
            "path": "clips.json",
            "sha256": _sha256_file(run_dir / "clips.json"),
        },
        "render_log": {"path": "render.log", "sha256": "0" * 64},
        "final": {"path": "output.mp4", "sha256": "0" * 64, "probe": {}},
        "model_params": dict(
            harness.FROZEN_SAMPLING,
            sample_steps=contract["frozen_parameters"]["sample_steps"],
        ),
        "total_audio_samples": 242000,
        "failure_layer": None,
        "created_at": "2026-08-09T12:00:00+08:00",
    }
    (run_dir / "generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# remote_sample
# ---------------------------------------------------------------------------


def test_remote_sample_success_exact_argv_and_runtime_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)
    config = _default_runner_config(tmp_path)
    config_path = _write_runner_config(tmp_path, config)
    output = tmp_path / "sample-output"

    remote_sample.cmd_run(package, config_path, output)

    assert (output / "valid_sample_complete.json").is_file()
    assert (output / "sampling_receipt.json").is_file()
    assert (output / "raw_shot.mp4").is_file()
    assert (output / "preflight.json").is_file()
    assert (output / "result.json").is_file()
    assert (output / "runtime_input.txt").is_file()
    assert (output / "runner-output" / "0.mp4").is_file()

    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    request = json.loads(
        (package / "request.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (output / "sampling_receipt.json").read_text(encoding="utf-8")
    )
    harness._validate_receipt(
        receipt,
        request["request_id"],
        manifest["request_sha256"],
        _sha256_file(package / "package_manifest.json"),
        _sha256_file(package / "sampling_contract.json"),
        manifest["start_sha256"],
        manifest["end_sha256"],
    )
    raw_sha = _sha256_file(output / "raw_shot.mp4")
    assert receipt["raw_sha256"] == raw_sha
    valid = json.loads(
        (output / "valid_sample_complete.json").read_text(encoding="utf-8")
    )
    assert valid["raw_sha256"] == raw_sha
    assert valid["runner_invocations"] == 1

    summary = harness._probe_summary(
        harness.FFmpegToolkit(), output / "raw_shot.mp4"
    )
    assert summary["video_codec"] == "h264"
    assert (summary["width"], summary["height"]) == (1280, 704)
    assert summary["r_frame_rate"] == "16/1"
    assert summary["avg_frame_rate"] == "16/1"
    assert summary["nb_read_frames"] == 81
    assert summary["audio_streams"] == 0

    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "success"
    assert result["raw"]["sha256"] == raw_sha
    assert result["runner"]["invoked"] is True
    assert result["failure"] is None
    argv = result["runner"]["argv"]
    assert argv[:2] == [
        config["python_executable"],
        config["bf16_runner_script"],
    ]
    assert argv[2:8] == [
        "--task",
        "i2v-14B",
        "--size",
        "1280*720",
        "--ckpt_dir",
        config["checkpoint_dir"],
    ]
    assert argv[8] == "--image"
    image_dir = Path(argv[9])
    assert image_dir.name == "runner-output"
    assert argv[10:12] == [
        "--prompt",
        str(image_dir / "anisora_input.txt"),
    ]
    assert argv[12:] == [
        "--base_seed",
        "4096",
        "--frame_num",
        "81",
        "--sample_steps",
        "40",
        "--sample_shift",
        "5",
        "--sample_guide_scale",
        "5",
        "--offload_model",
        "True",
    ]
    invoked = json.loads(
        (output / "runner-output" / "invoked.json").read_text(
            encoding="utf-8"
        )
    )
    assert invoked["argv"] == argv[2:]
    assert invoked["argv"][0:2] == ["--task", "i2v-14B"]

    contract = json.loads(
        (package / "sampling_contract.json").read_text(encoding="utf-8")
    )
    assert contract["frozen_parameters"]["sample_steps"] == 40
    k0_abs = str((package / "inputs" / "k0.png").resolve())
    k_end_abs = str((package / "inputs" / "k_end.png").resolve())
    expected_line = (
        f"{contract['resolved_prompt']}@@{k0_abs},{k_end_abs}&&0,1"
    )
    runtime_text = (output / "runtime_input.txt").read_text(encoding="utf-8")
    assert runtime_text == expected_line + "\n"
    assert invoked["prompt_text"] == runtime_text

    preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["passed"] is True
    assert preflight["first_formal_gate"]["active"] is True
    assert preflight["probes"]["python_cuda"]["python_version"]
    assert preflight["probes"]["ffmpeg_version"]
    assert preflight["probes"]["ffprobe_version"]
    assert preflight["probes"]["nvidia_smi"]

    for name in (
        "preflight.json",
        "result.json",
        "valid_sample_complete.json",
        "sampling_receipt.json",
        "runtime_input.txt",
    ):
        text = (output / name).read_text(encoding="utf-8", errors="replace")
        assert "API_KEY" not in text
        assert "SECRET" not in text
        assert "TOKEN" not in text
        assert "os.environ" not in text

    recovery_mp4 = tmp_path / "sample-output.raw-recovery.mp4"
    recovery_manifest = tmp_path / "sample-output.raw-recovery.json"
    assert recovery_mp4.read_bytes() == (
        output / "raw_shot.mp4"
    ).read_bytes()
    recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    assert recovery["schema_version"] == "g1-mk1-raw-recovery-v1"
    assert recovery["request_id"] == manifest["request_id"]
    assert recovery["request_sha256"] == manifest["request_sha256"]
    assert recovery["package_manifest_sha256"] == _sha256_file(
        package / "package_manifest.json"
    )
    assert recovery["sampling_contract_sha256"] == _sha256_file(
        package / "sampling_contract.json"
    )
    assert recovery["raw_sha256"] == raw_sha
    assert recovery["size_bytes"] == recovery_mp4.stat().st_size
    assert recovery["sample_steps"] == 40
    assert recovery["validation_status"] == "valid"
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_rejects_second_sample_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    remote_sample.cmd_run(package, config_path, output)
    with pytest.raises(harness.HarnessError, match="already exists"):
        remote_sample.cmd_run(package, config_path, output)
    valid = json.loads(
        (output / "valid_sample_complete.json").read_text(encoding="utf-8")
    )
    assert valid["runner_invocations"] == 1
    assert (output / "raw_shot.mp4").is_file()


def test_steps10_package_remote_finalize_qa_no_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid steps-10 package drives remote/finalize/QA with no shim."""

    package = _make_package(tmp_path, monkeypatch, sample_steps=10)
    contract = json.loads(
        (package / "sampling_contract.json").read_text(encoding="utf-8")
    )
    assert contract["frozen_parameters"]["sample_steps"] == 10

    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    remote_sample.cmd_run(package, config_path, output)

    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    argv = result["runner"]["argv"]
    assert argv[argv.index("--sample_steps") + 1] == "10"
    receipt = json.loads(
        (output / "sampling_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["sample_steps"] == 10
    assert receipt["sampling_contract_sha256"] == _sha256_file(
        package / "sampling_contract.json"
    )
    raw_sha = _sha256_file(output / "raw_shot.mp4")
    recovery_mp4 = tmp_path / "sample-output.raw-recovery.mp4"
    recovery_manifest = tmp_path / "sample-output.raw-recovery.json"
    assert recovery_mp4.read_bytes() == (output / "raw_shot.mp4").read_bytes()
    recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    assert recovery["sample_steps"] == 10
    assert recovery["raw_sha256"] == raw_sha
    assert recovery["validation_status"] == "valid"

    run = tmp_path / "finalized-run"
    harness.cmd_finalize(
        package,
        output / "raw_shot.mp4",
        output / "sampling_receipt.json",
        run,
    )
    generation = json.loads(
        (run / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert generation["model_params"]["sample_steps"] == 10

    qa_output = tmp_path / "qa-output"
    qa_evidence.cmd_qa(
        package,
        output / "raw_shot.mp4",
        run,
        qa_output,
    )
    metrics = json.loads(
        (qa_output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["raw_sha256"] == raw_sha
    assert recovery_manifest.is_file()
    assert recovery_mp4.is_file()
    assert _staging_leftovers(tmp_path) == []


def test_remote_sample_decode_failure_preserves_unverified_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)

    def fail_decode(toolkit, raw):
        raise harness.HarnessError(
            "media_normalization", "injected decode failure"
        )

    monkeypatch.setattr(harness, "_decode_check_raw", fail_decode)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "injected decode failure" in str(exc.value)

    recovery_mp4 = tmp_path / "sample-output.raw-recovery.mp4"
    recovery_manifest = tmp_path / "sample-output.raw-recovery.json"
    assert recovery_mp4.is_file()
    assert recovery_manifest.is_file()
    recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    assert recovery["validation_status"] == "unverified"
    assert "validated_at" not in recovery
    assert recovery["raw_sha256"] == _sha256_file(recovery_mp4)
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "sampling_technical"
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_unexpected_post_recovery_exception_keeps_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)

    def boom(receipt, *args, **kwargs):
        raise RuntimeError("injected post-copy exception")

    monkeypatch.setattr(harness, "_validate_receipt", boom)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(RuntimeError, match="injected post-copy exception"):
        remote_sample.cmd_run(package, config_path, output)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []
    recovery_mp4 = tmp_path / "sample-output.raw-recovery.mp4"
    recovery_manifest = tmp_path / "sample-output.raw-recovery.json"
    assert recovery_mp4.is_file()
    assert recovery_manifest.is_file()
    recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    assert recovery["validation_status"] == "valid"


@pytest.mark.parametrize("suffix", ["mp4", "json"])
def test_remote_sample_rejects_preexisting_recovery_path_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    recovery_path = tmp_path / f"sample-output.raw-recovery.{suffix}"
    recovery_path.write_bytes(b"existing")
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "input_contract"
    assert "raw recovery path already exists" in str(exc.value)
    assert not output.exists()
    assert not list(tmp_path.rglob("invoked.json"))
    assert _staging_leftovers(tmp_path) == []


@pytest.mark.parametrize("mode", ["fail", "no_output", "extra_mp4"])
def test_remote_sample_no_recovery_without_exact_regular_0mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", mode)
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert not (tmp_path / "sample-output.raw-recovery.mp4").exists()
    assert not (tmp_path / "sample-output.raw-recovery.json").exists()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "raw_shot.mp4").exists()


def test_remote_sample_no_recovery_for_symlink_0mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    real_target = tmp_path / "real-0.mp4"
    _make_raw_mp4(real_target)

    def fake_run(argv, **kwargs):
        image = Path(argv[argv.index("--image") + 1])
        image.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(real_target, image / "0.mp4")
        except OSError:
            pytest.skip("platform cannot create symlinks")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(remote_sample.subprocess, "run", fake_run)
    monkeypatch.setattr(
        remote_sample,
        "_preflight",
        lambda *args, **kwargs: {"passed": True},
    )
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "no regular 0.mp4" in str(exc.value)
    assert not (tmp_path / "sample-output.raw-recovery.mp4").exists()
    assert not (tmp_path / "sample-output.raw-recovery.json").exists()
    assert not (output / "valid_sample_complete.json").exists()


def _tamper_manifest_hash(package: Path) -> None:
    path = package / "package_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["files"]["inputs/k0.png"]["sha256"] = "0" * 64
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _tamper_prompt(package: Path) -> None:
    path = package / "sampling_contract.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["resolved_prompt"] = data["resolved_prompt"] + " DRIFT"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.mark.parametrize(
    "tamper, layer",
    [
        (
            lambda p: (p / "inputs" / "k0.png").write_bytes(
                (p / "inputs" / "k0.png").read_bytes() + b"x"
            ),
            "evidence_incomplete",
        ),
        (_tamper_manifest_hash, "evidence_incomplete"),
        (_tamper_prompt, "evidence_incomplete"),
    ],
)
def test_remote_sample_rejects_package_guide_prompt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper,
    layer: str,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    tamper(package)
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == layer
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.update(extra_field=1), "unknown fields"),
        (lambda c: c.pop("ffprobe"), "missing fields"),
        (
            lambda c: c.update(schema_version="g1-mk1-runner-config-v0"),
            "schema_version",
        ),
        (lambda c: c.update(checkpoint_files=[]), "non-empty"),
        (
            lambda c: c["checkpoint_files"][0].update(
                relative_path="../escape.txt"
            ),
            "traverse",
        ),
        (
            lambda c: c["checkpoint_files"][0].update(size_bytes=True),
            "positive integer",
        ),
        (
            lambda c: c.update(bf16_runner_script="relative/runner.py"),
            "absolute path",
        ),
    ],
)
def test_remote_sample_rejects_invalid_runner_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    config = _default_runner_config(tmp_path)
    mutate(config)
    config_path = _write_runner_config(tmp_path, config)
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "input_contract"
    assert message in str(exc.value)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda c: c["checkpoint_files"][0].update(size_bytes=999999),
            "checkpoint size mismatch",
        ),
        (
            lambda c: c.update(
                python_executable=str(c["python_executable"]) + "-missing"
            ),
            "does not exist",
        ),
        (
            lambda c: c.update(
                cgroup_memory_events=str(c["cgroup_memory_events"]) + "-missing"
            ),
            "is not a file",
        ),
    ],
)
def test_remote_sample_preflight_failure_invokes_runner_zero_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    config = _default_runner_config(tmp_path)
    mutate(config)
    config_path = _write_runner_config(tmp_path, config)
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "remote_environment"
    assert message in str(exc.value)
    assert output.exists()
    preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["passed"] is False
    assert preflight["failures"]
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "preflight_failed"
    assert result["runner"]["invoked"] is False
    assert not (output / "runner.log").exists()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    assert not (output / "runner-output" / "invoked.json").exists()
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_zero_sample_technical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "fail")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "exit code 23" in str(exc.value)
    assert output.exists()
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "sampling_technical"
    assert result["runner"]["invoked"] is True
    assert result["runner"]["exit_code"] == 23
    assert "simulated technical failure" in (
        output / "runner.log"
    ).read_text(encoding="utf-8", errors="replace")
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    assert (output / "gpu_samples.csv").exists()
    assert (output / "memory_samples.csv").exists()
    assert (output / "memory_events.txt").exists()
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_invalid_raw_no_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "invalid_frames")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "sampling_technical"
    assert result["runner"]["exit_code"] == 0
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_publish_failure_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"

    def fail_publish(staging, target):
        raise harness.HarnessError(
            "evidence_incomplete", "injected publish failure"
        )

    monkeypatch.setattr(harness, "_publish_dir", fail_publish)
    with pytest.raises(harness.HarnessError, match="injected publish failure"):
        remote_sample.cmd_run(package, config_path, output)
    assert not output.exists()
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_default_deny_no_directory_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "success")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"

    def boom(*args, **kwargs):
        raise AssertionError("directory discovery is forbidden")

    monkeypatch.setattr(remote_sample.os, "walk", boom)
    monkeypatch.setattr(Path, "rglob", boom)
    real_iterdir = Path.iterdir

    def guarded_iterdir(self):
        if self.name == "runner-output":
            return real_iterdir(self)
        raise AssertionError("directory discovery is forbidden")

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    remote_sample.cmd_run(package, config_path, output)
    assert (output / "valid_sample_complete.json").is_file()


def test_remote_sample_cli_via_frozen_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen CLI revalidates the run-scoped gate in a fresh process.

    The package carries its own exact path/SHA/image binding, so a fresh
    process accepts it and proceeds to the environment preflight, which fails
    here because the local development Python reports no torch/CUDA.
    """

    package = _make_package(tmp_path, monkeypatch)
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "manual_keyframe_mvp"
        / "remote_sample.py"
    )
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output-cli"
    env = os.environ.copy()
    env["ANISORA_FAKE_MODE"] = "success"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--package",
            str(package),
            "--runner-config",
            str(config_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 2, completed.stderr
    assert "approval_blocked" not in completed.stderr
    assert "remote_environment" in completed.stderr
    preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["first_formal_gate"]["active"] is True
    assert preflight["passed"] is False
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "preflight_failed"


# ---------------------------------------------------------------------------
# R1: hard python/CUDA/GPU preflight gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"python_version": "3.13.5"},
        {
            "python_version": "3.13.5",
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "cuda_version": "12.4",
        },
        {
            "python_version": "3.13.5",
            "torch_available": True,
            "cuda_available": False,
            "device_count": 0,
            "cuda_version": "12.4",
        },
        {
            "python_version": "3.13.5",
            "torch_available": True,
            "cuda_available": True,
            "device_count": 0,
            "cuda_version": "12.4",
        },
        {
            "python_version": "3.13.5",
            "torch_available": True,
            "cuda_available": True,
            "device_count": True,
            "cuda_version": "12.4",
        },
        {
            "python_version": "3.13.5",
            "torch_available": True,
            "cuda_available": True,
            "device_count": 1,
            "cuda_version": "",
        },
    ],
)
def test_remote_sample_preflight_rejects_bad_cuda_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload,
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    config = _default_runner_config(tmp_path)
    config_path = _write_runner_config(tmp_path, config)
    monkeypatch.setattr(remote_sample, "_probe_python_cuda", lambda p: payload)
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "remote_environment"
    assert output.exists()
    preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["passed"] is False
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "preflight_failed"
    assert result["runner"]["invoked"] is False
    assert not (output / "runner-output" / "invoked.json").exists()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    assert _staging_leftovers(output.parent) == []


def test_remote_sample_preflight_rejects_empty_nvidia_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    config = _default_runner_config(tmp_path)
    config_path = _write_runner_config(tmp_path, config)
    monkeypatch.setattr(
        remote_sample,
        "_probe_python_cuda",
        lambda python_path: {
            "python_version": "3.13.5",
            "platform": "synthetic",
            "torch_available": True,
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_count": 1,
        },
    )
    monkeypatch.setattr(remote_sample, "_probe_nvidia_smi", lambda exe: [])
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "remote_environment"
    assert "no GPU rows" in str(exc.value)
    assert output.exists()
    preflight = json.loads(
        (output / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["passed"] is False
    assert not (output / "runner-output" / "invoked.json").exists()
    assert not (output / "valid_sample_complete.json").exists()


# ---------------------------------------------------------------------------
# R4: checkpoint symlink/reparse escape rejection
# ---------------------------------------------------------------------------


def test_remote_sample_rejects_checkpoint_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    payload = outside / "payload.bin"
    payload.write_bytes(b"\x00" * 4096)
    config = _default_runner_config(tmp_path)
    checkpoint_dir = Path(config["checkpoint_dir"])
    link = checkpoint_dir / "escaped.safetensors"
    try:
        os.symlink(payload, link)
    except OSError:
        pytest.skip("platform cannot create symlinks")
    config["checkpoint_files"] = [
        {"relative_path": "escaped.safetensors", "size_bytes": 4096}
    ]
    config_path = _write_runner_config(tmp_path, config)
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "remote_environment"
    assert output.exists()
    assert not (output / "runner-output" / "invoked.json").exists()
    assert not (output / "valid_sample_complete.json").exists()


def test_remote_sample_rejects_checkpoint_symlink_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    config = _default_runner_config(tmp_path)
    checkpoint_dir = Path(config["checkpoint_dir"])
    sub = checkpoint_dir / "sub"
    sub.mkdir()
    (sub / "model.bin").write_bytes(b"\x00" * 4096)
    config["checkpoint_files"] = [
        {"relative_path": "sub/model.bin", "size_bytes": 4096}
    ]
    config_path = _write_runner_config(tmp_path, config)
    real_is_link = harness._is_link_or_reparse

    def fake_is_link(path):
        return Path(path) == sub or real_is_link(path)

    monkeypatch.setattr(harness, "_is_link_or_reparse", fake_is_link)
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "remote_environment"
    assert "symlink" in str(exc.value) or "reparse" in str(exc.value)
    assert not (output / "runner-output" / "invoked.json").exists()
    assert not (output / "valid_sample_complete.json").exists()


# ---------------------------------------------------------------------------
# R5: one successful invocation and exactly one sample
# ---------------------------------------------------------------------------


def test_remote_sample_fails_when_runner_exit_nonzero_with_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "fail_with_output")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "exit code 23" in str(exc.value)
    assert output.exists()
    assert (output / "runner-output" / "0.mp4").is_file()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "sampling_technical"
    assert result["runner"]["exit_code"] == 23


def test_remote_sample_fails_on_extra_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "extra_mp4")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "exactly one MP4" in str(exc.value)
    assert "1.mp4" in str(exc.value)
    assert output.exists()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()


# ---------------------------------------------------------------------------
# R6: runtime-input mutation detection
# ---------------------------------------------------------------------------


def test_remote_sample_fails_on_runtime_input_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "mutate_input")
    _mock_passing_probes(monkeypatch)
    config_path = _write_runner_config(
        tmp_path, _default_runner_config(tmp_path)
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "evidence_incomplete"
    assert "mutated" in str(exc.value)
    assert output.exists()
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()
    assert not (output / "raw_shot.mp4").exists()
    contract = json.loads(
        (package / "sampling_contract.json").read_text(encoding="utf-8")
    )
    k0_abs = str((package / "inputs" / "k0.png").resolve())
    k_end_abs = str((package / "inputs" / "k_end.png").resolve())
    expected = f"{contract['resolved_prompt']}@@{k0_abs},{k_end_abs}&&0,1\n"
    assert (output / "runtime_input.txt").read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# R8: best-effort failure resource evidence
# ---------------------------------------------------------------------------


def test_resource_evidence_best_effort_records_unavailable(
    tmp_path: Path,
) -> None:
    config = _default_runner_config(tmp_path)
    missing = str(tmp_path / "missing-cgroup")
    config["cgroup_memory_current"] = missing
    config["cgroup_memory_peak"] = missing
    config["cgroup_memory_events"] = missing
    staging = tmp_path / "staging"
    staging.mkdir()
    summary = remote_sample._resource_evidence_best_effort(config, staging)
    assert len(summary["unavailable"]) == 3
    assert (staging / "memory_events.txt").read_text(encoding="utf-8") == "\n"


def test_remote_sample_failure_publishes_best_effort_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    monkeypatch.setenv("ANISORA_FAKE_MODE", "fail")
    _mock_passing_probes(monkeypatch)
    config = _default_runner_config(tmp_path)
    config_path = _write_runner_config(tmp_path, config)
    monkeypatch.setenv(
        "ANISORA_FAKE_DELETE_FILE", str(config["cgroup_memory_events"])
    )
    output = tmp_path / "sample-output"
    with pytest.raises(harness.HarnessError) as exc:
        remote_sample.cmd_run(package, config_path, output)
    assert exc.value.layer == "sampling_technical"
    assert "exit code 23" in str(exc.value)
    assert output.exists()
    result = json.loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "sampling_technical"
    assert result["failure"]["message"] == str(exc.value)
    assert result["resources"]["unavailable"]
    assert not (output / "valid_sample_complete.json").exists()
    assert not (output / "sampling_receipt.json").exists()


# ---------------------------------------------------------------------------
# qa_evidence
# ---------------------------------------------------------------------------


def test_qa_evidence_exact_endpoint_metrics_and_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(
        tmp_path,
        monkeypatch,
        k0_color=(255, 255, 255),
        k_end_color=(0, 0, 0),
    )
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "white")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    output = tmp_path / "qa-output"

    qa_evidence.cmd_qa(package, raw, finalized, output)

    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["schema_version"] == "g1-mk1-qa-evidence-v1"
    first = metrics["endpoints"]["k0_vs_frame0"]
    assert first["frame_index"] == 0
    assert first["canvas"] == [1280, 720]
    assert first["mae"] == 0.0
    assert first["mse"] == 0.0
    assert first["psnr"] == "infinite"
    last = metrics["endpoints"]["k_end_vs_frame120"]
    assert last["frame_index"] == 120
    assert last["canvas"] == [1280, 720]
    assert last["mae"] == 255.0
    assert last["mse"] == 65025.0
    assert last["psnr"] == 0.0
    assert metrics["capability_verdict"] is None
    assert metrics["contact_sheet_order"] == {
        "raw_contact_sheet.png": [0, 10, 20, 30, 40, 50, 60, 70, 80],
        "endpoint_comparison.png": ["k0", "k_end", "raw_start", "raw_end"],
    }

    artifacts = json.loads(
        (output / "artifacts.json").read_text(encoding="utf-8")
    )
    assert artifacts["schema_version"] == "g1-mk1-qa-artifacts-v1"
    contact = artifacts["artifacts"]["raw_contact_sheet.png"]
    assert contact["sha256"] == _sha256_file(output / "raw_contact_sheet.png")
    assert contact["width"] == 1272
    assert contact["height"] == 696
    assert contact["frames"] == [0, 10, 20, 30, 40, 50, 60, 70, 80]
    endpoint = artifacts["artifacts"]["endpoint_comparison.png"]
    assert endpoint["sha256"] == _sha256_file(
        output / "endpoint_comparison.png"
    )
    assert endpoint["width"] == 2560
    assert endpoint["height"] == 1440
    assert endpoint["order"] == ["k0", "k_end", "raw_start", "raw_end"]
    assert _png_size(output / "raw_contact_sheet.png") == (1272, 696)
    assert _png_size(output / "endpoint_comparison.png") == (2560, 1440)
    assert _staging_leftovers(output.parent) == []


def test_qa_evidence_rejects_raw_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    drifted = tmp_path / "raw-drifted.mp4"
    shutil.copyfile(raw, drifted)
    with drifted.open("ab") as handle:
        handle.write(b"x")
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, drifted, finalized, output)
    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()


def test_qa_evidence_rejects_finalized_run_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    with (finalized / "generated_clip.mp4").open("ab") as handle:
        handle.write(b"x")
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()


def test_qa_evidence_accepts_valid_generated_clip_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    output = tmp_path / "qa-output"

    qa_evidence.cmd_qa(package, raw, finalized, output)

    clips = json.loads(
        (finalized / "clips.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (finalized / "timeline.json").read_text(encoding="utf-8")
    )
    generation_manifest = json.loads(
        (finalized / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert clips["schema_version"] == "1.9"
    assert len(clips["clips"]) == 1
    clip = clips["clips"][0]
    item = timeline["items"][0]
    assert clip["id"] == item["source_asset_id"]
    assert clip["path"] == item["source_path"] == "generated_clip.mp4"
    generated_bytes = (finalized / "generated_clip.mp4").read_bytes()
    assert item["source_sha256"] == _sha256_bytes(generated_bytes)
    assert item["source_size_bytes"] == len(generated_bytes)
    assert generation_manifest["clips"] == {
        "path": "clips.json",
        "sha256": _sha256_file(finalized / "clips.json"),
    }
    assert generation_manifest["timeline"] == {
        "path": "timeline.json",
        "sha256": _sha256_file(finalized / "timeline.json"),
    }
    assert (output / "metrics.json").exists()
    assert _staging_leftovers(output.parent) == []


def test_qa_evidence_rejects_tampered_clips_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    clips_path = finalized / "clips.json"
    data = json.loads(clips_path.read_text(encoding="utf-8"))
    data["clips"][0]["path"] = "other.mp4"
    clips_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    manifest_path = finalized / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"]["sha256"] = _sha256_file(clips_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "evidence_incomplete"
    assert "clips asset path" in str(exc.value)
    assert not output.exists()


def test_qa_evidence_rejects_tampered_clips_manifest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    manifest_path = finalized / "generation_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["clips"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()


def test_qa_evidence_rejects_tampered_timeline_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    timeline_path = finalized / "timeline.json"
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    data["items"][0]["source_asset_id"] = "g1mk1-tampered-id"
    timeline_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    manifest_path = finalized / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["timeline"]["sha256"] = _sha256_file(timeline_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "evidence_incomplete"
    assert "source_asset_id" in str(exc.value)
    assert not output.exists()


def test_qa_evidence_rejects_tampered_timeline_source_media_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic handoff QA catches timeline source SHA/size drift.

    The generation-manifest timeline SHA is refreshed after each tamper so the
    rejection must come from the semantic handoff check itself, not the
    manifest binding.
    """

    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    for index, (field, delta) in enumerate(
        (("source_sha256", "tampered-sha"), ("source_size_bytes", -1))
    ):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        finalized = _build_finalized_run(
            case_dir, package, raw, generated
        )
        timeline_path = finalized / "timeline.json"
        data = json.loads(timeline_path.read_text(encoding="utf-8"))
        original = data["items"][0][field]
        data["items"][0][field] = (
            delta if field == "source_sha256" else original + delta
        )
        timeline_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        manifest_path = finalized / "generation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["timeline"]["sha256"] = _sha256_file(timeline_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        output = tmp_path / f"qa-output-{index}"
        with pytest.raises(harness.HarnessError) as exc:
            qa_evidence.cmd_qa(package, raw, finalized, output)
        assert exc.value.layer == "evidence_incomplete"
        assert field in str(exc.value)
        assert not output.exists()


def test_qa_evidence_rejects_existing_output_and_publish_failure_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    existing = tmp_path / "qa-existing"
    existing.mkdir()
    with pytest.raises(harness.HarnessError, match="already exists"):
        qa_evidence.cmd_qa(package, raw, finalized, existing)

    output = tmp_path / "qa-output"

    def fail_publish(staging, target):
        raise harness.HarnessError(
            "evidence_incomplete", "injected qa publish failure"
        )

    monkeypatch.setattr(harness, "_publish_dir", fail_publish)
    with pytest.raises(harness.HarnessError, match="injected qa publish failure"):
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_qa_evidence_cli_via_frozen_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen QA CLI revalidates the run-scoped gate in a fresh process.

    The synthetic package is genuinely valid under the run-scoped gate, so a
    fresh process accepts it from its own bytes and completes QA with no
    in-process constants.
    """

    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "manual_keyframe_mvp"
        / "qa_evidence.py"
    )
    output = tmp_path / "qa-output-cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package",
            str(package),
            "--raw",
            str(raw),
            "--finalized-run",
            str(finalized),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["schema_version"] == "g1-mk1-qa-evidence-v1"
    assert metrics["capability_verdict"] is None
    assert (output / "artifacts.json").exists()
    assert _staging_leftovers(output.parent) == []


# ---------------------------------------------------------------------------
# R2: FFmpeg reference canvases
# ---------------------------------------------------------------------------


def test_qa_reference_canvas_matches_independent_ffmpeg(
    tmp_path: Path,
) -> None:
    toolkit = harness.FFmpegToolkit()
    chain = (
        "scale=1280:720:force_original_aspect_ratio=decrease:"
        "force_divisible_by=2,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1,"
        "format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:"
        "colorspace=bt709:field_mode=prog"
    )
    cases = [
        ("nonuniform-rgb", _gradient_png_bytes(1672, 941)),
        ("rgba", _rgba_png_bytes(1280, 720)),
    ]
    for name, png_bytes in cases:
        png = tmp_path / f"{name}.png"
        png.write_bytes(png_bytes)
        tool_canvas = qa_evidence._png_reference_canvas(toolkit, png)
        assert len(tool_canvas) == 1280 * 720 * 3
        completed = subprocess.run(
            [
                _ffmpeg(),
                "-v",
                "error",
                "-i",
                str(png),
                "-vf",
                chain,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode(
            errors="replace"
        )
        assert tool_canvas == completed.stdout


# ---------------------------------------------------------------------------
# R3: QA consumes only captured bytes
# ---------------------------------------------------------------------------


def test_qa_uses_captured_generated_bytes_after_live_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "white")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    original_bytes = (finalized / "generated_clip.mp4").read_bytes()
    original_sha = _sha256_bytes(original_bytes)
    real_capture = harness._capture_bytes
    rewritten = {"count": 0}

    def capture_then_rewrite(path, layer="input_contract"):
        p = Path(path)
        data = real_capture(p, layer)
        if p.name == "generated_clip.mp4" and rewritten["count"] == 0:
            rewritten["count"] += 1
            p.write_bytes(b"REPLACED-LIVE-" * 100)
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_rewrite)
    output = tmp_path / "qa-output"
    qa_evidence.cmd_qa(package, raw, finalized, output)
    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["generated_clip_sha256"] == original_sha
    assert metrics["generated_clip_sha256"] != _sha256_bytes(
        b"REPLACED-LIVE-" * 100
    )
    assert metrics["endpoints"]["k0_vs_frame0"]["mse"] == 0.0
    assert _staging_leftovers(output.parent) == []


def test_qa_uses_captured_guide_bytes_after_live_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(
        tmp_path,
        monkeypatch,
        k0_color=(255, 255, 255),
        k_end_color=(0, 0, 0),
    )
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "black")
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    blue = _solid_png_bytes(1280, 720, (0, 0, 255))
    real_capture = harness._capture_bytes
    rewritten = {"count": 0}

    def capture_then_rewrite(path, layer="input_contract"):
        p = Path(path)
        data = real_capture(p, layer)
        if p.name == "k0.png" and rewritten["count"] == 0:
            rewritten["count"] += 1
            p.write_bytes(blue)
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_rewrite)
    output = tmp_path / "qa-output"
    qa_evidence.cmd_qa(package, raw, finalized, output)
    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )
    # captured white K0 vs frame 0 white; reopening the live blue guide would
    # produce a finite error, so mse == 0 proves only captured bytes were used.
    assert metrics["endpoints"]["k0_vs_frame0"]["mse"] == 0.0
    assert metrics["endpoints"]["k0_vs_frame0"]["psnr"] == "infinite"
    assert _staging_leftovers(output.parent) == []


# ---------------------------------------------------------------------------
# R7: full finalized generated-media contract
# ---------------------------------------------------------------------------


def test_qa_rejects_generated_clip_omitted_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    generated = tmp_path / "generated.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(generated, "white", "white", conformant=False)
    finalized = _build_finalized_run(tmp_path, package, raw, generated)
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "media_normalization"
    assert not output.exists()


def test_qa_rejects_generated_clip_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _make_package(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    good = tmp_path / "generated-good.mp4"
    _make_raw_mp4(raw)
    _make_generated_mp4(good, "white", "white")
    corrupt = tmp_path / "generated-corrupt.mp4"
    corrupt.write_bytes(good.read_bytes()[:2048])
    finalized = _build_finalized_run(tmp_path, package, raw, corrupt)
    output = tmp_path / "qa-output"
    with pytest.raises(harness.HarnessError) as exc:
        qa_evidence.cmd_qa(package, raw, finalized, output)
    assert exc.value.layer == "media_normalization"
    assert not output.exists()


# ---------------------------------------------------------------------------
# regression of the already-PASS G1-MK1 harness
# ---------------------------------------------------------------------------


def test_g1_mk1_harness_regression_unmodified() -> None:
    """The already-PASS G1-MK1 harness schemas/bindings remain intact."""

    assert harness.RECEIPT_SCHEMA == "g1-mk1-sampling-receipt-v1"
    assert set(harness.PACKAGE_MEMBERS) == {
        "request.json",
        "inputs/k0.png",
        "inputs/k_end.png",
        "inputs/k0.provenance.json",
        "inputs/k_end.provenance.json",
        "inspection.json",
        "approval.json",
        "sampling_contract.json",
        "anisora_input.txt",
    }
    receipt = dict(harness.FROZEN_SAMPLING)
    receipt.update(
        {
            "schema_version": harness.RECEIPT_SCHEMA,
            "request_id": "g1mk1-synthetic-001",
            "request_sha256": "a" * 64,
            "package_manifest_sha256": "b" * 64,
            "sampling_contract_sha256": "c" * 64,
            "start_sha256": "d" * 64,
            "end_sha256": "e" * 64,
            "raw_sha256": "f" * 64,
            "status": "success",
        }
    )
    harness._validate_receipt(
        receipt,
        "g1mk1-synthetic-001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )
