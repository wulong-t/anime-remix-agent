"""Offline tests for the Phase 3 DashScope experiment runner.

The runner is a real-mode experiment script, so these tests only exercise
the no-network ``--dry-run`` path and the fail-before-network guard for real
mode.  The DashScope SDK call is monkeypatched to explode: any accidental
network path fails loudly instead of waiting on a timeout.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import dashscope
import pytest

_PNG = b"\x89PNG\r\n\x1a\n" + b"runner-fixture"


def _load_runner(monkeypatch: pytest.MonkeyPatch):
    runner_dir = (
        Path(__file__).resolve().parents[2] / "experiments" / "phase3"
    )
    monkeypatch.syspath_prepend(str(runner_dir))
    runner_path = runner_dir / "run_dashscope_synthesis.py"
    spec = importlib.util.spec_from_file_location(
        "run_dashscope_synthesis",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    identity = tmp_path / "identity.png"
    identity.write_bytes(_PNG)
    pose = tmp_path / "pose.png"
    pose.write_bytes(_PNG)
    return identity, pose


def _argv(
    *,
    identity: Path,
    pose: Path,
    run_dir: Path,
    dry_run: bool,
    visual_how: bool = False,
) -> list[str]:
    args = [
        "run_dashscope_synthesis.py",
        "--identity",
        str(identity),
        "--pose",
        str(pose),
        "--run-dir",
        str(run_dir),
    ]
    if dry_run:
        args.append("--dry-run")
    if visual_how:
        args.append("--visual-how")
    return args


def _explode_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*args, **kwargs):
        raise AssertionError("the real DashScope SDK must not be called")

    monkeypatch.setattr(dashscope.MultiModalConversation, "call", _explode)


def test_dry_run_completes_without_key_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner(monkeypatch)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    _explode_sdk(monkeypatch)
    identity, pose = _write_inputs(tmp_path)
    run_dir = tmp_path / "dry-run-out"
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            identity=identity,
            pose=pose,
            run_dir=run_dir,
            dry_run=True,
        ),
    )

    module.main()

    manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["outcome"] == "dry_run"
    assert manifest["dry_run"] is True
    assert "no DashScope request was made" in manifest["detail"]
    assert manifest["executor"]["status"] == "succeeded"
    assert manifest["executor"]["request_id"] == "dry-run"
    assert manifest["executor"]["usage"]["output_width"] == 1280
    assert "character_candidate" in manifest["ports"]
    assert (run_dir / "execution-ledger.jsonl").exists()

    summary = manifest["request_summary"]
    assert summary["model"] == "qwen-image-3.0"
    assert summary["roles"] == ["WHO"]
    assert [entry["role"] for entry in summary["inputs"]] == ["WHO"]
    for entry in summary["inputs"]:
        assert set(entry) == {"role", "bytes", "sha256", "base64_chars"}
        assert entry["bytes"] == len(_PNG)
        assert entry["sha256"] == hashlib.sha256(_PNG).hexdigest()
        assert entry["base64_chars"] == len(
            base64.b64encode(_PNG).decode("ascii")
        )
    ledger_lines = (run_dir / "execution-ledger.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    prompt_record = next(
        json.loads(line)
        for line in ledger_lines
        if json.loads(line)["record_type"] == "model_render_request_created"
    )
    prompt = prompt_record["payload"]["prompt"]
    assert prompt_record["payload"]["adapter_id"] == (
        "qwen-image-30-adapter-v6-reference-first"
    )
    assert "Image 1 is the only visual character reference" in prompt
    assert "Image 2" not in prompt
    assert "both eyes fully closed" in prompt
    assert "fingertips touching the right temple" in prompt
    assert "Target gaze and eye state: eyes fully closed" in prompt
    assert "Do not redesign, restyle" in prompt
    assert "hairstyle and hair color, clothing and accessories" in prompt
    assert "Background: classroom, afternoon" in prompt
    assert "Foreground: desk occludes lower body" in prompt
    assert summary["prompt_sha256"] == hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert summary["parameters"]["seed"] == 0
    assert summary["parameters"]["size"] == "1280*720"

    serialized = json.dumps(manifest, ensure_ascii=False)
    assert base64.b64encode(_PNG).decode("ascii") not in serialized
    assert prompt not in serialized
    assert "Keep the character identity" not in serialized
    assert "DASHSCOPE_API_KEY" not in serialized
    assert str(identity) not in serialized
    assert str(pose) not in serialized


def test_visual_how_dry_run_sends_both_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner(monkeypatch)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    _explode_sdk(monkeypatch)
    identity, pose = _write_inputs(tmp_path)
    run_dir = tmp_path / "visual-how-dry-run-out"
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            identity=identity,
            pose=pose,
            run_dir=run_dir,
            dry_run=True,
            visual_how=True,
        ),
    )

    module.main()

    manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["request_summary"]["roles"] == ["WHO", "HOW"]
    assert manifest["executor"]["usage"]["input_image_count"] == 2
    records = [
        json.loads(line)
        for line in (run_dir / "execution-ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    request = next(
        record
        for record in records
        if record["record_type"] == "model_render_request_created"
    )
    assert request["payload"]["adapter_id"] == (
        "qwen-image-30-adapter-v4-visual-how"
    )
    assert "Image 2 is a pose-only HOW control reference" in (
        request["payload"]["prompt"]
    )


def test_real_mode_without_key_records_environment_capability_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner(monkeypatch)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    _explode_sdk(monkeypatch)
    identity, pose = _write_inputs(tmp_path)
    run_dir = tmp_path / "real-out"
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            identity=identity,
            pose=pose,
            run_dir=run_dir,
            dry_run=False,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["outcome"] == "error"
    assert manifest["dry_run"] is False
    assert "DASHSCOPE_API_KEY" in manifest["detail"]
    assert manifest["request_summary"]["model"] == "qwen-image-3.0"
    serialized = json.dumps(manifest)
    assert base64.b64encode(_PNG).decode("ascii") not in serialized
    assert str(identity) not in serialized
    assert str(pose) not in serialized


def test_dry_run_still_validates_frozen_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner(monkeypatch)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    _explode_sdk(monkeypatch)
    identity, pose = _write_inputs(tmp_path)
    run_dir = tmp_path / "dry-run-invalid"
    args = _argv(
        identity=identity,
        pose=pose,
        run_dir=run_dir,
        dry_run=True,
    )
    args += ["--seed", "-1"]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    manifest = json.loads(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["outcome"] == "error"
    assert manifest["dry_run"] is True
    assert "seed" in manifest["detail"]
