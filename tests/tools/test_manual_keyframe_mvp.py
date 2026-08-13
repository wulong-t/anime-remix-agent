"""G1-MK1-L harness unit tests: request/path/hash/approval/package gates."""

from __future__ import annotations

import hashlib
import json
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from experiments.manual_keyframe_mvp import manual_keyframe_mvp as harness
from experiments.manual_keyframe_mvp.manual_keyframe_mvp import (
    APPROVAL_SCHEMA,
    FROZEN_SAMPLING,
    INSPECTION_SCHEMA,
    PACKAGE_MEMBERS,
    RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    HarnessError,
    _first_gate_state,
    cmd_inspect,
    cmd_package,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_bytes(
    width: int,
    height: int,
    *,
    color_type: int = 2,
    seed: int = 0,
    animated: bool = False,
) -> bytes:
    channels = 3 if color_type in (2, 3) else 4

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(
        ">IIBBBBB", width, height, 8, color_type, 0, 0, 0
    )
    raw = bytearray()
    rng = random.Random(seed)
    for _ in range(height):
        raw.append(0)
        for _ in range(width):
            raw.extend(
                bytes(
                    (
                        rng.randrange(256),
                        rng.randrange(256),
                        rng.randrange(256),
                    )
                )
            )
            if channels == 4:
                raw.append(255)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + (chunk(b"acTL", struct.pack(">II", 1, 0)) if animated else b"")
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    return payload


def _make_inputs(
    root: Path,
    *,
    k0_bytes: bytes | None = None,
    k_end_bytes: bytes | None = None,
    k0_size: tuple[int, int] = (1280, 720),
    k_end_size: tuple[int, int] = (1280, 720),
) -> dict[str, object]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    k0 = inputs / "k0.png"
    k_end = inputs / "k_end.png"
    k0.write_bytes(
        k0_bytes if k0_bytes is not None else _png_bytes(*k0_size, seed=11)
    )
    k_end.write_bytes(
        k_end_bytes
        if k_end_bytes is not None
        else _png_bytes(*k_end_size, seed=22)
    )
    return {
        "k0": k0,
        "k_end": k_end,
        "k0_sha": _sha256(k0),
        "k_end_sha": _sha256(k_end),
    }


def _base_request(info: dict[str, object], **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
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
                "sha256": _sha256(image),
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


def _write_request(root: Path, data: dict[str, object]) -> Path:
    path = root / "request.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_request_root(
    tmp_path: Path,
    *,
    overrides: dict[str, object] | None = None,
    k0_bytes: bytes | None = None,
    k_end_bytes: bytes | None = None,
    k0_size: tuple[int, int] = (1280, 720),
    k_end_size: tuple[int, int] = (1280, 720),
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "request-root"
    info = _make_inputs(
        root,
        k0_bytes=k0_bytes,
        k_end_bytes=k_end_bytes,
        k0_size=k0_size,
        k_end_size=k_end_size,
    )
    _write_provenance(root, "k0.provenance.json", info["k0"])  # type: ignore[arg-type]
    _write_provenance(root, "k_end.provenance.json", info["k_end"])  # type: ignore[arg-type]
    request = _base_request(info, **(overrides or {}))
    _write_request(root, request)
    return root, request


def _run_inspect(
    request_root: Path, tmp_path: Path, name: str = "inspection"
) -> Path:
    output = tmp_path / name
    cmd_inspect(request_root / "request.json", output)
    return output


def _approve(pending: Path) -> Path:
    data = json.loads(pending.read_text(encoding="utf-8"))
    rights = data["rights"]
    for key in rights:
        rights[key] = True
    visual = data["visual_review"]
    for key in (
        "identity",
        "endpoint_pose",
        "body_camera_background",
        "style",
        "artifact",
    ):
        visual[key] = "pass"
    visual["accept_borderline"] = False
    visual["overall"] = "approved"
    data["approved_at"] = "2026-08-09T12:00:00+08:00"
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return pending


def _staging_leftovers(parent: Path) -> list[Path]:
    return [
        p
        for p in parent.iterdir()
        if p.name.startswith(".staging-") or ".staging-" in p.name
    ]


def test_inspect_valid_emits_pending_approval(
    tmp_path: Path,
) -> None:
    request_root, request = _make_request_root(tmp_path)
    output = _run_inspect(request_root, tmp_path)
    inspection = json.loads(
        (output / "inspection.json").read_text(encoding="utf-8")
    )
    assert inspection["schema_version"] == INSPECTION_SCHEMA
    assert inspection["request_id"] == request["request_id"]
    assert inspection["status"] == "pending"
    assert inspection["first_formal_gate"]["active"] is True
    assert inspection["first_formal_gate"]["problems"] == []
    assert inspection["images"]["start_keyframe"]["width"] == 1280
    assert inspection["images"]["start_keyframe"]["height"] == 720
    assert inspection["images"]["start_keyframe"]["sha256"] == request[
        "start_sha256"
    ]
    assert inspection["images"]["end_keyframe"]["sha256"] == request[
        "end_sha256"
    ]
    approval = json.loads((output / "approval.json").read_text(encoding="utf-8"))
    assert approval["schema_version"] == APPROVAL_SCHEMA
    assert approval["request_id"] == request["request_id"]
    assert approval["request_sha256"] == hashlib.sha256(
        (request_root / "request.json").read_bytes()
    ).hexdigest()
    assert approval["rights"]["start_owned_or_authorized"] is False
    assert approval["visual_review"]["overall"] == "pending"
    assert approval["approved_at"] is None


@pytest.mark.parametrize(
    "override, message",
    [
        ({"schema_version": "g1-mk1-request-v0"}, "schema_version"),
        ({"request_id": "bad id!"}, "request_id"),
        ({"request_id": ""}, "request_id"),
        ({"start_keyframe": "C:/abs/k0.png"}, "relative"),
        ({"start_keyframe": "https://example.com/k0.png"}, "relative"),
        ({"start_keyframe": "..\\escape.png"}, "traverse"),
        ({"start_keyframe": "CON"}, "device"),
        ({"start_keyframe": "\\\\server\\share\\k0.png"}, "relative"),
        ({"camera": "handheld"}, "camera"),
        ({"duration_seconds": 4}, "duration_seconds"),
        ({"aspect_ratio": "4:3"}, "aspect_ratio"),
        ({"emotion": "angry2"}, "emotion"),
        ({"emotion": []}, "emotion"),
        ({"shot_scale": "full"}, "shot_scale"),
        ({"shot_scale": {"nested": 1}}, "shot_scale"),
        ({"start_sha256": "0" * 64}, "start_sha256"),
        ({"subject_description": "line\nbreak"}, "control"),
        ({"scene_description": "tab\there"}, "control"),
        ({"action": "move@@fast"}, "separators"),
        ({"start_state": "calm&&still"}, "separators"),
        ({"extra_field": 1}, "unknown"),
        ({"end_sha256": "not-a-hash"}, "end_sha256"),
    ],
)
def test_inspect_rejects_invalid_request_fields(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    request_root, _ = _make_request_root(tmp_path, overrides=override)
    output = tmp_path / "inspection"
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", output)
    assert exc.value.layer == "input_contract"
    assert message in str(exc.value)
    assert not output.exists()


def test_inspect_rejects_same_keyframe_hash(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path,
        overrides={"end_sha256": hashlib.sha256(b"x").hexdigest()},
    )
    # Make the declared end hash equal the declared start hash.
    data = json.loads(
        (request_root / "request.json").read_text(encoding="utf-8")
    )
    data["end_sha256"] = data["start_sha256"]
    (request_root / "request.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    with pytest.raises(HarnessError, match="must differ"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_declared_sha_mismatch(tmp_path: Path) -> None:
    """A declared SHA256 that does not match the inspected bytes must fail."""

    request_root, _ = _make_request_root(
        tmp_path,
        overrides={"start_sha256": "f" * 64},
    )
    output = tmp_path / "inspection"
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", output)
    assert exc.value.layer == "input_contract"
    assert "start_sha256 does not match the captured start keyframe bytes" in str(
        exc.value
    )
    assert not output.exists()


def test_inspect_rejects_non_png(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k0_bytes=b"definitely not a png"
    )
    with pytest.raises(HarnessError, match="PNG"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_oversized_png(tmp_path: Path) -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (26 * 1024 * 1024)
    request_root, _ = _make_request_root(tmp_path, k0_bytes=oversized)
    with pytest.raises(HarnessError, match="exceeds"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_dimension_mismatch(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k_end_size=(1024, 576)
    )
    with pytest.raises(HarnessError, match="identical"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_bad_aspect_ratio(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k0_size=(512, 512), k_end_size=(512, 512)
    )
    with pytest.raises(HarnessError, match="16:9"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_small_canvas(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k0_size=(100, 100), k_end_size=(100, 100)
    )
    with pytest.raises(HarnessError, match="outside"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_animated_png(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k_end_bytes=_png_bytes(1280, 720, seed=2, animated=True)
    )
    with pytest.raises(HarnessError, match="acTL"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_trailing_data(tmp_path: Path) -> None:
    valid = _png_bytes(1280, 720, seed=1)
    request_root, _ = _make_request_root(
        tmp_path, k0_bytes=valid + b"trailing-garbage"
    )
    with pytest.raises(HarnessError, match="trailing"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_indexed_color_type(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(
        tmp_path, k_end_bytes=_png_bytes(1280, 720, color_type=3, seed=2)
    )
    with pytest.raises(HarnessError, match="color_type"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_symlink_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)

    def fake_is_symlink(self: Path) -> bool:
        return self.name == "k_end.png"

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(HarnessError, match="symlink"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")


def test_inspect_rejects_provenance_sha_mismatch(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k0.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["sha256"] = "f" * 64
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"


def test_inspect_rejects_provenance_missing_keys(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k_end.provenance.json"
    provenance.write_text(
        json.dumps({"asset": "inputs/k_end.png"}, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "missing key" in str(exc.value)


def test_first_formal_gate_active_for_arbitrary_valid_pair(
    tmp_path: Path,
) -> None:
    """The run-scoped gate activates for any valid pair, not a frozen K0."""

    # A deliberately non-frozen canvas: 1024x576 is 16:9 and in 512..4096.
    request_root, _ = _make_request_root(
        tmp_path, k0_size=(1024, 576), k_end_size=(1024, 576)
    )
    inspection = _run_inspect(request_root, tmp_path)
    gate = json.loads(
        (inspection / "inspection.json").read_text(encoding="utf-8")
    )["first_formal_gate"]
    assert gate["active"] is True
    assert gate["problems"] == []


def test_first_formal_gate_state_is_run_scoped() -> None:
    """Gate state derives only from the run's declared hashes/image evidence."""

    active, problems = _first_gate_state(
        {"start_sha256": "a" * 64, "end_sha256": "b" * 64},
        {"width": 1024, "height": 576, "color_type": "RGB"},
    )
    assert active is True
    assert problems == []

    inactive, problems = _first_gate_state(
        {"start_sha256": "a" * 64, "end_sha256": "a" * 64},
        {"width": 1024, "height": 576, "color_type": "RGB"},
    )
    assert inactive is False
    assert any("must differ" in p for p in problems)

    inactive, problems = _first_gate_state(
        {"start_sha256": "a" * 64, "end_sha256": "b" * 64},
        {"width": None, "height": None, "color_type": None},
    )
    assert inactive is False
    assert any("image contract" in p for p in problems)


def _approved_chain(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    request_root, _ = _make_request_root(tmp_path)
    inspection = _run_inspect(request_root, tmp_path)
    approval = _approve(inspection / "approval.json")
    return request_root, inspection, approval


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["visual_review"].update(overall="rejected"), "overall"),
        (lambda d: d["visual_review"].update(identity="fail"), "=fail"),
        (
            lambda d: d["visual_review"].update(
                endpoint_pose="borderline", accept_borderline=False
            ),
            "accept_borderline",
        ),
        (
            lambda d: d["rights"].update(start_owned_or_authorized=False),
            "start_owned_or_authorized",
        ),
        (lambda d: d.update(approved_at="2026-08-09T12:00:00"), "RFC 3339"),
        (lambda d: d.update(request_sha256="f" * 64), "request_sha256"),
    ],
)
def test_package_rejects_invalid_approval(
    tmp_path: Path, mutate, message: str
) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(approval.read_text(encoding="utf-8"))
    mutate(data)
    approval.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert message in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_publishes_atomically(tmp_path: Path) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    output = tmp_path / "package"
    result = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        output,
    )
    assert result == output
    for name in (
        "package_manifest.json",
        "sampling_contract.json",
        "anisora_input.txt",
        "request.json",
        "inspection.json",
        "approval.json",
        "inputs/k0.png",
        "inputs/k_end.png",
        "inputs/k0.provenance.json",
        "inputs/k_end.provenance.json",
    ):
        assert (output / name).is_file(), name
    manifest = json.loads(
        (output / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "g1-mk1-package-v1"
    assert manifest["request_id"] == "g1mk1-synthetic-001"
    assert set(manifest["files"]) == {
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
    for relative, record in manifest["files"].items():
        assert not Path(relative).is_absolute()
        assert ".." not in Path(relative).parts
        assert _sha256(output / relative) == record["sha256"]
        assert (output / relative).stat().st_size == record["size_bytes"]
    approval_data = json.loads(
        (output / "approval.json").read_text(encoding="utf-8")
    )
    assert manifest["start_sha256"] == approval_data["start_sha256"]
    assert manifest["end_sha256"] == approval_data["end_sha256"]
    assert _sha256(output / "inputs/k0.png") == approval_data["start_sha256"]
    assert _sha256(output / "inputs/k_end.png") == approval_data["end_sha256"]
    contract = json.loads(
        (output / "sampling_contract.json").read_text(encoding="utf-8")
    )
    assert contract["frozen_parameters"] == FROZEN_SAMPLING
    input_line = (output / "anisora_input.txt").read_text(
        encoding="utf-8"
    ).strip()
    assert input_line == contract["input_line"]
    assert "@@inputs/k0.png,inputs/k_end.png&&0,1" in input_line
    assert _staging_leftovers(tmp_path) == []


def test_package_records_explicit_sample_steps(tmp_path: Path) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    output = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / "package",
        10,
    )
    contract = json.loads(
        (output / "sampling_contract.json").read_text(encoding="utf-8")
    )
    assert contract["frozen_parameters"]["sample_steps"] == 10
    manifest = json.loads(
        (output / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sampling_contract_sha256"] == _sha256(
        output / "sampling_contract.json"
    )
    assert _staging_leftovers(tmp_path) == []


def test_package_succeeds_for_arbitrary_valid_pair(tmp_path: Path) -> None:
    """A non-frozen valid pair activates the gate through the full package."""

    request_root, _ = _make_request_root(
        tmp_path, k0_size=(1024, 576), k_end_size=(1024, 576)
    )
    inspection = _run_inspect(request_root, tmp_path)
    approval = _approve(inspection / "approval.json")
    output = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / "package",
    )
    contract = json.loads(
        (output / "sampling_contract.json").read_text(encoding="utf-8")
    )
    assert contract["first_formal_gate"]["active"] is True
    assert contract["first_formal_gate"]["problems"] == []
    assert _staging_leftovers(tmp_path) == []


def test_package_rejects_existing_output(tmp_path: Path) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    output = tmp_path / "package"
    output.mkdir()
    with pytest.raises(HarnessError, match="already exists"):
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert _staging_leftovers(tmp_path) == []


@pytest.mark.parametrize("value", [0, -1, 101, True, "10", 10.5, None])
def test_package_rejects_invalid_sample_steps(
    tmp_path: Path, value: object
) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
            value,  # type: ignore[arg-type]
        )
    assert exc.value.layer == "input_contract"
    assert "sample_steps" in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_rejects_hash_drift_after_approval(tmp_path: Path) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    with (request_root / "inputs" / "k0.png").open("ab") as handle:
        handle.write(b"drift")
    output = tmp_path / "package"
    with pytest.raises(HarnessError, match="start_sha256"):
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_toctou_during_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    real_write = harness._write_staged_bytes

    def corrupt(path, data):
        real_write(path, data)
        if Path(path).name == "k0.png":
            Path(path).write_bytes(Path(path).read_bytes() + b"x")

    monkeypatch.setattr(harness, "_write_staged_bytes", corrupt)
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_outputs_contain_no_absolute_paths_or_secrets(
    tmp_path: Path,
) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    output = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / "package",
    )
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix in {".png"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "DEEPSEEK" not in text
        assert "API_KEY" not in text
        assert "://" not in text
        assert "C:" not in text
        assert "\\" not in text
        assert not text.lstrip().startswith("/")


def test_cli_inspect_via_frozen_command(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    script = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "manual_keyframe_mvp"
        / "manual_keyframe_mvp.py"
    )
    output = tmp_path / "inspection-cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "inspect",
            "--request",
            str(request_root / "request.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "inspection.json").is_file()
    assert (output / "approval.json").is_file()


def test_inspect_rejects_provenance_missing_named_references(
    tmp_path: Path,
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k0.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    del data["named_references"]
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "named_references" in str(exc.value)


def test_inspect_rejects_provenance_unknown_field(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k_end.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["extra_field"] = 1
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "unknown keys" in str(exc.value)


def test_inspect_rejects_provenance_wrong_named_reference_type(
    tmp_path: Path,
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k0.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["named_references"]["artists"] = "not-a-list"
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "named_references.artists" in str(exc.value)


def test_inspect_rejects_provenance_asset_mismatch(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k0.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["asset"] = "inputs/other.png"
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "keyframe path" in str(exc.value)


def test_inspect_rejects_provenance_absolute_asset(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    provenance = request_root / "inputs" / "k0.provenance.json"
    data = json.loads(provenance.read_text(encoding="utf-8"))
    data["asset"] = "C:\\abs\\k0.png"
    provenance.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(HarnessError) as exc:
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert exc.value.layer == "rights_blocked"
    assert "canonical" in str(exc.value)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(extra_field=1), "approval unknown fields"),
        (
            lambda d: d["rights"].update(extra=1),
            "rights unknown fields",
        ),
        (
            lambda d: d["visual_review"].update(extra=1),
            "visual_review unknown fields",
        ),
    ],
)
def test_package_rejects_approval_unknown_fields(
    tmp_path: Path, mutate, message: str
) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(approval.read_text(encoding="utf-8"))
    mutate(data)
    approval.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert message in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_rejects_inspection_unknown_field(tmp_path: Path) -> None:
    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(
        (inspection / "inspection.json").read_text(encoding="utf-8")
    )
    data["extra_field"] = 1
    (inspection / "inspection.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert "inspection unknown fields" in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda d: d["rights"].update(public_demo_allowed=1),
            "public_demo_allowed must be a boolean",
        ),
        (
            lambda d: d["visual_review"].update(accept_borderline=0),
            "accept_borderline must be a boolean",
        ),
    ],
)
def test_package_rejects_boolean_impostors_in_approval(
    tmp_path: Path, mutate, message: str
) -> None:
    """B6: JSON 0/1 must not be accepted as booleans via Python equality."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(approval.read_text(encoding="utf-8"))
    mutate(data)
    approval.write_text(json.dumps(data, indent=2), encoding="utf-8")
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert message in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def _published_package(
    tmp_path: Path, name: str = "package"
) -> Path:
    request_root, inspection, approval = _approved_chain(tmp_path)
    return cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / name,
    )


def _tamper_manifest(
    package: Path, mutate
) -> dict[str, object]:
    manifest_path = package / "package_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(data)
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def test_package_manifest_rejects_extra_member(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    (package / "extra.txt").write_text("x", encoding="utf-8")
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"].update(
            {
                "extra.txt": {
                    "sha256": _sha256(package / "extra.txt"),
                    "size_bytes": 1,
                }
            }
        ),
    )
    with pytest.raises(HarnessError, match="extra="):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_missing_member(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package, lambda d: d["files"].pop("anisora_input.txt")
    )
    with pytest.raises(HarnessError, match="missing="):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_absolute_path_key(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"].update(
            {
                "C:\\evil\\x": {
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                }
            }
        ),
    )
    with pytest.raises(HarnessError, match="canonical forward-slash"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_traversal_key(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("o", encoding="utf-8")
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"].update(
            {
                "../outside.txt": {
                    "sha256": _sha256(outside),
                    "size_bytes": outside.stat().st_size,
                }
            }
        ),
    )
    with pytest.raises(HarnessError, match="traverse"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_backslash_key(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"].update(
            {"inputs\\k0.png": d["files"]["inputs/k0.png"]}
        ),
    )
    with pytest.raises(HarnessError, match="canonical forward-slash"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _published_package(tmp_path)
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(self: Path) -> bool:
        if self.name == "k0.png":
            return True
        return real_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(HarnessError, match="symlink"):
        harness._validate_package_manifest(manifest, package)


def test_package_captures_request_once_and_uses_original_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: request is captured exactly once; a later disk rewrite is ignored."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    original_request = (request_root / "request.json").read_bytes()
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def capture_then_rewrite(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "request.json" and counts[p.name] == 1:
            replaced = json.loads(original_request)
            replaced["subject_description"] = "REPLACED AFTER CAPTURE"
            (request_root / "request.json").write_bytes(
                json.dumps(replaced, indent=2).encode("utf-8")
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_rewrite)
    output = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / "package",
    )
    assert (output / "request.json").read_bytes() == original_request
    assert counts.get("request.json") == 1
    manifest = json.loads(
        (output / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["request_sha256"] == hashlib.sha256(
        original_request
    ).hexdigest()
    assert _staging_leftovers(tmp_path) == []


def test_package_captures_k0_once_and_uses_original_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: K0 is captured exactly once; a later disk rewrite is ignored."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    original_k0 = (request_root / "inputs" / "k0.png").read_bytes()
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def capture_then_rewrite(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "k0.png" and counts[p.name] == 1:
            (request_root / "inputs" / "k0.png").write_bytes(
                original_k0 + b"REPLACED"
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_rewrite)
    output = cmd_package(
        request_root / "request.json",
        inspection / "inspection.json",
        approval,
        tmp_path / "package",
    )
    assert (output / "inputs" / "k0.png").read_bytes() == original_k0
    assert counts.get("k0.png") == 1
    manifest = json.loads(
        (output / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["files"]["inputs/k0.png"]["sha256"] == hashlib.sha256(
        original_k0
    ).hexdigest()
    assert _staging_leftovers(tmp_path) == []


def test_package_rejects_k0_replaced_before_first_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: request captured, K0 rewritten before its first capture -> fail."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    original_k0 = (request_root / "inputs" / "k0.png").read_bytes()
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def rewrite_k0_after_request(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "request.json" and counts[p.name] == 1:
            (request_root / "inputs" / "k0.png").write_bytes(
                original_k0 + b"DRIFT"
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", rewrite_k0_after_request)
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "input_contract"
    assert "start_sha256" in str(exc.value)
    assert counts.get("request.json") == 1
    assert counts.get("k0.png") == 1
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_inspect_evidence_matches_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1 (inspect): evidence derives from the one captured bytes snapshot."""

    request_root, _ = _make_request_root(tmp_path)
    original_k0 = (request_root / "inputs" / "k0.png").read_bytes()
    real_capture = harness._capture_bytes
    counts: dict[str, int] = {}

    def capture_then_rewrite(path, layer="input_contract"):
        p = Path(path)
        counts[p.name] = counts.get(p.name, 0) + 1
        data = real_capture(p, layer)
        if p.name == "k0.png" and counts[p.name] == 1:
            (request_root / "inputs" / "k0.png").write_bytes(
                original_k0 + b"REPLACED"
            )
        return data

    monkeypatch.setattr(harness, "_capture_bytes", capture_then_rewrite)
    output = cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    inspection = json.loads(
        (output / "inspection.json").read_text(encoding="utf-8")
    )
    assert (
        inspection["images"]["start_keyframe"]["sha256"]
        == hashlib.sha256(original_k0).hexdigest()
    )
    assert counts.get("request.json") == 1
    assert counts.get("k0.png") == 1
    assert (request_root / "inputs" / "k0.png").read_bytes() != original_k0
    assert _staging_leftovers(tmp_path) == []


def test_inspect_second_write_failure_leaves_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    real_dump = harness.dump_json_atomic

    def fail_on_approval(path, data, **kwargs):
        if Path(path).name == "approval.json":
            raise OSError("injected second write failure")
        real_dump(path, data, **kwargs)

    monkeypatch.setattr(harness, "dump_json_atomic", fail_on_approval)
    output = tmp_path / "inspection"
    with pytest.raises(OSError, match="injected"):
        cmd_inspect(request_root / "request.json", output)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_inspect_publish_failure_leaves_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)

    def fail_publish(staging, output):
        raise HarnessError("evidence_incomplete", "injected publish failure")

    monkeypatch.setattr(harness, "_publish_dir", fail_publish)
    output = tmp_path / "inspection"
    with pytest.raises(HarnessError, match="injected"):
        cmd_inspect(request_root / "request.json", output)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def _png_bytes_with_filters(
    width: int,
    height: int,
    filter_bytes: list[int],
    *,
    seed: int = 0,
) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = bytearray()
    rng = random.Random(seed)
    for row in range(height):
        raw.append(filter_bytes[row] if row < len(filter_bytes) else 0)
        for _ in range(width):
            raw.extend(
                bytes(
                    (
                        rng.randrange(256),
                        rng.randrange(256),
                        rng.randrange(256),
                    )
                )
            )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def test_inspect_rejects_invalid_scanline_filter_byte(tmp_path: Path) -> None:
    # 1024x576 is exactly 16:9 and inside the 512..4096 canvas range; the
    # first scanline carries filter byte 5 with a correct CRC/length, so only
    # the per-scanline filter gate can reject it.
    bad = _png_bytes_with_filters(1024, 576, [5], seed=3)
    request_root, _ = _make_request_root(
        tmp_path,
        k_end_bytes=bad,
        k0_size=(1024, 576),
        k_end_size=(1024, 576),
    )
    with pytest.raises(HarnessError, match="filter byte"):
        cmd_inspect(request_root / "request.json", tmp_path / "inspection")
    assert not (tmp_path / "inspection").exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda d: d.update(request_file="C:\\abs\\request.json"),
            "request_file",
        ),
        (
            lambda d: d["images"]["start_keyframe"].update(width=100),
            "images",
        ),
        (
            lambda d: d["images"]["start_keyframe"].update(color_type="RGBA"),
            "images",
        ),
        (
            lambda d: d["images"]["start_keyframe"].update(size_bytes=1),
            "images",
        ),
        (
            lambda d: d["provenance"]["start"].update(
                path="C:\\abs\\k0.provenance.json"
            ),
            "provenance",
        ),
        (
            lambda d: d["first_formal_gate"].update(active=False),
            "first_formal_gate",
        ),
        (
            lambda d: d.update(checked_at="2026-08-09T12:00:00"),
            "checked_at",
        ),
        (lambda d: d.update(checked_at="line\nbreak"), "checked_at"),
    ],
)
def test_package_rejects_forged_inspection_evidence(
    tmp_path: Path, mutate, message: str
) -> None:
    """B3: inspection known fields must match the captured bytes evidence."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(
        (inspection / "inspection.json").read_text(encoding="utf-8")
    )
    mutate(data)
    (inspection / "inspection.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert message in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_manifest_rejects_root_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B4: package root symlink/junction/reparse is rejected before resolve."""

    package = _published_package(tmp_path)
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    real = harness._is_link_or_reparse

    def fake_is_link(path: Path) -> bool:
        return Path(path) == package or real(path)

    monkeypatch.setattr(harness, "_is_link_or_reparse", fake_is_link)
    with pytest.raises(HarnessError, match="package root"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_unknown_top_level(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package, lambda d: d.update(extra_field=1)
    )
    with pytest.raises(HarnessError, match="unknown fields"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_missing_top_level(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package, lambda d: d.pop("created_at")
    )
    with pytest.raises(HarnessError, match="missing fields"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_invalid_created_at(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package, lambda d: d.update(created_at="2026-08-09T12:00:00")
    )
    with pytest.raises(HarnessError, match="RFC 3339"):
        harness._validate_package_manifest(manifest, package)

def test_parse_rfc3339_rejects_frozen_negatives() -> None:
    """R3: the five Chief-probe negatives are not RFC 3339 timestamps.

    Each value previously passed the loose fromisoformat check (see the
    G1-MK1-L-R2 Chief Gate probe), plus one regex-valid but
    datetime-invalid calendar date to prove the datetime layer is real.
    """

    frozen_negatives = [
        "2026-08-09 12:00:00+08:00",
        "2026-08-09T12:00+08:00",
        "2026-W32-7T12:00:00+08:00",
        "2026-08-09T12:00:00+0800",
        "2026-08-09T12:00:00+08:00:30",
    ]
    for value in frozen_negatives:
        assert harness._parse_rfc3339(value) is False, value
    assert harness._parse_rfc3339("2026-02-30T12:00:00+08:00") is False


def test_parse_rfc3339_accepts_frozen_positives() -> None:
    """R3: frozen grammar YYYY-MM-DDTHH:MM:SS[.fraction](Z|+-HH:MM)."""

    frozen_positives = [
        "2026-08-09T12:00:00+08:00",
        "2026-08-09T12:00:00Z",
        "2026-08-09T12:00:00.123456+08:00",
    ]
    for value in frozen_positives:
        assert harness._parse_rfc3339(value) is True, value


def test_package_rejects_frozen_rfc3339_checked_at(tmp_path: Path) -> None:
    """B3: inspection checked_at is gated by the strict RFC 3339 helper."""

    request_root, inspection, approval = _approved_chain(tmp_path)
    data = json.loads(
        (inspection / "inspection.json").read_text(encoding="utf-8")
    )
    data["checked_at"] = "2026-08-09T12:00:00+0800"
    (inspection / "inspection.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    output = tmp_path / "package"
    with pytest.raises(HarnessError) as exc:
        cmd_package(
            request_root / "request.json",
            inspection / "inspection.json",
            approval,
            output,
        )
    assert exc.value.layer == "approval_blocked"
    assert "checked_at" in str(exc.value)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_package_manifest_rejects_frozen_rfc3339_created_at(
    tmp_path: Path,
) -> None:
    """B5: package_manifest created_at is gated by the strict RFC 3339 helper."""

    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package, lambda d: d.update(created_at="2026-08-09 12:00:00+08:00")
    )
    with pytest.raises(HarnessError, match="RFC 3339"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_files_record_unknown_field(
    tmp_path: Path,
) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"]["request.json"].update(extra=1),
    )
    with pytest.raises(HarnessError, match="files.request.json unknown"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_rejects_bool_size(tmp_path: Path) -> None:
    package = _published_package(tmp_path)
    manifest = _tamper_manifest(
        package,
        lambda d: d["files"]["request.json"].update(size_bytes=True),
    )
    with pytest.raises(HarnessError, match="size_bytes"):
        harness._validate_package_manifest(manifest, package)


def test_package_manifest_bindings_reject_member_hash_mismatch(
    tmp_path: Path,
) -> None:
    """B2/B5: manifest header/records must cross-bind to captured members."""

    package = _published_package(tmp_path)
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    members = {
        relative: (package / relative).read_bytes()
        for relative in PACKAGE_MEMBERS
    }
    members["request.json"] = b'{"replaced": true}\n'
    with pytest.raises(HarnessError, match="request_sha256"):
        harness._validate_package_manifest_bindings(manifest, members)


def _sample_info() -> dict[str, object]:
    return {
        "data": {
            "subject_description": "An original 2D cel-animation woman",
            "scene_description": "A quiet observatory control room at dusk",
            "action": "turn head slightly to the right",
            "start_state": "near-frontal calm",
            "end_state": "three-quarter calm",
            "emotion": "calm",
            "shot_scale": "medium",
        },
        "request_id": "g1mk1-synthetic-001",
        "request_sha256": "a" * 64,
        "start_sha256": "b" * 64,
        "end_sha256": "c" * 64,
    }


def test_sampling_contract_rejects_unknown_fields() -> None:
    info = _sample_info()
    contract = harness._sampling_contract(info, False, [])
    contract["extra_field"] = 1
    with pytest.raises(HarnessError, match="unknown fields"):
        harness._validate_sampling_contract(contract, info, False, [])


def test_sampling_contract_rejects_missing_fields() -> None:
    info = _sample_info()
    contract = harness._sampling_contract(info, False, [])
    del contract["input_line"]
    with pytest.raises(HarnessError, match="missing fields"):
        harness._validate_sampling_contract(contract, info, False, [])


def test_sampling_contract_rejects_strict_type_or_value_mismatch() -> None:
    info = _sample_info()
    contract = harness._sampling_contract(info, False, [])
    contract["guide_positions"] = [1]
    with pytest.raises(HarnessError, match="guide_positions"):
        harness._validate_sampling_contract(contract, info, False, [])
    contract = harness._sampling_contract(info, False, [])
    contract["guide_positions"] = [0, "1"]
    with pytest.raises(HarnessError, match="guide_positions"):
        harness._validate_sampling_contract(contract, info, False, [])


@pytest.mark.parametrize("value", [True, 0, -1, 101, "10", 10.5, None])
def test_sampling_contract_rejects_invalid_sample_steps(value: object) -> None:
    info = _sample_info()
    contract = harness._sampling_contract(info, False, [])
    contract["frozen_parameters"]["sample_steps"] = value
    with pytest.raises(HarnessError) as exc:
        harness._validate_sampling_contract(contract, info, False, [])
    assert exc.value.layer == "evidence_incomplete"
    assert "sample_steps" in str(exc.value)


def _valid_receipt_dict() -> dict[str, object]:
    data: dict[str, object] = dict(FROZEN_SAMPLING)
    data.update(
        {
            "schema_version": RECEIPT_SCHEMA,
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
    return data


def test_receipt_rejects_numeric_boolean_confusion() -> None:
    args = (
        "g1mk1-synthetic-001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )
    data = _valid_receipt_dict()
    data["offload_model"] = 1
    with pytest.raises(HarnessError, match="offload_model"):
        harness._validate_receipt(data, *args)
    data = _valid_receipt_dict()
    data["aesthetic_score"] = 5
    with pytest.raises(HarnessError, match="aesthetic_score"):
        harness._validate_receipt(data, *args)
    data = _valid_receipt_dict()
    data["sample_guide_scale"] = "5"
    with pytest.raises(HarnessError, match="sample_guide_scale"):
        harness._validate_receipt(data, *args)
    data = _valid_receipt_dict()
    data["guide_positions"] = [1]
    with pytest.raises(HarnessError, match="guide_positions"):
        harness._validate_receipt(data, *args)


@pytest.mark.parametrize("value", [True, 0, -1, 101, "10", 10.5, None])
def test_receipt_rejects_invalid_sample_steps(value: object) -> None:
    args = (
        "g1mk1-synthetic-001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )
    data = _valid_receipt_dict()
    data["sample_steps"] = value
    with pytest.raises(HarnessError) as exc:
        harness._validate_receipt(data, *args)
    assert exc.value.layer == "evidence_incomplete"
    assert "sample_steps" in str(exc.value)


def test_receipt_uses_packaged_sample_steps() -> None:
    args = (
        "g1mk1-synthetic-001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )
    data = _valid_receipt_dict()
    data["sample_steps"] = 10
    harness._validate_receipt(data, *args, sample_steps=10)
    data = _valid_receipt_dict()
    data["sample_steps"] = 10
    with pytest.raises(HarnessError, match="sample_steps"):
        harness._validate_receipt(data, *args, sample_steps=40)
    data = _valid_receipt_dict()
    data["status"] = 1
    with pytest.raises(HarnessError, match="status"):
        harness._validate_receipt(data, *args)


def _png_from_idat(
    width: int,
    height: int,
    idat_payload: bytes,
    *,
    color_type: int = 2,
    bit_depth: int = 8,
) -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(
        ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat_payload)
        + chunk(b"IEND", b"")
    )


def _raw_1024x576(seed: int = 0) -> bytes:
    raw = bytearray()
    rng = random.Random(seed)
    for _ in range(576):
        raw.append(0)
        for _ in range(1024):
            raw.extend(
                bytes(
                    (
                        rng.randrange(256),
                        rng.randrange(256),
                        rng.randrange(256),
                    )
                )
            )
    return bytes(raw)


def test_png_small_canvas_oversized_payload_is_bounded() -> None:
    """B7: in-bounds IHDR but IDAT inflates beyond expected -> bounded."""

    expected = 512 * (1 + 512 * 3)
    png = _png_from_idat(512, 512, zlib.compress(bytes(expected + 1537)))
    info = harness._png_details_bytes(png)
    assert any("beyond the expected" in p for p in info["problems"])


def test_png_out_of_bounds_ihdr_never_decompresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B7: out-of-range IHDR must fail before any decompression is attempted."""

    def boom(*args, **kwargs):
        raise AssertionError("decompression must not be attempted")

    monkeypatch.setattr(harness.zlib, "decompressobj", boom)
    png = _png_from_idat(10000, 10000, b"garbage")
    info = harness._png_details_bytes(png)
    assert any("outside" in p for p in info["problems"])


def test_png_rejects_trailing_compressed_data() -> None:
    payload = zlib.compress(_raw_1024x576(seed=4)) + b"\x00\x00"
    png = _png_from_idat(1024, 576, payload)
    info = harness._png_details_bytes(png)
    assert any("trailing compressed data" in p for p in info["problems"])


def test_png_rejects_truncated_zlib_stream() -> None:
    payload = zlib.compress(_raw_1024x576(seed=4))[:-4]
    png = _png_from_idat(1024, 576, payload)
    info = harness._png_details_bytes(png)
    assert any("truncated" in p for p in info["problems"])
