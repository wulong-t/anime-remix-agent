"""G1-MK4-L tests: thin session_agent coordinator (synthetic only).

The session agent coordinates the already-PASS manual-keyframe tools, so the
happy path runs the real ``inspect -> package -> finalize -> QA`` chain
against synthetic media produced by FFmpeg inside pytest temporary
directories.  The agent itself never invokes FFmpeg/a shell and never reads
real media, ``.tmp``, runs, secrets, environment values or Remote files.
Tests cover the minimal session state, exact binding validation, and
recoverable transitions (injected failures keep the prior phase clean).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from experiments.manual_keyframe_mvp import (
    manual_keyframe_mvp as harness,
)
from experiments.manual_keyframe_mvp import (
    qa_evidence,
    remote_sample,
    session_agent,
)


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


def _make_inputs(
    root: Path,
    *,
    k0_color: tuple[int, int, int] = (255, 255, 255),
    k_end_color: tuple[int, int, int] = (0, 0, 0),
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


def _base_request(info: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": harness.REQUEST_SCHEMA,
        "request_id": "g1mk4-synthetic-session-001",
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


def _make_request_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "request-root"
    info = _make_inputs(root)
    _write_provenance(root, "k0.provenance.json", info["k0"])  # type: ignore[arg-type]
    _write_provenance(root, "k_end.provenance.json", info["k_end"])  # type: ignore[arg-type]
    request = _base_request(info)
    (root / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    return root, request


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
    data["approved_at"] = "2026-08-10T12:00:00+08:00"
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return pending


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


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


def _make_fake_remote_output(
    tmp_path: Path, package: Path, *, name: str = "remote-output"
) -> Path:
    out = tmp_path / name
    out.mkdir()
    raw_path = out / "raw_shot.mp4"
    _make_raw_mp4(raw_path)
    raw_sha = _sha256_file(raw_path)
    receipt = _receipt_for(package, raw_sha)
    (out / "sampling_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return out


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _staging_leftovers(parent: Path) -> list[Path]:
    return [
        path
        for path in parent.iterdir()
        if ".staging-" in path.name or ".probe-staging-" in path.name
    ]


def _to_awaiting_remote(
    tmp_path: Path, *, sample_steps: int = 40
) -> tuple[Path, Path]:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(
        request_root / "request.json", workspace, sample_steps=sample_steps
    )
    _approve(workspace / "inspection/approval.json")
    session_agent.cmd_advance(workspace)
    return workspace, request_root


def _to_complete(tmp_path: Path) -> Path:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")
    session_agent.cmd_advance(workspace, remote_output=remote)
    return workspace


def _mutate_session(workspace: Path, mutator) -> None:
    path = workspace / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutator(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_default40_minimal_session_and_pending_approval(
    tmp_path: Path,
) -> None:
    request_root, request = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"

    session_agent.cmd_init(request_root / "request.json", workspace)

    session = json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )
    assert set(session) == session_agent.SESSION_FIELDS
    assert session["schema_version"] == session_agent.SESSION_SCHEMA
    assert session["request"] == {
        "path": str((request_root / "request.json").resolve()),
        "id": request["request_id"],
        "sha256": _sha256_file(request_root / "request.json"),
    }
    assert session["sample_steps"] == 40
    assert session["phase"] == "awaiting_approval"
    assert session["package_manifest_sha256"] is None
    assert session["completion"] is None
    assert (workspace / "inspection/inspection.json").is_file()
    assert (workspace / "inspection/approval.json").is_file()
    approval = json.loads(
        (workspace / "inspection/approval.json").read_text(encoding="utf-8")
    )
    assert approval["schema_version"] == harness.APPROVAL_SCHEMA
    assert approval["visual_review"]["overall"] == "pending"

    payload = session_agent.cmd_status(workspace)
    assert payload["next_action"]["action"] == "approve"
    assert payload["next_action"]["target"] == "inspection/approval.json"
    text = (workspace / "session.json").read_text(encoding="utf-8")
    for forbidden in ("API_KEY", "SECRET", "TOKEN", "os.environ", "ssh", "scp"):
        assert forbidden not in text


def test_init_explicit10_and_rejects_invalid_steps_or_existing_workspace(
    tmp_path: Path,
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace10"

    session_agent.cmd_init(
        request_root / "request.json", workspace, sample_steps=10
    )
    session = json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )
    assert session["sample_steps"] == 10

    for bad in (0, 101):
        other = tmp_path / f"workspace-bad-{bad}"
        with pytest.raises(harness.HarnessError) as exc:
            session_agent.cmd_init(
                request_root / "request.json", other, sample_steps=bad
            )
        assert exc.value.layer == "input_contract"
        assert not other.exists()

    with pytest.raises(harness.HarnessError, match="already exists"):
        session_agent.cmd_init(request_root / "request.json", workspace)


def test_init_failure_leaves_no_requested_workspace(tmp_path: Path) -> None:
    request_root, request = _make_request_root(tmp_path)
    request["start_sha256"] = "0" * 64
    (request_root / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(harness.HarnessError):
        session_agent.cmd_init(request_root / "request.json", workspace)

    assert not workspace.exists()
    assert _staging_leftovers(tmp_path) == []


def test_init_session_write_failure_leaves_no_requested_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"

    def _boom(*args, **kwargs):
        raise OSError("injected session write failure")

    monkeypatch.setattr(session_agent, "_write_session", _boom)
    with pytest.raises(OSError, match="injected"):
        session_agent.cmd_init(request_root / "request.json", workspace)

    assert not workspace.exists()
    assert _staging_leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# minimal session drift rejection
# ---------------------------------------------------------------------------


DRIFT_MUTATIONS = [
    (
        "schema_version",
        lambda data: data.update(schema_version="g1-mk4-other-v1"),
    ),
    (
        "request path",
        lambda data: data["request"].update(
            path=str(Path("Z:/does/not/exist/request.json"))
        ),
    ),
    (
        "request sha",
        lambda data: data["request"].update(sha256="0" * 64),
    ),
    ("phase", lambda data: data.update(phase="complete")),
    ("extra key", lambda data: data.update(sneaky=True)),
    ("sample_steps", lambda data: data.update(sample_steps=101)),
    (
        "package hash before package",
        lambda data: data.update(package_manifest_sha256="0" * 64),
    ),
    (
        "completion before completion",
        lambda data: data.update(
            completion={
                "generation_manifest_sha256": "0" * 64,
                "qa_metrics_sha256": "0" * 64,
                "qa_artifacts_sha256": "0" * 64,
            }
        ),
    ),
]


@pytest.mark.parametrize(("name", "mutator"), DRIFT_MUTATIONS)
def test_session_schema_hash_path_drift_rejected(
    tmp_path: Path, name: str, mutator
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(request_root / "request.json", workspace)
    _mutate_session(workspace, mutator)

    with pytest.raises(harness.HarnessError):
        session_agent.cmd_status(workspace)
    with pytest.raises(harness.HarnessError):
        session_agent.cmd_advance(workspace)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_read_only_derives_single_next_action(tmp_path: Path) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(request_root / "request.json", workspace)

    before = _tree(workspace)
    payload = session_agent.cmd_status(workspace)
    assert _tree(workspace) == before

    assert payload["schema_version"] == session_agent.SESSION_SCHEMA
    assert payload["phase"] == "awaiting_approval"
    assert payload["package_manifest_sha256"] is None
    assert payload["completion"] is None
    assert payload["next_action"]["action"] == "approve"
    assert payload["next_action"]["target"] == "inspection/approval.json"

    # Editing the pending approval is the intended flow and must not break
    # read-only status.
    _approve(workspace / "inspection/approval.json")
    payload2 = session_agent.cmd_status(workspace)
    assert payload2["phase"] == "awaiting_approval"


def test_status_rejects_missing_or_file_workspace(tmp_path: Path) -> None:
    with pytest.raises(harness.HarnessError, match="exact existing directory"):
        session_agent.cmd_status(tmp_path / "missing")
    regular = tmp_path / "regular-file"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="exact existing directory"):
        session_agent.cmd_status(regular)


def test_status_validates_exact_request_and_package(tmp_path: Path) -> None:
    workspace, request_root = _to_awaiting_remote(tmp_path)

    session_agent.cmd_status(workspace)

    request_path = request_root / "request.json"
    request_path.write_text(
        request_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(harness.HarnessError, match="path/hash drift"):
        session_agent.cmd_status(workspace)

    request_path.write_text(
        request_path.read_text(encoding="utf-8").rstrip(),
        encoding="utf-8",
    )
    manifest = workspace / "package/package_manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(harness.HarnessError, match="package_manifest.json hash drift"):
        session_agent.cmd_status(workspace)


def test_status_validates_completion_hashes(tmp_path: Path) -> None:
    workspace = _to_complete(tmp_path)

    session_agent.cmd_status(workspace)

    metrics = workspace / "qa/metrics.json"
    metrics.write_text(metrics.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="qa/metrics.json hash drift"):
        session_agent.cmd_status(workspace)


# ---------------------------------------------------------------------------
# advance: awaiting_approval -> awaiting_remote
# ---------------------------------------------------------------------------


def test_pending_approval_advance_fails_without_phase_change(
    tmp_path: Path,
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(request_root / "request.json", workspace)
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    with pytest.raises(harness.HarnessError):
        session_agent.cmd_advance(workspace)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_approval"
    assert not (workspace / "package").exists()
    assert _staging_leftovers(workspace) == []


def test_approved_advance_packages_and_transitions_once(
    tmp_path: Path,
) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path, sample_steps=10)

    contract = json.loads(
        (workspace / "package/sampling_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["frozen_parameters"]["sample_steps"] == 10
    session = json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )
    assert session["phase"] == "awaiting_remote"
    assert session["sample_steps"] == 10
    assert session["package_manifest_sha256"] == (
        _sha256_file(workspace / "package/package_manifest.json")
    )
    assert session["completion"] is None

    # The second advance (no remote output) is read-only: no rerun, no
    # finalized/qa, session bytes unchanged.
    before = _tree(workspace)
    payload = session_agent.cmd_advance(workspace)
    assert _tree(workspace) == before
    assert payload["phase"] == "awaiting_remote"
    assert payload["next_action"]["action"] == "provide_remote_output"
    assert payload["missing"] == {
        "option": "--remote-output",
        "required": "exact successful remote-output directory",
        "files": ["raw_shot.mp4", "sampling_receipt.json"],
    }
    assert not (workspace / "finalized").exists()
    assert not (workspace / "qa").exists()


def test_awaiting_remote_missing_remote_dir_rejected_without_work(
    tmp_path: Path,
) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    before = _tree(workspace)

    with pytest.raises(harness.HarnessError, match="exact existing directory"):
        session_agent.cmd_advance(
            workspace, remote_output=tmp_path / "missing-remote"
        )

    assert _tree(workspace) == before
    session = json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )
    assert session["phase"] == "awaiting_remote"
    assert not (workspace / "finalized").exists()
    assert not (workspace / "qa").exists()


def test_validate_failure_after_package_creation_keeps_phase_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(request_root / "request.json", workspace)
    _approve(workspace / "inspection/approval.json")
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    original = remote_sample.validate_package
    state = {"calls": 0}

    def _boom_once(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise harness.HarnessError(
                "evidence_incomplete", "injected validation failure"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(remote_sample, "validate_package", _boom_once)
    with pytest.raises(harness.HarnessError, match="injected"):
        session_agent.cmd_advance(workspace)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_approval"
    assert not (workspace / "package").exists()
    assert _staging_leftovers(workspace) == []

    session_agent.cmd_advance(workspace)
    assert (workspace / "package").is_dir()
    assert json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )["phase"] == "awaiting_remote"


def test_session_write_failure_after_package_publish_keeps_phase_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"
    session_agent.cmd_init(request_root / "request.json", workspace)
    _approve(workspace / "inspection/approval.json")
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    original = session_agent._write_session
    state = {"calls": 0}

    def _boom_once(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("injected session write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(session_agent, "_write_session", _boom_once)
    with pytest.raises(OSError, match="injected"):
        session_agent.cmd_advance(workspace)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_approval"
    assert not (workspace / "package").exists()
    assert _staging_leftovers(workspace) == []

    session_agent.cmd_advance(workspace)
    assert (workspace / "package").is_dir()
    assert json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )["phase"] == "awaiting_remote"


# ---------------------------------------------------------------------------
# advance: awaiting_remote -> complete (finalize + QA)
# ---------------------------------------------------------------------------


def test_successful_fake_remote_output_finalize_qa_complete(
    tmp_path: Path,
) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")

    payload = session_agent.cmd_advance(workspace, remote_output=remote)

    assert payload["phase"] == "complete"
    assert payload["next_action"]["action"] == "none"
    assert (workspace / "finalized/generation_manifest.json").is_file()
    assert (workspace / "finalized/raw_shot.mp4").is_file()
    assert (workspace / "finalized/output.mp4").is_file()
    assert (workspace / "qa/metrics.json").is_file()
    assert (workspace / "qa/artifacts.json").is_file()

    session = json.loads(
        (workspace / "session.json").read_text(encoding="utf-8")
    )
    assert session["phase"] == "complete"
    assert session["package_manifest_sha256"] == (
        _sha256_file(workspace / "package/package_manifest.json")
    )
    assert session["completion"] == {
        "generation_manifest_sha256": _sha256_file(
            workspace / "finalized/generation_manifest.json"
        ),
        "qa_metrics_sha256": _sha256_file(workspace / "qa/metrics.json"),
        "qa_artifacts_sha256": _sha256_file(workspace / "qa/artifacts.json"),
    }
    assert _staging_leftovers(workspace) == []


def test_finalize_failure_keeps_awaiting_remote_clean(tmp_path: Path) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")
    # The receipt must fail the frozen sample_steps contract inside
    # cmd_finalize: package contract is 40, receipt now claims 99.
    receipt_path = remote / "sampling_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sample_steps"] = 99
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    with pytest.raises(harness.HarnessError, match="sample_steps"):
        session_agent.cmd_advance(workspace, remote_output=remote)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_remote"
    assert not (workspace / "finalized").exists()
    assert not (workspace / "qa").exists()
    assert _staging_leftovers(workspace) == []


def test_qa_failure_after_finalize_keeps_awaiting_remote_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    original = qa_evidence.cmd_qa
    state = {"calls": 0}

    def _boom_once(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise harness.HarnessError(
                "media_normalization", "injected QA failure"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(qa_evidence, "cmd_qa", _boom_once)
    with pytest.raises(harness.HarnessError, match="injected"):
        session_agent.cmd_advance(workspace, remote_output=remote)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_remote"
    assert not (workspace / "finalized").exists()
    assert not (workspace / "qa").exists()
    assert _staging_leftovers(workspace) == []

    # The same valid remote output can be passed again.
    payload = session_agent.cmd_advance(workspace, remote_output=remote)
    assert payload["phase"] == "complete"
    assert (workspace / "finalized").is_dir()
    assert (workspace / "qa").is_dir()


def test_session_write_failure_after_publish_keeps_awaiting_remote_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    original = session_agent._write_session
    state = {"calls": 0}

    def _boom_once(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("injected session write failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(session_agent, "_write_session", _boom_once)
    with pytest.raises(OSError, match="injected"):
        session_agent.cmd_advance(workspace, remote_output=remote)

    assert session_path.read_bytes() == before
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["phase"] == "awaiting_remote"
    assert session["package_manifest_sha256"] is not None
    assert session["completion"] is None
    assert not (workspace / "finalized").exists()
    assert not (workspace / "qa").exists()
    assert _staging_leftovers(workspace) == []

    # The same valid remote output can be passed again.
    payload = session_agent.cmd_advance(workspace, remote_output=remote)
    assert payload["phase"] == "complete"
    assert (workspace / "finalized").is_dir()
    assert (workspace / "qa").is_dir()


def test_complete_advance_idempotent_no_tool_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _to_complete(tmp_path)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("finalize/QA must not run again in complete phase")

    monkeypatch.setattr(harness, "cmd_finalize", _must_not_run)
    monkeypatch.setattr(qa_evidence, "cmd_qa", _must_not_run)
    session_path = workspace / "session.json"
    before = session_path.read_bytes()

    payload = session_agent.cmd_advance(workspace)

    assert payload["phase"] == "complete"
    assert payload["next_action"]["action"] == "none"
    assert session_path.read_bytes() == before
    assert _staging_leftovers(workspace) == []


def test_exact_paths_only_no_directory_discovery(tmp_path: Path) -> None:
    workspace, _ = _to_awaiting_remote(tmp_path)
    remote = _make_fake_remote_output(tmp_path, workspace / "package")
    # Extra remote files are ignored: the agent reads only the two fixed
    # files needed by finalize + QA, never enumerates the directory.
    (remote / "result.json").write_text("{}", encoding="utf-8")
    extra_dir = remote / "runner-output"
    extra_dir.mkdir()
    (extra_dir / "0.mp4").write_bytes(b"not used")
    (remote / "extra.bin").write_bytes(b"\x00" * 16)

    payload = session_agent.cmd_advance(workspace, remote_output=remote)
    assert payload["phase"] == "complete"

    text = (workspace / "session.json").read_text(encoding="utf-8")
    for forbidden in (
        "iterdir",
        "listdir",
        "rglob",
        "glob(",
        "API_KEY",
        "SECRET",
        "TOKEN",
        "os.environ",
        "ssh",
        "scp",
    ):
        assert forbidden not in text


def test_source_has_no_recursive_discovery_or_subprocess_ssh() -> None:
    text = Path(session_agent.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "iterdir",
        "listdir",
        "rglob",
        "glob(",
        "subprocess",
        "ssh",
        "scp",
        "os.environ",
        "API_KEY",
        "SECRET",
        "TOKEN",
    ):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_json_stdout_and_existing_style_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request_root, _ = _make_request_root(tmp_path)
    workspace = tmp_path / "workspace"

    rc = session_agent.main(
        [
            "init",
            "--request",
            str(request_root / "request.json"),
            "--workspace",
            str(workspace),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["phase"] == "awaiting_approval"
    assert payload["next_action"]["action"] == "approve"

    rc = session_agent.main(["status", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "awaiting_approval"

    rc = session_agent.main(
        [
            "init",
            "--request",
            str(request_root / "request.json"),
            "--workspace",
            str(workspace),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert captured.out == ""

    _approve(workspace / "inspection/approval.json")
    rc = session_agent.main(["advance", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "awaiting_remote"
    rc = session_agent.main(["advance", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "awaiting_remote"
    assert payload["missing"]["option"] == "--remote-output"
