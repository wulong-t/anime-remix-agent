"""G1-MK6-L tests: queue -> manual session -> resolved timeline bridge.

The bridge only reuses already-PASS modules (generation_queue, session_agent,
manual-keyframe harness).  The happy path runs one real synthetic session
(init -> approve -> package -> finalize+QA) against FFmpeg-generated media in
pytest temporary directories, then resolves the completed generated clip into
a strict Timeline copy rendered by the existing render workflow.  All drift
rejections are atomic; the bridge source contains no media tools, no external
worker references and no directory discovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from anime_remix.domain.models import RenderProfile, Timeline
from anime_remix.workflows.render_workflow import render_timeline
from experiments.manual_keyframe_mvp import (
    generation_bridge,
    generation_queue,
    session_agent,
)
from experiments.manual_keyframe_mvp import (
    manual_keyframe_mvp as harness,
)

ACTION = "turn head slightly to the right"
EMOTION = "calm"
SHOT_SCALE = "medium"
SHOT_ID = "shot_003"


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


def _make_inputs(root: Path) -> dict[str, object]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    k0 = inputs / "k0.png"
    k_end = inputs / "k_end.png"
    k0.write_bytes(_solid_png_bytes(1280, 720, (255, 255, 255)))
    k_end.write_bytes(_solid_png_bytes(1280, 720, (0, 0, 0)))
    return {
        "k0": k0,
        "k_end": k_end,
        "k0_sha": _sha256_file(k0),
        "k_end_sha": _sha256_file(k_end),
    }


def _base_request(info: dict[str, object], shot_id: str) -> dict[str, object]:
    return {
        "schema_version": harness.REQUEST_SCHEMA,
        "request_id": shot_id,
        "start_keyframe": "inputs/k0.png",
        "end_keyframe": "inputs/k_end.png",
        "start_provenance": "inputs/k0.provenance.json",
        "end_provenance": "inputs/k_end.provenance.json",
        "start_sha256": info["k0_sha"],
        "end_sha256": info["k_end_sha"],
        "subject_description": "An original 2D cel-animation woman",
        "scene_description": "A quiet observatory control room at dusk",
        "action": ACTION,
        "start_state": "near-frontal calm",
        "end_state": "three-quarter calm",
        "emotion": EMOTION,
        "shot_scale": SHOT_SCALE,
        "camera": "fixed",
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
    }


def _provenance_asset_for(name: str) -> str:
    return "inputs/k0.png" if name.startswith("k0.") else "inputs/k_end.png"


def _write_provenance(root: Path, name: str, image: Path) -> None:
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


def _make_request_root(root: Path, shot_id: str) -> tuple[Path, dict[str, object]]:
    info = _make_inputs(root)
    _write_provenance(root, "k0.provenance.json", info["k0"])  # type: ignore[arg-type]
    _write_provenance(root, "k_end.provenance.json", info["k_end"])  # type: ignore[arg-type]
    request = _base_request(info, shot_id)
    (root / "request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    return root, request


def _requirement(shot_id: str, order: int, target_frames: int) -> dict[str, object]:
    return {
        "id": shot_id,
        "order": order,
        "source_text": f"source text for {shot_id}",
        "action": ACTION,
        "target_frames": target_frames,
        "dialogue": None,
        "emotion": EMOTION,
        "shot_scale": SHOT_SCALE,
        "characters": [],
        "location_id": None,
        "location_name": None,
    }


def _placeholder_item(
    shot_id: str, order: int, target_frames: int = 121
) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": _requirement(shot_id, order, target_frames),
        "strategy": "placeholder",
        "source_asset_id": None,
        "source_path": None,
        "source_size_bytes": None,
        "source_sha256": None,
        "source_in_frame": 0,
        "source_frame_count": 0,
        "target_frames": target_frames,
        "score": None,
        "reason_code": "no_candidate",
        "reason": "no candidate",
    }


def _clip_item(shot_id: str = "shot_001", order: int = 1) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": _requirement(shot_id, order, 72),
        "strategy": "clip",
        "source_asset_id": f"asset_{shot_id}",
        "source_path": f"clips/{shot_id}.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "a" * 64,
        "source_in_frame": 0,
        "source_frame_count": 72,
        "target_frames": 72,
        "score": None,
        "reason_code": "exact_length",
        "reason": "exact_length",
    }


def _freeze_item(shot_id: str = "shot_002", order: int = 3) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": _requirement(shot_id, order, 72),
        "strategy": "freeze_frame",
        "source_asset_id": f"asset_{shot_id}",
        "source_path": f"clips/{shot_id}.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "b" * 64,
        "source_in_frame": 0,
        "source_frame_count": 24,
        "target_frames": 72,
        "score": None,
        "reason_code": "short_source_freeze",
        "reason": "short_source_freeze",
    }


def _timeline_doc(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.9",
        "path_base": "timeline_dir",
        "render_profile": RenderProfile().model_dump(mode="json"),
        "items": items,
    }


def _write_timeline(root: Path, items: list[dict[str, object]]) -> Path:
    path = root / "timeline.json"
    path.write_text(json.dumps(_timeline_doc(items), indent=2), encoding="utf-8")
    return path


def _approve(pending: Path) -> None:
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


def _make_fake_remote_output(root: Path, package: Path) -> Path:
    out = root / "remote-output"
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_shot.mp4"
    _make_raw_mp4(raw_path)
    raw_sha = _sha256_file(raw_path)
    (out / "sampling_receipt.json").write_text(
        json.dumps(_receipt_for(package, raw_sha), indent=2), encoding="utf-8"
    )
    return out


def _make_flow(root: Path, target_frames: int) -> dict[str, Path]:
    timeline_dir = root / "timeline-dir"
    timeline_dir.mkdir()
    timeline = _write_timeline(
        timeline_dir,
        [
            _clip_item("shot_001", 1),
            _placeholder_item(SHOT_ID, 2, target_frames),
            _freeze_item("shot_002", 3),
        ],
    )
    queue_dir = timeline_dir / "queue"
    generation_queue.cmd_plan(timeline, queue_dir)
    request_root, _ = _make_request_root(root / "request-root", SHOT_ID)
    workspace = timeline_dir / "session-ws"
    generation_bridge.cmd_start(
        timeline,
        queue_dir / generation_queue.QUEUE_FILE,
        SHOT_ID,
        request_root / "request.json",
        workspace,
    )
    return {
        "timeline_dir": timeline_dir,
        "timeline": timeline,
        "queue": queue_dir / generation_queue.QUEUE_FILE,
        "request_root": request_root,
        "request": request_root / "request.json",
        "workspace": workspace,
    }


def _complete(flow: dict[str, Path], root: Path) -> None:
    _approve(flow["workspace"] / "inspection/approval.json")
    session_agent.cmd_advance(flow["workspace"])
    remote = _make_fake_remote_output(root, flow["workspace"] / "package")
    session_agent.cmd_advance(flow["workspace"], remote_output=remote)


@pytest.fixture(scope="module")
def completed_flow(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("g1mk6-module")
    flow = _make_flow(root, 121)
    _complete(flow, root)
    return flow


def _clone_flow(
    tmp_path: Path, flow: dict[str, Path], *, target_frames: int | None = None
) -> dict[str, Path]:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    timeline_dir = clone_root / "timeline-dir"
    shutil.copytree(flow["timeline_dir"], timeline_dir)
    request_root = clone_root / "request-root"
    shutil.copytree(flow["request_root"], request_root)
    if target_frames is not None and target_frames != 121:
        timeline = _write_timeline(
            timeline_dir,
            [
                _clip_item("shot_001", 1),
                _placeholder_item(SHOT_ID, 2, target_frames),
                _freeze_item("shot_002", 3),
            ],
        )
        queue_dir = timeline_dir / "queue"
        shutil.rmtree(queue_dir)
        generation_queue.cmd_plan(timeline, queue_dir)
        workspace = timeline_dir / "session-ws"
        shutil.rmtree(workspace)
        shutil.copytree(flow["workspace"], workspace)
    else:
        timeline = timeline_dir / "timeline.json"
        queue_dir = timeline_dir / "queue"
        workspace = timeline_dir / "session-ws"
    return {
        "timeline_dir": timeline_dir,
        "timeline": timeline,
        "queue": queue_dir / generation_queue.QUEUE_FILE,
        "request_root": request_root,
        "request": request_root / "request.json",
        "workspace": workspace,
    }


def _staging_leftovers(root: Path) -> list[Path]:
    return [path for path in root.iterdir() if ".staging-" in path.name]


def test_start_valid_mapping_creates_awaiting_approval_session(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    timeline_bytes = flow["timeline"].read_bytes()
    queue_bytes = flow["queue"].read_bytes()
    new_ws = flow["timeline_dir"] / "new-session"

    payload = generation_bridge.cmd_start(
        flow["timeline"], flow["queue"], SHOT_ID, flow["request"], new_ws
    )

    assert payload["status"] == "ok"
    assert payload["phase"] == "awaiting_approval"
    assert payload["request_id"] == SHOT_ID
    assert payload["next_action"]["action"] == "approve"
    assert flow["timeline"].read_bytes() == timeline_bytes
    assert flow["queue"].read_bytes() == queue_bytes
    assert (new_ws / "session.json").is_file()
    assert (new_ws / "inspection/approval.json").is_file()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda doc: doc.update(request_id="other-shot"), "request_id"),
        (lambda doc: doc.update(action="wave hello"), "action mismatch"),
        (lambda doc: doc.update(emotion="happy"), "emotion mismatch"),
        (lambda doc: doc.update(shot_scale="wide"), "shot_scale mismatch"),
    ],
)
def test_start_rejects_mismatched_request_semantics(
    tmp_path: Path,
    completed_flow: dict[str, Path],
    mutator,
    match: str,
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    request_path = flow["request"]
    doc = json.loads(request_path.read_text(encoding="utf-8"))
    mutator(doc)
    request_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    workspace = flow["timeline_dir"] / "bad-ws"

    with pytest.raises(harness.HarnessError, match=match):
        generation_bridge.cmd_start(
            flow["timeline"], flow["queue"], SHOT_ID, request_path, workspace
        )

    assert not workspace.exists()
    assert _staging_leftovers(flow["timeline_dir"]) == []


def test_start_rejects_stale_queue_hash(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    doc = json.loads(flow["timeline"].read_text(encoding="utf-8"))
    doc["items"][0]["reason"] = "tampered"
    flow["timeline"].write_text(json.dumps(doc, indent=2), encoding="utf-8")
    workspace = flow["timeline_dir"] / "bad-ws"

    with pytest.raises(harness.HarnessError) as exc:
        generation_bridge.cmd_start(
            flow["timeline"], flow["queue"], SHOT_ID, flow["request"], workspace
        )

    assert exc.value.layer == "evidence_incomplete"
    assert not workspace.exists()


def test_start_rejects_non_child_workspace(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    outside = tmp_path / "outside-ws"

    with pytest.raises(harness.HarnessError, match="direct child"):
        generation_bridge.cmd_start(
            flow["timeline"], flow["queue"], SHOT_ID, flow["request"], outside
        )

    assert not outside.exists()


def test_start_rejects_existing_workspace(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    existing = flow["timeline_dir"] / "existing-ws"
    existing.mkdir()

    with pytest.raises(harness.HarnessError, match="already exists"):
        generation_bridge.cmd_start(
            flow["timeline"], flow["queue"], SHOT_ID, flow["request"], existing
        )


@pytest.mark.parametrize(
    ("target_frames", "strategy", "start", "count", "reason_code"),
    [
        (121, "clip", 0, 121, "exact_length"),
        (72, "clip", 24, 72, "center_trim"),
        (144, "freeze_frame", 0, 121, "short_source_freeze"),
    ],
)
def test_resolve_maps_generated_clip_and_preserves_other_items(
    tmp_path: Path,
    completed_flow: dict[str, Path],
    target_frames: int,
    strategy: str,
    start: int,
    count: int,
    reason_code: str,
) -> None:
    flow = _clone_flow(tmp_path, completed_flow, target_frames=target_frames)
    original = flow["timeline"].read_bytes()
    original_doc = json.loads(original)
    output = flow["timeline_dir"] / f"resolved-{target_frames}.json"

    payload = generation_bridge.cmd_resolve(
        flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], output
    )

    generated = flow["workspace"] / "finalized/generated_clip.mp4"
    assert payload["status"] == "ok"
    assert payload["strategy"] == strategy
    assert payload["reason_code"] == reason_code
    assert payload["source_in_frame"] == start
    assert payload["source_frame_count"] == count
    assert payload["target_frames"] == target_frames
    assert payload["source_path"] == "session-ws/finalized/generated_clip.mp4"
    assert payload["source_sha256"] == _sha256_file(generated)
    assert payload["source_size_bytes"] == generated.stat().st_size

    resolved = json.loads(output.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == "1.9"
    assert resolved["render_profile"] == original_doc["render_profile"]
    resolved_items = resolved["items"]
    assert len(resolved_items) == 3
    for index, original_item in enumerate(original_doc["items"]):
        if original_item["shot_id"] == SHOT_ID:
            assert resolved_items[index]["strategy"] == strategy
            assert resolved_items[index]["source_asset_id"] == SHOT_ID
            assert resolved_items[index]["source_path"] == (
                "session-ws/finalized/generated_clip.mp4"
            )
            assert resolved_items[index]["source_in_frame"] == start
            assert resolved_items[index]["source_frame_count"] == count
            assert resolved_items[index]["target_frames"] == target_frames
            assert resolved_items[index]["reason_code"] == reason_code
            assert resolved_items[index]["score"] is None
            assert resolved_items[index]["requirement"] == original_item["requirement"]
            assert resolved_items[index]["source_sha256"] == _sha256_file(generated)
            assert resolved_items[index]["source_size_bytes"] == generated.stat().st_size
        else:
            assert resolved_items[index] == original_item

    Timeline.model_validate(resolved)
    assert flow["timeline"].read_bytes() == original


def test_resolve_rejects_incomplete_session(tmp_path: Path) -> None:
    flow = _make_flow(tmp_path, 121)
    output = flow["timeline_dir"] / "resolved.json"

    with pytest.raises(harness.HarnessError, match="not complete"):
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], output
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("relative", "mutator"),
    [
        ("finalized/generated_clip.mp4", lambda data: data + b"x"),
        (
            "finalized/generation_manifest.json",
            lambda data: data.replace('"generated_clip.mp4"', '"tampered.mp4"', 1),
        ),
        (
            "finalized/clips.json",
            lambda data: data.replace('"generated_clip.mp4"', '"tampered.mp4"', 1),
        ),
    ],
)
def test_resolve_rejects_tampered_finalized_artifacts(
    tmp_path: Path,
    completed_flow: dict[str, Path],
    relative: str,
    mutator,
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    path = flow["workspace"] / relative
    if path.suffix == ".mp4":
        path.write_bytes(mutator(path.read_bytes()))
    else:
        path.write_text(mutator(path.read_text(encoding="utf-8")), encoding="utf-8")
    output = flow["timeline_dir"] / "resolved.json"

    with pytest.raises(harness.HarnessError) as exc:
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], output
        )

    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()


def test_resolve_rejects_stale_timeline(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    doc = json.loads(flow["timeline"].read_text(encoding="utf-8"))
    doc["items"][0]["reason"] = "tampered"
    flow["timeline"].write_text(json.dumps(doc, indent=2), encoding="utf-8")
    output = flow["timeline_dir"] / "resolved.json"

    with pytest.raises(harness.HarnessError) as exc:
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], output
        )

    assert exc.value.layer == "evidence_incomplete"
    assert not output.exists()


def test_resolve_rejects_wrong_outside_or_existing_output(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)

    outside = tmp_path / "outside.json"
    with pytest.raises(harness.HarnessError, match="original timeline directory"):
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], outside
        )
    assert not outside.exists()

    existing = flow["timeline_dir"] / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(harness.HarnessError, match="already exists"):
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], existing
        )
    assert existing.read_text(encoding="utf-8") == "keep"

    with pytest.raises(harness.HarnessError, match="original timeline"):
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], flow["timeline"]
        )


def test_resolve_rejects_symlink_output_where_supported(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    target = flow["timeline_dir"] / "target.json"
    target.write_text("x", encoding="utf-8")
    link = flow["timeline_dir"] / "link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create symlinks: {exc}")

    with pytest.raises(harness.HarnessError, match="link"):
        generation_bridge.cmd_resolve(
            flow["timeline"], flow["queue"], SHOT_ID, flow["workspace"], link
        )


def test_resolved_timeline_renders_with_existing_workflow(
    tmp_path: Path, completed_flow: dict[str, Path]
) -> None:
    timeline_dir = tmp_path / "render-case"
    timeline_dir.mkdir()
    timeline = _write_timeline(timeline_dir, [_placeholder_item(SHOT_ID, 1, 121)])
    queue_dir = timeline_dir / "queue"
    generation_queue.cmd_plan(timeline, queue_dir)
    workspace = timeline_dir / "session-ws"
    shutil.copytree(completed_flow["workspace"], workspace)
    resolved = timeline_dir / "resolved.json"
    generation_bridge.cmd_resolve(
        timeline,
        queue_dir / generation_queue.QUEUE_FILE,
        SHOT_ID,
        workspace,
        resolved,
    )

    output = tmp_path / "rendered.mp4"
    render_timeline(resolved, output, log_path=tmp_path / "render.log")

    assert output.is_file()
    assert output.stat().st_size > 0


def test_cli_start_and_resolve_json_stdout_and_error_codes(
    tmp_path: Path, completed_flow: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    flow = _clone_flow(tmp_path, completed_flow)
    new_ws = flow["timeline_dir"] / "cli-ws"

    rc = generation_bridge.main(
        [
            "start",
            "--timeline",
            str(flow["timeline"]),
            "--queue",
            str(flow["queue"]),
            "--shot-id",
            SHOT_ID,
            "--request",
            str(flow["request"]),
            "--workspace",
            str(new_ws),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["phase"] == "awaiting_approval"

    output = flow["timeline_dir"] / "cli-resolved.json"
    rc = generation_bridge.main(
        [
            "resolve",
            "--timeline",
            str(flow["timeline"]),
            "--queue",
            str(flow["queue"]),
            "--shot-id",
            SHOT_ID,
            "--workspace",
            str(flow["workspace"]),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert output.is_file()

    rc = generation_bridge.main(
        [
            "resolve",
            "--timeline",
            str(flow["timeline"]),
            "--queue",
            str(flow["queue"]),
            "--shot-id",
            SHOT_ID,
            "--workspace",
            str(flow["workspace"]),
            "--output",
            str(output),
        ]
    )
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_bridge_source_has_no_media_remote_subprocess_or_discovery() -> None:
    text = Path(generation_bridge.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "ssh",
        "scp",
        "remote",
        "anisora",
        "subprocess",
        "iterdir",
        "listdir",
        "rglob",
        "glob(",
        "ffmpeg",
        "ffprobe",
        "os.environ",
        "api_key",
        "secret",
        "token",
    ):
        assert forbidden not in text
