#!/usr/bin/env python
"""G1-MK6-L: queue -> manual session -> resolved timeline bridge.

Resolves one completed manual-keyframe session's generated clip back into a
strict copy of the original Timeline for the existing renderer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import ClipsDocument, Timeline
from anime_remix.json_io import dump_json_atomic
from experiments.manual_keyframe_mvp import (
    generation_queue as queue_tool,
)
from experiments.manual_keyframe_mvp import (
    manual_keyframe_mvp as harness,
)
from experiments.manual_keyframe_mvp import (
    session_agent,
)


def _sha256_bytes(data: bytes) -> str:
    return harness._sha256_bytes(data)


def _link(path: Path) -> bool:
    return queue_tool._is_link_or_reparse(path)


def _capture(path: Path, what: str) -> bytes:
    if _link(path):
        raise harness.HarnessError("input_contract", f"{what} must not be a link: {path}")
    return harness._capture_bytes(path, "input_contract")


def _read_json(path: Path, what: str) -> Any:
    return harness._load_json_bytes(_capture(path, what), what, "input_contract")


def _exact(data: Any, fields: set[str], what: str) -> dict[str, Any]:
    data = harness._require_object(data, what, "input_contract")
    if set(data) != fields:
        raise harness.HarnessError(
            "input_contract", f"{what}: keys must be exactly {sorted(fields)}"
        )
    return data


_FACT_FIELDS = (
    "order", "target_frames", "source_text", "action", "emotion", "shot_scale",
    "characters", "location_id", "location_name",
)


def _facts(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, dict):
        return {key: item[key] for key in _FACT_FIELDS}
    return {
        "order": item.order,
        "target_frames": item.target_frames,
        "source_text": item.source_text,
        "action": item.action,
        "emotion": item.emotion.value if item.emotion is not None else None,
        "shot_scale": item.shot_scale.value if item.shot_scale is not None else None,
        "characters": [c.model_dump(mode="json") for c in item.characters],
        "location_id": item.location_id,
        "location_name": item.location_name,
    }


def _load_binding(
    timeline: Path, queue_path: Path, shot_id: str
) -> tuple[bytes, dict[str, Any]]:
    timeline = Path(timeline)
    if _link(timeline) or not timeline.is_file():
        raise harness.HarnessError("input_contract", f"timeline must be an exact regular file: {timeline}")
    if _link(timeline.parent):
        raise harness.HarnessError("input_contract", f"timeline directory must not be a link: {timeline.parent}")
    timeline_bytes = _capture(timeline, "timeline.json")
    try:
        timeline_model = Timeline.model_validate(json.loads(timeline_bytes))
    except Exception as exc:
        raise harness.HarnessError("input_contract", f"invalid timeline.json: {exc}") from exc

    queue_doc = _exact(
        _read_json(queue_path, "generation_queue.json"),
        queue_tool.QUEUE_FIELDS,
        "generation_queue.json",
    )
    if queue_doc["schema_version"] != queue_tool.QUEUE_SCHEMA:
        raise harness.HarnessError("input_contract", "queue schema drift")
    if queue_doc["timeline_sha256"] != _sha256_bytes(timeline_bytes):
        raise harness.HarnessError(
            "evidence_incomplete", "queue timeline_sha256 does not match timeline bytes"
        )
    if not isinstance(queue_doc["items"], list):
        raise harness.HarnessError("input_contract", "queue items must be a list")
    items = [_exact(item, queue_tool.ITEM_FIELDS, "queue item") for item in queue_doc["items"]]
    if (
        queue_doc["timeline_schema_version"] != timeline_model.schema_version
        or queue_doc["total_timeline_items"] != len(timeline_model.items)
        or queue_doc["pending_count"] != len(items)
    ):
        raise harness.HarnessError("evidence_incomplete", "queue metadata drift")
    selected = [item for item in items if item["shot_id"] == shot_id]
    if len(selected) != 1:
        raise harness.HarnessError(
            "input_contract", f"queue must contain exactly one item for {shot_id}"
        )
    queue_item = selected[0]
    if queue_item["status"] != queue_tool.STATUS_NEEDS_MANUAL_KEYFRAMES:
        raise harness.HarnessError("input_contract", "queue item status drift")

    matches = [item for item in timeline_model.items if item.shot_id == shot_id]
    if len(matches) != 1 or matches[0].strategy is not TimelineStrategy.PLACEHOLDER:
        raise harness.HarnessError(
            "evidence_incomplete",
            f"timeline must contain exactly one placeholder for {shot_id}",
        )
    target = matches[0]
    if _facts(queue_item) != _facts(target.requirement):
        raise harness.HarnessError(
            "evidence_incomplete", f"timeline item {shot_id} no longer matches its queue item"
        )
    return timeline_bytes, queue_item


def _validate_workspace(timeline: Path, workspace: Path, *, existing: bool) -> None:
    workspace = Path(workspace)
    if os.path.normcase(str(workspace.parent.resolve())) != os.path.normcase(
        str(timeline.parent.resolve())
    ):
        raise harness.HarnessError(
            "input_contract", "workspace must be a direct child of the timeline directory"
        )
    if not workspace.name or workspace.name in (".", "..") or _link(workspace):
        raise harness.HarnessError(
            "input_contract", f"workspace must be a plain non-link path: {workspace}"
        )
    if existing:
        if not workspace.is_dir():
            raise harness.HarnessError(
                "input_contract", f"workspace must be an exact existing directory: {workspace}"
            )
    elif workspace.exists() or workspace.is_symlink():
        raise harness.HarnessError("input_contract", f"workspace already exists: {workspace}")


def cmd_start(
    timeline: Path,
    queue_path: Path,
    shot_id: str,
    request: Path,
    workspace: Path,
    sample_steps: int = harness.SAMPLE_STEPS_DEFAULT,
) -> dict[str, Any]:
    sample_steps = harness._validate_sample_steps(sample_steps)
    _, queue_item = _load_binding(timeline, queue_path, shot_id)
    _validate_workspace(timeline, workspace, existing=False)
    request_bytes = _capture(request, "request.json")
    info = harness._validate_request_bytes(request_bytes, request)
    if info["request_id"] != queue_item["shot_id"]:
        raise harness.HarnessError("input_contract", "request_id must equal the queued shot_id")
    for field in ("action", "emotion", "shot_scale"):
        if info["data"][field] != queue_item[field]:
            raise harness.HarnessError("input_contract", f"request {field} mismatch")
    session_agent.cmd_init(request, workspace, sample_steps=sample_steps)
    session = _read_json(Path(workspace) / session_agent.SESSION_FILE, "session.json")
    if (
        session["phase"] != session_agent.PHASE_AWAITING_APPROVAL
        or session["request"]["id"] != shot_id
        or session["request"]["sha256"] != _sha256_bytes(request_bytes)
    ):
        session_agent._remove_exact_child(Path(timeline).parent.resolve(), Path(workspace))
        raise harness.HarnessError("evidence_incomplete", "session start binding drift")
    payload = dict(session_agent.cmd_status(workspace))
    payload["workspace"] = str(Path(workspace).resolve())
    payload["status"] = "ok"
    return payload


def _validate_finalized(
    shot_id: str,
    generated_bytes: bytes,
    manifest: dict[str, Any],
    clips_bytes: bytes,
) -> None:
    if (
        manifest.get("schema_version") != harness.GENERATION_MANIFEST_SCHEMA
        or manifest.get("request_id") != shot_id
    ):
        raise harness.HarnessError("evidence_incomplete", "generation manifest schema/id drift")
    normalized = manifest.get("normalized")
    clips_ref = manifest.get("clips")
    if (
        not isinstance(normalized, dict)
        or normalized.get("path") != "generated_clip.mp4"
        or normalized.get("sha256") != _sha256_bytes(generated_bytes)
    ):
        raise harness.HarnessError("evidence_incomplete", "generated clip binding drift")
    probe = normalized.get("probe")
    if not isinstance(probe, dict) or probe.get("nb_read_frames") != harness.TARGET_FRAMES:
        raise harness.HarnessError(
            "evidence_incomplete", "normalized generated source must have 121 frames"
        )
    if (
        not isinstance(clips_ref, dict)
        or clips_ref.get("path") != "clips.json"
        or clips_ref.get("sha256") != _sha256_bytes(clips_bytes)
    ):
        raise harness.HarnessError("evidence_incomplete", "clips.json binding drift")
    try:
        clips_doc = ClipsDocument.model_validate(json.loads(clips_bytes))
    except Exception as exc:
        raise harness.HarnessError(
            "evidence_incomplete", f"invalid finalized clips.json: {exc}"
        ) from exc
    if len(clips_doc.clips) != 1:
        raise harness.HarnessError(
            "evidence_incomplete", "finalized clips.json must contain exactly one asset"
        )
    asset = clips_doc.clips[0]
    if asset.id != shot_id or asset.path.as_posix() != "generated_clip.mp4":
        raise harness.HarnessError(
            "evidence_incomplete", "finalized clips asset must map shot_id to generated_clip.mp4"
        )


def _resolved_item(
    item: dict[str, Any], shot_id: str, workspace_name: str, generated_bytes: bytes
) -> dict[str, Any]:
    target_frames = item["target_frames"]
    if target_frames == harness.TARGET_FRAMES:
        strategy, start, count, code = "clip", 0, harness.TARGET_FRAMES, "exact_length"
        reason = "generated clip matches target length"
    elif target_frames < harness.TARGET_FRAMES:
        strategy, count, code = "clip", target_frames, "center_trim"
        start = (harness.TARGET_FRAMES - target_frames) // 2
        reason = "generated clip longer than target; centered trim"
    else:
        strategy, start, count, code = (
            "freeze_frame",
            0,
            harness.TARGET_FRAMES,
            "short_source_freeze",
        )
        reason = "generated clip shorter than target; freeze final frame"
    resolved = dict(item)
    resolved.update({
        "strategy": strategy,
        "source_asset_id": shot_id,
        "source_path": f"{workspace_name}/finalized/generated_clip.mp4",
        "source_size_bytes": len(generated_bytes),
        "source_sha256": _sha256_bytes(generated_bytes),
        "source_in_frame": start,
        "source_frame_count": count,
        "score": None,
        "reason_code": code,
        "reason": reason,
    })
    return resolved


def cmd_resolve(
    timeline: Path,
    queue_path: Path,
    shot_id: str,
    workspace: Path,
    output: Path,
) -> dict[str, Any]:
    timeline_bytes, _ = _load_binding(timeline, queue_path, shot_id)
    _validate_workspace(timeline, workspace, existing=True)
    status = session_agent.cmd_status(workspace)
    if status["phase"] != session_agent.PHASE_COMPLETE:
        raise harness.HarnessError("input_contract", "session is not complete")
    if status["request_id"] != shot_id:
        raise harness.HarnessError("evidence_incomplete", "session request_id drift")
    workspace = Path(workspace)
    generated_bytes = _capture(
        workspace / "finalized" / "generated_clip.mp4", "generated_clip.mp4"
    )
    manifest = _read_json(
        workspace / "finalized" / "generation_manifest.json",
        "generation_manifest.json",
    )
    clips_bytes = _capture(workspace / "finalized" / "clips.json", "clips.json")
    _validate_finalized(shot_id, generated_bytes, manifest, clips_bytes)

    doc = json.loads(timeline_bytes)
    target_index = next(
        i for i, item in enumerate(doc["items"]) if item["shot_id"] == shot_id
    )
    items = [dict(item) for item in doc["items"]]
    items[target_index] = _resolved_item(
        items[target_index], shot_id, workspace.name, generated_bytes
    )
    resolved_doc = dict(doc)
    resolved_doc["items"] = items
    try:
        model = Timeline.model_validate(resolved_doc)
    except Exception as exc:
        raise harness.HarnessError(
            "input_contract", f"invalid resolved timeline: {exc}"
        ) from exc
    output = Path(output)
    if output.resolve() == Path(timeline).resolve():
        raise harness.HarnessError(
            "input_contract", "output must not overwrite the original timeline"
        )
    if _link(output) or output.exists():
        raise harness.HarnessError(
            "input_contract", f"output already exists or is a link: {output}"
        )
    if os.path.normcase(str(output.parent.resolve())) != os.path.normcase(
        str(Path(timeline).parent.resolve())
    ):
        raise harness.HarnessError(
            "input_contract", "output must be in the original timeline directory"
        )
    dump_json_atomic(Path(output), resolved_doc)

    item = model.items[target_index]
    return {
        "status": "ok", "schema_version": model.schema_version,
        "shot_id": shot_id, "strategy": item.strategy.value,
        "reason_code": item.reason_code, "source_path": item.source_path,
        "source_sha256": item.source_sha256,
        "source_size_bytes": item.source_size_bytes,
        "source_in_frame": item.source_in_frame,
        "source_frame_count": item.source_frame_count,
        "target_frames": item.target_frames,
        "output": str(Path(output).resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generation_bridge",
        description="G1-MK6-L queue -> session -> resolved timeline bridge",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser(
        "start", help="start a session for one queued placeholder shot"
    )
    for name in ("--timeline", "--queue", "--request", "--workspace"):
        start.add_argument(name, required=True, type=Path)
    start.add_argument("--shot-id", required=True)
    start.add_argument(
        "--sample-steps",
        type=int,
        default=harness.SAMPLE_STEPS_DEFAULT,
        metavar="1..100",
        help=f"packaged sampling steps, 1..100 (default {harness.SAMPLE_STEPS_DEFAULT})",
    )
    resolve = subparsers.add_parser(
        "resolve", help="resolve a completed session into a new timeline JSON"
    )
    for name in ("--timeline", "--queue", "--workspace", "--output"):
        resolve.add_argument(name, required=True, type=Path)
    resolve.add_argument("--shot-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "start":
            payload = cmd_start(
                args.timeline, args.queue, args.shot_id,
                args.request, args.workspace, args.sample_steps,
            )
        else:
            payload = cmd_resolve(
                args.timeline, args.queue, args.shot_id,
                args.workspace, args.output,
            )
    except harness.HarnessError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
