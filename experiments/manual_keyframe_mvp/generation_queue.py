#!/usr/bin/env python
"""G1-MK5-L: deterministic placeholder -> manual-generation queue handoff.

Reads one explicitly passed existing ``timeline.json``, captures its bytes
exactly once, strictly validates that captured object with the existing
``Timeline`` Pydantic model, and atomically publishes a deterministic queue
``g1-mk5-generation-queue-v1`` listing only ``placeholder`` items in timeline
array order.  Opt-in read-only handoff: no reopen, no source media reads, no
render, no write-back, no generation session; self-contained so the documented
``python experiments/.../generation_queue.py plan`` works directly from repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import Timeline, TimelineItem
from anime_remix.json_io import dump_json_atomic

QUEUE_SCHEMA = "g1-mk5-generation-queue-v1"
QUEUE_FILE = "generation_queue.json"
STATUS_NEEDS_MANUAL_KEYFRAMES = "needs_manual_keyframes"
REQUIRED_HUMAN_INPUTS = [
    "subject_description",
    "scene_description",
    "start_state",
    "end_state",
    "k0_png",
    "k_end_png",
    "k0_provenance",
    "k_end_provenance",
    "rights_and_visual_approval",
]
QUEUE_FIELDS = {
    "schema_version",
    "timeline_sha256",
    "timeline_schema_version",
    "total_timeline_items",
    "pending_count",
    "items",
    "next_action",
}
ITEM_FIELDS = {
    "shot_id",
    "order",
    "target_frames",
    "source_text",
    "action",
    "emotion",
    "shot_scale",
    "characters",
    "location_id",
    "location_name",
    "status",
    "required_human_inputs",
}


class QueueError(Exception):
    """Queue failure classified with the G1-MK harness error taxonomy."""

    def __init__(self, layer: str, message: str) -> None:
        self.layer = layer
        super().__init__(f"{layer}: {message}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    try:
        st = path.lstat()
    except OSError:
        return False
    return bool(
        getattr(st, "st_file_attributes", 0)
        & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _capture_bytes(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise QueueError("input_contract", f"cannot read {path}: {exc}") from exc


def _validate_timeline(data_bytes: bytes, what: str) -> Timeline:
    try:
        data = json.loads(data_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise QueueError("input_contract", f"invalid JSON in {what}: {exc}") from exc
    if not isinstance(data, dict):
        raise QueueError("input_contract", f"{what} must be a JSON object")
    try:
        return Timeline.model_validate(data)
    except Exception as exc:
        raise QueueError("input_contract", f"invalid {what}: {exc}") from exc


def _queue_item(item: TimelineItem) -> dict[str, Any]:
    requirement = item.requirement.model_dump(mode="json")
    return {
        "shot_id": item.shot_id,
        "order": item.order,
        "target_frames": item.target_frames,
        "source_text": requirement["source_text"],
        "action": requirement["action"],
        "emotion": requirement["emotion"],
        "shot_scale": requirement["shot_scale"],
        "characters": requirement["characters"],
        "location_id": requirement["location_id"],
        "location_name": requirement["location_name"],
        "status": STATUS_NEEDS_MANUAL_KEYFRAMES,
        "required_human_inputs": list(REQUIRED_HUMAN_INPUTS),
    }


def _reject_output(output: Path) -> None:
    if _is_link_or_reparse(output) or output.exists():
        raise QueueError(
            "input_contract",
            f"output already exists or is a symlink/reparse point: {output}",
        )


def _publish_dir(staging: Path, output: Path) -> None:
    try:
        os.replace(staging, output)
    except OSError:
        try:
            os.rename(staging, output)
        except OSError as exc:
            raise QueueError(
                "evidence_incomplete", f"atomic publication failed: {exc}"
            ) from exc


def _remove_staging(parent: Path, staging: Path) -> None:
    if not staging.exists() and not staging.is_symlink():
        return
    if staging.is_symlink():
        raise QueueError(
            "evidence_incomplete", f"refusing to remove symlink: {staging}"
        )
    if os.path.normcase(str(staging.resolve().parent)) != os.path.normcase(
        str(parent.resolve())
    ):
        raise QueueError(
            "evidence_incomplete",
            f"refusing to remove staging outside parent: {staging}",
        )
    shutil.rmtree(staging)


def cmd_plan(timeline: Path, output: Path) -> dict[str, Any]:
    timeline = Path(timeline)
    output = Path(output)
    if _is_link_or_reparse(timeline) or not timeline.is_file():
        raise QueueError(
            "input_contract",
            f"timeline must be an exact regular file "
            f"(no symlink/reparse point): {timeline}",
        )
    _reject_output(output)
    timeline_bytes = _capture_bytes(timeline)
    timeline_sha256 = _sha256_bytes(timeline_bytes)
    model = _validate_timeline(timeline_bytes, "timeline.json")

    items = [
        _queue_item(item)
        for item in model.items
        if item.strategy is TimelineStrategy.PLACEHOLDER
    ]
    next_action = (
        {"action": "provide_manual_keyframes", "target_shot_id": items[0]["shot_id"]}
        if items
        else {"action": "none", "target_shot_id": None}
    )
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "timeline_sha256": timeline_sha256,
        "timeline_schema_version": model.schema_version,
        "total_timeline_items": len(model.items),
        "pending_count": len(items),
        "items": items,
        "next_action": next_action,
    }
    parent = output.resolve().parent
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent)
    )
    try:
        dump_json_atomic(staging / QUEUE_FILE, queue)
        _publish_dir(staging, output)
    except BaseException:
        _remove_staging(parent, staging)
        raise
    return queue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generation_queue",
        description="G1-MK5-L placeholder -> manual-generation queue",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser(
        "plan", help="emit the manual-generation queue for placeholder shots"
    )
    plan_parser.add_argument("--timeline", required=True, type=Path)
    plan_parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        queue = cmd_plan(args.timeline, args.output)
    except QueueError as exc:
        print(f"ERROR {exc.layer}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR unexpected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(dict(queue, status="ok"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
