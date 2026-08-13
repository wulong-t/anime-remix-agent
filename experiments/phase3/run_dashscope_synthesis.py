"""Phase 3 Real, work package 5: character_synthesis only via DashScope.

Runs the frozen compose chain up to and including the real
``character_synthesis`` model node on the DashScope-hosted
``qwen-image-3.0-pro``.  Identity-safe text-HOW mode is the default: only the
WHO image is sent and the complete HOW state is textual.  ``--visual-how``
also sends the pose image and is reserved for identity-stripped pose controls.
The full synthesis lifecycle
(node_run_started -> render_intent_created -> model_render_request_created ->
render_attempt_started -> artifact_registered -> render_attempt_finished ->
node_run_finished -> port_bound) is recorded in the Execution Ledger, then
the run finishes without touching the deterministic / inpaint stages.

Usage::

    python experiments/phase3/run_dashscope_synthesis.py \
      --identity path/to/identity.png \
      --pose path/to/pose.png \
      --run-dir runs/phase3_dashscope_synthesis --seed 0 --size 1280*720

    python experiments/phase3/run_dashscope_synthesis.py \
      --identity path/to/identity.png --pose path/to/pose.png \
      --run-dir runs/phase3_dashscope_dry_run --dry-run

Real runs require the ``DASHSCOPE_API_KEY`` environment variable (never
logged or stored).  The executor resolves SDK availability before reading
the key and reports either gap as an ``EnvironmentCapabilityError``
recorded in the run manifest.  Exactly one model request is made; there is
no retry.
``--dry-run`` validates the full chain locally without any network access:
no key is required, no request is sent, and no media leaves this machine.

WARNING - cost and data egress
------------------------------
In real mode (without ``--dry-run``) this script makes exactly one PAID
DashScope API request (``qwen-image-3.0-pro``, billed to the account that owns
``DASHSCOPE_API_KEY``).  Default text-HOW mode uploads only the identity image;
``--visual-how`` uploads both identity and pose images.  Uploaded media leaves
this machine.  Run only with images you own or are licensed to use.  A failed
request is not retried; re-running the script deliberately is a new request
and a new cost.
``--dry-run`` makes no request and uploads nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_remote import (
    _keyframe_plan,
    _reference_package,
    _scene_geometry,
    _shot_spec,
)

from anime_remix.errors import (
    EnvironmentCapabilityError,
    InputValidationError,
    RenderError,
)
from anime_remix.services.execution.adapter import QwenImage30ProAdapter
from anime_remix.services.execution.dashscope_executor import (
    DashScopeQwenExecutor,
)
from anime_remix.services.execution.orchestrator import run_compose_keyframe

_DRY_RUN_PNG = b"\x89PNG\r\n\x1a\n" + b"dry-run-local-synthetic-bytes"
_DRY_RUN_IMAGE_URL = "https://dashscope.aliyuncs.com/dry-run.png"


def _dry_run_call(request: dict, api_key: str | None) -> dict:
    """Local stand-in for the SDK response; never touches the network.

    The shape mirrors the DashScope SDK response (status_code, request_id,
    output.choices[*].message.content[*].image, usage) so the executor's
    output URL validation and usage sanitization still run.
    """

    input_image_count = sum(
        1
        for message in request["input"]["messages"]
        for item in message["content"]
        if "image" in item
    )
    return {
        "status_code": 200,
        "request_id": "dry-run",
        "code": "",
        "message": "",
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [{"image": _DRY_RUN_IMAGE_URL}]
                    }
                }
            ],
            "usage": {
                "output_width": 1280,
                "output_height": 720,
                "input_image_count": input_image_count,
                "input_image_type": "image",
                "output_image_count": 1,
                "output_image_type": "image",
            },
        },
    }


def _dry_run_download(url: str) -> bytes:
    return _DRY_RUN_PNG


def _write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    outcome: str,
    detail: str,
    executor: DashScopeQwenExecutor,
    ports: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phase3-dashscope-manifest-v1",
        "run_id": run_id,
        "stage": "character_synthesis",
        "outcome": outcome,
        "detail": detail,
        "dry_run": dry_run,
        "executor": executor.last_metadata,
        "ports": ports or {},
    }
    if executor.last_request_summary is not None:
        manifest["request_summary"] = executor.last_request_summary
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "WARNING (real mode only): this script makes exactly one PAID "
            "DashScope API "
            "request (qwen-image-3.0-pro). Default text-HOW mode uploads only "
            "the identity image; --visual-how also uploads the pose image. "
            "Run only with images you own or are licensed to use. No "
            "automatic retry. "
            "Pass --dry-run for a no-network local validation run."
        ),
    )
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument(
        "--run-dir",
        default="runs/phase3_dashscope_synthesis",
        type=Path,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument(
        "--visual-how",
        action="store_true",
        help=(
            "upload the pose image as an identity-stripped visual HOW "
            "control; default keeps the pose image local and uses textual HOW"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "no network: run the full chain with a local transport, write a "
            "dry_run manifest, and make no DashScope request"
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        executor = DashScopeQwenExecutor(
            call_fn=_dry_run_call,
            download_fn=_dry_run_download,
        )
        print(
            "DRY RUN: no network, no DashScope request, no cost; the key is "
            "not required.",
            file=sys.stderr,
        )
    else:
        upload_description = (
            "identity and visual-HOW pose images"
            if args.visual_how
            else "the identity image only; the pose image stays local"
        )
        print(
            "WARNING: this run makes one paid DashScope qwen-image-3.0-pro "
            f"request and uploads {upload_description} to Alibaba Cloud "
            "DashScope; no automatic retry.",
            file=sys.stderr,
        )
        executor = DashScopeQwenExecutor()

    identity_ref = "asset://anime-remix/character/asuna@v1"
    pose_ref = "asset://anime-remix/pose/asuna_sitting@v1"
    run_id = f"phase3_dashscope_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    try:
        result = run_compose_keyframe(
            run_dir=args.run_dir,
            run_id=run_id,
            shot_spec=_shot_spec(),
            keyframe_plan=_keyframe_plan(),
            reference_package=_reference_package(identity_ref, pose_ref),
            # orchestrator never consumes the scene before
            # stop_after="character_synthesis"; the value is a placeholder.
            scene_image_path=args.identity,
            scene_geometry=_scene_geometry(),
            scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
            asset_map={
                identity_ref: args.identity,
                pose_ref: args.pose,
            },
            adapter=QwenImage30ProAdapter(
                seed=args.seed,
                size=args.size,
                visual_how=args.visual_how,
            ),
            executor=executor,
            canvas=(1280, 720),
            stop_after="character_synthesis",
        )
    except (EnvironmentCapabilityError, InputValidationError, RenderError) as exc:
        _write_manifest(
            args.run_dir,
            run_id=run_id,
            outcome="error",
            detail=str(exc),
            executor=executor,
            dry_run=args.dry_run,
        )
        print(f"DASHSCOPE SYNTHESIS FAILED: {exc}")
        raise SystemExit(1) from exc

    outcome = "dry_run" if args.dry_run else "success"
    detail = (
        "dry run completed without network; no DashScope request was made"
        if args.dry_run
        else "character_synthesis real run completed; ledger + port bound"
    )
    _write_manifest(
        args.run_dir,
        run_id=run_id,
        outcome=outcome,
        detail=detail,
        executor=executor,
        ports=result.ports,
        dry_run=args.dry_run,
    )
    prefix = "DRY RUN: " if args.dry_run else ""
    print(f"{prefix}character_candidate -> {result.ports['character_candidate']}")
    print(f"ledger -> {result.ledger_path}")
    print(f"{prefix}manifest -> {args.run_dir / 'run-manifest.json'}")


if __name__ == "__main__":
    main()
