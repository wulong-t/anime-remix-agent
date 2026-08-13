"""Phase 3 Real: full compose keyframe chain with the Replicate executor.

Expected Phase 3 Real outcome: ``character_synthesis`` runs on the
Replicate-hosted ``qwen/qwen-image-edit-2511`` and every deterministic stage
(extract / validate / geometry / layout / mask / composite) records real
artifacts; then ``local_inpaint`` stops with an EnvironmentCapabilityError
because the hosted API cannot consume an independent mask (finding F-013).
The script writes a run manifest and reports BLOCKED instead of pretending an
unconstrained edit happened.

Usage::

    python experiments/phase3/run_replicate.py \
      --identity path/to/identity.png \
      --pose path/to/pose.png \
      --scene path/to/scene.png \
      --run-dir runs/phase3_replicate --seed 0

Requires the ``REPLICATE_API_TOKEN`` environment variable (never logged).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from run_remote import (
    _keyframe_plan,
    _reference_package,
    _scene_geometry,
    _shot_spec,
)

from anime_remix.errors import EnvironmentCapabilityError, RenderError
from anime_remix.services.execution.adapter import QwenImageEditAdapter
from anime_remix.services.execution.orchestrator import run_compose_keyframe
from anime_remix.services.execution.replicate_executor import (
    REPLICATE_TOKEN_ENV,
    ReplicateQwenExecutor,
)


def _write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    outcome: str,
    detail: str,
    executor: ReplicateQwenExecutor,
    ports: dict[str, str] | None = None,
    final_keyframe_ref: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phase3-replicate-manifest-v1",
        "run_id": run_id,
        "stage": "full_compose_keyframe",
        "outcome": outcome,
        "detail": detail,
        "executor": executor.last_metadata,
        "ports": ports or {},
        "final_keyframe_ref": final_keyframe_ref,
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--geometry-json", type=Path, default=None)
    parser.add_argument("--run-dir", default="runs/phase3_replicate", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--temp-dir", type=Path, default=None)
    args = parser.parse_args()

    if not os.environ.get(REPLICATE_TOKEN_ENV):
        raise SystemExit(
            f"missing {REPLICATE_TOKEN_ENV}: set it before running the "
            "Replicate experiment (the token is never logged or stored)"
        )

    scene_ref = "asset://anime-remix/scene/classroom_01@v1"
    identity_ref = "asset://anime-remix/character/asuna@v1"
    pose_ref = "asset://anime-remix/pose/asuna_sitting@v1"
    scene_geometry = (
        json.loads(args.geometry_json.read_text(encoding="utf-8"))
        if args.geometry_json is not None
        else _scene_geometry()
    )
    run_id = f"phase3_replicate_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    executor = ReplicateQwenExecutor(temp_dir=args.temp_dir)
    try:
        result = run_compose_keyframe(
            run_dir=args.run_dir,
            run_id=run_id,
            shot_spec=_shot_spec(),
            keyframe_plan=_keyframe_plan(),
            reference_package=_reference_package(identity_ref, pose_ref),
            scene_image_path=args.scene,
            scene_geometry=scene_geometry,
            scene_crop={"source_rect": [0.0, 0.0, 1.0, 1.0]},
            asset_map={
                scene_ref: args.scene,
                identity_ref: args.identity,
                pose_ref: args.pose,
            },
            adapter=QwenImageEditAdapter(seed=args.seed, steps=args.steps),
            executor=executor,
            canvas=(1280, 720),
        )
    except EnvironmentCapabilityError as exc:
        message = str(exc)
        if "F-013" in message:
            _write_manifest(
                args.run_dir,
                run_id=run_id,
                outcome="blocked_local_inpaint",
                detail=message,
                executor=executor,
            )
            print("PHASE 3 REAL: BLOCKED at local_inpaint (finding F-013)")
            print(message)
            raise SystemExit(2) from exc
        _write_manifest(
            args.run_dir,
            run_id=run_id,
            outcome="error",
            detail=message,
            executor=executor,
        )
        print(f"PHASE 3 REAL: ERROR -> {message}")
        raise SystemExit(1) from exc
    except RenderError as exc:
        _write_manifest(
            args.run_dir,
            run_id=run_id,
            outcome="error",
            detail=str(exc),
            executor=executor,
        )
        print(f"PHASE 3 REAL: ERROR -> {exc}")
        raise SystemExit(1) from exc

    _write_manifest(
        args.run_dir,
        run_id=run_id,
        outcome="success",
        detail="full compose keyframe completed with real Replicate model",
        executor=executor,
        ports=result.ports,
        final_keyframe_ref=result.final_keyframe_ref,
    )
    print(f"final_keyframe -> {result.final_keyframe_ref}")
    print(f"ledger -> {result.ledger_path}")
    print(f"manifest -> {args.run_dir / 'run-manifest.json'}")


if __name__ == "__main__":
    main()
