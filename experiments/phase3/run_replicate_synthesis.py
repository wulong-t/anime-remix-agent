"""Phase 3 Real, work package 4: character_synthesis only via Replicate.

Runs the frozen compose chain up to and including the real
``character_synthesis`` model node (WHO + HOW images + compiled prompt) on
the Replicate-hosted ``qwen/qwen-image-edit-2511``.  The full synthesis
lifecycle (node_run_started -> render_intent_created ->
model_render_request_created -> render_attempt_started -> artifact_registered
-> render_attempt_finished -> node_run_finished -> port_bound) is recorded in
the Execution Ledger, then the run finishes without touching the
deterministic / inpaint stages.

Usage::

    python experiments/phase3/run_replicate_synthesis.py \
      --identity path/to/identity.png \
      --pose path/to/pose.png \
      --run-dir runs/phase3_replicate_synthesis --seed 0

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
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phase3-replicate-manifest-v1",
        "run_id": run_id,
        "stage": "character_synthesis",
        "outcome": outcome,
        "detail": detail,
        "executor": executor.last_metadata,
        "ports": ports or {},
    }
    (run_dir / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--run-dir", default="runs/phase3_replicate_synthesis", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--temp-dir", type=Path, default=None)
    args = parser.parse_args()

    if not os.environ.get(REPLICATE_TOKEN_ENV):
        raise SystemExit(
            f"missing {REPLICATE_TOKEN_ENV}: set it before running the "
            "Replicate experiment (the token is never logged or stored)"
        )

    identity_ref = "asset://anime-remix/character/asuna@v1"
    pose_ref = "asset://anime-remix/pose/asuna_sitting@v1"
    run_id = f"phase3_replicate_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    executor = ReplicateQwenExecutor(temp_dir=args.temp_dir)
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
            adapter=QwenImageEditAdapter(seed=args.seed, steps=args.steps),
            executor=executor,
            canvas=(1280, 720),
            stop_after="character_synthesis",
        )
    except (EnvironmentCapabilityError, RenderError) as exc:
        _write_manifest(
            args.run_dir,
            run_id=run_id,
            outcome="error",
            detail=str(exc),
            executor=executor,
        )
        print(f"REPLICATE SYNTHESIS FAILED: {exc}")
        raise SystemExit(1) from exc

    _write_manifest(
        args.run_dir,
        run_id=run_id,
        outcome="success",
        detail="character_synthesis real run completed; ledger + port bound",
        executor=executor,
        ports=result.ports,
    )
    print(f"character_candidate -> {result.ports['character_candidate']}")
    print(f"ledger -> {result.ledger_path}")
    print(f"manifest -> {args.run_dir / 'run-manifest.json'}")


if __name__ == "__main__":
    main()
