from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from anime_remix.errors import InputValidationError, SourceDriftError
from anime_remix.json_io import dump_json_atomic, load_json_object, sha256_file
from anime_remix.services.execution.generated_shot_pipeline import (
    parse_generated_shot_inputs,
    run_generated_shot_pipeline,
)


class FakeToolkit:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.normalize_calls: list[str] = []
        self.concat_calls = 0

    def normalize_generated_source(
        self,
        source: Path,
        output: Path,
        *,
        target_frames: int,
        profile: object | None = None,
    ) -> None:
        del profile
        self.normalize_calls.append(source.parent.name)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic normalization failure")
        output.write_bytes(
            b"normalized:" + source.read_bytes() + f":{target_frames}".encode()
        )

    def validate_segment(
        self,
        path: Path,
        *,
        target_frames: int,
        shot_id: str,
    ) -> dict[str, object]:
        assert path.is_file()
        assert target_frames > 0
        assert shot_id
        return {"codec": "fake-h264", "profile": "stable"}

    def concat_signatures_equal(
        self,
        signatures: list[dict[str, object]],
    ) -> None:
        assert signatures
        assert all(item == signatures[0] for item in signatures)

    def concat_video(
        self,
        segments: list[Path],
        output: Path,
        *,
        durations: list[Decimal] | None = None,
    ) -> None:
        self.concat_calls += 1
        assert durations is not None
        assert len(durations) == len(segments)
        output.write_bytes(
            b"joined:" + b"|".join(path.read_bytes() for path in segments)
        )


def _anchor(
    anchor_id: str,
    *,
    order: int,
    time_seconds: float,
    total_seconds: float,
    previous_id: str | None,
    final: bool,
) -> dict:
    if order == 1:
        return {
            "anchor_id": anchor_id,
            "order": order,
            "time_seconds": 0.0,
            "position": 0.0,
            "roles": ["master_start"],
            "control_reasons": ["shot_start"],
            "target_state": "start",
            "composition": "locked composition",
            "camera": "fixed camera",
            "base_anchor_id": None,
            "generation_method": "approved_first_frame",
            "reference_asset_id": None,
            "reference_role": None,
            "reference_attributes": [],
            "delta_instruction": None,
            "locked_attributes": ["identity"],
            "information_added": [],
            "generation_risk": "low",
            "risk_factors": [],
            "grounding": "approved_first_frame",
        }
    reasons = ["shot_end"] if final else ["model_duration_limit"]
    roles = ["final"] if final else ["continuity"]
    return {
        "anchor_id": anchor_id,
        "order": order,
        "time_seconds": time_seconds,
        "position": time_seconds / total_seconds,
        "roles": roles,
        "control_reasons": reasons,
        "target_state": f"state {order}",
        "composition": "locked composition",
        "camera": "fixed camera",
        "base_anchor_id": previous_id,
        "generation_method": "edit_previous",
        "reference_asset_id": None,
        "reference_role": None,
        "reference_attributes": [],
        "delta_instruction": f"change to state {order}",
        "locked_attributes": ["identity"],
        "information_added": [],
        "generation_risk": "low",
        "risk_factors": [],
        "grounding": "previous_frame_and_action_delta",
    }


def _plan(*durations: float) -> dict:
    total = sum(durations)
    anchors: list[dict] = []
    elapsed = 0.0
    for index in range(len(durations) + 1):
        anchor_id = f"anchor_{index}"
        anchors.append(
            _anchor(
                anchor_id,
                order=index + 1,
                time_seconds=elapsed,
                total_seconds=total,
                previous_id=f"anchor_{index - 1}" if index else None,
                final=index == len(durations),
            )
        )
        if index < len(durations):
            elapsed += durations[index]
    segments = []
    for index, duration in enumerate(durations):
        end = anchors[index + 1]
        segments.append(
            {
                "segment_id": f"segment_{index + 1}",
                "order": index + 1,
                "start_anchor_id": anchors[index]["anchor_id"],
                "end_anchor_id": end["anchor_id"],
                "duration_seconds": duration,
                "process_description": f"motion {index + 1}",
                "dominant_motion": "head turn",
                "camera_motion": "fixed",
                "end_control_reasons": end["control_reasons"],
                "required_visible_fact_ids": [],
                "continuity_mode": "shared_boundary",
            }
        )
    return {
        "schema_version": "generation-segment-plan-v1",
        "plan_id": "shot_001-generation-segments",
        "shot_id": "shot_001",
        "shot_duration_seconds": total,
        "first_frame_plan_id": "shot_001-first-frame",
        "policy": "shared-boundary-reference-first-v1",
        "review_status": "approved",
        "decision": "ready",
        "information_ledger": [],
        "anchors": anchors,
        "segments": segments,
        "warnings": [],
    }


def _write_anchor_manifest(
    root: Path,
    plan: dict,
    *,
    approved_count: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, anchor in enumerate(plan["anchors"]):
        approved = index < approved_count
        output = None
        if approved:
            relative = f"anchors/{anchor['anchor_id']}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"anchor:{anchor['anchor_id']}".encode())
            output = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        records.append(
            {
                "anchor_id": anchor["anchor_id"],
                "status": "completed" if approved else "pending",
                "approved": approved,
                "output": output,
            }
        )
    manifest = root / "handoff-frame-compose-run.json"
    dump_json_atomic(
        manifest,
        {
            "schema_version": "handoff-frame-compose-run-v1",
            "run_id": "handoff-test",
            "plan_id": plan["plan_id"],
            "shot_id": plan["shot_id"],
            "anchors": records,
        },
    )
    return manifest


def _write_video_input(root: Path, segment_id: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / f"{segment_id}.mp4"
    raw.write_bytes(f"raw-video:{segment_id}".encode())
    provider = root / f"{segment_id}.provider.json"
    dump_json_atomic(
        provider,
        {
            "schema_version": "provider-run-v1",
            "run_id": f"provider-{segment_id}",
            "provider": "stub-provider",
            "model": "stub-video-model",
            "human_review": {"status": "accepted"},
            "execution": {"output": {"sha256": sha256_file(raw)}},
        },
    )
    return {
        "segment_id": segment_id,
        "raw_video_path": raw.name,
        "provider_manifest_path": provider.name,
        "human_review": "approved",
    }


def _inputs(*segments: dict) -> dict:
    return {
        "schema_version": "generated-shot-inputs-v1",
        "shot_id": "shot_001",
        "segments": list(segments),
    }


def test_progressive_resume_preserves_shared_anchor_and_completed_work(
    tmp_path: Path,
) -> None:
    plan = _plan(1.0, 1.0)
    anchors_dir = tmp_path / "handoff"
    anchor_manifest = _write_anchor_manifest(
        anchors_dir,
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    first_input = _write_video_input(inputs_dir, "segment_1")
    toolkit = FakeToolkit()
    run_dir = tmp_path / "generated"

    first = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="i7-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(first_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )

    assert first.status == "awaiting_anchors"
    assert first.completed_segment_ids == ("segment_1",)
    assert first.next_segment_id == "segment_2"
    assert len(toolkit.normalize_calls) == 1

    anchor_manifest = _write_anchor_manifest(
        anchors_dir,
        plan,
        approved_count=3,
    )
    second_input = _write_video_input(inputs_dir, "segment_2")
    second = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="i7-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(first_input, second_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )

    assert second.status == "completed"
    assert second.completed_segment_ids == ("segment_1", "segment_2")
    assert second.next_segment_id is None
    assert second.shot_video_path == run_dir / "generated_shot.mp4"
    assert second.shot_video_path.is_file()
    assert len(toolkit.normalize_calls) == 2
    assert toolkit.concat_calls == 1
    manifest = load_json_object(second.manifest_path)
    left, right = manifest["segments"]
    assert left["end_anchor"] == right["start_anchor"]
    assert left["end_anchor"]["anchor_id"] == "anchor_1"
    assert left["end_anchor"]["path"] == "anchors/002_anchor_1/frame.png"
    assert manifest["policy"]["video_model_calls"] == 0
    assert manifest["summary"]["normalization_attempt_count"] == 2

    resumed = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="i7-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(first_input, second_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )
    assert resumed.status == "completed"
    assert len(toolkit.normalize_calls) == 2
    assert toolkit.concat_calls == 1


def test_missing_end_anchor_blocks_video_registration(tmp_path: Path) -> None:
    plan = _plan(1.0, 1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    second_input = _write_video_input(inputs_dir, "segment_2")

    with pytest.raises(InputValidationError, match="before both anchors"):
        run_generated_shot_pipeline(
            run_dir=tmp_path / "run",
            run_id="missing-anchor",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(second_input),
            inputs_base_dir=inputs_dir,
            toolkit=FakeToolkit(),
        )


def test_two_second_boundary_is_enforced(tmp_path: Path) -> None:
    plan = _plan(2.5)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )

    with pytest.raises(InputValidationError, match="verified 2 second"):
        run_generated_shot_pipeline(
            run_dir=tmp_path / "run",
            run_id="too-long",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(),
            toolkit=FakeToolkit(),
        )


def test_registered_raw_video_drift_is_rejected(tmp_path: Path) -> None:
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    segment_input = _write_video_input(inputs_dir, "segment_1")
    run_dir = tmp_path / "run"
    run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="drift-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment_input),
        inputs_base_dir=inputs_dir,
        toolkit=FakeToolkit(),
    )
    (inputs_dir / "segment_1.mp4").write_bytes(b"changed")

    with pytest.raises(SourceDriftError, match="provider manifest output"):
        run_generated_shot_pipeline(
            run_dir=run_dir,
            run_id="drift-test",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(segment_input),
            inputs_base_dir=inputs_dir,
            toolkit=FakeToolkit(),
        )


def test_failed_local_normalization_requires_explicit_retry(tmp_path: Path) -> None:
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    segment_input = _write_video_input(inputs_dir, "segment_1")
    toolkit = FakeToolkit(fail_once=True)
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="synthetic normalization"):
        run_generated_shot_pipeline(
            run_dir=run_dir,
            run_id="retry-test",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(segment_input),
            inputs_base_dir=inputs_dir,
            toolkit=toolkit,
        )
    failed = load_json_object(run_dir / "generated_shot_manifest.json")
    assert failed["status"] == "failed"
    assert failed["segments"][0]["normalization_attempts"] == 1

    paused = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="retry-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )
    assert paused.status == "failed"
    assert len(toolkit.normalize_calls) == 1

    completed = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="retry-test",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
        retry_failed_segment_ids={"segment_1"},
    )
    assert completed.status == "completed"
    assert len(toolkit.normalize_calls) == 2
    final = load_json_object(completed.manifest_path)
    assert final["segments"][0]["normalization_attempts"] == 2


def test_zero_normalization_budget_pauses_without_work(tmp_path: Path) -> None:
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    segment_input = _write_video_input(inputs_dir, "segment_1")
    toolkit = FakeToolkit()

    result = run_generated_shot_pipeline(
        run_dir=tmp_path / "run",
        run_id="preflight",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment_input),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
        max_new_normalizations=0,
    )

    assert result.status == "paused_limit"
    assert result.completed_segment_ids == ()
    assert toolkit.normalize_calls == []


def test_one_normalization_per_invocation_is_recoverable(tmp_path: Path) -> None:
    plan = _plan(1.0, 1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=3,
    )
    inputs_dir = tmp_path / "inputs"
    first = _write_video_input(inputs_dir, "segment_1")
    second = _write_video_input(inputs_dir, "segment_2")
    toolkit = FakeToolkit()
    run_dir = tmp_path / "run"

    paused = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="limited",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(first, second),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )
    assert paused.status == "paused_limit"
    assert paused.completed_segment_ids == ("segment_1",)
    assert len(toolkit.normalize_calls) == 1

    completed = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="limited",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(first, second),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )
    assert completed.status == "completed"
    assert len(toolkit.normalize_calls) == 2


def test_interrupted_normalization_requires_explicit_retry(tmp_path: Path) -> None:
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    segment = _write_video_input(inputs_dir, "segment_1")
    toolkit = FakeToolkit()
    run_dir = tmp_path / "run"
    run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="interrupted",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
        max_new_normalizations=0,
    )
    manifest_path = run_dir / "generated_shot_manifest.json"
    manifest = load_json_object(manifest_path)
    manifest["segments"][0]["status"] = "normalizing"
    manifest["segments"][0]["normalization_attempts"] = 1
    dump_json_atomic(manifest_path, manifest, sort_keys=True)

    stopped = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="interrupted",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
    )
    assert stopped.status == "failed"
    assert toolkit.normalize_calls == []

    completed = run_generated_shot_pipeline(
        run_dir=run_dir,
        run_id="interrupted",
        plan=plan,
        anchor_manifest_path=anchor_manifest,
        inputs=_inputs(segment),
        inputs_base_dir=inputs_dir,
        toolkit=toolkit,
        retry_failed_segment_ids={"segment_1"},
    )
    assert completed.status == "completed"
    assert len(toolkit.normalize_calls) == 1


def test_provider_manifest_must_record_human_acceptance(tmp_path: Path) -> None:
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )
    inputs_dir = tmp_path / "inputs"
    segment = _write_video_input(inputs_dir, "segment_1")
    provider_path = inputs_dir / str(segment["provider_manifest_path"])
    provider = load_json_object(provider_path)
    provider["human_review"]["status"] = "pending"
    dump_json_atomic(provider_path, provider)

    with pytest.raises(InputValidationError, match="accepted human review"):
        run_generated_shot_pipeline(
            run_dir=tmp_path / "run",
            run_id="unreviewed",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(segment),
            inputs_base_dir=inputs_dir,
            toolkit=FakeToolkit(),
        )


def test_generated_shot_inputs_are_strict_and_unique() -> None:
    segment = {
        "segment_id": "segment_1",
        "raw_video_path": "raw.mp4",
        "provider_manifest_path": "provider.json",
        "human_review": "approved",
    }
    with pytest.raises(InputValidationError, match="ids must be unique"):
        parse_generated_shot_inputs(_inputs(segment, segment))
    invalid = _inputs(segment)
    invalid["unknown"] = True
    with pytest.raises(InputValidationError, match="invalid generated-shot"):
        parse_generated_shot_inputs(invalid)


def test_symlink_run_directory_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real-run"
    target.mkdir()
    linked = tmp_path / "linked-run"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlink on this Windows host: {exc}")
    plan = _plan(1.0)
    anchor_manifest = _write_anchor_manifest(
        tmp_path / "handoff",
        plan,
        approved_count=2,
    )

    with pytest.raises(InputValidationError, match="real directory"):
        run_generated_shot_pipeline(
            run_dir=linked,
            run_id="symlink-run",
            plan=plan,
            anchor_manifest_path=anchor_manifest,
            inputs=_inputs(),
            toolkit=FakeToolkit(),
        )
