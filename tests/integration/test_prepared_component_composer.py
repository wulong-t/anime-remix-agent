"""Offline integration tests for recoverable WHO/HOW component generation."""

from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError, RenderError
from anime_remix.services.execution.adapter import QwenImage30Adapter
from anime_remix.services.execution.prepared_component_composer import (
    run_prepared_component_composition,
)
from anime_remix.services.script.prepared_component_plan import (
    parse_prepared_component_plan,
)

runner = CliRunner()


def _png(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 48), colour).save(output, format="PNG")
    return output.getvalue()


class RecordingExecutor:
    provider = "offline-recording"

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls: list[dict] = []
        self.last_metadata: dict | None = None

    def execute(
        self,
        *,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        self.calls.append(
            {
                "request": copy.deepcopy(request_payload),
                "operation": operation,
                "inputs": dict(inputs),
            }
        )
        if self.fail_on_call == len(self.calls):
            self.last_metadata = {
                "provider": "offline-recording",
                "request_id": f"failed-{len(self.calls)}",
                "status": "failed",
            }
            raise RenderError("injected component failure")
        self.last_metadata = {
            "provider": "offline-recording",
            "model": "stub",
            "request_id": f"success-{len(self.calls)}",
            "status": "succeeded",
        }
        return _png((40 * len(self.calls), 90, 130))


def _plan(*, approved: bool = True, two_tasks: bool = True) -> dict:
    tasks = [
        {
            "task_id": "task.character.character_001",
            "kind": "character_plate",
            "component_ids": ["character_001"],
            "subjects": ["Mira"],
            "target_state": "Mira kneels with eyes closed",
            "model_inputs": [
                {"asset_id": "mira_who", "function": "who"},
                {"asset_id": "mira_how", "function": "how"},
            ],
            "control_evidence_asset_ids": [],
            "preserve_attributes": ["face", "hair", "outfit", "native style"],
            "allowed_text_fallbacks": {
                "spatial_placement": "left foreground"
            },
            "output_asset_id": "prep_mira",
            "result": "pending",
            "result_review_notes": None,
        }
    ]
    if two_tasks:
        tasks.append(
            {
                "task_id": "task.interaction.mira_grips_key",
                "kind": "interaction_plate",
                "component_ids": ["character_001", "prop_001"],
                "subjects": ["Mira", "red key"],
                "target_state": "Mira's fingers visibly wrap around the red key",
                "model_inputs": [
                    {"asset_id": "mira_who", "function": "who"},
                    {"asset_id": "key_visual", "function": "prop_visual"},
                ],
                "control_evidence_asset_ids": ["layout_only"],
                "preserve_attributes": ["Mira identity", "red key appearance"],
                "allowed_text_fallbacks": {
                    "contact_relation": (
                        "Mira's fingers visibly wrap around the red key"
                    ),
                    "prop_affordance": (
                        "grip only at circular bow; active end is shaft tip"
                    ),
                    "action_axis": "shaft tip follows right-to-left",
                    "external_target": "approaches the empty reactor socket",
                    "required_visibility": "shaft tip, empty socket, insertion gap",
                },
                "prop_affordances": [
                    {
                        "component_id": "prop_001",
                        "subject": "red key",
                        "grip_zone": "circular bow",
                        "active_end": "shaft tip",
                        "native_action_axis": "bow-to-shaft",
                    }
                ],
                "external_attachments": [
                    {
                        "attachment_id": "key_approaches_socket",
                        "source_component_id": "prop_001",
                        "source_anchor": "shaft tip",
                        "source_anchor_x": 0.46,
                        "source_anchor_y": 0.56,
                        "target_component_id": "scene",
                        "target_subject": "observatory",
                        "target_anchor": "empty socket",
                        "target_anchor_x": 0.42,
                        "target_anchor_y": 0.56,
                        "relation": "approaches",
                        "action_axis": "right-to-left",
                        "initial_gap": "four percent of frame width",
                        "required_visible_state": "tip points toward socket",
                        "must_remain_visible": [
                            "shaft tip",
                            "empty socket",
                            "insertion gap",
                        ],
                        "source_must_remain_visible": ["shaft tip"],
                        "target_must_remain_visible": [
                            "empty socket",
                            "insertion gap",
                        ],
                        "hard_gate": True,
                    }
                ],
                "review_gates": [
                    {
                        "gate_id": "gate.attachment.key_approaches_socket",
                        "criterion": "Confirm the key tip approaches the visible socket.",
                    }
                ],
                "output_asset_id": "prep_mira_key",
                "result": "pending",
                "result_review_notes": None,
                "gate_results": [],
            }
        )
    source_ids = ["mira_who", "mira_how"]
    if two_tasks:
        source_ids.extend(["key_visual", "layout_only"])
    return {
        "schema_version": "prepared-component-plan-v1",
        "plan_id": "shot_001-prepared-components",
        "shot_id": "shot_001",
        "content_plan_id": "shot_001-first-frame-content",
        "review_status": "approved" if approved else "draft",
        "completion_status": "pending",
        "decision": "ready",
        "max_primary_visual_references_per_model_call": 2,
        "source_asset_ids": source_ids,
        "tasks": tasks,
        "warnings": [],
    }


def _assets(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for index, asset_id in enumerate(
        ("mira_who", "mira_how", "key_visual"), start=1
    ):
        path = tmp_path / f"{asset_id}.png"
        path.write_bytes(_png((20 * index, 30, 40)))
        result[asset_id] = path
    return result


def _manifest(tmp_path: Path) -> dict:
    return json.loads(
        (tmp_path / "run" / "prepared-component-compose-run.json").read_text(
            encoding="utf-8"
        )
    )


def test_runner_pauses_after_one_task_and_resumes_without_control_pixels(
    tmp_path: Path,
) -> None:
    assets = _assets(tmp_path)
    executor = RecordingExecutor()
    adapter = QwenImage30Adapter(size="512*512")

    first = run_prepared_component_composition(
        run_dir=tmp_path / "run",
        run_id="component-run-001",
        plan=_plan(),
        asset_map=assets,
        adapter=adapter,
        executor=executor,
    )

    assert first.status == "paused_limit"
    assert first.completed_task_ids == ("task.character.character_001",)
    assert first.next_task_id == "task.interaction.mira_grips_key"
    assert len(executor.calls) == 1
    request = executor.calls[0]["request"]
    assert request["adapter_id"].endswith("synthesize_component-identity-pose")
    assert list(executor.calls[0]["inputs"]) == [
        "cond_task_character_character_001_1",
        "cond_task_character_character_001_2",
    ]
    assert "component plate" in request["prompt"]
    assert "action, pose and expression geometry only" in request["prompt"]
    assert "Mira kneels" not in request["prompt"]
    manifest = _manifest(tmp_path)
    assert manifest["control_evidence"] == {
        "asset_ids": ["layout_only"],
        "resolved_as_model_input": False,
        "uploaded_to_model": False,
    }

    second = run_prepared_component_composition(
        run_dir=tmp_path / "run",
        run_id="component-run-001",
        plan=_plan(),
        asset_map=assets,
        adapter=adapter,
        executor=executor,
    )

    assert second.status == "completed"
    assert second.next_task_id is None
    assert len(executor.calls) == 2
    assert executor.calls[1]["request"]["adapter_id"].endswith(
        "synthesize_component-identity-prop"
    )
    assert "right-to-left" in executor.calls[1]["request"]["prompt"]
    assert "empty reactor socket" in executor.calls[1]["request"]["prompt"]
    assert set(second.output_paths) == {
        ("prep_mira", tmp_path / "run" / "candidates" / "prep_mira.png"),
        (
            "prep_mira_key",
            tmp_path / "run" / "candidates" / "prep_mira_key.png",
        ),
    }
    assert all(path.is_file() for _, path in second.output_paths)
    assert all(
        item["requires_manual_review"] for item in _manifest(tmp_path)["tasks"]
    )
    interaction_manifest = _manifest(tmp_path)["tasks"][1]
    assert interaction_manifest["review_criteria"]["structured_gates"] == [
        {
            "criterion": "Confirm the key tip approaches the visible socket.",
            "gate_id": "gate.attachment.key_approaches_socket",
        }
    ]
    assert interaction_manifest["review_criteria"]["external_attachments"][0][
        "target_anchor"
    ] == "empty socket"


def test_manifest_review_criteria_are_plate_scoped(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    executor = RecordingExecutor()
    run_prepared_component_composition(
        run_dir=tmp_path / "run",
        run_id="component-review-scope",
        plan=_plan(),
        asset_map=assets,
        adapter=QwenImage30Adapter(size="512*512"),
        executor=executor,
        max_new_model_tasks=2,
    )
    manifest = _manifest(tmp_path)
    criteria = manifest["tasks"][1]["review_criteria"]
    assert "plate_scope" in criteria
    assert "must be absent from the plate" in criteria["plate_scope"]
    reject = " ".join(criteria["reject"])
    assert "target scene object rendered in the plate" in reject
    assert "hidden external target" not in reject


def test_failure_is_recorded_and_retried_only_on_explicit_resume(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(fail_on_call=1)
    kwargs = {
        "run_dir": tmp_path / "run",
        "run_id": "component-run-failure",
        "plan": _plan(two_tasks=False),
        "asset_map": _assets(tmp_path),
        "adapter": QwenImage30Adapter(size="512*512"),
        "executor": executor,
    }

    with pytest.raises(RenderError, match="injected"):
        run_prepared_component_composition(**kwargs)
    failed = _manifest(tmp_path)
    assert failed["status"] == "failed"
    assert failed["tasks"][0]["attempts"] == 1
    assert failed["tasks"][0]["request"]["provider"]["status"] == "failed"

    resumed = run_prepared_component_composition(**kwargs)
    assert resumed.status == "completed"
    assert _manifest(tmp_path)["tasks"][0]["attempts"] == 2


def test_runner_rejects_draft_plan_and_completed_output_drift(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    kwargs = {
        "run_dir": tmp_path / "run",
        "run_id": "component-run-contract",
        "asset_map": assets,
        "adapter": QwenImage30Adapter(size="512*512"),
        "executor": RecordingExecutor(),
    }
    with pytest.raises(InputValidationError, match="must be approved"):
        run_prepared_component_composition(plan=_plan(approved=False), **kwargs)

    result = run_prepared_component_composition(
        plan=_plan(two_tasks=False), **kwargs
    )
    result.output_paths[0][1].write_bytes(_png((255, 0, 0)))
    with pytest.raises(InputValidationError, match="drifted"):
        run_prepared_component_composition(
            plan=_plan(two_tasks=False), **kwargs
        )


def test_cli_stub_runs_offline_and_dashscope_requires_paid_confirmation(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    assets = []
    for asset_id, asset_type, subject in (
        ("mira_who", "character", "Mira"),
        ("mira_how", "character", "Mira"),
        ("key_visual", "prop", "red key"),
        ("layout_only", "character", "Mira and red key"),
    ):
        path = image_dir / f"{asset_id}.png"
        path.write_bytes(_png((30, 40, 50)))
        assets.append(
            {
                "asset_id": asset_id,
                "path": f"images/{asset_id}.png",
                "asset_type": asset_type,
                "rights_status": "user-owned",
                "subject_or_scene_id": subject,
                "analysis_status": "analyzed",
            }
        )
    catalog_path = tmp_path / "image_assets.json"
    catalog_path.write_text(
        json.dumps({"schema_version": "image-assets-v1", "assets": assets}),
        encoding="utf-8",
    )
    plan_path = tmp_path / "prepared_plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")

    rejected = runner.invoke(
        app,
        [
            "director",
            "component-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(tmp_path / "paid-run"),
            "--run-id",
            "paid-run",
            "--executor",
            "dashscope",
        ],
    )
    assert rejected.exit_code != 0
    assert "--confirm-paid" in rejected.output
    assert not (tmp_path / "paid-run").exists()

    offline = runner.invoke(
        app,
        [
            "director",
            "component-compose",
            "--plan",
            str(plan_path),
            "--image-assets",
            str(catalog_path),
            "--run-dir",
            str(tmp_path / "offline-run"),
            "--run-id",
            "offline-run",
            "--executor",
            "stub",
            "--size",
            "512*512",
        ],
    )
    assert offline.exit_code == 0, offline.output
    assert "status=paused_limit" in offline.output
    assert (tmp_path / "offline-run" / "candidates" / "prep_mira.png").is_file()


def test_plan_parser_fixture_is_strict() -> None:
    assert parse_prepared_component_plan(_plan()).review_status == "approved"
