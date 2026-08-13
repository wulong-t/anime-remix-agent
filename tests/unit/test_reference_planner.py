"""KF-PRODUCT-1 Reference Planner / Condition Compiler tests.

All images are synthetic PNG bytes; every catalog is built under
``tmp_path``.  The suite pins the I4-I7 product constraint: at most two
visual references (WHO + HOW), everything else textified.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from anime_remix.cli import app
from anime_remix.errors import InputValidationError
from anime_remix.services.image_assets import (
    ImageAssetCatalog,
    load_image_assets,
)
from anime_remix.services.script.reference_planner import (
    KeyframeGenerationPlan,
    _semantic_issues,
    plan_keyframe_generation,
    write_generation_artifacts,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry, parse_shot_plan

_REFERENCE_PLANNER_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "reference_planner_golden"
)
if str(_REFERENCE_PLANNER_GOLDEN) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_PLANNER_GOLDEN))
from generate_golden_i7 import (
    GOLDEN_OUTPUT,
    GOLDEN_PROMPT,
    build_golden_catalog,
    build_golden_shot,
    golden_bundle,
    normalize_plan_paths,
)

runner = CliRunner()

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png(width: int, height: int) -> bytes:
    """Minimal valid PNG bytes for a synthetic fixture."""

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        _PNG_MAGIC
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _asset(
    asset_id: str,
    asset_type: str = "character",
    *,
    roles: list[str] | None = None,
    tier: str = "canonical",
    analysis_status: str = "pending",
    **overrides: Any,
) -> dict[str, Any]:
    """One image_assets.json entry with deterministic defaults."""

    data: dict[str, Any] = {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "subject_or_scene_id": asset_id,
        "quality_notes": f"参考：{asset_id}",
        "rights_status": "synthetic-fixture",
        "source_tier": tier,
        "reference_roles": roles or [],
        "analysis_status": analysis_status,
    }
    data.update(overrides)
    return data


def _write_catalog(tmp_path: Path, assets: list[dict[str, Any]]) -> ImageAssetCatalog:
    """Write synthetic images + manifest and load the strict catalog."""

    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    for asset in assets:
        entry = dict(asset)
        entry.setdefault("path", f"images/{asset['asset_id']}.png")
        (images / f"{asset['asset_id']}.png").write_bytes(_png(24, 24))
        entries.append(entry)
    manifest = tmp_path / "image_assets.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": "image-assets-v1", "assets": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return load_image_assets(manifest)


def _bundle(*asset_ids: str) -> list[dict[str, str]]:
    """ReferenceBundle entries shaped like validate_binding_against_plan."""

    return [
        {
            "asset_id": asset_id,
            "asset_type": "character",
            "path": "irrelevant.png",
            "note": "synthetic bundle entry",
        }
        for asset_id in asset_ids
    ]


def _shot(**overrides: Any) -> ShotPlanEntry:
    """One strict ShotPlanEntry with deterministic defaults."""

    data: dict[str, Any] = {
        "shot_id": "shot_001",
        "scene_id": "scene_01",
        "order": 1,
        "narrative_purpose": "测试镜头",
        "duration_seconds": 4.0,
        "shot_scale": "medium",
        "composition": "林夏居中，背景天台",
        "camera_position": "正面平视",
        "camera_motion": "fixed",
        "subjects": ["林夏"],
        "setting": "黄昏天台",
        "props": [],
        "start_state": "林夏站在栏杆旁",
        "action_beats": [{"time_seconds": 0.0, "description": "林夏抬头看向远方"}],
        "end_state": "林夏望向远方",
        "emotion_arc": "平静",
    }
    data.update(overrides)
    document = parse_shot_plan({"schema_version": "shot-plan-v1", "shots": [data]})
    return document.shots[0]


def _ref_ids(plan: KeyframeGenerationPlan) -> list[str]:
    return [ref.asset_id for ref in plan.visual_refs]


def _ref_slots(plan: KeyframeGenerationPlan) -> list[str]:
    return [ref.slot for ref in plan.visual_refs]


def test_i7_regression_slot_assignment(tmp_path: Path) -> None:
    """I7 oracle: P1=F1, P2=k_end, F3 textified, exactly two visual refs."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                subject_or_scene_id="loc_night_rooftop",
                quality_notes="夜间城市住宅建筑外景；wide establishing；暖黄窗灯",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert plan.decision == "READY"
    assert _ref_ids(plan) == ["F1", "k_end"]
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    assert plan.visual_refs[0].tier == "canonical"
    assert plan.visual_refs[0].confidence == "high"
    assert "F3" not in _ref_ids(plan)
    assert "loc_night_rooftop" not in plan.prompt
    assert "wide establishing" in plan.text_constraints["SCENE"]
    assert "夜间城市住宅建筑外景" in plan.text_constraints["SCENE"]
    assert "F3" not in plan.prompt
    assert "F3" in plan.source_asset_ids
    assert plan.source_asset_ids == ["F1", "F3", "k_end"]
    assert set(plan.source_hashes) == {"F1", "F3", "k_end"}


def test_visual_refs_never_exceed_two(tmp_path: Path) -> None:
    """Hard slot rule: many WHO/HOW candidates still yield <= 2 refs."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("A", roles=["identity_reference"]),
            _asset("B", roles=["identity_reference"]),
            _asset("C", roles=["identity_reference"]),
            _asset("D", roles=["pose_reference"]),
            _asset("E", roles=["expression_reference"]),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(),
        _bundle("A", "B", "C", "D", "E"),
        catalog,
    )

    assert len(plan.visual_refs) <= 2
    assert len(set(_ref_slots(plan))) == len(plan.visual_refs)
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]


def test_tier_priority_for_identity(tmp_path: Path) -> None:
    """P1 prefers canonical over derived over generated_candidate."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1_candidate", roles=["identity_reference"], tier="generated_candidate"),
            _asset("F1_derived", roles=["identity_reference"], tier="derived"),
            _asset("F1", roles=["identity_reference"], tier="canonical"),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(),
        _bundle("F1_candidate", "F1_derived", "F1"),
        catalog,
    )

    assert _ref_ids(plan) == ["F1"]
    assert plan.visual_refs[0].tier == "canonical"
    assert plan.decision == "READY"


def test_pose_and_expression_same_asset_priority(tmp_path: Path) -> None:
    """P2 prefers one asset carrying both pose and expression roles."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset("pose_only", roles=["pose_reference"]),
            _asset(
                "both_roles",
                roles=["pose_reference", "expression_reference"],
            ),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(),
        _bundle("F1", "pose_only", "both_roles"),
        catalog,
    )

    assert _ref_ids(plan) == ["F1", "both_roles"]
    assert plan.visual_refs[1].slot == "p2_pose_expression"


def test_scene_metadata_compiled_to_text(tmp_path: Path) -> None:
    """Scene/camera/composition/style facts live in text, never a visual slot."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                subject_or_scene_id="loc_rooftop",
                quality_notes="黄昏天台；栏杆；晚霞",
                time_of_day="dusk",
            ),
            _asset(
                "style_watercolor",
                "style",
                roles=["style_reference"],
                quality_notes="水彩风格",
            ),
        ],
    )
    shot = _shot(setting="黄昏的学校天台，晚霞下的城市")
    plan = plan_keyframe_generation(shot, _bundle("F1", "F3", "style_watercolor"), catalog)

    assert "黄昏的学校天台，晚霞下的城市" in plan.text_constraints["SCENE"]
    assert "loc_rooftop" not in plan.prompt
    assert "黄昏天台；栏杆；晚霞" in plan.text_constraints["SCENE"]
    assert "dusk" in plan.text_constraints["LIGHTING"]
    assert "水彩风格" in plan.text_constraints["STYLE"]
    assert "shot_scale: medium" in plan.text_constraints["CAMERA"]
    assert "正面平视" in plan.text_constraints["CAMERA"]
    assert plan.text_constraints["COMPOSITION"] == shot.composition
    assert "F3" not in _ref_ids(plan)


def test_generated_candidate_does_not_beat_canonical(tmp_path: Path) -> None:
    """Trusted canonical P1 wins even when the candidate appears first."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("cand", roles=["identity_reference"], tier="generated_candidate"),
            _asset("F1", roles=["identity_reference"], tier="canonical"),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("cand", "F1"), catalog)

    assert _ref_ids(plan) == ["F1"]
    assert plan.visual_refs[0].tier == "canonical"
    assert plan.decision == "READY"


def test_ambiguous_who_requires_review(tmp_path: Path) -> None:
    """Two indistinguishable identity candidates -> NEEDS_REVIEW."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("char_a", roles=["identity_reference"], subject_or_scene_id="hero"),
            _asset("char_b", roles=["identity_reference"], subject_or_scene_id="hero"),
        ],
    )
    plan = plan_keyframe_generation(_shot(subjects=["hero"]), _bundle("char_a", "char_b"), catalog)

    assert plan.decision == "NEEDS_REVIEW"
    assert _ref_ids(plan) == ["char_a"]
    assert plan.visual_refs[0].confidence == "low"
    assert "ambiguous identity" in plan.visual_refs[0].selection_reason


def test_missing_who_is_unresolved(tmp_path: Path) -> None:
    """Shot requires a character but no identity asset exists -> UNRESOLVED."""

    catalog = _write_catalog(
        tmp_path,
        [_asset("F3", "background", roles=["scene_reference"])],
    )
    plan = plan_keyframe_generation(_shot(subjects=["林夏"]), _bundle("F3"), catalog)

    assert plan.decision == "UNRESOLVED"
    assert plan.visual_refs == []
    assert "缺少身份参考图" in plan.prompt


def test_plan_is_deterministic(tmp_path: Path) -> None:
    """Same inputs produce byte-identical plans and hashes."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset("F3", "background", roles=["scene_reference"]),
        ],
    )
    shot = _shot()
    bundle = _bundle("F1", "k_end", "F3")
    first = plan_keyframe_generation(shot, bundle, catalog)
    second = plan_keyframe_generation(shot, bundle, catalog)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.source_hashes == second.source_hashes


def test_prompt_is_deterministic_and_i7_shaped(tmp_path: Path) -> None:
    """Prompt compile is deterministic and keeps the I7 WHO/HOW/text shape."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="夜间城市住宅建筑外景",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)
    again = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert plan.prompt == again.prompt
    assert "Picture 1" in plan.prompt
    assert "Picture 2" in plan.prompt
    assert "SCENE (WHERE)" in plan.prompt
    assert "CAMERA" in plan.prompt
    assert "夜间城市住宅建筑外景" in plan.prompt
    assert "林夏" in plan.prompt


def test_generated_output_registers_as_candidate(tmp_path: Path) -> None:
    """Artifacts carry lineage; register-candidate records generated_candidate."""

    catalog = _write_catalog(tmp_path, [_asset("F1", roles=["identity_reference"])])
    plan = plan_keyframe_generation(_shot(), _bundle("F1"), catalog)
    artifact_dir = tmp_path / "artifacts"
    paths = write_generation_artifacts(artifact_dir, plan)

    generated = tmp_path / "generated.png"
    generated.write_bytes(_png(16, 16))
    result = runner.invoke(
        app,
        [
            "assets",
            "register-candidate",
            "--catalog",
            str(tmp_path / "image_assets.json"),
            "--paths",
            str(generated),
            "--generated-from",
            "F1",
        ],
    )
    assert result.exit_code == 0, result.output

    updated = load_image_assets(tmp_path / "image_assets.json")
    candidate = updated.get("generated")
    assert candidate is not None
    assert candidate.source_tier == "generated_candidate"
    assert candidate.provenance is not None
    assert candidate.provenance["parent_asset_id"] == "F1"
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    assert request["model"] == "Qwen-Image-Edit-2511"
    assert request["sampling"]["seed"] == 0
    assert request["source_asset_ids"] == ["F1"]
    assert request["source_hashes"]["F1"] == plan.source_hashes["F1"]


def test_unapproved_candidate_is_not_auto_promoted(tmp_path: Path) -> None:
    """A generated_candidate-only WHO stays NEEDS_REVIEW and unpromoted."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("cand", roles=["identity_reference"], tier="generated_candidate"),
            _asset("F3", "background", roles=["scene_reference"]),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("cand", "F3"), catalog)

    assert plan.decision == "NEEDS_REVIEW"
    assert _ref_ids(plan) == ["cand"]
    assert plan.visual_refs[0].tier == "generated_candidate"
    assert plan.visual_refs[0].confidence == "low"
    assert catalog.get("cand").source_tier == "generated_candidate"
    assert "未批准的 generated_candidate" in plan.prompt


def test_text_how_fallback_keeps_ready(tmp_path: Path) -> None:
    """No pose asset: text HOW from the shot still yields READY."""

    catalog = _write_catalog(tmp_path, [_asset("F1", roles=["identity_reference"])])
    shot = _shot(
        start_state="林夏低头",
        end_state="林夏抬头微笑",
        action_beats=[
            {"time_seconds": 0.0, "description": "林夏低头沉默"},
            {"time_seconds": 2.0, "description": "她轻轻握紧罐身"},
        ],
    )
    plan = plan_keyframe_generation(shot, _bundle("F1"), catalog)

    assert plan.decision == "READY"
    assert _ref_slots(plan) == ["p1_identity"]
    assert "无姿态/表情参考图" in plan.text_constraints["POSE_EXPRESSION"]
    assert "林夏低头" in plan.text_constraints["POSE_EXPRESSION"]
    assert "No Picture 2" in plan.prompt


def test_write_generation_artifacts(tmp_path: Path) -> None:
    """Writes plan JSON, qwen request JSON and prompt.txt deterministically."""

    catalog = _write_catalog(tmp_path, [_asset("F1", roles=["identity_reference"])])
    plan = plan_keyframe_generation(_shot(), _bundle("F1"), catalog)
    paths = write_generation_artifacts(tmp_path / "out", plan)

    assert paths["plan"].is_file()
    assert paths["request"].is_file()
    assert paths["prompt"].is_file()
    saved_plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    assert saved_plan == plan.model_dump(mode="json")
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    assert request["schema_version"] == "qwen-request-v1"
    assert request["visual_refs"][0]["slot"] == "p1_identity"
    assert request["visual_refs"][0]["sha256"] == plan.source_hashes["F1"]
    assert request["text_constraints"] == plan.text_constraints
    assert paths["prompt"].read_text(encoding="utf-8") == plan.prompt


def test_bundle_references_unregistered_asset_raises(tmp_path: Path) -> None:
    """Unregistered bundle assets are rejected, never silently ignored."""

    catalog = _write_catalog(tmp_path, [_asset("F1", roles=["identity_reference"])])
    with pytest.raises(InputValidationError):
        plan_keyframe_generation(_shot(), _bundle("F1", "missing"), catalog)


def test_empty_bundle_raises(tmp_path: Path) -> None:
    """An empty bundle is an input error, not a silent empty plan."""

    catalog = _write_catalog(tmp_path, [_asset("F1", roles=["identity_reference"])])
    with pytest.raises(InputValidationError):
        plan_keyframe_generation(_shot(), [], catalog)


def test_pending_asset_uses_only_declared_metadata(tmp_path: Path) -> None:
    """pending assets contribute declared facts; analyzed adds visual fields."""

    pending_catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "P",
                roles=["identity_reference"],
                analysis_status="pending",
                subject_or_scene_id="char_p",
                quality_notes="参考正面",
                pose="standing",
                expression="calm",
                view_angle="front",
                outfit="uniform",
            )
        ],
    )
    pending_plan = plan_keyframe_generation(_shot(), _bundle("P"), pending_catalog)
    pending_character = pending_plan.text_constraints["CHARACTER"]
    assert "char_p" not in pending_plan.prompt
    assert "参考正面" in pending_character
    for hidden in ("standing", "calm", "front", "uniform"):
        assert hidden not in pending_character

    analyzed_catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "A",
                roles=["identity_reference"],
                analysis_status="analyzed",
                subject_or_scene_id="char_a",
                quality_notes="参考正面",
                pose="standing",
                expression="calm",
                view_angle="front",
                outfit="uniform",
            )
        ],
    )
    analyzed_plan = plan_keyframe_generation(_shot(), _bundle("A"), analyzed_catalog)
    analyzed_character = analyzed_plan.text_constraints["CHARACTER"]
    assert "char_a" not in analyzed_plan.prompt
    for visible in ("参考正面", "uniform"):
        assert visible in analyzed_character
    for hidden in ("standing", "calm", "front"):
        assert hidden not in analyzed_character
    assert "瞬时状态" in analyzed_character


def test_scene_only_bundle_has_no_visual_refs(tmp_path: Path) -> None:
    """A bundle with only scene assets never produces a fake visual slot."""

    catalog = _write_catalog(
        tmp_path,
        [_asset("F3", "background", roles=["scene_reference"])],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F3"), catalog)

    assert plan.visual_refs == []
    assert plan.decision == "UNRESOLVED"


def test_compiled_prompt_contains_no_question_mark(tmp_path: Path) -> None:
    """UTF-8 Chinese metadata survives into the compiled prompt with no '?'."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference", "outfit_reference"],
                quality_notes="用户提供：2D anime 角色身份参考，bob 发型",
            ),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="夜间城市住宅建筑外景；暖黄窗灯",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert "?" not in plan.prompt
    assert "用户提供：2D anime 角色身份参考；bob 发型" in plan.prompt


def test_selected_p2_background_stays_pose_expression_only(
    tmp_path: Path,
) -> None:
    """A P2 with asset_type=background never enters SCENE/CHARACTER/STYLE."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
                subject_or_scene_id="shot_visual_end",
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="夜间城市住宅建筑外景；暖黄窗灯",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    assert "姿态/表情参考：伸手、回望" in plan.text_constraints["POSE_EXPRESSION"]
    assert "姿态/表情参考：伸手、回望" not in plan.text_constraints["SCENE"]
    assert "k_end" not in plan.prompt
    assert "shot_visual_end" not in plan.prompt


def test_selected_p2_with_scene_role_stays_how(tmp_path: Path) -> None:
    """Even a scene_reference-flagged asset stays HOW when selected as P2."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                "background",
                roles=[
                    "pose_reference",
                    "expression_reference",
                    "scene_reference",
                ],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="夜间城市住宅建筑外景",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert _ref_ids(plan) == ["F1", "k_end"]
    assert "姿态/表情参考：伸手、回望" in plan.text_constraints["POSE_EXPRESSION"]
    assert "姿态/表情参考：伸手、回望" not in plan.text_constraints["SCENE"]


def test_source_scene_does_not_leak_into_target(tmp_path: Path) -> None:
    """P1's source-scene metadata stays out of SCENE/AVOID; target scene wins."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference"],
                quality_notes="角色参考：餐厅内景、暖黄灯光",
            ),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset(
                "bg_rooftop_dusk",
                "background",
                roles=["scene_reference"],
                quality_notes="黄昏天台、晚霞、栏杆",
            ),
        ],
    )
    shot = _shot(setting="黄昏的学校天台，铁门、栏杆、晚霞下的城市")
    plan = plan_keyframe_generation(
        shot,
        _bundle("F1", "k_end", "bg_rooftop_dusk"),
        catalog,
    )

    assert "餐厅内景" not in plan.text_constraints["SCENE"]
    assert "餐厅内景" not in plan.text_constraints["AVOID"]
    assert "餐厅内景" not in plan.text_constraints["CHARACTER"]
    assert "暖黄灯光" not in plan.text_constraints["CHARACTER"]
    assert "瞬时状态" in plan.text_constraints["CHARACTER"]
    assert "黄昏的学校天台，铁门、栏杆、晚霞下的城市" in plan.text_constraints["SCENE"]
    assert "黄昏天台、晚霞、栏杆" in plan.text_constraints["SCENE"]


def test_scene_change_has_no_preserve_avoid_conflict(tmp_path: Path) -> None:
    """Restaurant->rooftop must not emit any scene-preserve phrase."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="黄昏天台、晚霞、栏杆",
            ),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(setting="黄昏的学校天台，铁门、栏杆、晚霞下的城市"),
        _bundle("F1", "k_end", "F3"),
        catalog,
    )

    for phrase in ("保持场景", "保持已绑定场景", "保持背景", "不要改变场景"):
        assert phrase not in plan.text_constraints["PRESERVE"]
        assert phrase not in plan.text_constraints["AVOID"]
    assert "按目标约束改变" in plan.text_constraints["PRESERVE"]


def test_semantic_scene_preserve_contradiction_is_critical() -> None:
    """The validator must flag scene-preserve phrases as critical conflicts."""

    warnings, criticals = _semantic_issues(
        _shot(setting="黄昏的学校天台"),
        p1=None,
        p2=None,
        constraints={
            "PRESERVE": "保持已绑定场景、道具与风格事实",
            "AVOID": "不要改变身份、服装或场景",
        },
    )
    assert warnings == []
    assert any("PRESERVE" in item for item in criticals)
    assert any("AVOID" in item for item in criticals)


def test_identity_observability_risk_requires_review(tmp_path: Path) -> None:
    """Silhouette composition under a mandatory identity gate -> NEEDS_REVIEW."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
        ],
    )
    shot = _shot(
        composition="铁门位于画面右侧三分之一，林夏的剪影从门后踏入",
        shot_scale="wide",
    )
    plan = plan_keyframe_generation(shot, _bundle("F1", "k_end"), catalog)

    assert plan.decision == "NEEDS_REVIEW"
    assert "identity_observability" in plan.prompt
    assert "剪影" in plan.prompt
    assert _ref_ids(plan) == ["F1", "k_end"]


def test_p2_role_conflict_requires_review(tmp_path: Path) -> None:
    """A HOW asset also flagged identity_reference triggers NEEDS_REVIEW."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference"],
                subject_or_scene_id="hero",
            ),
            _asset(
                "k_end",
                roles=["pose_reference", "identity_reference"],
            ),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(subjects=["hero"]),
        _bundle("F1", "k_end"),
        catalog,
    )

    assert plan.decision == "NEEDS_REVIEW"
    assert "P2 role conflict" in plan.prompt
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]


def test_prompt_has_no_asset_id_hash_or_manifest_identifier(
    tmp_path: Path,
) -> None:
    """Prompt stays clean; ids/hashes live only in the JSON facts layer."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference"],
                quality_notes="角色身份参考：脸、发型、校服、色彩",
            ),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
                subject_or_scene_id="shot_visual_end",
            ),
            _asset(
                "F3",
                "background",
                roles=["scene_reference"],
                quality_notes="夜间城市住宅建筑外景；暖黄窗灯",
                subject_or_scene_id="loc_night_rooftop",
            ),
            _asset(
                "style_watercolor",
                "style",
                roles=["style_reference"],
                quality_notes="水彩风格",
                subject_or_scene_id="style_watercolor",
            ),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(),
        _bundle("F1", "k_end", "F3", "style_watercolor"),
        catalog,
    )

    for token in (
        "F1",
        "k_end",
        "F3",
        "style_watercolor",
        "char_f1",
        "loc_night_rooftop",
        "shot_visual_end",
    ):
        assert token not in plan.prompt
    assert not re.search(r"\b[0-9a-f]{64}\b", plan.prompt)
    assert set(plan.source_hashes) == {"F1", "F3", "k_end", "style_watercolor"}


def test_prompt_declares_change_and_not_must_not_be_changed(
    tmp_path: Path,
) -> None:
    """Picture 1 is an identity-feature preserve with explicit CHANGE."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    assert "must not be changed" not in plan.prompt
    assert "Preserve Picture 1's identity features" in plan.prompt
    assert "pose follows Picture 2" in plan.prompt
    assert "scene follows the SCENE constraint" in plan.prompt
    assert (
        "Do not copy Picture 2's identity, outfit, background or scene"
        in plan.prompt
    )
    assert "apply only its pose and expression" in plan.prompt


def test_i7_golden_matches_committed_artifacts(tmp_path: Path) -> None:
    """Repaired compiler reproduces the committed I7 golden exactly."""

    catalog = build_golden_catalog(tmp_path)
    shot = build_golden_shot()
    plan = plan_keyframe_generation(shot, golden_bundle(catalog), catalog)

    assert plan.decision == "READY"
    assert _ref_ids(plan) == ["F1", "k_end"]
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    assert "F3" not in _ref_ids(plan)
    assert len(plan.visual_refs) <= 2
    assert "夜间城市住宅建筑外景" in plan.text_constraints["SCENE"]
    assert "暖黄窗灯" in plan.text_constraints["SCENE"]
    assert "姿态/表情参考：伸手、回望" in plan.text_constraints["POSE_EXPRESSION"]
    assert "姿态/表情参考：伸手、回望" not in plan.text_constraints["SCENE"]
    for token in ("F1", "k_end", "F3", "char_main", "shot_visual_end", "loc_night_rooftop"):
        assert token not in plan.prompt
    assert "must not be changed" not in plan.prompt
    assert "?" not in plan.prompt

    assert plan.prompt == GOLDEN_PROMPT.read_text(encoding="utf-8")
    normalized = normalize_plan_paths(plan, catalog)
    assert normalized == json.loads(GOLDEN_OUTPUT.read_text(encoding="utf-8"))


def test_same_file_multiple_roles_are_legal(tmp_path: Path) -> None:
    """One asset may carry identity+outfit; another pose+expression."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference", "outfit_reference"],
            ),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
            _asset("F3", "background", roles=["scene_reference"]),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end", "F3"), catalog)

    assert plan.decision == "READY"
    assert _ref_ids(plan) == ["F1", "k_end"]
    assert plan.visual_refs[0].roles == [
        "identity_reference",
        "outfit_reference",
    ]
    assert plan.visual_refs[1].roles == [
        "pose_reference",
        "expression_reference",
    ]


def test_same_sha_assets_are_not_merged(tmp_path: Path) -> None:
    """Two assets with identical file bytes stay distinct by id and role."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("char_a", roles=["identity_reference"]),
            _asset("char_b", roles=["identity_reference"]),
            _asset("pose_a", roles=["pose_reference"]),
            _asset("pose_b", roles=["pose_reference"]),
        ],
    )
    plan = plan_keyframe_generation(
        _shot(),
        _bundle("char_a", "char_b", "pose_a", "pose_b"),
        catalog,
    )

    assert [ref.asset_id for ref in plan.visual_refs] == ["char_a", "pose_a"]
    ref_hashes = {ref.asset_id: ref.sha256 for ref in plan.visual_refs}
    assert ref_hashes["char_a"] == plan.source_hashes["char_b"]
    assert ref_hashes["pose_a"] == plan.source_hashes["pose_b"]
    assert plan.source_asset_ids == ["char_a", "char_b", "pose_a", "pose_b"]
    assert len(plan.source_hashes) == 4


def test_compiler_output_is_byte_deterministic(tmp_path: Path) -> None:
    """Repeated compiles produce byte-identical JSON and prompt files."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
            ),
            _asset("F3", "background", roles=["scene_reference"]),
        ],
    )
    shot = _shot()
    bundle = _bundle("F1", "k_end", "F3")
    first = write_generation_artifacts(
        tmp_path / "out1",
        plan_keyframe_generation(shot, bundle, catalog),
    )
    second = write_generation_artifacts(
        tmp_path / "out2",
        plan_keyframe_generation(shot, bundle, catalog),
    )

    for name in ("plan", "request", "prompt"):
        assert first[name].read_bytes() == second[name].read_bytes()


def test_p2_how_metadata_sufficient_is_ready(tmp_path: Path) -> None:
    """P2 with explicit HOW text (body + action + expression) keeps READY."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes=(
                    "用户提供：镜头结束画面参考（1920x1080）；动作描述："
                    "闭眼，右手抬起靠近右侧太阳穴/脸侧，手指位置相近，"
                    "平静、略疲惫、若有所思的表情，头部与上半身姿态相近"
                ),
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    assert plan.decision == "READY"
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    how = plan.text_constraints["POSE_EXPRESSION"]
    assert "闭眼" in how
    assert "太阳穴" in how
    assert "上半身姿态相近" in how
    assert "P2 HOW metadata insufficient" not in plan.prompt


def test_p2_how_metadata_missing_requires_review(tmp_path: Path) -> None:
    """A bare P2 pointer without HOW text is never silently weakened."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes="用户提供：镜头结束画面参考（1920x1080）",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    assert plan.decision == "NEEDS_REVIEW"
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    assert "P2 HOW metadata insufficient" in plan.prompt
    assert "镜头结束画面参考" in plan.text_constraints["POSE_EXPRESSION"]


def test_p1_transient_state_excluded_from_identity(tmp_path: Path) -> None:
    """Eye state / expression fragments stay out of CHARACTER (WHO)."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference"],
                quality_notes=(
                    "身份参考：短发偏蓬松 bob 头、深青绿色发色、大眼灰紫色虹膜；"
                    "平静面无表情、睁眼直视前方"
                ),
            ),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    character = plan.text_constraints["CHARACTER"]
    assert "bob 头" in character
    assert "灰紫色虹膜" in character
    for transient in ("面无表情", "睁眼直视前方", "平静"):
        assert transient not in character
    assert "瞬时状态" in character


def test_p1_source_scene_and_lighting_excluded_from_identity(
    tmp_path: Path,
) -> None:
    """Source-scene/lighting fragments never enter the WHO description."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference"],
                quality_notes="角色参考：餐厅内景、暖黄灯光",
            ),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    character = plan.text_constraints["CHARACTER"]
    assert "餐厅内景" not in character
    assert "暖黄灯光" not in character
    assert "瞬时状态" in character


def test_identity_preserve_keeps_outfit_and_face_features(
    tmp_path: Path,
) -> None:
    """Persistent identity/outfit facts remain in CHARACTER and PRESERVE."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference", "outfit_reference"],
                quality_notes="身份参考：蓝白配色上衣、bob 头发型、深青绿色发色",
            ),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    character = plan.text_constraints["CHARACTER"]
    preserve = plan.text_constraints["PRESERVE"]
    assert "蓝白配色上衣" in character
    assert "bob 头" in character
    assert "身份特征" in preserve
    assert "服装" in preserve
    assert "preserve HOW from Picture 2" in preserve


def test_prompt_pose_follows_p2_when_p2_present(tmp_path: Path) -> None:
    """P2 present: pose follows Picture 2, no 'or the text HOW' freedom."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="姿态/表情参考：伸手、回望",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    assert "or the text HOW" not in plan.prompt
    assert "pose follows Picture 2" in plan.prompt
    assert "Role summary: WHO from Picture 1; HOW from Picture 2" in plan.prompt


def test_prompt_text_how_and_role_summary_without_p2(tmp_path: Path) -> None:
    """No P2: text HOW fallback plus the no-P2 role summary."""

    catalog = _write_catalog(
        tmp_path,
        [_asset("F1", roles=["identity_reference"])],
    )
    shot = _shot(
        start_state="林夏低头",
        end_state="林夏抬头微笑",
        action_beats=[{"time_seconds": 0.0, "description": "林夏低头沉默"}],
    )
    plan = plan_keyframe_generation(shot, _bundle("F1"), catalog)

    assert plan.decision == "READY"
    assert (
        "No Picture 2 is provided; pose and expression are text-only"
        in plan.prompt
    )
    assert "Role summary: WHO from Picture 1; HOW from text" in plan.prompt


def test_shot003_repaired_v2_how_section(tmp_path: Path) -> None:
    """shot_003 shape with repaired metadata yields a HOW-complete READY plan."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset(
                "F1",
                roles=["identity_reference", "outfit_reference"],
                quality_notes=(
                    "用户提供真实身份参考：2D anime 女性角色，短发偏蓬松 bob 头、"
                    "脸侧有明显外翘发束、额前较厚刘海；深青绿色/蓝绿色偏深发色；"
                    "大眼灰紫色虹膜；平静面无表情、睁眼直视前方；蓝白配色上衣、"
                    "外层浅蓝色衣领/披层；原图为室内餐厅/店内环境"
                    "（仅作背景来源说明，不作为目标场景）"
                ),
                subject_or_scene_id="char_f1",
            ),
            _asset(
                "k_end",
                "background",
                roles=["pose_reference", "expression_reference"],
                quality_notes=(
                    "用户提供：镜头结束画面参考（1920x1080）；动作描述：闭眼，"
                    "右手抬起靠近右侧太阳穴/脸侧，手指位置相近，平静、略疲惫、"
                    "若有所思的表情，头部与上半身姿态相近"
                ),
                subject_or_scene_id="shot_visual_end",
            ),
            _asset(
                "bg_rooftop_dusk",
                "background",
                roles=["scene_reference"],
                quality_notes="合成背景参考：黄昏天台、晚霞、栏杆",
                subject_or_scene_id="loc_rooftop",
            ),
            _asset(
                "style_watercolor",
                "style",
                roles=["style_reference"],
                quality_notes="合成风格参考：水彩风格",
                subject_or_scene_id="style_watercolor",
            ),
        ],
    )
    shot = _shot(
        shot_id="shot_003",
        shot_scale="close_up",
        camera_position="正面平视的肩部特写，靠近林夏。",
        camera_motion="极缓慢推（push-in）",
        composition=(
            "林夏面部居中略偏右，汽水罐位于画面下方一角；背景虚化，"
            "晚霞侧光勾勒发丝轮廓。"
        ),
        setting="黄昏的学校天台，栏杆旁，晚霞逆光。",
        props=["汽水罐（林夏手中）"],
        start_state="林夏低头看着手中的汽水罐，沉默不语。",
        end_state="林夏说完，视线望向远处城市方向。",
        action_beats=[
            {"time_seconds": 0.0, "description": "林夏低头看着手中的汽水罐。"},
            {"time_seconds": 2.0, "description": "她沉默了几秒，轻轻握紧罐身。"},
        ],
    )
    plan = plan_keyframe_generation(
        shot,
        _bundle("F1", "k_end", "bg_rooftop_dusk", "style_watercolor"),
        catalog,
    )

    assert plan.decision == "READY"
    assert _ref_ids(plan) == ["F1", "k_end"]
    assert _ref_slots(plan) == ["p1_identity", "p2_pose_expression"]
    assert len(plan.visual_refs) == 2
    how = plan.text_constraints["POSE_EXPRESSION"]
    assert "闭眼" in how
    assert "太阳穴" in how
    assert "上半身姿态相近" in how
    character = plan.text_constraints["CHARACTER"]
    assert "bob 头" in character
    assert "灰紫色虹膜" in character
    assert "蓝白配色上衣" in character
    for transient in ("面无表情", "睁眼直视前方", "室内餐厅", "餐厅"):
        assert transient not in character
    assert "瞬时状态" in character
    assert "or the text HOW" not in plan.prompt
    assert "Role summary: WHO from Picture 1; HOW from Picture 2" in plan.prompt
    assert "preserve HOW from Picture 2" in plan.text_constraints["PRESERVE"]


def test_how_sufficiency_uses_declared_metadata_only(tmp_path: Path) -> None:
    """Pending P2 fields do not count as analyzed HOW text."""

    catalog = _write_catalog(
        tmp_path,
        [
            _asset("F1", roles=["identity_reference"]),
            _asset(
                "k_end",
                roles=["pose_reference", "expression_reference"],
                quality_notes="镜头结束画面参考",
                pose="standing",
                expression="calm",
            ),
        ],
    )
    plan = plan_keyframe_generation(_shot(), _bundle("F1", "k_end"), catalog)

    assert plan.decision == "NEEDS_REVIEW"
    assert "P2 HOW metadata insufficient" in plan.prompt
