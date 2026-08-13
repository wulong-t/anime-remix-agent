"""Generate the retained synthetic reference-planner regression artifacts.

The golden pins the repaired compiler's I7-shape output:

    P1 = F1 (WHO, identity_reference, canonical)
    P2 = k_end (HOW, pose_reference + expression_reference, canonical)
    F3 = scene asset, textified into SCENE, never a visual ref

and the KF-PRODUCT-1-R1 repair guarantees:

* no asset_id / sha256 / manifest identifier in the prompt text;
* no literal ``?`` (UTF-8 survival);
* no whole-pixel "must not be changed" wording: Picture 1 is an
  identity-feature preserve with explicit CHANGE (pose <- P2/text,
  scene <- SCENE, camera <- shot);
* PRESERVE keeps only identity; AVOID forbids identity fusion, P2 outfit
  leakage and source-scene leakage;
* deterministic: identical inputs -> identical plan and prompt bytes.

Run locally:

    python experiments/reference_planner_golden/generate_golden_i7.py
"""

from __future__ import annotations

import json
import struct
import tempfile
import zlib
from pathlib import Path

from anime_remix.json_io import dump_json_atomic
from anime_remix.services.image_assets import ImageAssetCatalog, load_image_assets
from anime_remix.services.script.reference_planner import (
    KeyframeGenerationPlan,
    plan_keyframe_generation,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry, parse_shot_plan

HERE = Path(__file__).resolve().parent
GOLDEN_OUTPUT = HERE / "golden_i7_compiler_output.json"
GOLDEN_PROMPT = HERE / "golden_i7_prompt.txt"

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_GOLDEN_ASSETS = [
    {
        "asset_id": "F1",
        "path": "images/F1.png",
        "asset_type": "character",
        "subject_or_scene_id": "char_main",
        "quality_notes": "角色身份参考：脸、发型、校服、色彩",
        "rights_status": "synthetic-fixture",
        "source_tier": "canonical",
        "reference_roles": ["identity_reference", "outfit_reference"],
        "analysis_status": "pending",
    },
    {
        "asset_id": "k_end",
        "path": "images/k_end.png",
        "asset_type": "background",
        "subject_or_scene_id": "shot_visual_end",
        "quality_notes": "姿态/表情参考：伸手、回望",
        "rights_status": "synthetic-fixture",
        "source_tier": "canonical",
        "reference_roles": ["pose_reference", "expression_reference"],
        "analysis_status": "pending",
    },
    {
        "asset_id": "F3",
        "path": "images/F3.png",
        "asset_type": "background",
        "subject_or_scene_id": "loc_night_rooftop",
        "quality_notes": "夜间城市住宅建筑外景；wide establishing；暖黄窗灯",
        "rights_status": "synthetic-fixture",
        "source_tier": "canonical",
        "reference_roles": ["scene_reference"],
        "analysis_status": "pending",
    },
]

_GOLDEN_SHOT = {
    "shot_id": "shot_001",
    "scene_id": "scene_01",
    "order": 1,
    "narrative_purpose": "I7 golden regression oracle",
    "duration_seconds": 4.0,
    "shot_scale": "wide",
    "composition": "人物位于画面中央，黄昏天台全景，栏杆与远处城市天际线",
    "camera_position": "中景平视",
    "camera_motion": "fixed",
    "subjects": ["林夏"],
    "setting": "夜间城市住宅建筑外景",
    "props": [],
    "start_state": "人物站在楼顶",
    "action_beats": [
        {"time_seconds": 0.0, "description": "人物望向远处"},
    ],
    "end_state": "人物回望镜头",
    "emotion_arc": "平静",
}


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


def build_golden_catalog(tmp: Path) -> ImageAssetCatalog:
    """Write the F1 / k_end / F3 synthetic catalog under ``tmp``."""

    images = tmp / "images"
    images.mkdir(parents=True, exist_ok=True)
    for asset in _GOLDEN_ASSETS:
        (images / asset["path"].rsplit("/", 1)[-1]).write_bytes(_png(24, 24))
    manifest = tmp / "image_assets.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": "image-assets-v1", "assets": _GOLDEN_ASSETS},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_image_assets(manifest)


def build_golden_shot() -> ShotPlanEntry:
    """The golden oracle shot (identity remains observable, so READY)."""

    return parse_shot_plan(
        {"schema_version": "shot-plan-v1", "shots": [_GOLDEN_SHOT]}
    ).shots[0]


def golden_bundle(catalog: ImageAssetCatalog) -> list[dict[str, str]]:
    """ReferenceBundle entries in the golden order F1, k_end, F3."""

    return [
        {
            "asset_id": record.asset_id,
            "asset_type": record.asset_type,
            "path": str(record.resolved_path),
            "note": "i7 golden regression bundle",
        }
        for record in (
            catalog.get("F1"),
            catalog.get("k_end"),
            catalog.get("F3"),
        )
    ]


def compile_golden_plan(
    tmp: Path,
) -> tuple[KeyframeGenerationPlan, ImageAssetCatalog]:
    """Compile the golden plan; the catalog is returned for path normalization."""

    catalog = build_golden_catalog(tmp)
    plan = plan_keyframe_generation(build_golden_shot(), golden_bundle(catalog), catalog)
    return plan, catalog


def normalize_plan_paths(
    plan: KeyframeGenerationPlan,
    catalog: ImageAssetCatalog,
) -> dict:
    """Return the plan dict with visual_ref paths rewritten to manifest-relative."""

    data = plan.model_dump(mode="json")
    for ref in data["visual_refs"]:
        record = catalog.get(ref["asset_id"])
        ref["path"] = record.path
    return data


def _write_golden() -> None:
    with tempfile.TemporaryDirectory(prefix="kf_i7_golden_") as tmp_name:
        plan, catalog = compile_golden_plan(Path(tmp_name))
        assert plan.decision == "READY", plan.decision
        assert len(plan.visual_refs) == 2
        dump_json_atomic(
            GOLDEN_OUTPUT,
            normalize_plan_paths(plan, catalog),
            sort_keys=True,
        )
        prompt_text = plan.prompt if plan.prompt.endswith("\n") else plan.prompt + "\n"
        GOLDEN_PROMPT.write_text(prompt_text, encoding="utf-8", newline="\n")
    print(f"wrote {GOLDEN_OUTPUT}")
    print(f"wrote {GOLDEN_PROMPT}")


if __name__ == "__main__":
    _write_golden()
