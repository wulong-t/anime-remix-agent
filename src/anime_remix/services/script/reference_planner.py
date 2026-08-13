"""KF-PRODUCT-1: legacy deterministic Reference Planner / Condition Compiler.

This module preserves the historical WHO/HOW planning contract and its
reproducibility tests.  It is not the active first/last-frame execution path.
New endpoint generation uses
``anime_remix.services.execution.shot_keyframe_runner`` and its
reference-first policy: referenced visual facts stay in images, while text is
limited to action/state changes and visual facts that have no reference.

This module is the thin glue between an approved ``ReferenceBundle`` and a
Qwen-Image-Edit keyframe generation request.  It reuses the existing
``ImageAssetCatalog`` / ``ShotPlanEntry`` contracts and never talks to a
model, the network or the Remote environment.

Product constraints implemented here (frozen by the I4-I7 single-sample
experiments):

* at most two visual references per keyframe generation:
  ``p1_identity`` (WHO) and ``p2_pose_expression`` (HOW);
* scene / background / camera / composition / lighting / style / prop
  information is always compiled into ``text_constraints`` instead of a
  third visual slot;
* ``generated_candidate`` assets never beat a trusted tier and are never
  promoted by this planner;
* ``analysis_status=pending`` assets are selected only from explicitly
  declared metadata (``reference_roles`` / ``subject_or_scene_id`` /
  ``quality_notes``); derived visual facts are never invented.
* facts and instructions are separated: asset ids, sha256 digests and
  manifest identifiers live in the JSON plan and request; the model prompt
  carries only clean descriptive text;
* deterministic keyword-based semantic checks raise ``NEEDS_REVIEW`` for
  identity-observability risks and PRESERVE/AVOID contradictions;
* a selected P2 without explicit pose/expression metadata raises
  ``NEEDS_REVIEW`` instead of silently weakening to a bare image pointer;
* P1 identity text excludes transient state (expression, eye state, gaze,
  pose, source scene/lighting); only persistent identity/outfit facts
  enter CHARACTER/PRESERVE.

The output ``KeyframeGenerationPlan`` is fully deterministic: identical
inputs produce identical plans, prompts and artifact bytes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic, sha256_file
from anime_remix.services.image_assets import (
    ImageAssetCatalog,
    ImageAssetRecord,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry

_SCHEMA_VERSION = "keyframe-generation-plan-v1"
_REQUEST_SCHEMA_VERSION = "qwen-request-v1"
_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

_TIER_RANK = {
    "canonical": 0,
    "derived": 1,
    "approved_generated": 2,
    "generated_candidate": 3,
}

_TEXT_KEYS = (
    "CHARACTER",
    "POSE_EXPRESSION",
    "SCENE",
    "CAMERA",
    "COMPOSITION",
    "LIGHTING",
    "STYLE",
    "PRESERVE",
    "AVOID",
)
_TEXT_LABELS = {
    "CHARACTER": "CHARACTER (WHO)",
    "POSE_EXPRESSION": "POSE/EXPRESSION (HOW)",
    "SCENE": "SCENE (WHERE)",
    "CAMERA": "CAMERA",
    "COMPOSITION": "COMPOSITION",
    "LIGHTING": "LIGHTING",
    "STYLE": "STYLE",
    "PRESERVE": "PRESERVE",
    "AVOID": "AVOID",
}

# Frozen keyword tokens for the deterministic semantic checks (no NLP).
_OBSERVABILITY_RISK_TOKENS = (
    "剪影",
    "背对",
    "背影",
    "背面",
    "极小",
    "silhouette",
    "back view",
    "from behind",
    "tiny",
    "far away",
)

# Phrases that preserve the source scene; contradictory when the shot
# defines a target scene (e.g. restaurant -> rooftop).
_SCENE_PRESERVE_PHRASES = (
    "保持场景",
    "保持已绑定场景",
    "保持背景",
    "不要改变场景",
    "不要改变身份、服装或场景",
    "不要改变背景",
    "不得改变场景",
    "keep the scene",
    "keep the background",
    "do not change the scene",
)

# Positive identity-change instructions; contradictory with P1 preserve.
_IDENTITY_CHANGE_PHRASES = (
    "改为另一个角色",
    "身份改为",
    "换一个角色",
    "替换角色",
    "change the identity",
    "change the character",
    "replace the character",
)

# Transient-state fragments excluded from P1's identity description.
# They describe a moment (expression, eye state, gaze, pose, action,
# source scene/lighting) that is target-changeable, not an
# identity-preserve fact.  Matching is fragment-based and deterministic
# (no NLP).
_P1_TRANSIENT_TOKENS = (
    # expression / emotional state
    "表情",
    "面无表情",
    "神情",
    "微笑",
    "皱眉",
    "平静",
    "疲惫",
    "沉思",
    "温柔",
    "冷淡",
    "惊讶",
    "难过",
    "开心",
    "严肃",
    # eye state / gaze
    "睁眼",
    "闭眼",
    "眨眼",
    "直视前方",
    "目光",
    "视线",
    "眼神",
    "看向",
    "望着",
    "凝视",
    "回望",
    # pose / posture / action
    "姿态",
    "姿势",
    "动作",
    "站立",
    "坐着",
    "坐姿",
    "伸手",
    "抬起",
    "放下",
    "转身",
    "行走",
    "走动",
    "倚靠",
    "侧身",
    "低头",
    "抬头",
    # source scene / background / source lighting
    "场景",
    "背景",
    "环境",
    "室内",
    "餐厅",
    "餐馆",
    "店内",
    "原图",
    "灯光",
    "光线",
    "光照",
    "侧光",
    "逆光",
    "暖光",
    "冷光",
    "光源",
)

# Deterministic HOW sufficiency groups for a selected P2.  A P2 whose
# declared metadata contains no explicit pose/expression text must raise
# NEEDS_REVIEW instead of being silently weakened to a bare pointer.
_HOW_BODY_TOKENS = (
    "手",
    "手指",
    "头部",
    "上半身",
    "肩",
    "手臂",
    "腿",
    "脚",
    "身体",
)
_HOW_ACTION_TOKENS = (
    "抬起",
    "举起",
    "闭眼",
    "睁眼",
    "低头",
    "抬头",
    "回望",
    "转身",
    "伸手",
    "放下",
    "握",
    "倚靠",
    "站立",
    "站",
    "坐",
    "看",
    "走",
)
_HOW_EXPRESSION_TOKENS = (
    "表情",
    "神情",
    "平静",
    "疲惫",
    "沉思",
    "若有所思",
    "微笑",
    "皱眉",
    "眼神",
    "目光",
    "温柔",
    "严肃",
)

# I7 frozen sampling values, reused as the deterministic default for the
# generated request artifact (no sampling search is ever performed here).
_DEFAULT_SAMPLING: dict[str, object] = {
    "seed": 0,
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "negative_prompt": " ",
    "num_images_per_prompt": 1,
}

Decision = Literal["READY", "NEEDS_REVIEW", "UNRESOLVED"]
VisualRefSlot = Literal["p1_identity", "p2_pose_expression"]
Confidence = Literal["high", "low", "unresolved"]


class VisualRef(BaseModel):
    """One visual reference occupying a hard slot (WHO or HOW)."""

    model_config = _STRICT_CONFIG

    slot: VisualRefSlot
    asset_id: str
    path: str
    sha256: str
    tier: str
    roles: list[str]
    selection_reason: str
    confidence: Confidence


class KeyframeGenerationPlan(BaseModel):
    """Deterministic plan for one Qwen keyframe generation request."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["keyframe-generation-plan-v1"] = _SCHEMA_VERSION
    decision: Decision
    shot_id: str
    visual_refs: list[VisualRef]
    text_constraints: dict[str, str]
    prompt: str
    source_asset_ids: list[str]
    source_hashes: dict[str, str]

    @field_validator("visual_refs", mode="before")
    @classmethod
    def _visual_refs(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("visual_refs must be a list")
        if len(value) > 2:
            raise ValueError("visual_refs must contain at most two references")
        slots = [
            entry.get("slot") if isinstance(entry, Mapping) else entry.slot
            for entry in value
        ]
        if len(slots) != len(set(slots)):
            raise ValueError("visual_ref slots must be unique")
        if slots == ["p2_pose_expression"]:
            raise ValueError("p2 slot requires a preceding p1 slot")
        return value

    @field_validator("text_constraints", mode="before")
    @classmethod
    def _text_constraints(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("text_constraints must be a dict")
        if set(value) != set(_TEXT_KEYS):
            raise ValueError(
                "text_constraints must contain exactly the keys "
                + ", ".join(_TEXT_KEYS)
            )
        for key, item in value.items():
            if not isinstance(item, str):
                raise TypeError(f"text_constraints[{key}] must be a string")
        return value

    @field_validator("source_asset_ids", mode="before")
    @classmethod
    def _source_ids(cls, value: object) -> object:
        if not isinstance(value, list) or not value:
            raise ValueError("source_asset_ids must be a non-empty list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("source_asset_ids entries must be non-empty strings")
            if item in cleaned:
                raise ValueError(f"duplicate source_asset_id {item!r}")
            cleaned.append(item)
        if cleaned != sorted(cleaned):
            raise ValueError("source_asset_ids must be sorted")
        return cleaned

    @field_validator("source_hashes", mode="before")
    @classmethod
    def _source_hashes(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("source_hashes must be a dict")
        for asset_id, digest in value.items():
            if not isinstance(asset_id, str):
                raise TypeError("source_hashes keys must be strings")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"source_hashes[{asset_id!r}] must be a sha256 hex digest"
                )
        return value

    @model_validator(mode="after")
    def _hashes_match_sources(self) -> KeyframeGenerationPlan:
        if set(self.source_hashes) != set(self.source_asset_ids):
            raise ValueError(
                "source_hashes keys must exactly match source_asset_ids"
            )
        return self


def _norm(text: str) -> str:
    """NFKC-normalize, casefold and strip for deterministic matching."""

    return unicodedata.normalize("NFKC", text).strip().lower()


def _tier_rank(record: ImageAssetRecord) -> int:
    """Deterministic trust rank; unknown tiers sort after every known tier."""

    return _TIER_RANK.get(record.source_tier, len(_TIER_RANK))


def _subject_match_score(subjects: list[str], record: ImageAssetRecord) -> int:
    """2 when metadata names a shot subject, 1 when notes mention it, else 0."""

    asset = _norm(record.asset_id)
    scene = _norm(record.subject_or_scene_id or "")
    notes = _norm(record.quality_notes or "")
    best = 0
    for subject in subjects:
        needle = _norm(subject)
        if not needle:
            continue
        if needle in scene or needle in asset or scene in needle:
            best = max(best, 2)
        elif needle in notes:
            best = max(best, 1)
    return best


def _bundle_records(
    bundle: list[Mapping[str, object]],
    catalog: ImageAssetCatalog,
) -> list[ImageAssetRecord]:
    """Resolve bundle entries to catalog records; strict and deterministic."""

    if not bundle:
        raise InputValidationError("bundle must contain at least one asset")
    records: list[ImageAssetRecord] = []
    seen: set[str] = set()
    for entry in bundle:
        if not isinstance(entry, Mapping):
            raise InputValidationError("bundle entries must be objects")
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise InputValidationError(
                "bundle entries need a non-empty asset_id"
            )
        if asset_id in seen:
            raise InputValidationError(
                f"bundle repeats asset {asset_id!r}",
                asset_id=asset_id,
            )
        seen.add(asset_id)
        record = catalog.get(asset_id)
        if record is None:
            raise InputValidationError(
                f"bundle references unregistered asset {asset_id!r}",
                asset_id=asset_id,
            )
        records.append(record)
    return records


def _select_identity(
    shot: ShotPlanEntry,
    records: list[ImageAssetRecord],
) -> tuple[ImageAssetRecord | None, str, bool]:
    """Pick P1 (WHO): best tier, then subject match, then role bonus.

    Returns ``(record, reason, conflict)``.  A conflict means two candidates
    are indistinguishable on tier/match/role and needs human review.
    """

    candidates = [
        record
        for record in records
        if "identity_reference" in record.reference_roles
    ]
    if not candidates:
        return (
            None,
            f"no identity_reference asset in bundle for shot {shot.shot_id}",
            False,
        )
    scored = sorted(
        (
            (
                _tier_rank(record),
                -_subject_match_score(shot.subjects, record),
                -int("outfit_reference" in record.reference_roles),
                record.asset_id,
                record,
            )
            for record in candidates
        ),
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    top = scored[0]
    conflict = False
    if len(scored) > 1:
        runner = scored[1]
        conflict = (
            runner[0],
            runner[1],
            runner[2],
        ) == (
            top[0],
            top[1],
            top[2],
        )
    record = top[4]
    reason = (
        f"identity_reference with best tier/match: {record.asset_id}"
        if not conflict
        else (
            "ambiguous identity candidates with equal tier/match "
            f"({', '.join(item[3] for item in scored[:2])}); "
            f"picked {record.asset_id}"
        )
    )
    return record, reason, conflict


def _select_pose_expression(
    records: list[ImageAssetRecord],
) -> tuple[ImageAssetRecord | None, str, bool]:
    """Pick P2 (HOW): best tier, then an asset carrying both roles."""

    candidates = [
        record
        for record in records
        if "pose_reference" in record.reference_roles
        or "expression_reference" in record.reference_roles
    ]
    if not candidates:
        return None, "no pose/expression reference asset; text HOW", False

    def role_bonus(record: ImageAssetRecord) -> int:
        return int("pose_reference" in record.reference_roles) + int(
            "expression_reference" in record.reference_roles
        )

    scored = sorted(
        (
            (_tier_rank(record), -role_bonus(record), record.asset_id, record)
            for record in candidates
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    top = scored[0]
    conflict = False
    if len(scored) > 1:
        runner = scored[1]
        conflict = (runner[0], runner[1]) == (top[0], top[1])
    record = top[3]
    reason = (
        f"pose/expression reference with best tier: {record.asset_id}"
        if not conflict
        else (
            "ambiguous pose/expression candidates with equal tier "
            f"({', '.join(item[2] for item in scored[:2])}); "
            f"picked {record.asset_id}"
        )
    )
    return record, reason, conflict


def _derive_pose_text(shot: ShotPlanEntry) -> str:
    """Deterministic text HOW from the shot's start/end state and beats."""

    parts = [shot.start_state, shot.end_state]
    parts.extend(beat.description for beat in shot.action_beats[:3])
    return "；".join(part.strip() for part in parts if part.strip())


def _declared_metadata(record: ImageAssetRecord) -> str:
    """Explicit manifest metadata usable as prompt description text.

    ``subject_or_scene_id`` stays in the facts layer (JSON), never in the
    prompt; ``pending`` assets contribute only declared fields
    (``quality_notes`` / ``time_of_day``); ``analyzed`` assets additionally
    contribute pose / expression / view angle / outfit facts.
    """

    parts: list[str] = []
    for field in (
        record.quality_notes,
        record.time_of_day,
    ):
        if field:
            parts.append(field)
    if record.analysis_status == "analyzed":
        for field in (
            record.view_angle,
            record.pose,
            record.expression,
            record.outfit,
        ):
            if field:
                parts.append(field)
    return "；".join(parts)


def _identity_metadata(record: ImageAssetRecord) -> tuple[str, bool]:
    """Persistent identity/outfit facts only, plus a transient-state flag.

    ``quality_notes`` is split into fragments; fragments describing
    expression, eye state, gaze, pose, action, source scene or source
    lighting are excluded from the WHO description.  ``analyzed`` assets
    contribute only ``outfit`` (pose / expression / view_angle are
    transient source facts, not identity-preserve facts).  The flag makes
    the compiler emit an explicit "transient state is not preserved" note.
    """

    parts: list[str] = []
    transient_found = False
    for fragment in re.split(r"[，,；;]", record.quality_notes or ""):
        fragment = fragment.strip()
        if not fragment:
            continue
        if any(token in fragment for token in _P1_TRANSIENT_TOKENS):
            transient_found = True
            continue
        parts.append(fragment)
    if record.analysis_status == "analyzed":
        if record.outfit:
            parts.append(record.outfit)
        if any((record.pose, record.expression, record.view_angle)):
            transient_found = True
    return "；".join(parts), transient_found


def _p2_how_sufficient(record: ImageAssetRecord) -> bool:
    """True when a selected P2 carries an explicit pose/expression text.

    The haystack is exactly the declared metadata that would be compiled
    into POSE_EXPRESSION.  At least one body-part token, one action token
    and one expression/state token must appear; otherwise the compiler
    must not silently ship a bare image pointer and the plan becomes
    NEEDS_REVIEW.
    """

    haystack = _norm(_declared_metadata(record))
    return all(
        any(token in haystack for token in group)
        for group in (
            _HOW_BODY_TOKENS,
            _HOW_ACTION_TOKENS,
            _HOW_EXPRESSION_TOKENS,
        )
    )


def _compile_constraints(
    shot: ShotPlanEntry,
    records: list[ImageAssetRecord],
    p1: ImageAssetRecord | None,
    p2: ImageAssetRecord | None,
    *,
    p1_conflict: bool,
    p2_conflict: bool,
) -> dict[str, str]:
    """Compile every non-visual condition into the nine text slots.

    Role isolation: the records already occupying a visual slot (P1 WHO /
    P2 HOW) are excluded from SCENE / STYLE, so a selected P2 stays
    POSE_EXPRESSION even when its ``asset_type`` is background.  No
    ``asset_id``, sha256 or manifest identifier is written into the prompt
    text; those facts live in the JSON plan/request.
    """

    selected_ids = {
        record.asset_id for record in (p1, p2) if record is not None
    }
    scene_assets = [
        record
        for record in records
        if record.asset_id not in selected_ids
        and (
            record.asset_type in ("background", "foreground")
            or "scene_reference" in record.reference_roles
        )
    ]
    style_assets = [
        record
        for record in records
        if record.asset_id not in selected_ids
        and (
            record.asset_type == "style"
            or "style_reference" in record.reference_roles
        )
    ]

    character_parts = ["主角：" + "、".join(shot.subjects)]
    if p1 is not None:
        meta, transient = _identity_metadata(p1)
        character_parts.append(
            "P1 提供角色身份视觉参考（WHO）：脸、五官、发型、发色、眼睛"
            + (f"，参考描述：{meta}" if meta else "")
        )
        if transient:
            character_parts.append(
                "P1 的瞬时状态（表情、睁眼/闭眼、视线、姿态与源图场景）不作为身份保留项"
            )
        if p1.source_tier == "generated_candidate":
            character_parts.append(
                "P1 是未批准的 generated_candidate，需人工评审"
            )
    if p1_conflict:
        character_parts.append("存在同级身份候选冲突，P1 选择需人工确认")
    if p2 is not None and p2_conflict:
        character_parts.append("存在同级姿态/表情候选冲突，P2 选择需人工确认")

    pose_parts: list[str] = []
    if p2 is not None:
        meta = _declared_metadata(p2)
        pose_parts.append(
            "P2 仅提供姿态与表情（HOW）；不采用 P2 的身份、服装或背景"
            + (f"，参考描述：{meta}" if meta else "")
        )
    else:
        pose_text = _derive_pose_text(shot)
        if pose_text:
            pose_parts.append(
                "无姿态/表情参考图；动作与表情仅由文本约束：" + pose_text
            )

    scene_parts = [f"shot.setting: {shot.setting}"]
    if shot.props:
        scene_parts.append("props: " + "、".join(shot.props))
    for record in scene_assets:
        meta = _declared_metadata(record)
        if meta:
            scene_parts.append(meta)

    camera_parts = [
        f"shot_scale: {shot.shot_scale}",
        f"camera_position: {shot.camera_position}",
        f"camera_motion: {shot.camera_motion}",
    ]

    light_parts: list[str] = []
    for record in scene_assets + style_assets:
        if record.time_of_day:
            light_parts.append(f"time_of_day: {record.time_of_day}")
        for fragment in (record.quality_notes or "").split("；"):
            if "光" in fragment:
                light_parts.append(fragment)
    if shot.setting and "光" in shot.setting:
        light_parts.append(f"shot.setting: {shot.setting}")

    style_parts: list[str] = []
    for record in style_assets:
        meta = _declared_metadata(record)
        if meta:
            style_parts.append(meta)

    preserve_parts: list[str] = []
    if p1 is not None:
        preserve_parts.append(
            "保持 P1 的身份特征：脸、五官、发型、发色、眼睛与服装"
            "（如 P1 服装需保留）；不保留 P1 的瞬时表情、视线或姿态状态；"
            "姿态、场景、机位与构图按目标约束改变"
        )
    if p2 is not None:
        preserve_parts.append(
            "姿态与表情遵循 Picture 2（preserve HOW from Picture 2）；"
            "P2 仅作为姿态与表情来源；不保留 P2 的服装、身份或背景"
        )
    else:
        preserve_parts.append("姿态与表情由文本 HOW 约束（无 Picture 2）")

    avoid_parts = [
        "不要融合或混用 P1 与 P2 的身份特征（identity fusion）",
        "不要复制 P2 的服装或外观细节（P2 outfit leakage）",
        "不要把 P1 源场景或其他参考图场景带入目标场景（source scene leakage）",
        "不要改变 P1 的身份特征",
        "不要引入未登记角色或道具",
        "不要添加文字、水印或 Logo",
        "本计划最多两张视觉参考，不要使用第三张参考图",
    ]

    return {
        "CHARACTER": "；".join(character_parts),
        "POSE_EXPRESSION": "；".join(pose_parts),
        "SCENE": "；".join(scene_parts),
        "CAMERA": "；".join(camera_parts),
        "COMPOSITION": shot.composition,
        "LIGHTING": "；".join(light_parts),
        "STYLE": "；".join(style_parts),
        "PRESERVE": "；".join(preserve_parts),
        "AVOID": "；".join(avoid_parts),
    }


def _compile_prompt(
    constraints: Mapping[str, str],
    *,
    has_p1: bool,
    has_p2: bool,
    review_notes: Sequence[str] = (),
) -> str:
    """Compile the I7-style deterministic Qwen prompt.

    Picture 1 is an identity-feature preserve (WHO), not a whole-pixel
    preserve: pose, scene and camera explicitly change toward the target
    shot.  ``review_notes`` are emitted only for NEEDS_REVIEW plans so a
    human can read the compiler's semantic concerns before anything is sent
    to a model.
    """

    lines = [
        (
            "Picture 1 defines the character identity (WHO). Preserve "
            "Picture 1's identity features: face, facial features, hairstyle, "
            "hair color and eye appearance, plus the outfit when it must be "
            "kept. Picture 1's pose, scene and camera are not preserved: "
            "when a Picture 2 is provided the pose follows Picture 2, "
            "otherwise the pose follows the text HOW; the scene follows the "
            "SCENE constraint, and camera/composition follow the shot "
            "constraints."
        )
    ]
    if has_p2:
        lines.append(
            "Picture 2 provides only the pose and expression (HOW). Do not "
            "copy Picture 2's identity, outfit, background or scene; apply "
            "only its pose and expression."
        )
    else:
        lines.append(
            "No Picture 2 is provided; pose and expression are text-only "
            "(HOW)."
        )
    lines.extend(
        [
            (
                "No third visual reference is used. Scene, camera, "
                "composition, lighting and style constraints are text-only "
                "(WHERE/CAMERA)."
            ),
            "",
            "Text constraints:",
        ]
    )
    for key in _TEXT_KEYS:
        value = constraints.get(key, "")
        label = _TEXT_LABELS[key]
        if not has_p1 and key == "CHARACTER":
            lines.append(f"- {label}: 缺少身份参考图，需要人工补充")
        else:
            lines.append(f"- {label}: {value if value else '（无）'}")
    lines.append("")
    if has_p2:
        lines.append(
            "Role summary: WHO from Picture 1; HOW from Picture 2; "
            "WHERE/CAMERA/STYLE from text."
        )
    else:
        lines.append(
            "Role summary: WHO from Picture 1; HOW from text; "
            "WHERE/CAMERA/STYLE from text."
        )
    if review_notes:
        lines.append("")
        lines.append("Review notes:")
        lines.extend(f"- {note}" for note in review_notes)
    return "\n".join(lines) + "\n"


def _semantic_issues(
    shot: ShotPlanEntry,
    p1: ImageAssetRecord | None,
    p2: ImageAssetRecord | None,
    constraints: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """Deterministic keyword checks; returns ``(warnings, criticals)``.

    Deliberately not NLP: only explicit, frozen tokens are matched.

    * identity gate + composition/camera that hides face or outfit
      (silhouette / back view / tiny) -> ``identity_observability`` warning;
    * a selected P2 that also carries identity/scene roles -> role-conflict
      warning (it still stays POSE_EXPRESSION only);
    * PRESERVE/AVOID that requires keeping the source scene while the shot
      defines a target scene -> critical contradiction;
    * AVOID that asks to change the identity while P1 must be preserved ->
      critical contradiction.
    """

    warnings: list[str] = []
    criticals: list[str] = []

    if p1 is not None:
        haystack = (
            f"{shot.composition} {shot.camera_position} "
            f"{shot.camera_motion}"
        ).lower()
        for token in _OBSERVABILITY_RISK_TOKENS:
            if token in haystack:
                warnings.append(
                    "identity_observability: composition/camera contains "
                    f"{token!r}, which reduces face/outfit observability; "
                    "this shot has a mandatory identity gate (subjects: "
                    f"{'、'.join(shot.subjects)}), so a human must confirm "
                    "the identity is observable or downgrade the identity gate"
                )
                break

    if p2 is not None:
        conflicting = sorted(
            role
            for role in ("identity_reference", "scene_reference")
            if role in p2.reference_roles
        )
        if conflicting:
            warnings.append(
                "P2 role conflict: the selected HOW asset also carries "
                + ", ".join(conflicting)
                + "; role isolation keeps it in POSE_EXPRESSION only"
            )

    if p2 is not None and not _p2_how_sufficient(p2):
        warnings.append(
            "P2 HOW metadata insufficient: the selected HOW asset has no "
            "explicit pose/expression description (body part, action and "
            "expression); provide a HOW description or replace P2 before "
            "sending to the model"
        )

    preserve = constraints.get("PRESERVE", "")
    avoid = constraints.get("AVOID", "")
    if shot.setting and any(
        phrase in preserve for phrase in _SCENE_PRESERVE_PHRASES
    ):
        criticals.append(
            "PRESERVE requires keeping the source scene while the shot "
            "defines a target scene; this is a critical contradiction"
        )
    if shot.setting and any(
        phrase in avoid for phrase in _SCENE_PRESERVE_PHRASES
    ):
        criticals.append(
            "AVOID forbids changing the scene while the shot defines a "
            "target scene; this is a critical contradiction"
        )
    if any(phrase in avoid for phrase in _IDENTITY_CHANGE_PHRASES):
        criticals.append(
            "AVOID asks to change the identity while PRESERVE must keep P1 "
            "identity; this is a critical contradiction"
        )
    return warnings, criticals


def plan_keyframe_generation(
    shot: ShotPlanEntry,
    bundle: list[Mapping[str, object]],
    catalog: ImageAssetCatalog,
) -> KeyframeGenerationPlan:
    """Compile one deterministic KeyframeGenerationPlan for a shot.

    ``bundle`` is the per-shot ReferenceBundle produced by
    ``validate_binding_against_plan`` (list of
    ``{asset_id, asset_type, path, note}``).  Only registered assets are
    considered; scene/style/prop assets never occupy a visual slot.
    """

    records = _bundle_records(bundle, catalog)
    p1, p1_reason, p1_conflict = _select_identity(shot, records)
    p2, p2_reason, p2_conflict = _select_pose_expression(records)

    issues: list[str] = []
    if p1 is None:
        decision: Decision = "UNRESOLVED"
    else:
        if p1_conflict:
            issues.append("ambiguous WHO")
        if p1.source_tier == "generated_candidate":
            issues.append("P1 is an unapproved generated_candidate")
        if p2 is not None:
            if p2_conflict:
                issues.append("ambiguous HOW")
            if p2.source_tier == "generated_candidate":
                issues.append("P2 is an unapproved generated_candidate")
        else:
            if not _derive_pose_text(shot):
                issues.append("missing HOW and no text derivation")

    constraints = _compile_constraints(
        shot,
        records,
        p1,
        p2,
        p1_conflict=p1_conflict,
        p2_conflict=p2_conflict,
    )
    warnings, criticals = _semantic_issues(shot, p1, p2, constraints)
    review_notes = warnings + criticals
    if p1 is not None:
        issues.extend(review_notes)
        decision = "READY" if not issues else "NEEDS_REVIEW"

    prompt = _compile_prompt(
        constraints,
        has_p1=p1 is not None,
        has_p2=p2 is not None,
        review_notes=review_notes,
    )

    visual_refs: list[VisualRef] = []
    if p1 is not None:
        visual_refs.append(
            VisualRef(
                slot="p1_identity",
                asset_id=p1.asset_id,
                path=str(p1.resolved_path),
                sha256=sha256_file(p1.resolved_path),
                tier=p1.source_tier,
                roles=list(p1.reference_roles),
                selection_reason=p1_reason,
                confidence=(
                    "low"
                    if p1_conflict or p1.source_tier == "generated_candidate"
                    else "high"
                ),
            )
        )
    if p2 is not None:
        visual_refs.append(
            VisualRef(
                slot="p2_pose_expression",
                asset_id=p2.asset_id,
                path=str(p2.resolved_path),
                sha256=sha256_file(p2.resolved_path),
                tier=p2.source_tier,
                roles=list(p2.reference_roles),
                selection_reason=p2_reason,
                confidence=(
                    "low"
                    if p2_conflict or p2.source_tier == "generated_candidate"
                    else "high"
                ),
            )
        )

    source_ids = sorted(record.asset_id for record in records)
    source_hashes = {
        record.asset_id: sha256_file(record.resolved_path)
        for record in records
    }
    return KeyframeGenerationPlan(
        decision=decision,
        shot_id=shot.shot_id,
        visual_refs=visual_refs,
        text_constraints=constraints,
        prompt=prompt,
        source_asset_ids=source_ids,
        source_hashes=source_hashes,
    )


def write_generation_artifacts(
    out_dir: Path,
    plan: KeyframeGenerationPlan,
    *,
    model: str = "Qwen-Image-Edit-2511",
    sampling: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write plan JSON, Qwen request JSON and prompt.txt deterministically."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "keyframe_generation_plan.json"
    request_path = out_dir / "qwen_request.json"
    prompt_path = out_dir / "prompt.txt"

    dump_json_atomic(
        plan_path,
        plan.model_dump(mode="json"),
        sort_keys=True,
    )
    repo_root = Path(__file__).resolve().parents[4]

    def _to_request_path(path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return path
        try:
            return candidate.relative_to(repo_root).as_posix()
        except ValueError:
            return path

    request = {
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "model": model,
        "sampling": dict(_DEFAULT_SAMPLING if sampling is None else sampling),
        "visual_refs": [
            {
                "slot": ref.slot,
                "asset_id": ref.asset_id,
                "path": _to_request_path(ref.path),
                "sha256": ref.sha256,
            }
            for ref in plan.visual_refs
        ],
        "text_constraints": plan.text_constraints,
        "prompt": plan.prompt,
        "source_asset_ids": plan.source_asset_ids,
        "source_hashes": plan.source_hashes,
    }
    dump_json_atomic(request_path, request, sort_keys=True)
    prompt_text = plan.prompt if plan.prompt.endswith("\n") else plan.prompt + "\n"
    prompt_path.write_text(prompt_text, encoding="utf-8", newline="\n")
    return {
        "plan": plan_path,
        "request": request_path,
        "prompt": prompt_path,
    }
