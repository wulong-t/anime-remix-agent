"""I3 Keyframe Planner LLM experiment harness.

Frozen experiment contract:

- model: ``deepseek-v4-flash`` via the ``codex-deepseek`` CLI
  (provider ``deepseek``, reasoning effort ``high``);
- prompt: ``KEYFRAME_PLANNER_PROMPT_V1`` below;
- input: one reviewed ``ShotPlanEntry`` (JSON) plus a deterministic text
  asset summary (IDs and types only, no image bytes);
- invalid-JSON repair: at most 2 retries carrying the validator error;
- the shot plan bytes are hashed and recorded in the run manifest.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic
from anime_remix.services.script.director import (
    _extract_json,
    _extract_model_identity,
    _extract_session_id,
    _extract_tokens,
)
from anime_remix.services.script.keyframe_plan import (
    KeyframePlanDocument,
    parse_keyframe_plan,
)
from anime_remix.services.script.shot_plan import ShotPlanEntry

PROMPT_VERSION = "keyframe-planner-prompt-v1"
MAX_REPAIR_TRIES = 2
COMMAND = ["cmd", "/c", "codex-deepseek", "exec", "-"]

KEYFRAME_PLANNER_PROMPT_V1 = """\
你是动画短片的关键帧规划师。你只根据一个镜头计划规划关键帧，不生成图片、不调用视频模型、不决定任何文件路径。

输入：
- 一个完整镜头计划（shot_plan.json 中的单个镜头对象，JSON 格式）；
- 可用的图片资产摘要（asset_id 与类型列表，只有文字，不包含图片）。

输出要求：
只输出一个 JSON 对象，不要输出任何其他文字、解释、markdown 代码块或前后缀。
JSON 必须严格符合以下 schema（schema_version 固定为 "keyframe-plan-v1"）：

{
  "schema_version": "keyframe-plan-v1",
  "shot_id": "镜头 ID，必须与输入一致",
  "shot_duration_seconds": 镜头时长（必须与输入一致）,
  "keyframes": [
    {
      "keyframe_id": "kf_001",
      "shot_id": "镜头 ID",
      "order": 1,
      "time_seconds": 0.0,
      "position": 0.0,
      "visual_description": "该时刻画面的完整视觉描述",
      "subject_pose": "人物姿态",
      "expression": "表情",
      "gaze": "视线方向",
      "composition": "构图",
      "camera": "机位与运动",
      "background_state": "背景状态",
      "foreground_state": "前景状态",
      "prop_state": "道具状态",
      "required_assets": [
        {"asset_id": "资产 ID", "asset_type": "character|background|foreground|prop|style", "locked_attributes": ["身份", "服装"]}
      ],
      "motion_from_previous": "相对上一关键帧发生的运动（第一帧写起点状态）"
    }
  ]
}

规则：
1. 第一帧必须 time_seconds=0、position=0；最后一帧必须 time_seconds=镜头时长、position=1。
2. time_seconds 严格递增且不重复；position 对应 time_seconds/镜头时长，范围 [0,1]。
3. 关键帧数量由内容复杂度决定，不固定：简单静止镜头可以 2～3 帧；有动作、表情、视线、道具、构图或机位变化时按变化点增加关键帧；复杂动作镜头可以更多。
4. 每个重要视觉状态变化（姿势、表情、视线、接触、道具、构图、机位）都需要锚点；删除语义重复、不能增加控制力的帧。
5. required_assets 只能引用输入资产摘要中存在的 asset_id，且 asset_type 必须与摘要一致。
6. 所有描述必须与输入镜头计划的事实一致，不新增镜头计划中没有的动作或道具。
7. motion_from_previous 描述相邻帧之间的运动；第一帧写镜头开始的静止或起始动作状态。

以下是镜头计划（JSON）：
---
{shot_json}
---

以下是可用图片资产摘要：
---
{assets_summary}
---
"""


@dataclass(frozen=True)
class PlannerResult:
    """One validated Keyframe Planner run for a shot."""

    shot_id: str
    shot_sha256: str
    prompt_version: str
    model_identity: str
    session_id: str | None
    tokens_used: int | None
    tries: int
    document: KeyframePlanDocument


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_asset_summary(assets: list[dict[str, str]]) -> str:
    """Render a deterministic text asset summary (IDs and types only)."""

    lines = [f"- {item['asset_id']} ({item['asset_type']})" for item in assets]
    return "\n".join(lines)


def _validate_asset_refs(
    document: KeyframePlanDocument,
    assets: list[dict[str, str]],
) -> None:
    by_id = {item["asset_id"]: item["asset_type"] for item in assets}
    for keyframe in document.keyframes:
        for ref in keyframe.required_assets:
            declared = by_id.get(ref.asset_id)
            if declared is None:
                raise InputValidationError(
                    f"keyframe {keyframe.keyframe_id} references unknown "
                    f"asset {ref.asset_id!r}",
                    asset_id=ref.asset_id,
                )
            if declared != ref.asset_type:
                raise InputValidationError(
                    f"keyframe {keyframe.keyframe_id} references asset "
                    f"{ref.asset_id!r} with wrong type "
                    f"({ref.asset_type!r} vs declared {declared!r})",
                    asset_id=ref.asset_id,
                )


def _call_planner(
    shot: ShotPlanEntry,
    assets_summary: str,
    error_hint: str | None = None,
) -> tuple[str, str]:
    """Invoke the frozen Keyframe Planner model once."""

    prompt = KEYFRAME_PLANNER_PROMPT_V1.replace(
        "{shot_json}", shot.model_dump_json()
    ).replace("{assets_summary}", assets_summary)
    if error_hint:
        prompt += (
            "\n\n你上一次的输出未通过校验，错误如下。请只重新输出修正后的 JSON：\n"
            f"{error_hint}\n"
        )
    completed = subprocess.run(
        COMMAND,
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=900,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    return stdout, stderr


def run_planner_for_shot(
    shot: ShotPlanEntry,
    *,
    assets: list[dict[str, str]],
    out_dir: Path,
) -> PlannerResult:
    """Run the Keyframe Planner on one shot and persist raw outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    shot_bytes = shot.model_dump_json().encode("utf-8")
    shot_sha = _sha256_bytes(shot_bytes)
    assets_summary = build_asset_summary(assets)

    last_stderr = ""
    last_error: str | None = None
    document: KeyframePlanDocument | None = None
    validated = False
    tries = 0
    for attempt in range(1, MAX_REPAIR_TRIES + 2):
        tries = attempt
        stdout, stderr = _call_planner(
            shot,
            assets_summary,
            error_hint=last_error,
        )
        last_stderr = stderr
        (out_dir / f"attempt_{attempt:02d}_raw_output.txt").write_text(
            stdout, encoding="utf-8"
        )
        try:
            document = parse_keyframe_plan(_extract_json(stdout))
            _validate_asset_refs(document, assets)
            validated = True
            break
        except (InputValidationError, KeyError, TypeError) as exc:
            last_error = str(exc)

    if not validated:
        raise InputValidationError(
            f"Keyframe Planner failed after {tries} tries: {last_error}",
            actual=shot.shot_id,
        )

    result = PlannerResult(
        shot_id=shot.shot_id,
        shot_sha256=shot_sha,
        prompt_version=PROMPT_VERSION,
        model_identity=_extract_model_identity(last_stderr),
        session_id=_extract_session_id(last_stderr),
        tokens_used=_extract_tokens(last_stderr),
        tries=tries,
        document=document,
    )
    manifest = {
        "schema_version": "keyframe-planner-run-v1",
        "prompt_version": PROMPT_VERSION,
        "shot_id": shot.shot_id,
        "shot_sha256": shot_sha,
        "model_identity": result.model_identity,
        "session_id": result.session_id,
        "tokens_used": result.tokens_used,
        "tries": tries,
        "keyframes": len(document.keyframes),
        "output": out_dir.name,
    }
    dump_json_atomic(out_dir / "run_manifest.json", manifest, sort_keys=True)
    dump_json_atomic(
        out_dir / "keyframe_plan.json",
        document.model_dump(mode="json"),
        sort_keys=True,
    )
    return result
