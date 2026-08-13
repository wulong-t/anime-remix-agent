"""I2 Director LLM experiment harness.

Frozen experiment contract:

- model: ``deepseek-v4-flash`` via the ``codex-deepseek`` CLI
  (provider ``deepseek``, reasoning effort ``high``);
- prompt: ``DIRECTOR_PROMPT_V1`` below;
- invalid-JSON repair: at most 2 retries carrying the parser/validator
  error back to the model; afterwards the run is a technical failure;
- the script bytes are hashed and recorded in the run manifest so a run
  can be reproduced and traced without copying user content.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic
from anime_remix.services.script.shot_plan import ShotPlanDocument, parse_shot_plan

PROMPT_VERSION = "director-prompt-v1"
MAX_REPAIR_TRIES = 2
COMMAND = ["cmd", "/c", "codex-deepseek", "exec", "-"]

DIRECTOR_PROMPT_V1 = """\
你是动画短片的导演。你只负责把剧本拆成镜头计划，不生成图片、不调用视频模型、不决定任何文件路径。

输入：
- 完整剧本（UTF-8 文本）。

输出要求：
只输出一个 JSON 对象，不要输出任何其他文字、解释、markdown 代码块或前后缀。
JSON 必须严格符合以下 schema（schema_version 固定为 "shot-plan-v1"）：

{
  "schema_version": "shot-plan-v1",
  "shots": [
    {
      "shot_id": "shot_001",
      "scene_id": "scene_01",
      "order": 1,
      "narrative_purpose": "本镜头的叙事目的",
      "duration_seconds": 4.5,
      "shot_scale": "close_up | medium | wide",
      "composition": "构图与画面元素布局",
      "camera_position": "机位（如：正面平视、侧面低机位）",
      "camera_motion": "镜头运动（如：固定、缓推、横移；静止写 fixed）",
      "subjects": ["在场人物或主体"],
      "setting": "场景与时间（如：黄昏的学校天台）",
      "props": ["关键道具"],
      "start_state": "镜头开始时画面与人物状态",
      "action_beats": [
        {"time_seconds": 0.0, "description": "起始动作"},
        {"time_seconds": 2.0, "description": "中间动作"}
      ],
      "end_state": "镜头结束时画面与人物状态",
      "emotion_arc": "本镜头情绪变化（如：平静到紧张）",
      "dialogue": "本镜头的对白原文；无对白则省略",
      "continuity_in": "承接上一镜头的事实（第一镜可省略）",
      "continuity_out": "传递给下一镜头的事实"
    }
  ]
}

规则：
1. 忠实于剧本事实，不编造剧本中没有的人物、事件或设定。
2. shot_id 唯一，order 从 1 连续递增。
3. duration_seconds 为正数，与动作、对白和叙事节奏匹配；不要机械地等长切镜。
4. action_beats 至少包含 time_seconds=0 的起始拍，时间非递减且不超过 duration_seconds；按需要增加中间拍，不固定数量。
5. 景别在 close_up / medium / wide 中三选一。
6. 描述必须具体到可以实际拍摄：构图、机位、运动、人物姿态与位置、场景光线。
7. 镜头之间保持空间、动作、视线与道具状态的连续性，用 continuity_in / continuity_out 说明承接事实。
8. 每个镜头只表达一个清晰的叙事节拍；避免过度切镜或一个镜头塞进过多事件。

以下是剧本：
---
__SCRIPT_TEXT__
---
"""


@dataclass(frozen=True)
class DirectorResult:
    """One validated Director run for a script."""

    script_name: str
    script_sha256: str
    prompt_version: str
    model_identity: str
    session_id: str | None
    tokens_used: int | None
    tries: int
    document: ShotPlanDocument


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_model_identity(stderr_text: str) -> str:
    model = re.search(r"^model:\s*(.+)$", stderr_text, re.MULTILINE)
    provider = re.search(r"^provider:\s*(.+)$", stderr_text, re.MULTILINE)
    reasoning = re.search(r"^reasoning effort:\s*(.+)$", stderr_text, re.MULTILINE)
    parts = []
    if model:
        parts.append(f"model={model.group(1).strip()}")
    if provider:
        parts.append(f"provider={provider.group(1).strip()}")
    if reasoning:
        parts.append(f"reasoning={reasoning.group(1).strip()}")
    return "; ".join(parts) or "unknown"


def _extract_session_id(stderr_text: str) -> str | None:
    match = re.search(r"session id:\s*([0-9a-f-]+)", stderr_text)
    return match.group(1) if match else None


def _extract_tokens(stderr_text: str) -> int | None:
    match = re.search(r"tokens used\s*([0-9,]+)", stderr_text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _extract_json(stdout_text: str) -> object:
    """Return the JSON object from model stdout, tolerating fenced noise."""

    stripped = stdout_text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise InputValidationError(
        "Director output is not valid JSON",
        field="stdout",
        actual=stdout_text[:400],
    )


def _call_director(script_text: str, error_hint: str | None = None) -> tuple[str, str]:
    """Invoke the frozen Director model once; return (stdout, stderr)."""

    prompt = DIRECTOR_PROMPT_V1.replace("__SCRIPT_TEXT__", script_text)
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


def run_director_for_script(
    script_path: Path,
    *,
    out_dir: Path,
) -> DirectorResult:
    """Run the Director on one script and persist raw outputs + manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    script_bytes = script_path.read_bytes()
    script_text = script_bytes.decode("utf-8-sig")
    if not script_text.strip():
        raise InputValidationError("script must not be empty", actual=script_path)
    script_sha = _sha256_bytes(script_bytes)

    last_stderr = ""
    last_error: str | None = None
    document: ShotPlanDocument | None = None
    tries = 0
    for attempt in range(1, MAX_REPAIR_TRIES + 2):
        tries = attempt
        stdout, stderr = _call_director(script_text, error_hint=last_error)
        last_stderr = stderr
        (out_dir / f"attempt_{attempt:02d}_raw_output.txt").write_text(
            stdout, encoding="utf-8"
        )
        try:
            document = parse_shot_plan(_extract_json(stdout))
            break
        except InputValidationError as exc:
            last_error = str(exc)

    if document is None:
        raise InputValidationError(
            f"Director failed after {tries} tries: {last_error}",
            actual=script_path,
        )

    result = DirectorResult(
        script_name=script_path.name,
        script_sha256=script_sha,
        prompt_version=PROMPT_VERSION,
        model_identity=_extract_model_identity(last_stderr),
        session_id=_extract_session_id(last_stderr),
        tokens_used=_extract_tokens(last_stderr),
        tries=tries,
        document=document,
    )
    manifest = {
        "schema_version": "director-run-v1",
        "prompt_version": PROMPT_VERSION,
        "script": script_path.name,
        "script_sha256": script_sha,
        "model_identity": result.model_identity,
        "session_id": result.session_id,
        "tokens_used": result.tokens_used,
        "tries": tries,
        "output": out_dir.name,
    }
    dump_json_atomic(out_dir / "run_manifest.json", manifest, sort_keys=True)
    dump_json_atomic(
        out_dir / "shot_plan.json",
        document.model_dump(mode="json"),
        sort_keys=True,
    )
    return result
