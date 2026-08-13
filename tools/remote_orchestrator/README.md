# Remote Codex Orchestrator（最小版）

这个工具把本机 Codex、SSH、远端 Git worktree 和远端 `codex exec` 串成一条受限的线性流水线：

```text
Local Orchestrator
        ↓ SSH（BatchMode）
Remote Git branch + isolated worktree
        ↓
Remote Codex（workspace-write）
        ↓
Commit + structured stage result
        ↓
Local Git evidence gate
        ↓
PASS 才进入下一个 Stage
```

它与 `src/anime_remix/` 完全隔离，只使用 Python 标准库、SSH、Git、Codex CLI、JSON 和 TOML。它不会运行 `git merge`、`git rebase`、`git push` 或任何自动 PR 操作。

## 前置条件

- 本机使用 Python 3.11～3.13；本仓库可直接使用 `.venv/Scripts/python.exe`。
- `ssh <配置中的 host> true` 必须可以在 `BatchMode=yes` 下无交互成功。VS Code 能通过密码提示登录并不等于无人值守 SSH 可用；夜间运行应配置公钥认证，但不要把私钥、密码或 token 写进仓库。
- 远端已安装 `git`、GNU `timeout` 和支持 `codex exec --sandbox workspace-write --json --output-schema --output-last-message` 的 Codex CLI。
- 远端主 checkout 默认必须干净；仅当 `allow_dirty_primary=true` 时，才允许保留一份经过审查并全程逐字节核验不变的脏基线。工具不会替你清理、stash、reset 或覆盖任何内容。
- 远端 Git 必须已有提交身份配置，否则 Remote Codex 的提交会被门禁阻止。

## 配置

复制示例，但不要提交本机目标配置：

```powershell
Copy-Item tools/remote_orchestrator/pipeline.example.toml `
  tools/remote_orchestrator/pipeline.local.toml
```

最小格式：

```toml
[remote]
host = "my-gpu"
repo = "/root/autodl-tmp/anime-remix-agent"
worktree_root = "/root/autodl-tmp/anime-remix-worktrees"
# 可选：当 SSH config 未绑定专用密钥时，在已忽略的 local TOML 中填写。
identity_file = "C:/Users/example/.ssh/dedicated_key"
allow_dirty_primary = false
connect_timeout_seconds = 15
stage_timeout_seconds = 900

[pipeline]
id = "remote-orchestrator-smoke"
base_branch = "main"
push = false

[[stages]]
id = "stage-01"
prompt = "prompts/example-stage-01.md"
depends_on = []

[[stages]]
id = "stage-02"
prompt = "prompts/example-stage-02.md"
depends_on = ["stage-01"]
```

第一版只接受严格线性依赖：第一个 Stage 的 `depends_on` 必须为空；之后每个 Stage 必须且只能依赖紧邻的前一个 Stage。`push=true` 会明确报 `CONFIG_ERROR`。

Stage ID 只允许 `[A-Za-z0-9_-]+`。远端路径必须是具体的绝对 POSIX 路径，仓库与 worktree root 不能相同或互相嵌套。Prompt 必须位于 pipeline TOML 所在目录内。

`identity_file` 是可选的本机绝对路径。配置后 SSH 固定增加 `IdentitiesOnly=yes` 和 `-i <path>`；dry-run 和日志只显示“已配置”，不打印路径或密钥内容。真实路径只应放在已忽略的 `pipeline.local.toml`，不要提交私钥。

`allow_dirty_primary` 默认为 `false`。只有当主 checkout 中存在经过人工识别、必须原样保留的既有成果时才可设为 `true`；此时 preflight 会对 tracked diff、Git status 和所有非 ignored untracked 文件内容建立 SHA256 baseline，并在每个 Stage 后重新计算。任一文件、HEAD、branch 或 status 改变都会停止 pipeline。Orchestrator 本身仍不修改 primary checkout。

## 命令

以下示例使用仓库的 Python 3.13 虚拟环境：

```powershell
.\.venv\Scripts\python.exe tools\remote_orchestrator\orchestrator.py `
  validate --pipeline tools\remote_orchestrator\pipeline.local.toml

.\.venv\Scripts\python.exe tools\remote_orchestrator\orchestrator.py `
  run --pipeline tools\remote_orchestrator\pipeline.local.toml --dry-run

.\.venv\Scripts\python.exe tools\remote_orchestrator\orchestrator.py `
  run --pipeline tools\remote_orchestrator\pipeline.local.toml

.\.venv\Scripts\python.exe tools\remote_orchestrator\orchestrator.py `
  status --pipeline tools\remote_orchestrator\pipeline.local.toml

# 仅用于受限的一次性错误续跑/已提交结果复核：
.\.venv\Scripts\python.exe tools\remote_orchestrator\orchestrator.py `
  retry --pipeline tools\remote_orchestrator\pipeline.local.toml stage-01
```

`--dry-run` 不连接 SSH、不创建本地 state、不创建远端 branch/worktree，也不运行 Codex。它只打印 host、repo、Stage 顺序、stacked base、branch、worktree 和 prompt。

Remote Codex 运行时，JSONL 与 stderr 会逐行脱敏后实时写入本地 Stage 目录。`status` 返回最近事件的 `type`、item 类型/状态和更新时间，不回显事件正文；需要人工深入检查时再打开对应的 `codex-events.jsonl`。

## Branch 与 worktree

每个 Stage 的分支固定为 `codex/<stage-id>`，worktree 固定为 `<worktree_root>/<stage-id>`。创建前同时检查 Git ref 和目标目录；任何已存在但无法由本地 PASS state 证明的对象都会触发 `USER_REVIEW_REQUIRED`，绝不 reset、force 或递归删除。

线性依赖使用 stacked branches：

```text
main
  ↓
codex/stage-01
  ↓
codex/stage-02
```

Stage 01 从 `pipeline.base_branch` 创建；Stage 02 从 `codex/stage-01` 创建。没有任何 Stage 会 merge 回 `main`。

## Remote Codex 调用与权限

每个 Stage 都是一个新的非交互执行上下文。Prompt 通过 SSH stdin 发送，不拼进远端 shell 字符串：

```text
codex exec
  (--approve-for-me，仅当远端 CLI 支持；其内置 workspace-write 自动审核)
  或 (--sandbox workspace-write)
  --json
  --output-schema <远端控制目录中的 schema>
  --output-last-message <远端 stage-result.json>
  -
```

CLI 支持时优先单独使用 `--approve-for-me`，因为当前 Codex CLI 将它与显式 `--sandbox` 设为互斥；该选项自身使用自动审核的 `workspace-write` 沙箱。不支持时才使用显式 `--sandbox workspace-write`。工具永远不会使用 `danger-full-access`、`--dangerously-bypass-approvals-and-sandbox` 或 `--yolo`。每个运行同时由远端 GNU `timeout` 和本机 subprocess timeout 限时。

## Stage result 与 PASS gate

`stage-result.schema.json` 要求以下字段：

```json
{
  "stage": "stage-01",
  "status": "pass",
  "branch": "codex/stage-01",
  "base_commit": "<git object id>",
  "head_commit": "<git object id>",
  "commit_created": true,
  "summary": "...",
  "tests": [{"command": "...", "passed": true}],
  "artifacts": ["relative/path"],
  "changed_files": ["relative/path"],
  "blocking_issue": null,
  "recommended_next_action": "continue"
}
```

只有 `status == "pass"` 才能继续。Local gate 还会独立核验：

- stage、branch、base SHA 与实际计划完全一致；
- `HEAD != base_commit`，且报告的 HEAD 等于实际 HEAD；
- `changed_files` 与实际 `git diff --name-only base..HEAD` 一致；
- 每个 artifact 路径真实存在；
- 所有报告测试均通过；
- `git status --porcelain` 为空；
- 远端主 checkout 的 branch、HEAD 和 status 从头到尾未变化。

`borderline`、`fail`、`blocked`、`needs_user_review`、无提交、脏 worktree、无效 JSON、SSH/Codex 失败都会立即停止，不跳过，也不自动重试。

## State 与 stage-level resume

本地状态保存在已忽略的：

```text
tools/remote_orchestrator/.state/<pipeline-id>/
  pipeline-state.json
  remote-preflight.json
  orchestrator.log
  <stage-id>/
    stage-state.json
    stage-result.json
    review.json
    effective-prompt.md
    codex-events.jsonl
    codex-stderr.log
    orchestrator.log
```

重新执行同一 pipeline 时，会先要求远端 primary 与上一次保存的 preflight 快照完全相同；随后已 PASS 的 Stage 会核验远端 branch、worktree、HEAD 和清洁状态后跳过，从下一个 Stage 继续。处于 error/FAIL/BLOCKED 的 Stage 不会偷偷重试。

`retry <stage-id>` 是唯一的受限例外，只处理 `status=error` 且 worktree 干净的两种可证明状态：① HEAD 仍等于 base 且没有结果文件，此时以新的 Codex 上下文运行一次；② HEAD 已产生提交且结果文件存在，此时不再运行 Codex，只重新校验原始结果、提交祖先关系、差异、artifact 和主目录基线。它会先证明远端 primary 的 branch、HEAD、status 和内容指纹与失败时完全相同，并证明 Stage branch/worktree 仍属于目标仓库。旧的 Stage state 会保存为 `retry-001-prior-state.json`。任一条件不满足、或这一次机会已经用过，都会停止并要求人工审查；不会循环重试，也不会 reset/delete 远端对象。终端 Stage 的 PASS 可以推荐 `stop`，非终端 Stage 的 PASS 仍必须推荐 `continue`。

修改 pipeline 或 prompt 后不能复用旧 state；使用新的 `pipeline.id`。这避免同名 Stage 在契约变化后被错误续跑。

## 人工 review 与 merge

在远端审查各分支：

```bash
git -C /root/autodl-tmp/anime-remix-agent log --oneline --decorate --graph --all
git -C /root/autodl-tmp/anime-remix-agent diff main..codex/stage-01
git -C /root/autodl-tmp/anime-remix-agent diff codex/stage-01..codex/stage-02
```

只有你完成人工审查后，才可自行决定如何 merge。Orchestrator 不提供 merge 子命令，也不会 push 分支。

## 当前限制

- 只支持一个远端、串行 Stage 和单依赖 stacked branches，不支持任意 DAG、并行执行或 session-level resume。
- 第一版不支持两个本机进程同时运行同一 pipeline；branch/worktree 的拒绝覆盖门禁可以防止静默破坏，但并发启动仍应视为需要人工 review。
- 除上述严格的一次性 error 续跑/结果复核外，不提供通用 retry；也不提供 push、PR、merge、Web UI、数据库、队列、daemon 或后台服务。
- 大型 ignored artifact 可以列在 `artifacts` 中，但 committed diff 必须完全由 `changed_files` 说明。
- v1 默认要求远端 primary checkout 干净；`allow_dirty_primary=true` 只负责证明既有内容全程未变，不会自动处理已有修改。
- 示例只验证两个无害 smoke commit。它不会下载模型、运行 GPU 或执行 Anime Remix G1-C1/G1-C2。

真实 Anime Remix pipeline 应在 smoke 链路人工验收后另建新的 TOML 和新的 Stage ID。建议先放“合同/任务包同步”Stage，再放单个受限实验 Stage；任何 capability gate 的 `borderline` 都应返回 `needs_user_review`，不能自动进入后续模型实验。
