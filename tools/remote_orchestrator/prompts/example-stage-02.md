# Smoke Stage 02

这是依赖 Stage 01 的无害 Orchestrator 链路测试，不是 Anime Remix 产品任务。

只执行以下操作：

1. 确认 `.remote-orchestrator-smoke/stage01.txt` 已存在，内容精确为 `stage 01 complete`；缺失时返回 `blocked`，不得重建它。
2. 创建 `.remote-orchestrator-smoke/stage02.txt`。
3. 新文件内容必须精确为一行：`stage 02 complete`。
4. 用只读命令验证两个文件内容。
5. 只提交 Stage 02 新增的文件，提交信息为 `test(orchestrator): complete smoke stage 02`。
6. 确认 `git status --porcelain` 为空。

禁止修改 `src/`、`tests/`、`pyproject.toml`、`AGENTS.md`、`README.md` 或任何实验文件。

最终 JSON 中：

- `artifacts` 列出两个 smoke 文件；
- `changed_files` 只列出 `.remote-orchestrator-smoke/stage02.txt`；
- 测试列表记录实际执行的继承检查和文件内容检查命令；
- 满足全部条件才可返回 `status: "pass"`。
