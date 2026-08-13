# Smoke Stage 01

这是无害的 Orchestrator 链路测试，不是 Anime Remix 产品任务。

只执行以下操作：

1. 确认当前 branch 是包装契约声明的 branch。
2. 创建 `.remote-orchestrator-smoke/stage01.txt`。
3. 文件内容必须精确为一行：`stage 01 complete`。
4. 用只读命令验证该文件内容。
5. 只提交这个文件，提交信息为 `test(orchestrator): complete smoke stage 01`。
6. 确认 `git status --porcelain` 为空。

禁止修改 `src/`、`tests/`、`pyproject.toml`、`AGENTS.md`、`README.md` 或任何实验文件。

最终 JSON 中：

- `artifacts` 与 `changed_files` 都只列出 `.remote-orchestrator-smoke/stage01.txt`；
- 测试列表记录实际执行的文件内容检查命令；
- 满足全部条件才可返回 `status: "pass"`。
