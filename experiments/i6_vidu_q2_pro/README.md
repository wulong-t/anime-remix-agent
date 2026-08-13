# Vidu start/end-frame runner（归档）

`run_i6_vidu.py` 是一个可复用的单任务安全 runner，用于执行已经冻结并获授权的
`vidu/viduq2-pro_start-end2video` 合同。仓库不提供真实媒体或已授权合同。

runner 会校验：

- 两个精确输入路径和 SHA256；
- 模型、地域、prompt、时长、分辨率、seed 与 watermark；
- 最多一个异步任务、一个输出和无自动重试；
- DashScope API Key 与 Workspace ID 只从环境变量读取；
- 下载后使用 ffprobe 校验媒体合同，并保存不含 secret/签名 URL 的 manifest。

只读 dry-run：

```powershell
uv run python experiments/i6_vidu_q2_pro/run_i6_vidu.py `
  --contract path/to/private-request-contract.json `
  --dry-run `
  --run-dir path/to/gitignored-run
```

真实执行必须由用户对精确输入、外部目的地、模型、任务数和费用重新授权，并显式
传入 `--execute-paid`。项目当前处于暂停状态，不存在可直接继续的授权任务。
