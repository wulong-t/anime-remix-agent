# Anime Remix Agent

> 状态：**PAUSED / 学习归档**（2026-08-13）

这是一个探索“剧本 + 用户自有图片 → 可编辑动漫视频”的个人学习项目。项目已经
建立了一批可复用的规划、媒体和可恢复执行能力，但连续生成彩色锚点会累积降低
线稿、人物和背景质量，当前模型组合不足以稳定完成目标，因此主动暂停。

仓库保留的是以后真正可能复用的代码和结论，不包含用户私人图片、视频、模型、
密钥、付费运行产物或特定作品素材。

## 保留下来的能力

- 严格的图片资产清单、路径与权利状态校验；
- 剧本拆镜、镜头评审、参考图绑定与关键帧/生成片段计划；
- Qwen 图片请求编译、DashScope 执行边界与可恢复首帧/交接帧执行器；
- GeneratedShot 导入、24 fps 标准化、逐段恢复与最终拼接；
- legacy Timeline 1.9、FFmpeg Renderer、源指纹与媒体合同；
- manual-keyframe 可恢复 harness、Vidu 通用单任务 runner；
- 确定性 2D 分层合成原型和完全合成的 planner regression oracle。

## 最重要的实验结论

不要采用下面的递归链路：

```text
Generated K0 → edit → Generated K1 → edit → Generated K2
```

每次编辑都会重新采样整张图。即使 prompt 要求背景、脸、线稿和色块不变，它们
也不是像素级锁定项，误差会逐代累积。相同首尾图再配合禁止运动的 prompt，则会
稳定地产生近似静止的视频。更换 seed 或不断增加锚点并不能解决这个架构问题。

如果未来恢复，优先尝试：

```text
Canonical assets + motion/audio/camera controls → one independent shot
```

也就是 `Asset-First + Motion-Driven`：静态合成与运镜、2D rig/Live2D、音频驱动、
动作参考迁移，以及最后才使用一次性生成视频。生成结果只能是流水线叶子，不能再
成为下一轮生成的视觉真值。

## 目录

```text
src/anime_remix/                         产品与媒体代码
tests/                                   合成 fixture 与回归测试
experiments/layered_video_g0/            确定性分层视频原型
experiments/manual_keyframe_mvp/         可恢复手动关键帧 harness
experiments/phase3/                      通用 provider 调用脚本
experiments/i6_vidu_q2_pro/              通用 Vidu 单任务 runner
experiments/reference_planner_golden/    合成 reference-planner oracle
tools/remote_orchestrator/               受限远端调度工具
```

## 本地使用

要求 Python 3.11～3.13、`uv` 和 FFmpeg：

```powershell
uv sync --extra dev
uv run anime-remix --help
uv run python -m pytest tests/unit -q
uv run ruff check src tests tools experiments/manual_keyframe_mvp experiments/phase3 experiments/i6_vidu_q2_pro experiments/reference_planner_golden
```

所有真实媒体放在 gitignored 的 `runs/` 或显式 run 目录。密钥只从本机环境变量
读取，不应进入代码、日志、manifest 或聊天记录。

## 暂停边界

- 没有当前 active milestone，也没有待执行的付费任务；
- 不再继续生成 K_end、Vidu 片段或 Timeline 1.9 变更；
- 不抓取、提交或发布第三方动漫素材；
- 恢复项目前，先重新选择生产范式，再决定是否复用现有执行器。

完整的中间实验历史仍保留在本地旧开发分支中；归档分支只保存瘦身后的可复用
快照，避免把失败媒体和私人运行数据上传到 GitHub。
