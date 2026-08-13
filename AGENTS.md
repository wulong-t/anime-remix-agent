# Anime Remix Agent：暂停归档契约

> 状态：PAUSED
> 生效日期：2026-08-13（Asia/Shanghai）
> 目的：保存有复用价值的代码与结论，停止当前生成路线，等待未来重新设计。

## 1. 当前真值

项目没有 active milestone，也没有默认“下一阶段”。除非用户明确恢复项目，只做
维护、安全修复、依赖修复、文档整理和只读分析；不得自动启动图片/视频模型、远端
GPU、付费 API 或新的能力实验。

暂停原因不是单个模型或 prompt 偶发失败，而是连续生成彩色锚点会逐代重绘整张图，
导致人物、线稿、背景和色块发生累积漂移。相同首尾端点无法增加运动控制力，也不应
作为新的付费视频片段。

## 2. 保留架构

以下能力仍有独立价值：

- 图片资产清单、exact-path 读取、权利与格式校验；
- Director ShotPlan、评审、图片绑定、ReferenceBundle 与 planning contracts；
- 首帧/交接帧、provider adapter/executor、执行账本与恢复语义；
- GeneratedShot、标准化、Timeline 1.9 与 FFmpeg Renderer；
- deterministic layered-video、manual-keyframe、provider runner 与合成 regression oracle。

这些能力被保留不等于当前 Image-First 端到端路线已通过。

## 3. 不再采用的默认路线

```text
GeneratedFrame → image edit → GeneratedFrame → image edit → GeneratedFrame
```

生成结果不得作为后续生成的长期视觉真值。`reuse_previous` 等无视觉增量锚点不恢复；
叙事静止使用 renderer 的 freeze/hold，而不是同图首尾的视频推理。

## 4. 未来恢复方向

重新启动时，先做一个独立、低成本样本验证：

```text
Canonical assets + motion/audio/camera controls → independent final shot
```

优先级：确定性静态合成与运镜 → 2D rig/Live2D → 音频驱动 → 动作参考迁移 →
一次性参考生视频。只有新的生产范式通过单样本 Gate，才重启阶段路线。

## 5. 数据与外部动作

- `runs/` 和其他 gitignored run 目录只存本机私人媒体，不得自动加入 Git；
- 不读取或发送未被当前任务明确授权的媒体；
- secret 只在获授权调用时从环境变量读取，不输出、不持久化；
- 新外部目的地、新付费服务、扩大成本或样本数必须重新询问；
- 私人实验授权不等于公开发布或再分发授权；
- 仓库不得包含第三方动漫图片、视频、音频、模型权重或私人运行产物。

## 6. Git

- 保留用户未提交内容；禁止 destructive reset/checkout/clean；
- 普通本地 commit 可以直接执行；
- push、main merge、release 或公开分发必须有当次明确授权；
- 归档快照只保留源码、合成 fixture、测试和泛化文档，不推送中间媒体 blob；
- 本次归档以前的完整详细历史可留在本地开发分支，归档分支不承载它们。

## 7. 恢复条件

用户明确要求恢复时，先确认：目标输出、输入权利、生产范式、模型/远端、费用上限、
单样本 stop rule。不得直接从已拒绝的 K_end 或未完成 I7 状态继续。
