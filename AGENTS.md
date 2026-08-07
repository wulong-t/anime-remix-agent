# Anime Remix Agent：Codex 实施契约（v1.15，Renderer-First MVR）

> 文件名：`AGENTS.md`
> 文档版本：`1.15`（第二阶段 G0 视频生成模型可行性实验契约；状态同步：第一阶段完成并冻结，阶段 B 完成并冻结，第二阶段开始规划，G0 产品契约已补齐、代码未开始；JSON `schema_version` 仍为 `1.9`；文档版本与 JSON schema_version 是两个概念，不得混为一谈。）
> 适用范围：使用 Codex 在 Python 仓库中实现 Anime Remix Agent。
> 当前目标：第一阶段与阶段 B 完成并冻结；第二阶段 G0（视频生成模型可行性实验）产品契约已补齐，等待用户审核后再进行最新模型调研与实验。
> 当前停止点：G0 产品契约完成；尚未选择具体视频生成模型；尚未调用视频生成 API；等待用户审核后进行模型调研；不开始正式集成。
> 核心原则：**先证明 renderer，再让 planner 驱动 renderer；先交付闭环，再增加策略和硬化。**

规范优先级：

```text
第 0～18 节的完成定义、边界、模型和不变量
> 第 19 节的实施批次
> 第 20 节的可复制指令
> 示例和说明文字
```

发生冲突时：

1. 执行更具体、可测试的规则；
2. 不自行扩大范围；
3. 在完成报告中指出冲突；
4. 若冲突会阻止当前批次，停止在当前批次，不进入下一批。

---

## 0. Grill-Me 后锁定的父决策

以下决策已经按“父决策 → 依赖问题 → 推荐答案 → 最小验证”逐分支审查。实现中不得自行反转。

### 0.1 产品边界

1. 第一阶段是**已有素材重混工具**，不是生成式动漫模型；
2. 首个版本只解决单机、离线、命令行流程；
3. 第一阶段不实现 Web、数据库、后台任务、多人协作或远程服务；
4. 用户只能导入其有权使用的素材，项目不抓取、提供或分发动漫素材。

### 0.2 最小成果边界

最小可运行版本必须具备：

```text
script.md + clips.json + 本地 MP4
→ rule parser
→ 稳定检索和选择
→ timeline.json
→ output.mp4
→ 修改 timeline.json 后独立 rerender
```

最小版本不以以下能力为完成条件：

```text
freeze_frame（第 3A 批单独授权，不属于第 2 批 MVR 门槛）
speed_adjust
复杂别名
情绪和景别评分（第 3B-2 批单独授权，不属于第 2 批 MVR 门槛）
任意来源视频兼容
VFR / 旋转 / 非方形 SAR / BT.601 转换
覆盖已有运行目录
source drift override
完整构建指纹
LLM
字幕、文字 placeholder、配音和音乐
```

### 0.3 批次顺序

```text
第 0 批：验证 FFmpeg 假设
第 1 批：先实现独立 render，使用手写 timeline 生成真实 MP4
第 2 批：实现 planner/build，驱动已经验证的 render，完成 MVR
第 3A 批：freeze_frame 产品策略与渲染闭环（已实现并人工验收）
第 3B-1 批：aliases.json 人物和地点别名（已实现并人工验收）
第 3B-2 批：情绪和景别提取与评分（已实现并人工验收）
第 3B-3 批：更完整 selection trace、count_frames 审计和 manifest 补强（已实现并人工验收）
阶段 B-1：检索质量集（已实现并验收）
阶段 B-2：30×1000 retrieval 压力测试（已实现并验收）
第二阶段 G0：视频生成模型可行性实验（产品契约已补齐，代码未开始）
```

原因：

- 媒体渲染、拼接、时间基和 AAC 是最高技术风险；
- parser 和 retrieval 即使正确，也不能证明最终 MP4 可用；
- `render` 是 `build` 的稳定下游接口，应先被真实验证；
- 手写 timeline 可以隔离 planner 与 renderer，失败时更容易定位。

### 0.4 策略模型

第 2 批 MVR 只保留两个 renderer 策略：

```text
clip
placeholder
```

第 3A 批契约新增第三个策略：

```text
freeze_frame
```

`reuse` 与 `trim` 在渲染层使用相同过滤图，差异只在 planner 选择的源帧区间，因此不应成为两个 renderer 分支。

planner 用 `reason_code` 区分：

```text
exact_length
center_trim
short_source_freeze
no_candidate
```

`freeze_frame` 由第 3A 批契约定义（§1.4、§9.1、§9.6、§11.6、§11.7、§12.4）；在代码实现完成前不得启用，也不影响已验收的第 0～2 批。

### 0.5 帧模型

最小版本固定：

```text
输出：1280×720、24 fps
输入：CFR 24 fps
```

因此时间线使用整数源帧和目标帧，不使用秒作为渲染真值：

```text
source_in_frame
source_frame_count
target_frames
```

这样可以删除首个版本中的多套秒数舍入、时长误差和二次量化。

### 0.6 素材时长

`clips.json` 不要求用户填写 `duration_seconds`。

媒体时长和帧数来自 `ffprobe`。删除人工时长字段可以避免：

- 重复标注；
- 声明值与媒体值漂移；
- 在 MVR 中增加无必要的 mismatch 失败分支。

### 0.7 媒体与音频

1. 所有标准化片段均为 video-only；
2. 不为每段编码 AAC；
3. 所有片段验证通过后，以 concat demuxer 流复制视频；
4. 全片只在最终 mux 时编码一次连续静音 AAC；
5. 帧数是视频真值，音频样本数由总帧数精确推导。

### 0.8 发布边界

1. `build --output` 在 MVR 中必须指向不存在的新目录；
2. preflight、probe、parse、retrieve 和 timeline 编译成功前不创建 staging；
3. 在 target 同父目录创建 staging；
4. staging 内完成渲染和验证后再 rename 为 target；
5. MVR 不实现 `build --overwrite`、backup 或 rollback；
6. 独立 `render` 输出单个 MP4，已存在时必须显式 `--overwrite`。

工程优先级：

```text
闭环可运行 > 媒体正确性 > 可测试性 > 可解释性 > 可复现性 > 性能 > 扩展性
```

删除优先原则：

```text
能通过删除功能、固定参数、限制输入或后移需求解决的问题，
不通过增加抽象层、状态机、插件、回滚分支或通用框架解决。
```

---

## 1. 完成定义

### 1.1 第 0 批：可行性门禁

必须在当前环境真实验证：

- FFmpeg、ffprobe、libx264、AAC 和所需过滤器存在；
- video-only 片段可以独立编码；
- 不同长度片段在其他 concat 签名字段一致时可以 concat demuxer + `-c:v copy`；
- 记录每个片段的实际 `avg_frame_rate`；不要求不同片段必须产生不同值；
- 总视频帧数等于片段帧数之和；
- 单个 `anullsrc` 可以按精确样本数编码一次 AAC；
- 最终音视频 start time 为 0；
- 最终音频为静音；
- 输出颜色标签完整为 BT.709 limited；
- `avg_frame_rate` 不作为 concat 兼容性字段（不得进入 concat signature）。

第 0 批不是产品完成点，但它是第 1 批的硬前置。

### 1.2 第 1 批：Renderer Walking Skeleton

输入一份手写 `timeline.json`：

```bash
python tests/fixtures/generate_render_smoke.py \
  --output .tmp/render-smoke-fixture

anime-remix render \
  --timeline .tmp/render-smoke-fixture/timeline.json \
  --output runs/render-smoke.mp4
```

必须生成真实 MP4，并满足：

- 支持 `clip` 和纯黑 `placeholder`；
- 片段均为 video-only H.264；
- 最终只编码一次 AAC；
- 每段和最终帧数实际校验；
- timeline 修改顺序或源帧区间后可以重新渲染；
- 不读取 script、clips.json、YAML、aliases 或 retrieval 代码；
- 媒体测试零跳过。

第 1 批完成后已经有真实可播放视频，但还不是完整 MVR，因为尚未从剧本自动生成 timeline。

### 1.3 第 2 批：最小可运行版本（已验收）

输入：

```text
demo/script.md
demo/clips.json
demo/clips/*.mp4
```

构建命令：

```bash
anime-remix build \
  --script demo/script.md \
  --clips demo/clips.json \
  --output runs/demo-001 \
  --parser rule
```

输出：

```text
runs/demo-001/
├── .anime-remix-run
├── run_manifest.json
├── parsed_script.json
├── retrieval_results.json
├── timeline.json
├── normalized/
├── render.log
└── output.mp4
```

独立重渲染：

```bash
anime-remix render \
  --timeline runs/demo-001/timeline.json \
  --output runs/demo-001-rerendered.mp4
```

第 2 批必须满足：

- 规则模式完全离线；
- 支持 3～10 个非空剧本段落；
- 支持最多 50 条人工标注素材；
- 每段稳定生成一个 `ShotRequirement`；
- 每段稳定选择一个可渲染素材或纯黑 placeholder；
- 源素材足够长时选择精确长度或中心裁剪区间；
- 源素材不足时继续扫描候选，不直接锁死第一名；
- 无候选通过时使用 placeholder；
- timeline 可编辑顺序和合法源帧区间；
- build 和独立 rerender 实际成功；
- 最终视频有一个 H.264 视频流和一个 AAC 静音音频流；
- 每段和最终 `nb_read_frames` 正确；
- 媒体测试不得跳过；
- 完成后停止，不自动进入第 3A 批。

### 1.4 第 3A 批：freeze_frame 产品策略与渲染闭环（已实现并人工验收）

第 3A 批在已验收的第 0～2 批之上增加 `freeze_frame` 产品策略。完成门槛：

- `TimelineStrategy.FREEZE_FRAME = "freeze_frame"` 可用，并与 clip、placeholder 共存；
- 检索在内容门槛通过后按帧数资格区分 clip eligible / freeze eligible / too_short；
- 产品优先级固定为 clip > freeze_frame > placeholder；
- planner 生成 freeze_frame 时写入 §11.7 固定源字段；
- 独立 render 使用 §12.4 独立过滤图，且只读取 timeline 和源文件；
- freeze_frame 片段仍为 video-only，最终只编码一次 AAC；
- 含 freeze_frame 的 timeline 仍使用 `schema_version: 1.9`，旧 clip/placeholder timeline 保持兼容；
- 媒体测试不得跳过。

第 3A 批已完成并人工验收，不再作为进行中批次。

### 1.5 第 3B-1 批：aliases.json 人物和地点别名（已实现并人工验收）

第 3B-1 批在已验收的第 3A 批之上增加 `aliases.json`。完成门槛：

- `validate` / `build` 支持可选 `--aliases`，省略时与第 3A 批行为完全一致；
- aliases 只扩展规则解析器的人物/地点词典，输出始终是 canonical ID/name；
- 人物与地点别名分别校验 target 存在性与别名唯一性；
- aliases 静态验证失败在 probe、parse、retrieve 和 staging 创建前失败；
- manifest 增加 `aliases_sha256`（未提供为 null），`core_artifact_sha256` 定义不变；
- 不改变评分、检索、renderer 与 source fingerprint；
- 媒体测试不得跳过。

第 3B-1 批已完成并人工验收，不再作为进行中批次。

### 1.6 第 3B-2 批：情绪与景别提取和评分（已实现并人工验收）

第 3B-2 批在已验收的第 3B-1 批之上增加两个可选语义维度 `emotion` 与 `shot_scale`。完成门槛：

- `Emotion` 与 `ShotScale` 是封闭枚举，只允许 §9.1 固定值；`null` 表示未指定/未识别，`null != calm`；
- `ClipAsset` 与 `ShotRequirement` 新增两个可选字段，省略等于 null；旧 clips.json / timeline.json 没有这两个字段时必须继续兼容；
- rule parser 使用代码内固定、封闭的小型词典做有限、确定性提取；不得使用 LLM、Embedding、sentiment model、CV、人脸识别、模糊匹配或用户自定义情绪词典；
- 检索为 exact categorical match：requirement 未指定时该维度 inactive 且 score 为 null；指定后素材完全相同得 1，不同或素材缺失得 0；
- 基础权重扩展为六维：character 0.20 / location 0.12 / action 0.36 / duration 0.12 / emotion 0.10 / shot_scale 0.10；两新维度均 inactive 时按活跃权重重新归一化必须精确恢复旧四维权重 0.25 / 0.15 / 0.45 / 0.15；
- 不新增 emotion / shot_scale 硬门槛；内容门槛保持 total >= 0.55、人物 character >= 0.50、action >= 0.25，emotion / shot_scale 只通过 total 影响排名和最终 gate；
- 稳定全量排序不新增 tie-break；emotion / shot_scale 已体现在 total 中；
- timeline 中 `requirement.emotion` / `requirement.shot_scale` 与 `score.emotion` / `score.shot_scale` 属于 planner metadata 和选择解释信息；renderer 不得读取它们决定 FFmpeg 行为；
- JSON `schema_version` 仍为 `1.9`；实现后正式 JSON 显式写入 null 字段，旧核心 JSON 字节不要求与第 3B-1 完全一致，但旧输入的规划语义、score 数值、selection、source 区间和 output.mp4 必须完全不变；包含第 3B-2 新字段的正式 JSON 需要已实现第 3B-2 的应用版本读取；
- 实现阶段增加无版权合成 demo（例如 `demo/semantic/`，至少三镜头：emotion 区分、shot_scale 区分、双 inactive 回归；可同时使用 aliases，但不得依赖 aliases 才能验证 emotion / shot_scale）；
- 媒体测试不得跳过。

第 3B-2 批已完成并人工验收，不再作为进行中批次。

### 1.7 第 3B-3 批：selection trace / count_frames / manifest 补强（已实现并人工验收）

第 3B-3 批只解决两个问题：

```text
为什么选了这个素材？
最终选中的媒体是否真的拥有 planner 依赖的帧数？
```

增加三个能力：

1. 完整、确定性的 selection trace（写入 `retrieval_results.json` 的每个 shot，不新建文件）；
2. 对最终选中的唯一源素材执行真实 `ffprobe -count_frames` 审计；
3. manifest 增加可独立复算的 `selected_source_frame_audit` 与 `core_artifact_member_sha256`。

完成门槛：

- 三个能力只用于审计、调试、可解释性和构建正确性门禁，不得提高或改变检索能力；
- 相同输入下 selection 语义完全不变：parsed_script、ScoreBreakdown、active_weights、total、global ranking、hard gates、frame gate、selected asset、strategy、source_in_frame、source_frame_count、target_frames 和 output.mp4 全部不变；selection trace 只观察已有选择过程，count_frames 只验证已经选出的源，manifest 只记录结果，任何新审计字段不得反向参与评分或 selection；
- `selection_trace` 位于 `retrieval_results.json` 每个 shot 结果中，结构见 §11.8；不创建 `selection_trace.json`，`core_artifact_sha256` 的“三个核心文件”定义不变；
- count_frames 只对最终选中的唯一 `asset_id` 执行（clip 与 freeze_frame；placeholder 不执行），按 `asset_id` UTF-8 字节序升序串行执行，最多 `min(非 placeholder shot 所用 unique asset 数, 10)` 次；
- `metadata_nb_frames == counted_nb_frames` 必须严格成立（不允许 ±1、duration 或 avg_frame_rate 替代）；不一致时构建失败，使用现有媒体/环境错误体系（退出码 3），且不创建 staging、不执行 render、不重新 retrieval、不 fallback、不 placeholder、不修改 freeze 的 source_frame_count；
- count_frames 审计发生在 staging 创建之前；count mismatch 不创建 staging 和 target；
- manifest 新增 `selected_source_frame_audit`（未选中任何真实源时为 `{}`，不是 null）和 `core_artifact_member_sha256`（running 时为 null，planning artifacts 写完后可计算，succeeded 时必须为最终值），详见 §15.4；
- 独立 render 不读取 retrieval_results.json、selection_trace、run_manifest.json 或 audit 字段，仍只读取 timeline 与 timeline 引用的源；build-only 的 count audit 不在 render 中重复；
- 第 3B-3 不新增任何 CLI 参数；trace 与 count_frames 由 build 默认执行，validate / render 行为不变；
- JSON `schema_version` 仍为 `1.9`；selection_trace 使 `retrieval_results.json` 字节变化，`core_artifact_sha256` 自然变化，但 selection、source ranges 和 output.mp4 必须不变；
- 媒体测试不得跳过。

第 3B-3 批已完成并人工验收，不再作为进行中批次。

### 1.8 第一阶段：Planner / Renderer 主闭环（完成并冻结）

第一阶段（第 0～2 批 + 第 3A + 第 3B-1 + 第 3B-2 + 第 3B-3）已全部验收：

```text
第 0～2 批：通过
第 3A freeze_frame：通过
第 3B-1 aliases：通过
第 3B-2 emotion / shot_scale：通过
第 3B-3 selection trace / count_frames / manifest：通过
Planner / Renderer 第一阶段主闭环：完成并冻结
```

JSON `schema_version` 仍为 `1.9`。阶段 B 不增加新的正式产品 JSON schema，也不修改第一阶段任何已验收语义。第一阶段产品规则冻结：clip / freeze_frame / placeholder、parser、aliases、emotion、shot_scale、scoring、ranking、gates、selection trace、renderer、FFmpeg 与 manifest 语义均不再因后续实验改变。

### 1.9 阶段 B：检索质量集 + 30×1000 压力测试（已实现并验收）

阶段 B 不增加产品功能，只回答两个问题：

```text
当前 deterministic retriever 到底“选得对不对”？
在 30 个 shot × 1000 个候选下是否仍然稳定、确定、性能可接受？
```

- B1 检索质量集：纯测试/benchmark 数据集（`tests/quality/` 或仓库现有测试结构下等价位置），至少 30 个人工锁定的 retrieval case；expected truth 必须由测试数据作者显式写出，不得用当前输出自动快照；初始门槛 `selection_accuracy` / `strategy_accuracy` / `reason_code_accuracy` 均为 100%，这只是“当前人工定义的最小规则质量集全部符合产品契约”，不代表真实世界检索准确率；
- B2 30×1000 压力测试：30 个 ShotRequirement × 1000 个内存 ProbedClip = 30,000 pair；不使用真实 MP4，不执行 ffprobe / ffmpeg / SHA256 / count_frames / render / staging / manifest；只测 scoring、normalization cache、stable sorting、gate scan 与 selection trace；
- 完整契约见 §18.10；实施批次见 §19.9；可复制指令见 §20.8；README 后续要求见 §22；路线图见 §23；
- 禁止为让 benchmark 数字变好而修改 scoring、weights、hard gates、sorting、SequenceMatcher 阈值、selection、selection trace、renderer、FFmpeg 或 manifest 产品语义；禁止并发、multiprocessing、numpy、pandas、psutil 或缓存框架；发现性能问题先记录可复现 benchmark，优化作为单独的 B3 批授权；
- 质量集与 30×1000 determinism 是普通 pytest 硬门禁；wall-clock 性能通过独立 benchmark marker（例如 `pytest -m benchmark`）在当前开发机真实执行，不作为跨机器严格 CI 门槛；
- 不新增 pytest-benchmark / asv / pyperf / pandas / numpy / psutil 等依赖；
- 阶段 B 不得接入视频生成 API、generate strategy、prompt compiler、reference image、模型 provider、云 API 或 API key；
- 阶段 B 完成后，第一阶段 Remix Planner / Renderer + retrieval baseline 正式冻结，下一阶段才允许规划“第二阶段：视频生成模型接入”。

阶段 B 已完成并验收，正式冻结：retrieval 产品规则不再修改，性能优化只能作为独立 B3 批单独授权。

### 1.10 第二阶段 G0：视频生成模型可行性实验（产品契约已定义，代码未开始）

G0 的唯一目标：

```text
当前可获得的视频生成模型是否能够为 Anime Remix Agent 生成可用的单镜头素材？
```

G0 不是完整 Agent 集成、自动生成动漫、多镜头生成、完整人物一致性系统或正式 provider abstraction；第一步只验证单镜头。

最小生成任务（GenerationRequest，实验用，不是正式产品 JSON）至少描述：

```text
character_reference
scene
action
emotion
shot_scale
duration_seconds
aspect_ratio
```

第一版固定：`duration_seconds = 3`、`aspect_ratio = 16:9`，对应当前系统的 72 frames @ 24fps。生成模型不要求直接输出 24fps，原始输出可以经过现有 normalization 思路转换。

第一轮只测试三类固定无版权镜头：

```text
Test A：静态人物 + 简单动作
  原创动漫风人物站在屋顶，轻微转头，平静表情，medium shot
  目标：验证基础人物稳定性和简单运动

Test B：人物明显动作
  原创角色向前跑几步，镜头保持 wide shot
  目标：验证人体运动、肢体稳定性

Test C：近景表情
  原创角色 close-up，从平静变为惊讶
  目标：验证脸部、表情和近景稳定性
```

不得使用受版权保护动漫角色。

每个模型至少记录：

```text
provider
model
generation mode
input type
reference-image support
requested duration
actual duration
requested aspect ratio
actual resolution
actual fps
actual frame count
generation latency
generation success/failure
```

如可获得还记录 `estimated cost`，但不得假造价格。

每个生成视频必须人工检查 10 项，使用简单等级 `pass | borderline | fail`，不伪装成精确视觉质量分数：

```text
1. 主体是否保持同一人物
2. 脸是否明显崩坏
3. 手和肢体是否明显异常
4. 动作是否符合要求
5. 背景是否出现明显漂移
6. shot_scale 是否符合要求
7. emotion 是否符合要求
8. 是否存在明显闪烁
9. 是否有突然换人
10. 是否适合作为后续 Remix pipeline 的源素材
```

最关键的实验是 reference consistency：

- 如果模型支持 image-to-video 或 reference image，必须优先测试；
- 至少要求：同一张原创人物参考图 → 分别生成 Test A / B / C；
- 人工比较头发、脸型、服装、颜色和主要身份特征；
- G0 不要求彻底解决人物一致性，只测出当前模型的一致性上限。

生成结果与现有 pipeline 的接口实验：

- 生成视频成功后不得立即修改 Anime Remix 产品代码；
- 先手工验证：generated video → ffprobe → 是否满足现有输入合同；
- 如果不满足，尝试通过现有 FFmpeg normalization 思路转换（分辨率、fps、SAR、pixel format、BT.709、去音频）；
- 目标：生成一个最终能成为 ClipAsset 源的受限 MP4；
- 必须区分“模型原始输出”与“标准化后的 pipeline input”。

G0 不要求模型直接满足 MVR media contract：模型可能输出 30fps、不同分辨率、不同 time_base、带音频或不同编码参数，这不直接说明模型不可用；G0 要回答这些输出能否稳定标准化为 H.264 1280×720 24fps CFR yuv420p SAR 1:1 BT.709 limited。暂时不修改现有“输入只接受 24fps”的产品契约；生成模型 adapter 的标准化留到后续阶段。

G0 不实现 `TimelineStrategy.GENERATE`，不修改 clip / freeze_frame / placeholder，现有 planner 完全冻结；生成视频只作为实验文件。

G0 不设计复杂 provider abstraction：第一轮最多一个很薄的实验脚本或人工调用模型；不建立 VideoGenerationProvider、ProviderRegistry、PluginSystem、Factory、DI container 或通用多模型框架；先证明一个模型真的能用。

模型选择原则：

- 执行 G0 前，先基于当时可用的模型做一次最新调研；
- 比较重点：程序化 API、text-to-video、image-to-video、reference image / character consistency、输出时长、输出分辨率、生成延迟、价格、API 使用限制、可否合法用于用户自己的原创/授权素材；
- 不得根据旧模型知识直接锁死 provider；模型信息必须在执行 G0 时重新核验。

实验产物建议放在：

```text
experiments/video-generation/
  README.md
  requests/
  outputs/
  probes/
  results.json
```

这些是研发实验产物，不是正式产品 schema。

`results.json` 至少记录：

```text
experiment_id
provider
model
request
success
latency_seconds
raw_output_probe
normalized_output_probe
manual_evaluation
notes
```

不得加入 API key、完整环境变量或敏感认证信息。

安全边界：

- 实验只使用原创人物、合成参考图和用户拥有权利的素材；
- 不得把受版权保护动漫角色作为官方 demo；
- 不得加入素材抓取逻辑。

G0 成功定义（不要求三个镜头全部完美）：

```text
至少一个当前可用模型能够：
1. 成功生成三个测试镜头；
2. 三个输出均能被稳定标准化；
3. 至少两个镜头人工质量为 pass 或 borderline；
4. reference image 对角色身份具有明显约束作用；
5. 没有系统性生成失败；
6. 生成成本和延迟处于用户可接受的实验范围。
```

如果失败：不要为了“完成路线图”强行进入正式集成，先报告模型限制。

G0 通过后，下一批才允许规划 G1（Single-Shot Generation Pipeline）：

```text
ShotRequirement
→ generation prompt
→ video generation model
→ raw video
→ normalization
→ generated ClipAsset
→ existing timeline / render
```

如果 G0 不通过：暂停生成集成，重新选择模型或调整产品目标。

G0 明确不做：

```text
generate strategy
自动 fallback generate
多镜头自动生成
character database
LoRA 训练
fine-tuning
prompt optimizer agent
LLM planner
自动 seed 搜索
无限重试
自动质量评分模型
lip sync
voice
subtitles
music
Web
DB
Worker
billing
provider framework
```

下一轮实施前必须先调研：

```text
G0 契约审核通过后，下一轮不能直接写 API 代码。
必须先进行一次截至执行当天的模型调研，比较当前可获得的视频生成模型/API，
再由用户确认首个实验 provider/model；只有确认后才执行真实生成实验。
```

JSON `schema_version` 不因 G0 修改，仍为 `1.9`。

---

## 2. 本阶段明确不做

Codex 不得自行增加：

- 自动切镜；
- 人脸识别、OCR；
- Embedding、向量数据库；
- PostgreSQL、Redis、Celery；
- FastAPI、Web 页面；
- LangGraph、多 Agent、ReAct、自主规划或无限循环；
- 视频生成模型；
- 自动配音、口型同步、背景音乐；
- 字幕烧录；
- 文字 placeholder、字体扫描或字体下载；
- 转场；
- 用户、权限、计费；
- 素材抓取、下载或分发；
- Docker、Kubernetes；
- `generate` 策略；
- 插件系统、注册中心或通用工作流引擎；
- “为了以后”建立的 Repository、Event Bus、DI 容器或抽象工厂。

第 3A 批 freeze_frame 契约明确不做：

```text
aliases
情绪或景别
speed_adjust
多种 freeze 模式
用户可配置 freeze 阈值
只冻结单张任意帧
首帧冻结
中间帧冻结
转场
25/30 fps
VFR
source drift override
overwrite / rollback
LLM
字幕、字体、配音或音乐
新工作流框架
```

第 3B-1 批 aliases 契约明确不做：

```text
动作别名
情绪别名
景别别名
模糊匹配
拼音匹配
同义词模型
Embedding
LLM 自动生成别名
自动从素材描述推导别名
多层别名或别名指向别名
多文件合并
aliases 热加载
Web 管理界面
数据库
selection trace 扩展
count_frames 审计
manifest 其他补强
speed_adjust
VFR
旋转
overwrite / rollback
字幕、配音或音乐
```

第 3B-2 批 emotion / shot_scale 契约明确不做：

```text
CV 情绪识别
人脸表情识别
自动镜头景别视觉分类
LLM
Embedding
sentiment model
用户自定义情绪词典
emotion aliases
shot_scale aliases
多情绪
情绪强度
情绪变化轨迹
景别距离矩阵
景别部分匹配
fuzzy matching
拼音
动作 aliases
selection trace 扩展
count_frames 审计
manifest 进一步补强
speed_adjust
VFR
转场
字幕、配音或音乐
Web / DB
无关重构
```

第 3B-3 批 selection trace / count_frames / manifest 契约明确不做：

```text
新检索算法
新 scoring 维度
scoring 权重调整
新 hard gate
emotion 改进
shot_scale 改进
aliases 改进
fuzzy matching
拼音
LLM
Embedding
reranking model
CV
对全部素材 count_frames
并发 ffprobe
background audit
新 CLI 参数
source drift override
build overwrite
rollback
dependency lock hash
normalized segment hash
joined_video hash
renderer trace
subtitles
voice
music
Web
DB
Worker
无关重构
```

阶段 B 检索质量集 / 压力测试契约明确不做：

```text
新产品功能
parser / aliases / emotion / shot_scale 产品规则修改
scoring 公式或权重修改
hard gates / stable ranking / strategy 优先级修改
selection trace / renderer / FFmpeg / manifest 产品语义修改
为改善 benchmark 数字而优化算法
并发 / multiprocessing
numpy / pandas / psutil / pytest-benchmark / asv / pyperf
缓存框架
视频生成 API / generate strategy / prompt compiler
reference image / 模型 provider / 云 API / API key
Web / DB / Worker
无关重构
```

第二阶段 G0 视频生成实验契约明确不做：

```text
generate strategy
自动 fallback generate
多镜头自动生成
character database
LoRA 训练
fine-tuning
prompt optimizer agent
LLM planner
自动 seed 搜索
无限重试
自动质量评分模型
lip sync
voice
subtitles
music
Web
DB
Worker
billing
provider framework
版权动漫角色素材
素材抓取逻辑
```

版权边界：

> 用户只能导入和处理其有权使用的素材。项目不提供、抓取或分发受版权保护的动漫素材。

---

## 3. 固定数据流和职责边界

### 3.1 第 1 批

```text
hand-written timeline.json
          │
          ▼
validate timeline + source fingerprints
          │
          ▼
render video-only segments
          │
          ▼
validate frames + concat signature
          │
          ▼
stream-copy concat video
          │
          ▼
one anullsrc → one AAC encode
          │
          ▼
validate final MP4
```

### 3.2 第 2 批

```text
script.md                 clips.json + local MP4
    │                              │
    ├─rule parser                  ├─static path validation
    │  （人物/地点/对白/动作/       │
    │   情绪/景别固定词典提取）     │
    │                              └─serial ffprobe
    ▼
ShotRequirement[]          ProbedClip[]
            \                 /
             \               /
              ▼             ▼
           bounded deterministic retrieval
                        │
                        ▼
           selected-source count_frames audit（staging 前）
                        │
                        ▼
                 Timeline 1.9
                        │
                        ▼
             已验证的 render workflow
                        │
                        ▼
                   output.mp4
```

职责：

```text
普通 Python：输入、解析、评分、选择、时间线、路径和工作流
FFmpeg/ffprobe：媒体探测、标准化、拼接、AAC 和验证
JSON：阶段边界、人工检查和独立重渲染输入
Workflow：固定顺序，不自主规划
```

禁止：

- parser 改变段落数量；
- parser 或 LLM 生成 FFmpeg 命令；
- renderer 回读 script、clips.json 或 retrieval 结果；
- 隐式全局状态；
- 无限重试；
- 跳过失败步骤后返回成功。

---

## 4. 技术栈和参考环境

运行依赖：

```text
Python >=3.11,<3.14
Pydantic 2
Typer
FFmpeg / ffprobe
```

开发依赖：

```text
pytest
pytest-cov
ruff
```

当前参考验收环境：

```text
Linux x86_64
Python 3.13.x
FFmpeg 7.1.3
ffprobe 7.1.3
```

`pyproject.toml` 至少包含：

```toml
[project.scripts]
anime-remix = "anime_remix.cli:app"

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff"]
```

MVR 不依赖 OpenAI SDK、PyYAML、数据库驱动或 Web 框架。

---

## 5. 推荐目录

```text
anime-remix-agent/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── anime_remix/
│       ├── __init__.py
│       ├── cli.py
│       ├── errors.py
│       ├── json_io.py
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── enums.py
│       │   └── models.py
│       ├── services/
│       │   ├── input_loader.py
│       │   ├── script_parser.py
│       │   ├── clip_retriever.py
│       │   └── timeline_compiler.py
│       ├── adapters/
│       │   └── ffmpeg.py
│       └── workflows/
│           ├── build_workflow.py
│           └── render_workflow.py
│
├── demo/
│   ├── generate_media.py
│   ├── script.md
│   ├── clips.json
│   └── clips/
│
├── tests/
│   ├── unit/
│   ├── workflow/
│   ├── integration/
│   └── fixtures/
│       └── generate_render_smoke.py
│
└── runs/
```

约束：

- 唯一 CLI 入口是 `src/anime_remix/cli.py`；
- 所有外部命令集中在 `adapters/ffmpeg.py`；
- 不为单个 JSON 文件加载增加 Repository 接口；
- 不创建当前授权批次以后才会使用的产品模块（G0 代码实现开始前不得创建 GENERATE strategy、provider abstraction 或正式集成模块；实验产物只允许放在 `experiments/video-generation/`，后续功能的模块同样不得提前创建）。

---

## 6. 严格模型和 JSON 规则

所有正式 Pydantic 模型：

```python
model_config = ConfigDict(
    extra="forbid",
    strict=True,
    allow_inf_nan=False,
)
```

要求：

- 未知键失败；
- `NaN`、`Infinity`、`-Infinity` 失败；
- 布尔值不得冒充整数；
- ID 去除首尾空白后必须非空；
- 所有列表使用 `default_factory`；
- 正式 JSON 不使用顶层裸列表；
- 核心 JSON 不写当前时间、随机 UUID 或绝对输出目录。

正式 JSON：

```text
UTF-8
ensure_ascii=False
indent=2
稳定字段顺序
单个结尾换行
```

原子写文件：

```text
同目录临时文件
→ flush
→ os.fsync（普通文件）
→ os.replace
```

第 3B-2 批实现后，`emotion` / `shot_scale` 为 null 时仍按现有 `model_dump` 行为显式写入正式 JSON。因此旧核心 JSON 字节不要求与第 3B-1 完全一致，但旧输入的规划语义、score 数值、selection、source 区间和 output.mp4 必须完全不变。

第 3B-3 批实现后，`selection_trace` 是 `retrieval_results.json` 中每个 shot 的正式结构化字段（§11.8），不是 render.log 文本；manifest 新增字段见 §15.4。

阶段 B 的质量报告与 benchmark 输出（例如 `.tmp/quality-report.json`、`.tmp/retrieval-benchmark.json`）不属于正式产品 JSON：不进入 manifest、timeline 或 runs managed entries，可以被 .gitignore 忽略。

第二阶段 G0 的 `experiments/video-generation/results.json` 同样不属于正式产品 JSON：不使用 schema_version 1.9，不进入 manifest、timeline 或 runs managed entries。

运行日志和 manifest 可以包含时间，不参与核心 JSON 字节比较。

---

## 7. 输入规范

### 7.1 `script.md`

一个非空段落对应一个镜头需求。

规则：

- 先以 `utf-8-sig` 读取；
- 换行统一为 `\n`；
- 一个或多个空行分段；
- 去除段落首尾空白；
- 忽略空段落；
- 不允许空剧本；
- MVR 支持 3～10 段；
- 单段最多 5,000 Unicode code point；
- parser 不得增加、删除、合并或拆分段落；
- `source_text` 保存换行规范化后的原文。

稳定 ID：

```text
shot_001
shot_002
shot_003
```

### 7.2 `clips.json`

正式格式必须是对象，不是顶层数组：

```json
{
  "schema_version": "1.9",
  "clips": [
    {
      "id": "clip_001",
      "path": "clips/clip_001.mp4",
      "characters": [
        {"id": "char_lin_xia", "name": "林夏"}
      ],
      "location_id": "loc_school_rooftop",
      "location_name": "学校天台",
      "action": "独自站立",
      "description": "黄昏时，林夏独自站在学校天台。",
      "emotion": "calm",
      "shot_scale": "medium"
    }
  ]
}
```

MVR 不要求人工 `duration_seconds`。

规则：

- `schema_version` 必须为 `1.9`；
- `id` 全局唯一，匹配 `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`；
- `path` 必须是普通相对文件路径；
- 拒绝 URL、FFmpeg 协议、设备路径、绝对路径和 `..`；
- path 以 `clips.json` 所在目录为基准；
- `resolve(strict=True)` 后仍必须位于该目录内；
- 指向目录外的符号链接拒绝；
- 文件必须存在且为普通文件；
- `action`、`description` 非空；
- `emotion`、`shot_scale` 可选，省略等于 null；必须使用 §9.1 封闭枚举，非法值拒绝；
- 两字段是人工标注的素材事实；不从视频画面自动识别，不使用 CV 模型、人脸识别或 LLM，不根据 description 自动修改用户填写的素材 metadata；
- MVR 最多 50 条素材；
- 超限直接失败，不静默截断。

### 7.3 `CharacterRef`

```python
class CharacterRef(BaseModel):
    id: str | None = None
    name: str | None = None
```

规则：

1. `id`、`name` 至少一个非空；
2. 首次出现位置决定顺序；
3. 同 ID 可以补全缺失 name；
4. 同 ID 的两个非空规范化 name 不一致时失败；
5. name-only 遇到后续 `{id, same_name}` 时升级；
6. 同一规范化 name 映射到多个 ID 时失败；
7. 同一人物最终只保留一个 canonical 引用。

### 7.4 两类相对路径必须区分

`clips.json.path`：

```text
不得包含 ..
必须限制在 clips.json 所在目录内
```

`timeline.json.source_path`：

```text
相对 timeline.json 所在目录
允许包含 ..
不得是绝对路径、URL、协议或设备路径
resolve 后必须等于 planner 选中的原始源文件
render 时必须校验 size 和 SHA256
```

允许 timeline 相对路径包含 `..` 是为了让运行目录引用项目中的原始素材。它不等同于接受任意 FFmpeg 输入字符串。

### 7.5 `aliases.json`（第 3B-1 批）

正式顶层对象：

```json
{
  "schema_version": "1.9",
  "character_aliases": [
    {
      "target_id": "char_lin_xia",
      "aliases": ["小夏", "林同学"]
    }
  ],
  "location_aliases": [
    {
      "target_id": "loc_school_rooftop",
      "aliases": ["天台", "学校楼顶"]
    }
  ]
}
```

规则：

- 顶层必须是对象，禁止裸数组；
- `schema_version` 必须为 `1.9`；
- 模型配置 `extra="forbid"`、`strict=True`、`allow_inf_nan=False`；
- `character_aliases`、`location_aliases` 使用 `default_factory=list`；
- `target_id` 去除首尾空白后必须非空；
- `aliases` 必须是非空列表；
- 每个 alias 去除首尾空白后必须非空；
- 单个 alias 最多 128 Unicode code point；
- 单个 target 最多 32 个 alias；
- 人物和地点别名项各自最多 200 项；
- 文件最大 1 MiB；
- 以 UTF-8-SIG 读取；
- 不允许未知字段。

canonical target 规则：

- 人物 `target_id` 必须对应 clips.json canonical character dictionary 中真实存在的人物 ID；不得用人物 name 作为 target；不得指向 name-only 且没有 ID 的人物；
- 地点 `target_id` 必须对应 clips.json 中真实存在的非空 `location_id`；不得用 location_name 作为 target；
- target 不存在时静态验证失败；
- aliases.json 必须在 clips.json 完成 canonical merge 后再验证 target。

别名规范化与冲突（仅用于唯一性检查）：

```python
alias_key = Unicode NFKC → strip → casefold
```

- 同一类别内，相同 `alias_key` 只能映射到一个 target_id；
- 同一 `alias_key` 映射到不同人物 ID 时失败；映射到不同地点 ID 时失败；
- 同一 target_id 下重复 `alias_key` 失败，不静默去重；
- 人物类别和地点类别分别建立冲突域；同一个 `alias_key` 同时作为人物别名和地点别名暂时允许（人物解析与地点解析是独立字段）；
- alias 与该 target 的 canonical ID 或 canonical name 规范化后完全相同时，视为冗余配置并失败；
- alias 与其他 target 的 canonical ID 或 canonical name 冲突时失败。

错误消息必须指出：aliases 文件阶段、alias 原值、规范化 alias_key、涉及的 target_id、冲突类别；不输出完整剧本文本。

---

## 8. MVR 媒体输入契约

首个版本故意只支持受限输入。任一素材不满足时，`build` 明确失败并报告 asset ID 和字段。

要求：

```text
恰好一个视频流
H.264
8-bit yuv420p
CFR 24/1 fps
r_frame_rate = 24/1
avg_frame_rate = 24/1
nb_frames 为正整数
有限正时长
abs(duration * 24 - nb_frames) <= 1
progressive 或 unknown field_order
SAR = 1:1
无旋转元数据
宽高为正偶数
宽度 <= 3840，高度 <= 2160
时长 <= 120 秒
文件 <= 1 GiB
SDR
color_space 为 bt709 或缺失
color_primaries 为 bt709 或缺失
color_transfer 为 bt709 或缺失
color_range 为 tv 或缺失
chroma_location 为 left 或 unknown
```

源音频是否存在不影响接受；所有源音频均丢弃。

缺失颜色字段在 MVR 中按 BT.709 limited 假设，并写入 manifest 审计。

MVR 直接拒绝：

- 23.976、25、30 或其他源帧率；
- VFR；
- 非方形 SAR；
- 旋转；
- HDR；
- BT.601、BT.2020 或冲突颜色标签；
- full range；
- 已知隔行；
- 已知非 left chroma；
- 无 `nb_frames`；
- 多视频流或无视频流。

注：限制为 24 fps 是 MVR 的删除式设计，不是长期兼容范围。

---

## 9. 核心领域模型

模型可拆分文件，但语义不得改变。

### 9.1 枚举

```python
class TimelineStrategy(str, Enum):
    CLIP = "clip"
    FREEZE_FRAME = "freeze_frame"
    PLACEHOLDER = "placeholder"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Emotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    TENSE = "tense"
    CALM = "calm"


class ShotScale(str, Enum):
    CLOSE_UP = "close_up"
    MEDIUM = "medium"
    WIDE = "wide"
```

`Emotion` / `ShotScale` 是封闭枚举，不得使用任意字符串。第 3B-2 批不增加 neutral、mixed、other、unknown 等值；未指定/未识别统一使用 `null`，且 `null != calm`。没有检测到情绪时不得猜测为 calm。

`ShotScale` 粗粒度语义固定：

```text
close_up：特写或近景
medium：中景或半身构图
wide：全景或远景
```

本批故意不增加 extreme_close_up、medium_close_up、medium_wide、full_shot、extreme_wide。

### 9.2 素材和 probe

```python
class ClipsDocument(BaseModel):
    schema_version: Literal["1.9"] = "1.9"
    clips: list[ClipAsset]


class ClipAsset(BaseModel):
    id: str
    path: Path
    characters: list[CharacterRef] = Field(default_factory=list)
    location_id: str | None = None
    location_name: str | None = None
    action: str
    description: str
    emotion: Emotion | None = None
    shot_scale: ShotScale | None = None


class ProbedClip(BaseModel):
    asset: ClipAsset
    resolved_path: Path
    size_bytes: int
    width: int
    height: int
    fps_num: int
    fps_den: int
    nb_frames: int
    duration_seconds: Decimal
    assumed_color_metadata: bool = False
```

`resolved_path` 只在内存模型和 manifest 中使用，不写入核心 timeline。

`emotion` / `shot_scale`（第 3B-2 批起）是人工标注的素材事实：两字段均可省略，省略等于 null；只影响检索评分与选择，不改变 renderer。

### 9.3 镜头需求

```python
class ShotRequirement(BaseModel):
    id: str
    order: int = Field(ge=1)
    source_text: str
    characters: list[CharacterRef] = Field(default_factory=list)
    location_id: str | None = None
    location_name: str | None = None
    action: str
    target_frames: int = Field(gt=0)
    dialogue: str | None = None
    emotion: Emotion | None = None
    shot_scale: ShotScale | None = None
```

`emotion` / `shot_scale`（第 3B-2 批起）由 rule parser 从 `source_text` 做有限、确定性关键词提取，见 §10.7。不得使用 LLM、Embedding、sentiment model、NLP 模型、模糊匹配、拼音匹配或用户自定义词典；aliases.json 仍然只处理人物和地点。

MVR 默认目标帧数：

```text
无对白：72 帧（3 秒）
有对白：max(72, ceil((字符数 / 4.5 + 0.6) * 24))
最后夹在 24～192 帧之间
```

这里直接产出整数帧，不先生成 float 秒再二次量化。

### 9.4 检索分数

未参与评分的字段为 `null`。

```python
class ScoreBreakdown(BaseModel):
    character: Decimal | None = None
    location: Decimal | None = None
    action: Decimal
    duration: Decimal
    emotion: Decimal | None = None
    shot_scale: Decimal | None = None
    active_weights: dict[str, Decimal]
    total: Decimal
```

`emotion` / `shot_scale`（第 3B-2 批起）：requirement 未指定该维度时 score 为 `null` 且不参与 active weights；requirement 指定后为 exact categorical match，见 §11.3。

统一量化：

```python
SCORE_QUANTUM = Decimal("0.000001")
value.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP)
```

禁止使用 Python `round()`。

### 9.5 RenderProfile

MVR 不暴露未经测试的 profile 选项：

```python
class RenderProfile(BaseModel):
    width: Literal[1280] = 1280
    height: Literal[720] = 720
    fps: Literal[24] = 24
    video_codec: Literal["libx264"] = "libx264"
    pixel_format: Literal["yuv420p"] = "yuv420p"
    video_preset: Literal["medium"] = "medium"
    video_crf: Literal[20] = 20
    max_b_frames: Literal[0] = 0
    gop_frames: Literal[48] = 48
    video_track_timescale: Literal[48000] = 48000
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate_kbps: Literal[128] = 128
    audio_sample_rate: Literal[48000] = 48000
    audio_channels: Literal[2] = 2
```

### 9.6 时间线

```python
class TimelineItem(BaseModel):
    shot_id: str
    order: int
    requirement: ShotRequirement
    strategy: TimelineStrategy

    source_asset_id: str | None = None
    source_path: str | None = None
    source_size_bytes: int | None = None
    source_sha256: str | None = None
    source_in_frame: int = 0
    source_frame_count: int = 0

    target_frames: int
    score: ScoreBreakdown | None = None
    reason_code: Literal[
        "exact_length",
        "center_trim",
        "short_source_freeze",
        "no_candidate",
    ]
    reason: str


class Timeline(BaseModel):
    schema_version: Literal["1.9"] = "1.9"
    path_base: Literal["timeline_dir"] = "timeline_dir"
    render_profile: RenderProfile
    items: list[TimelineItem]
```

固定不变量：

- `Timeline.items` 数组顺序是唯一播放顺序；
- `item.order` 从 1 开始、唯一、连续，且必须等于数组索引 + 1；
- `shot_id` 唯一；
- `item.shot_id == requirement.id`；
- `item.order == requirement.order`；
- strategy 与 reason_code 固定映射：clip → `exact_length` | `center_trim`；freeze_frame → `short_source_freeze`；placeholder → `no_candidate`；其他组合必须模型校验失败；
- `reason_code` 仍是封闭 Literal，不得恢复为任意 str；
- `item.target_frames == requirement.target_frames`；
- `target_frames > 0`；
- `clip` 必须有完整源字段；
- `clip.source_in_frame >= 0`；
- `clip.source_frame_count == target_frames`；
- `clip.source_sha256` 是 64 位小写十六进制；
- `freeze_frame` 必须有完整源字段（`source_asset_id`、`source_path`、`source_size_bytes`、`source_sha256`）；
- `freeze_frame.source_sha256` 是 64 位小写十六进制；
- `freeze_frame.score` 必须保留被选候选的真实评分；
- planner 生成的 `freeze_frame`：`source_in_frame == 0`、`source_frame_count == 选中素材 probe 得到的 nb_frames`、`source_frame_count >= 24`、`source_frame_count < target_frames`；
- 人工编辑的 `freeze_frame`：`source_in_frame >= 0`、`source_frame_count >= 1`、`source_frame_count < target_frames`；render 必须探测真实源媒体并验证 `source_in_frame + source_frame_count <= 实际 nb_frames`，不满足时失败，不得截断、修正或静默回退；
- `placeholder` 不得携带任何源字段；
- `placeholder.source_in_frame == 0`；
- `placeholder.source_frame_count == 0`；
- `score`、`reason` 和 `source_asset_id` 不参与画面变换；
- `requirement.emotion` / `requirement.shot_scale` 属于 planner metadata，`score.emotion` / `score.shot_scale` 属于选择解释信息，均不参与画面变换；renderer 不得读取这两个字段决定 FFmpeg 行为，仍只依据 strategy、source path / fingerprint、source frame range、target_frames 和 render_profile 等现有渲染字段；
- renderer 只使用 timeline 中的渲染字段和源指纹。
- renderer 按 `items` 数组顺序渲染，不得按 `order` 字段重新排序。

时间线不得包含：

```text
绝对输出目录
staging 路径
worker 数
timeout
API Key
LLM 请求
字体路径
字幕配置
transition
generate
normalized 临时文件路径
```

独立 render 必须从原源文件重新生成片段，不依赖旧 `normalized/`。

---

## 10. MVR 规则解析器

规则解析器只做完成闭环所需的最小提取。

### 10.1 每段固定产生一个 requirement

```text
段落 1 → shot_001
段落 2 → shot_002
...
```

不得合并、拆分、删除或新增段落。

### 10.2 人物和地点词典

词典来源（第 3B-1 批起）：

人物词典：

- canonical character ID；
- canonical character name；
- character aliases（提供 `--aliases` 时）。

地点词典：

- canonical location ID；
- canonical location name；
- location aliases（提供 `--aliases` 时）。

未提供 aliases 时，词典只来自 `clips.json`，与第 3A 批行为完全一致。

别名命中后，输出必须仍然是 canonical 值：人物输出 `{"id": "char_lin_xia", "name": "林夏"}`；地点输出 `{"location_id": "loc_school_rooftop", "location_name": "学校天台"}`。正式 parsed_script.json、ShotRequirement 和 timeline 中不得保存 alias 字符串作为身份真值。aliases 只影响文本识别，不改变素材元数据、评分公式、source fingerprint 或 renderer。

### 10.3 最长不重叠匹配

对人物和地点：

候选词集合扩展为：

```text
canonical ID
+ canonical name
+ aliases（第 3B-1 批起，提供 --aliases 时）
```

排序固定为：

```text
1. 起点升序
2. 区间长度降序
3. canonical target_id 字节序升序
4. term 类型稳定顺序：canonical ID → canonical name → alias
5. 原始 term 字节序升序
```

然后：

```text
收集全部命中区间
→ 贪心选择不重叠区间
→ 按文本位置输出
```

- 所有用户提供的 alias 在进入正则前必须转义；
- ASCII-only alias 使用与 ASCII ID 相同的词边界；非 ASCII alias 使用字面匹配；
- 不进行模糊匹配、分词或子串相似度；
- 同一 canonical target 在一个段落中被多个 term 命中时，最终只输出一个 canonical 引用；首次命中位置决定最终顺序。

### 10.4 对白

提取成对引号：

```text
“……”
「……」
『……』
"……"
```

- 不支持嵌套；
- 未闭合不提取；
- 多段对白按原顺序以换行连接。

### 10.5 动作

```text
移除已识别的对白区间
→ 压缩空白
→ 若为空，回退到原段落
```

情绪与景别提取在第 3B-2 批由 §10.7 单独定义，输入为 `source_text` 原始规范化段落，与动作提取相互独立。

### 10.6 目标帧数

```python
DEFAULT_FRAMES = 72
MIN_FRAMES = 24
MAX_FRAMES = 192
DIALOGUE_CHARS_PER_SECOND = Decimal("4.5")
DIALOGUE_PADDING_SECONDS = Decimal("0.6")

if not dialogue:
    target_frames = DEFAULT_FRAMES
else:
    seconds = (
        Decimal(dialogue_codepoint_count) / DIALOGUE_CHARS_PER_SECOND
        + DIALOGUE_PADDING_SECONDS
    )
    dialogue_frames = int(
        (seconds * Decimal(24)).to_integral_value(rounding=ROUND_CEILING)
    )
    target_frames = max(DEFAULT_FRAMES, dialogue_frames)

target_frames = min(MAX_FRAMES, max(MIN_FRAMES, target_frames))
```

parser 是唯一 requirement 目标帧生成位置。

### 10.7 情绪与景别提取（第 3B-2 批）

`emotion` 与 `shot_scale` 独立提取。输入均为 `ShotRequirement.source_text` 原始规范化段落。

固定情绪词典（代码内固定、封闭，不得扩展，不实现配置文件或自定义情绪词）：

```text
happy:
- 开心
- 高兴
- 喜悦
- 微笑
- 大笑

sad:
- 难过
- 悲伤
- 伤心
- 哭泣
- 落泪

angry:
- 生气
- 愤怒
- 恼火

fearful:
- 害怕
- 恐惧
- 惊恐

surprised:
- 惊讶
- 震惊
- 吃惊

tense:
- 紧张
- 焦急
- 不安

calm:
- 平静
- 冷静
- 镇定
```

固定景别词典：

```text
close_up:
- 特写
- 近景

medium:
- 中景
- 半身

wide:
- 全景
- 远景
```

提取算法：分别对 emotion 与 shot_scale 收集该类别全部词典命中，然后：

```text
起点升序
→ term 长度降序
→ enum 固定顺序
→ term 字节序升序
→ 选择第一项
```

Emotion enum 固定顺序：

```text
happy
sad
angry
fearful
surprised
tense
calm
```

ShotScale enum 固定顺序：

```text
close_up
medium
wide
```

因此如果一个段落同时包含“林夏先惊讶，随后开心地笑了。”，第一处合法 emotion 命中决定结果。本批不尝试理解“主要情绪”“最终情绪”或情绪变化。

如果没有任何命中：

```text
emotion = null
shot_scale = null
```

不得从 action similarity 或其他字段推断。

匹配安全：

- 所有词典 term 按字面量处理；如实现使用正则，必须 `re.escape`；
- 不进行 fuzzy matching、SequenceMatcher、bigram similarity、词形扩展或分词；
- 情绪/景别提取与人物/地点 aliases 匹配系统相互独立。

---

## 11. 有界确定性检索

MVR 不使用 Embedding。

### 11.1 文本规范化

`normalize_for_match()`：

- Unicode NFKC；
- 英文小写；
- 去除 Unicode 空白和标点类别；
- 保留字母和数字；
- 最多保留 2,048 code point。

### 11.2 文本相似度

```python
SEQUENCE_MATCHER_MAX_CODEPOINTS = 256
```

```python
def text_similarity(a: str, b: str) -> Decimal:
    a = normalize_for_match(a)[:2048]
    b = normalize_for_match(b)[:2048]

    if not a or not b:
        return Decimal("0.000000")

    jaccard = bigram_jaccard(a, b)

    if max(len(a), len(b)) <= SEQUENCE_MATCHER_MAX_CODEPOINTS:
        sequence = Decimal(str(
            SequenceMatcher(None, a, b, autojunk=False).ratio()
        ))
    else:
        sequence = Decimal("0")

    return quantize_score(max(jaccard, sequence))
```

要求：

- clip 规范化文本和 bigram 集合在加载时缓存；
- requirement 每段只预处理一次；
- 不得在每个 shot×clip 对中重复规范化；
- 超过 256 code point 不运行 `SequenceMatcher`。

### 11.3 分项评分

人物：确定性一对一最大匹配，使用 F2：

```text
recall = matched / required
precision = matched / asset
F2 = 5 * precision * recall / (4 * precision + recall)
```

需求未指定人物时为 `null`。

人物匹配身份规则：

```text
双方都有非空 ID：只有 ID 相同才匹配；
至少一方没有 ID：才允许规范化 name 精确匹配；
两个不同非空 ID 不得仅凭同名合并为同一人物。
```

地点：

```text
ID 精确匹配优先
否则使用 location_name 文本相似度
需求未指定地点时为 null
```

动作：

```text
max(
  similarity(requirement.action, clip.action),
  0.90 * similarity(requirement.action, clip.description),
  0.80 * similarity(requirement.source_text, clip.description)
)
```

时长：以帧计算：

```text
若 clip.nb_frames >= target_frames：
    1 - min((clip.nb_frames - target_frames) / max(target_frames, 1), 1)
否则：
    clip.nb_frames / target_frames
```

情绪（第 3B-2 批）：

```text
requirement.emotion == null：
    emotion score = null，该维度不参与 active weights

requirement.emotion != null 且 clip.emotion == requirement.emotion：
    emotion score = 1.000000

其他情况（包括 clip.emotion == null）：
    emotion score = 0.000000
```

景别（第 3B-2 批）与情绪完全对称：

```text
requirement.shot_scale == null：
    shot_scale score = null，该维度不参与 active weights

requirement.shot_scale != null 且 clip.shot_scale == requirement.shot_scale：
    shot_scale score = 1.000000

其他情况（包括 clip.shot_scale == null）：
    shot_scale score = 0.000000
```

不得实现情绪相似度、情绪距离矩阵、close_up 与 medium 的部分分、adjacent shot scale 分数或 fuzzy score；第一版只做 exact categorical match。

### 11.4 权重

默认权重：

```yaml
character: 0.20
location: 0.12
action: 0.36
duration: 0.12
emotion: 0.10
shot_scale: 0.10
```

这是刻意按 0.8 缩放旧四维权重（0.25 / 0.15 / 0.45 / 0.15 → 0.20 / 0.12 / 0.36 / 0.12，其和为 0.80），再补上 emotion 0.10 与 shot_scale 0.10，总和为 1.00。不得直接在旧权重之上再加 0.10 + 0.10。

只对活跃字段重新归一化。所有权重大于 0；六维全活跃时总和恰好为 1.00，归一化后活跃权重总和在 `1e-6` 内等于 1：

- character：requirement 未指定人物时 inactive；
- location：requirement 未指定地点时 inactive；
- emotion：requirement.emotion == null 时 inactive；
- shot_scale：requirement.shot_scale == null 时 inactive；
- action 和 duration：继续沿用现有规则。

当 emotion 与 shot_scale 都 inactive 时，归一化后必须精确恢复旧四维权重：

```text
character 0.25
location 0.15
action 0.45
duration 0.15
```

这是第 3B-2 的兼容性硬门禁。输出 `active_weights` 必须保持稳定键顺序，不得使用 set 的非稳定遍历。

### 11.5 排序

稳定全量排序：

```text
total 降序
character 降序；null 按 -1
action 降序
asset_id 升序
```

第 3B-2 不得因为加入 emotion / shot_scale 再增加新的 tie-break；两者已经体现在 total 中。

`top_k=3` 只控制 `retrieval_results.json` 展示，不是选择边界。

### 11.6 硬门槛和可渲染性

默认内容门槛（clip 与 freeze_frame 候选必须先通过，与 clip 相同）：

```text
total >= 0.55
若 requirement 指定人物：character >= 0.50
action >= 0.25
```

第 3B-2 不新增 emotion gate 或 shot_scale gate。emotion mismatch 或 shot_scale mismatch 只通过 total score 影响排名和最终 gate，不产生独立 hard rejection；不得增加 emotion >= x 或 shot_scale >= x。

内容门槛通过后，按帧数资格判定：

```text
nb_frames >= target_frames：clip eligible
24 <= nb_frames < target_frames：freeze_frame eligible
nb_frames < 24：不可渲染（too_short），不能使用 freeze_frame
```

固定常量：

```python
MIN_FREEZE_SOURCE_FRAMES = 24
```

不得增加按比例阈值、配置文件或用户选项。

扫描（clip > freeze_frame > placeholder 确定性优先级）：

```text
按全局 rank：
    total < 0.55：停止
    人物门槛失败：记录并继续
    动作门槛失败：记录并继续
    帧 gate 记录为 clip_eligible / freeze_eligible / too_short
    遇到第一个 freeze_frame eligible 候选：保存为 freeze fallback，不得立即选择，继续扫描
    遇到第一个 clip eligible 候选：立即选择 clip 并停止
    扫描结束仍无 clip：选择此前保存的 freeze fallback
    没有 clip 也没有 freeze fallback：placeholder
```

- 候选按全局 rank 排序，第一个 freeze eligible 候选就是最高排名 freeze fallback，不需要维护多个 fallback 或再次排序；
- 不得因为排名第一的素材帧数不足就立即 freeze 或 placeholder；仍须继续寻找后续完整 clip。

第 3B-3 批起，扫描过程按 §11.8 写入 `selection_trace`：每个实际扫描候选记录 global_rank、复用 ScoreBreakdown 的量化 score、content_gate、frame_gate 和封闭 decision；`total < 0.55` 触发 `stop_total_below_threshold`，不作为候选的 content gate 失败。

### 11.7 源帧区间

选中素材后：

```text
若 nb_frames == target_frames：
    source_in_frame = 0
    reason_code = exact_length

若 nb_frames > target_frames：
    source_in_frame = (nb_frames - target_frames) // 2
    reason_code = center_trim

source_frame_count = target_frames
```

选中 freeze_frame 后固定写入：

```text
source_in_frame = 0
source_frame_count = selected_clip.nb_frames
target_frames = requirement.target_frames
reason_code = short_source_freeze
```

并在 timeline 中携带完整 `source_asset_id`、`source_path`、`source_size_bytes`、`source_sha256` 和真实 score。

第 3A 批固定播放整个短素材（planner 不截取其中一部分），再冻结其最后一帧。

`retrieval_results.json` 至少保存：

- 总候选数；
- Top 3；
- 最终选中候选；
- 最终候选全局 rank；
- 每个实际检查候选的 gate 结果；
- 每个实际检查候选的帧 gate：`clip_eligible` | `freeze_eligible` | `too_short`；
- 最终选择记录：`selected_asset_id`、`selected_global_rank`、`selected_strategy`（`clip` | `freeze_frame` | `placeholder`）、`reason_code`；
- 唯一跳过原因；
- 每个 shot 的 `selection_trace`：`scanned_candidates`、`stop_reason`、`freeze_fallback_asset_id`、`final_decision`（第 3B-3 批起，§11.8）。

freeze 审计语义：

- 当 freeze 候选被暂存但后续找到完整 clip 时：该候选帧 gate 仍记录 `freeze_eligible`，最终 `selected_strategy` 为 `clip`，不需要保存所有 fallback 历史；
- 当最终选择 freeze 时：`selected_strategy = freeze_frame`，`reason_code = short_source_freeze`；
- 当没有任何可用候选时：`selected_strategy = placeholder`，`reason_code = no_candidate`。

### 11.8 selection trace（第 3B-3 批）

每个 shot 的 retrieval 结果固定保存：

```json
{
  "selection_trace": {
    "scanned_candidates": [],
    "stop_reason": "exhausted_candidates",
    "freeze_fallback_asset_id": null,
    "final_decision": {}
  }
}
```

`scanned_candidates` 是“真正被选择算法扫描过”的候选列表，顺序必须就是实际 global rank 扫描顺序；未实际扫描的后续候选不得伪装成 scanned。

每个实际扫描候选（ScannedCandidateTrace）至少保存：

```json
{
  "global_rank": 1,
  "asset_id": "clip_001",
  "total": "0.770000",
  "character": "1.000000",
  "location": "1.000000",
  "action": "0.600000",
  "duration": "0.666667",
  "emotion": null,
  "shot_scale": null,
  "content_gate": "passed",
  "frame_gate": "clip_eligible",
  "decision": "selected_clip"
}
```

规则：

- score 字段必须复用已有 ScoreBreakdown 的最终量化值，不得在 trace 中重新计算第二套 score；
- `content_gate` 使用封闭值：`passed` | `failed_character` | `failed_action`；`total < 0.55` 不作为一个 candidate 的 content_gate；
- 触发提前停止的 `total < 0.55` 候选：`content_gate = null`、`frame_gate = null`、`decision = stop_total_below_threshold`；
- `frame_gate` 继续使用封闭值 `clip_eligible` | `freeze_eligible` | `too_short`；只有 `content_gate == passed` 时才有 frame_gate，content gate 失败或 total early stop 时 `frame_gate = null`；
- 不得修改现有 frame gate 判定：`nb_frames >= target_frames → clip_eligible`；`24 <= nb_frames < target_frames → freeze_eligible`；`nb_frames < 24 → too_short`。

candidate `decision` 使用封闭值：

```text
selected_clip
saved_freeze_fallback
freeze_eligible_not_saved
too_short
skipped_character_gate
skipped_action_gate
stop_total_below_threshold
```

语义：

```text
selected_clip：该候选成为最终 clip，扫描立即结束
saved_freeze_fallback：遇到的第一个 freeze_eligible 候选被保存为最高排名 fallback
freeze_eligible_not_saved：已有更高排名 freeze fallback，后续 freeze eligible 不替换
too_short：内容 gate 通过但源帧 < 24
skipped_character_gate：人物门槛失败
skipped_action_gate：action 门槛失败
stop_total_below_threshold：total < 0.55，触发提前停止
```

不得增加模糊 free-text decision。

每个 shot 最终必须有一个封闭 `stop_reason`：

```text
selected_clip：扫描遇到完整 clip 并结束
total_below_threshold：稳定排序下遇到 total < 0.55 提前结束
exhausted_candidates：候选全部扫描完
```

`stop_reason` 描述“扫描为什么结束”，不描述最终 strategy；例如扫描因 `exhausted_candidates` 结束，最终仍可能选择先前保存的 freeze fallback。

`freeze_fallback_asset_id: str | null`：

- 没有保存 fallback → null；
- 保存过 → 记录第一个 freeze_eligible candidate 的 asset_id；
- 后续 freeze candidate 不替换；
- 即使最终找到 clip，该字段仍保留，用于说明扫描过程中曾经保存 fallback；
- 不得保存多个 fallback，不得创建 fallback stack。

`final_decision` 固定包含：

```json
{
  "selected_asset_id": "clip_001",
  "selected_global_rank": 1,
  "selected_strategy": "clip",
  "reason_code": "center_trim",
  "source_in_frame": 12,
  "source_frame_count": 72,
  "target_frames": 72
}
```

- 必须与当前 retrieval 最终选择字段以及 timeline 完全一致，禁止 trace 一套结果、timeline 另一套结果；需要模型/测试保证二者一致；
- placeholder：`selected_asset_id = null`、`selected_global_rank = null`、`source_in_frame = 0`、`source_frame_count = 0`。

trace 与 Top 3 的关系：

- Top 3 只是展示结果，保持不变；
- `selection_trace.scanned_candidates` 是实际 selection 扫描过程；
- 不得因为 Top 3 只展示三个候选而停止 selection trace；例如第 5 名才找到完整 clip，Top 3 仍只有前三名，而 selection_trace 必须包含实际扫描过的 rank 1～5。

trace 确定性：

- 相同 script / clips / aliases / emotion / shot_scale metadata 下重复 build，`retrieval_results.json` 必须字节一致；
- 禁止时间、UUID、内存地址、unordered set/dict、FFmpeg stderr、duration wall clock、random ID；
- `scanned_candidates` 顺序必须就是实际 global rank 扫描顺序。

trace 内容边界（不保存）：

```text
完整剧本文本
clip description
clip action 原文
absolute path
SHA256
FFmpeg command
环境变量
aliases 内容
```

trace 只记录：候选 ID、rank、量化 score、gate、decision、final selection，避免 `retrieval_results.json` 无限制膨胀。

---

## 12. FFmpeg 适配器

所有外部命令集中在 `adapters/ffmpeg.py`。

调用规则：

- 参数列表；
- `shell=False`；
- 捕获 stdout/stderr；
- 硬超时；
- 显式 `-map`；
- 支持空格、单引号和中文路径；
- 不把剧本文本写入日志；
- stderr 摘要限制长度；
- 不记录环境变量或密钥。

必需能力：

```text
ffmpeg
ffprobe
libx264
aac
scale
pad
setsar
format
setparams
fps
trim
setpts
atrim
asetpts
color
anullsrc
concat demuxer
```

第 3A 批起，`tpad` 属于 required FFmpeg capability，必须加入 REQUIRED_FILTERS 检查。

### 12.1 video-only 标准化规格

```text
H.264 / libx264
yuv420p
1280×720
24 fps CFR
SAR = 1:1
BT.709 limited
progressive
chroma_location = left
无音频流
```

固定编码参数：

```text
-profile:v high
-level:v 3.1
-pix_fmt yuv420p
-preset medium
-crf 20
-bf 0
-g 48
-keyint_min 48
-sc_threshold 0
-video_track_timescale 48000
-color_primaries bt709
-color_trc bt709
-colorspace bt709
-color_range tv
-chroma_sample_location left
-map_metadata -1
-map_chapters -1
```

仅设置编码器参数不足以保证 primaries 和 transfer 被写出。过滤链必须显式包含：

```text
setparams=
  range=limited:
  color_primaries=bt709:
  color_trc=bt709:
  colorspace=bt709:
  field_mode=prog
```

### 12.2 `clip` 过滤图

```text
trim=start_frame=<source_in_frame>:
     end_frame=<source_in_frame + source_frame_count>
→ setpts=PTS-STARTPTS
→ scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2
→ pad=1280:720:(ow-iw)/2:(oh-ih)/2:black
→ setsar=1
→ format=yuv420p
→ setparams=range=limited:
            color_primaries=bt709:
            color_trc=bt709:
            colorspace=bt709:
            field_mode=prog
→ fps=fps=24:start_time=0:round=near
→ trim=end_frame=<target_frames>
→ setpts=N/(24*TB)
```

renderer 不从秒重新推导源区间或目标帧数。

`force_divisible_by=2` 保证缩放后的中间尺寸为偶数；最终经 pad 固定为 1280×720。

### 12.3 `placeholder` 过滤图

```text
color=c=black:s=1280x720:r=24
→ trim=end_frame=<target_frames>
→ setpts=N/(24*TB)
→ format=yuv420p
→ setparams=range=limited:
            color_primaries=bt709:
            color_trc=bt709:
            colorspace=bt709:
            field_mode=prog
```

placeholder 始终纯黑，不写文字。

### 12.4 `freeze_frame`（第 3A 批）

第 3A 批实现必须使用独立过滤图，不得复用 clip 分支后再偷偷补帧：

```text
trim source frames
→ setpts=PTS-STARTPTS
→ scale / pad / setsar / format / setparams
→ fps=24:start_time=0:round=near
→ tpad=stop_mode=clone:stop=<target_frames>
→ trim=end_frame=<target_frames>
→ setpts=N/(24*TB)
```

- `tpad` 必须位于第一次目标 `fps` 之后；
- `stop=<target_frames>` 故意过量补帧，不计算“缺少多少帧”作为 stop；
- 最终 `trim=end_frame=<target_frames>` 精确封口，避免源区间实际输出帧数的边界误差；
- 冻结后的最后一帧必须来自所选源区间的最后一帧；
- freeze 标准化片段仍为 video-only H.264；
- 编码参数、BT.709、SAR、time_base、profile、level、pix_fmt 和 concat signature 沿用第 0～2 批已验收实现；
- 每段必须实际验证 `nb_read_frames == target_frames`；
- `avg_frame_rate` 不进入 concat signature；
- extradata SHA256 必须来自 `ffprobe -show_data_hash sha256` 的真实 `extradata_hash`；
- 最终仍只编码一次连续静音 AAC；
- `tpad` 从第 3A 批开始属于 required FFmpeg capability。

### 12.5 单片段验证

每个标准化片段必须：

- 存在且非空；
- ffprobe 可读；
- 恰好一个视频流；
- 无音频流；
- H.264、High profile、yuv420p；
- 1280×720；
- SAR 1:1；
- `r_frame_rate=24/1`；
- `time_base=1/48000`；
- BT.709 primaries、transfer、space；
- limited range；
- progressive；
- chroma left；
- start time 为 0，允许一个 time-base tick；
- `nb_read_frames == target_frames`。

不得要求：

```text
stream.duration == target_frames / 24
avg_frame_rate == 24/1
```

MP4 中 N 个帧的 stream duration 可能显示为 `(N-1)/fps`，且 `avg_frame_rate` 可能随片段帧数变化（也可能在不同 FFmpeg 版本下保持 `24/1`）。无论哪种情况，`avg_frame_rate` 都不得进入 concat signature。MVR 以 `nb_read_frames` 为视频长度真值。

### 12.6 concat signature

只有所有片段通过帧数和签名校验后才允许 stream-copy concat。

签名包含：

```text
codec_name
profile
level
codec_tag_string
extradata SHA256（真实数据哈希，见下）
width
height
pix_fmt
sample_aspect_ratio
r_frame_rate
time_base
color_range
color_space
color_transfer
color_primaries
field_order
chroma_location
```

`extradata SHA256` 必须来自 `ffprobe -show_data_hash sha256` 输出的真实 extradata 哈希（`extradata_hash`）。不得在 extradata 缺失或为空时对空字符串计算 SHA256 后视为有效签名。

签名明确排除：

```text
avg_frame_rate
duration
nb_frames
nb_read_frames
bit_rate
文件大小
```

原因：这些字段可以随合法片段长度或内容变化，不决定 H.264 stream-copy 兼容性。

### 12.7 拼接和最终 AAC

流程：

```text
video-only 标准化片段
→ concat demuxer + -c:v copy → joined_video.mp4
→ 单个 anullsrc 48 kHz stereo
→ atrim=end_sample=<total_samples>
→ asetpts=PTS-STARTPTS
→ copy joined video + 编码一次 AAC
→ final.mp4
```

总样本数：

```python
total_frames = sum(item.target_frames for item in timeline.items)
total_samples = total_frames * 48000 // 24
```

ffconcat 清单必须为每个片段写入按帧推导的 duration：

```python
duration = Decimal(target_frames) / Decimal(24)
```

该 duration 仅用于 concat demuxer 的时间戳平移；视频总帧数仍以各片段和最终 MP4 的 `nb_read_frames` 为最终真值，不得用清单 duration 代替帧数验证。

不得：

- 为每段编码 AAC；
- 拼接逐段 AAC；
- 使用 `-shortest`；
- 使用 `-t` 作为第三套截断机制；
- 依赖默认流选择。

### 12.8 selected-source count_frames 审计（第 3B-3 批）

第 3B-3 批只对最终选中的源执行真实帧计数：

- 包含 `strategy == clip` 与 `strategy == freeze_frame`；
- 不包含 `placeholder`；
- 使用 `ffprobe -count_frames` 读取真实 `nb_read_frames`；
- 不对全部最多 50 个素材执行 count_frames；这是 selected-source correctness audit，不是全库昂贵扫描。

唯一源去重：

- 多个 shot 最终选择同一 `asset_id` 时，count_frames 只执行一次；
- 去重键固定为 `asset_id`，不是 source path、SHA256 或 inode；
- 与 `selected_source_sha256` 的审计身份语义保持一致；
- 不同 `asset_id` 即使指向同一物理文件，仍分别作为两个 asset 审计项。

count 顺序：

- 所有 selected-source count_frames 按 `asset_id` UTF-8 字节序升序串行执行；
- 不得并发；
- 不得按 shot 出现顺序；
- 保证日志稳定、错误顺序稳定、manifest 稳定。

count_frames 门禁：

```text
initial metadata probe.nb_frames == count_frames.nb_read_frames
```

- 必须严格相等，不允许 ±1；
- 不允许 duration 替代；
- 不允许 avg_frame_rate 替代；
- 不允许警告后继续；
- 不一致时构建失败，错误类别为媒体输入/媒体契约错误，使用现有环境/媒体错误体系，对应退出码 3；
- 错误至少包含：asset_id、metadata_nb_frames、counted_nb_frames、audit stage；
- 不回显剧本文本。

边界：

- selection 仍使用 initial probe.nb_frames；不得先 count 全部素材再用 counted frames 排名；
- count audit 是选择之后的验证；selected source 不一致直接失败，不重新 retrieval、不寻找第二候选、不 fallback、不 placeholder；
- freeze_frame 的 `source_frame_count == selected_clip.nb_frames`，因此真实 count 必须满足 `counted_nb_frames == selected_clip.nb_frames`，否则失败；不得自动把 `source_frame_count = counted_nb_frames`，那会静默改变已编译选择语义。

性能边界：

- 对每个 shot×candidate 调用 ffprobe 是禁止的；
- metadata probe 仍每个素材一次；
- 真实 count_frames 只对最终 selected unique asset_id 一次；
- 最多 `min(非 placeholder shot 所用 unique asset 数, 10)` 次（当前 script 最多 10 个 shot）；
- 不增加并发。

---

## 13. 最终媒体验证

最终 MP4 必须：

- 存在且非空；
- ffprobe 可读；
- 恰好一个视频流和一个音频流；
- 视频为 H.264、yuv420p、1280×720；
- 视频 SAR 1:1、24 fps、BT.709 limited、progressive、left；
- 视频 `nb_read_frames == sum(target_frames)`；
- 视频 start time 为 0，允许一个 time-base tick；
- 音频为 AAC、48,000 Hz、2 声道；
- 音频 start time 为 0；
- 音频时长与 `total_frames / 24` 的误差不超过：

```text
max(1024 / 48000, 1 / 24)
```

- `volumedetect` 或等价验证 `max_volume <= -90 dB`；
- 容器时长与 `total_frames / 24` 的误差不超过 0.25 秒。

不要用以下条件替代真实验证：

```text
ffprobe 能打开
ffmpeg 退出码为 0
文件大小大于 0
stream.duration 看起来接近
```

---

## 14. 源指纹和独立 render 信任边界

### 14.1 选中源指纹

planner 对每个选中源（clip 与 freeze_frame）写入：

```text
source_size_bytes
source_sha256
```

只哈希选中的源，不在 MVR 中哈希全部素材。

freeze_frame 作为带源字段策略，与 clip 一样必须携带并校验 `source_size_bytes` / `source_sha256`。

### 14.2 独立 `render`

`render`：

- 只读取 `timeline.json`；
- 按 `timeline_dir` 解析相对 source path；
- 校验源存在、普通文件、size 和 SHA256；
- 重新生成临时 video-only 片段；
- 重新 concat 并编码一次 AAC；
- 不读取 script、clips.json、parser、retriever 或 manifest；
- 不依赖旧 `normalized/`。

第 3B-3 批起，独立 `render` 仍不得读取 `retrieval_results.json`、`selection_trace`、`run_manifest.json`、`selected_source_frame_audit`、parsed_script、clips 或 aliases，仍然只读取 timeline 与 timeline 引用的源；不要求独立 render 再执行 build-only 的 selected-source count audit，render 现有的媒体/帧区间验证保持不变。

公共 CLI `render` 必须执行全部输出安全限制。build 可以调用同一个底层渲染核心，但只能通过未暴露给 CLI 的私有参数 `allow_managed_output=True` 写入本次 staging。该私有参数不得由用户输入控制。

MVR 不支持忽略 source drift。指纹不一致必须失败。

### 14.3 输出安全

独立 render 的输出路径不得：

- 等于 timeline 文件；
- 等于任何源素材；
- 与任何源素材是同一 inode；
- 位于含合法 `.anime-remix-run` 的祖先目录内；
- 通过符号链接绕过上述规则。

已存在输出必须显式 `--overwrite`。覆盖单文件时：

```text
同目录临时 MP4
→ 完整验证
→ os.replace
```

---

## 15. Build 工作流和发布

### 15.1 固定顺序

```text
validate_static_inputs（script + clips.json + canonical merge）
→ validate_new_output_target
→ load_and_validate_aliases_if_provided
→ check_ffmpeg_capabilities
→ probe_assets_serially_in_asset_id_order
→ parse_script
→ retrieve_and_select
→ hash_selected_sources
→ count_selected_source_frames_in_asset_id_order（第 3B-3 批）
→ validate_counted_frames
→ compile_and_validate_timeline
→ create_target_sibling_staging
→ write_marker_and_running_manifest
→ write_planning_artifacts
→ call_render_core_into_staging(allow_managed_output=True)
→ validate_final_output
→ write_succeeded_manifest
→ rename(staging, target)
```

### 15.2 副作用边界

- preflight、probe、parse、retrieve 或 timeline 编译失败：不创建 staging；
- aliases 文件缺失、格式错误、target 不存在或冲突时，在 probe、parse、retrieve 和 staging 创建前失败；
- aliases 静态验证失败不得创建 staging；
- count_frames 审计失败（metadata_nb_frames != counted_nb_frames）时不得创建 staging、不得创建 target、不得执行 render；不得为了写 failed manifest 提前创建 staging；
- staging 创建后的失败：写 `failed` manifest，保留 staging 用于诊断；
- target 必须不存在；
- staging 与 target 必须同父；
- target 第一次可见时 manifest 已是 `succeeded`；
- public `render` 的 managed-run 禁写规则仍然有效；
- 只有 build 内部私有调用可以写当前 staging，且该权限不得通过 CLI 暴露；
- build 不实现 `--overwrite`；
- target 不得等于或包含 script、clips.json 或任何源素材；
- planner 计算 timeline source path 时，以最终 target 目录为基准。

### 15.3 运行目录标记

`.anime-remix-run`：

```json
{
  "schema_version": "1.9",
  "application": "anime-remix",
  "managed_entries": [
    ".anime-remix-run",
    "normalized",
    "output.mp4",
    "parsed_script.json",
    "render.log",
    "retrieval_results.json",
    "run_manifest.json",
    "timeline.json"
  ]
}
```

`managed_entries` 按字节序排序、无重复、只含顶层名称。

### 15.4 运行清单

MVR manifest 至少包含：

```text
schema_version
status: running | succeeded | failed
application_version
python_version
ffmpeg_version
ffprobe_version
requested_parser
actual_parser
script_sha256
clips_json_sha256
aliases_sha256
selected_source_sha256
core_artifact_sha256
selected_source_frame_audit（第 3B-3 批）
core_artifact_member_sha256（第 3B-3 批）
output_sha256
assumed_color_metadata_asset_ids
started_at
finished_at
failed_stage
error_type
```

要求：

- `running.finished_at = null`；
- `succeeded` 或 `failed` 才有 finished_at；
- 不记录完整剧本文本；
- 不记录环境变量或密钥；
- `aliases_sha256`：未提供 aliases 文件时为 null；提供时为原始 aliases.json 文件字节的 SHA256；
- 不记录 aliases 文件绝对路径，不记录完整 alias 内容；
- `aliases_sha256` 不加入 `core_artifact_sha256` 的现有拼接定义；`core_artifact_sha256` 仍为 `SHA256(parsed_script.json 字节 || retrieval_results.json 字节 || timeline.json 字节)`；
- `selected_source_sha256` 保留现有字段名，语义为按 `asset_id` 字节序升序序列化的 `asset_id → source SHA256` 映射；
- 同一 `asset_id` 被多个 shot 重复选中时只保留一个映射项；
- 不同 `asset_id` 即使解析到同一物理文件，也分别保留各自的映射项；
- `core_artifact_sha256` 的稳定定义：`SHA256(parsed_script.json 字节 || retrieval_results.json 字节 || timeline.json 字节)`；
- `selected_source_frame_audit`：按 `asset_id` UTF-8 字节序排列的对象，每个值固定为 `{"metadata_nb_frames": 96, "counted_nb_frames": 96}`；未选择任何真实源（全 placeholder）时为 `{}`，不是 null；不写 `matches` 之类的冗余字段，因为两个整数已经足够复算；
- `core_artifact_member_sha256`：成功 manifest 中固定为 `{"parsed_script.json": "<sha256>", "retrieval_results.json": "<sha256>", "timeline.json": "<sha256>"}`，键按 UTF-8 字节序，每个值是对应最终文件字节的 SHA256；不得把 member hashes 的字符串拼接替代现有 `core_artifact_sha256`；
- manifest 生命周期：running manifest 创建时 `selected_source_frame_audit` 已经可以在 staging 前完成，因此写入最终 mapping，`core_artifact_member_sha256` 为 null（planning artifacts 尚未全部写完）；planning artifacts 写完后可计算 member hashes；succeeded 时两个字段都必须是最终值；如果 staging 创建后、planning artifacts 写完之后 render 失败，failed manifest 保留已经计算出的 `selected_source_frame_audit` 与 `core_artifact_member_sha256`；如果在 planning artifacts 完成前失败，`core_artifact_member_sha256` 可以仍为 null；不得为了补字段读取不存在的文件；
- 现有字段语义全部不变：`selected_source_sha256` 仍是 `asset_id → source SHA256` mapping，`core_artifact_sha256` 仍是三个核心文件原始字节直接拼接 SHA256；
- 第 3B-3 的 manifest hardening 只覆盖 planning core artifacts 与 selected source frame audit；不对 `normalized/*.mp4`、`joined_video.mp4`、临时 AAC、staging 临时文件建立 manifest hash；
- probe 顺序固定为 asset ID，不引入并发非确定性。

---

## 16. CLI

唯一应用：

```text
anime-remix
```

### 16.1 全局

```text
--help
--version
--verbose
```

### 16.2 `validate`

```bash
anime-remix validate \
  --script demo/script.md \
  --clips demo/clips.json
```

默认只做静态校验。

```bash
anime-remix validate ... --probe-media
```

才执行 FFprobe 和 MVR 媒体契约校验。

第 3B-1 批实现完成后允许：

```bash
anime-remix validate \
  --script demo/script.md \
  --clips demo/clips.json \
  --aliases demo/aliases.json
```

- `--aliases` 可省略；省略时行为与第 3A 批完全一致；
- 未加 `--probe-media` 时也必须验证 aliases schema 和 target；`--probe-media` 只额外执行媒体探测；
- aliases 路径由用户显式传入，不自动搜索当前目录；不实现默认 aliases 文件名发现、多文件叠加或配置目录。

### 16.3 `render`

```text
--timeline PATH
--output PATH
--overwrite
```

第 1 批必须完成。

render 不得增加 `--aliases`；render 永远不读取 aliases.json。
render 也不得读取 `emotion` / `shot_scale` 字段决定 FFmpeg 行为（第 3B-2 批起语义不变）。

### 16.4 `build`

```text
--script PATH
--clips PATH
--output PATH
--parser rule
--aliases PATH（第 3B-1 批实现后可选）
```

MVR 仍不支持：

```text
--config
--overwrite
--project-id
--allow-source-drift
```

第 3B-2 批不新增 CLI 选项；emotion / shot_scale 只作为 clips.json 可选字段和 script 解析结果存在。

第 3B-3 批也不新增任何 CLI 参数；不得新增 `--trace`、`--audit`、`--count-frames`、`--manifest-level`、`--debug-selection`；selection trace 与 selected-source frame audit 由 build 默认执行，validate / render 行为保持现状。

`--aliases` 的可用性：

- 第 3B-1 批实现完成前不得出现；
- 第 3B-1 批完成后，validate / build 支持可选 `--aliases`；
- render 始终不支持 `--aliases`；
- 除第 3B-1 完成后的 `--aliases` 外，上述选项不得提前出现在 help 中。

### 16.5 退出码

```text
0 成功
2 输入或 schema 错误
3 环境或媒体不支持
4 检索或时间线错误
5 渲染或输出验证错误
6 发布错误
```

CLI 默认不显示 Python traceback；`--verbose` 时可以显示。

---

## 17. 错误类型

至少定义：

```text
AnimeRemixError
InputValidationError
UnsafePathError
EnvironmentCapabilityError
MediaProbeError
UnsupportedMediaError
RetrievalError
TimelineValidationError
SourceDriftError
RenderError
OutputValidationError
PublicationError
UnsupportedStrategyError
```

错误消息要求：

- 指出阶段；
- 指出 asset ID 或 shot ID；
- 指出失败字段和实际值；
- 不泄露完整剧本文本；
- 不把原始完整 FFmpeg stderr 无限制输出。

---

## 18. 测试策略

### 18.1 单元测试

至少覆盖：

- strict 模型和未知键拒绝；
- NaN/Infinity 拒绝；
- `clips.json` 顶层数组拒绝；
- CharacterRef 合并和冲突；
- clips path 逃逸和符号链接；
- timeline source path 相对解析；
- 段落、对白、动作、目标帧；
- 最长不重叠人物/地点匹配；
- 有界 SequenceMatcher；
- bigram 缓存；
- Decimal HALF_UP；
- 人物 F2；
- 活跃权重；
- Emotion / ShotScale 枚举与 null 语义；
- 固定词典提取与多命中选择；
- emotion / shot_scale exact categorical score；
- 六维权重归一化与旧四维精确恢复；
- 稳定排序和平分；
- Top 3 只展示；
- gate 扫描继续查找；
- 中心帧裁剪；
- timeline 策略不变量；
- source drift；
- 旧版和未来版 schema 拒绝。

### 18.2 第 0 批媒体原型

必须动态生成素材并验证：

```text
24 帧 clip
36 帧 freeze_frame 原型
12 帧 placeholder
总计 72 帧
24 fps
最终 AAC 48 kHz stereo
最终时长 3 秒
静音 <= -90 dB
```

该原型还必须断言：

- 三个片段的真实 extradata SHA256 相同（`-show_data_hash sha256`）；
- 记录三个片段的实际 `avg_frame_rate`，不要求不同片段必须产生不同值；
- 不同长度片段在其他签名字段一致时 concat 仍然成功；
- 不使用 `avg_frame_rate` 作为签名字段；
- 缺少 `setparams` 时颜色 primaries/transfer 可能缺失；
- 加入 `setparams` 后完整为 BT.709。

第 0 批原型可以是测试脚本或临时命令，但结果必须写入完成报告。不得伪造。

### 18.3 第 1 批 renderer 集成测试

动态生成受限 24 fps 媒体，覆盖：

- 有源音频和无源音频；
- 横屏和竖屏；
- 空格、单引号和中文路径；
- `clip` 精确长度；
- `clip` 中心裁剪；
- 全 placeholder；
- clip + placeholder 混合；
- 损坏源；
- 指纹漂移；
- timeline 非连续 order；
- 非法源帧区间；
- 逐段 video-only；
- 每段实际帧数；
- concat signature；
- 最终一次 AAC；
- 音视频 start time 0；
- 最终静音；
- renderer 不调用 parser、retriever 或 clip loader。

媒体测试不得 skip。

### 18.4 第 2 批 build 端到端测试

必须覆盖：

- 3 段 demo 成功；
- 内容第一名源帧不足，继续选择较低 rank 的可渲染素材；
- 无候选时 placeholder；
- build 输出目录不存在时成功；
- target 已存在时失败且不修改 target；
- preflight 失败不创建 staging；
- staging 后失败不发布 target；
- selected source SHA256 真实写入；
- timeline 修改顺序后独立 rerender；
- rerender 不读取 script/clips/retrieval；
- final frame count 等于各 item 之和。

### 18.5 当前 MVR 门禁

第 2 批必须真实运行：

```bash
python -m pip install -e ".[dev]"
anime-remix --help
anime-remix --version
pytest -q
ruff check .
python demo/generate_media.py
anime-remix validate \
  --script demo/script.md \
  --clips demo/clips.json \
  --probe-media
anime-remix build \
  --script demo/script.md \
  --clips demo/clips.json \
  --output runs/demo-001 \
  --parser rule
anime-remix render \
  --timeline runs/demo-001/timeline.json \
  --output runs/demo-001-rerendered.mp4
```

要求：

- 媒体测试零跳过；
- 缺少 FFmpeg 不算通过；
- demo build 和 rerender 必须实际产生并验证 MP4；
- 完成后停止。

### 18.6 第 3A 批 freeze_frame 测试契约

最低覆盖（优先使用动态合成媒体，不依赖用户真实素材）：

1. 合法 freeze_frame 模型通过；
2. 非法 reason_code / strategy 组合失败；
3. freeze 缺少源字段失败；
4. `source_frame_count <= 0` 失败；
5. `source_frame_count >= target_frames` 失败（freeze_frame 分支）；
6. planner 只把至少 24 帧且短于 target 的候选视为 freeze eligible；
7. 高排名 freeze 候选不得抢在后续完整 clip 前被选择；
8. 没有完整 clip 时选择最高排名 freeze；
9. 没有 clip 和 freeze 时使用 placeholder；
10. render 验证源帧区间没有越界；
11. tpad 位于目标 fps 后（过滤图顺序断言或等价验证）；
12. freeze 片段实际帧数等于 target_frames；
13. 验证最后一帧被克隆延长，而不是黑帧或额外源帧；
14. 独立 render 只读取 timeline 和源文件；
15. build 与独立 rerender 均真实成功；
16. clip、placeholder 和第 0～2 批全部测试继续通过；
17. 媒体测试不得跳过。

### 18.7 第 3B-1 批 aliases 测试契约

最低覆盖：

模型和输入：

1. 合法 aliases.json 通过；
2. 顶层裸数组失败；
3. 未知字段失败；
4. 空 alias 失败；
5. 超长 alias 失败（超过 128 code point）；
6. 空 aliases 列表失败；
7. target 不存在失败；
8. 人物 target 使用 name 而非 ID 失败；
9. 地点 target 使用 name 而非 ID 失败；
10. 同一 alias_key 映射不同人物失败；
11. 同一 alias_key 映射不同地点失败；
12. 同一 target 下重复 alias_key 失败；
13. alias 与 canonical term 冲突失败；
14. Unicode NFKC / casefold 冲突失败；
15. 文件超限（> 1 MiB）失败。

解析：

16. 人物 alias 命中后输出 canonical CharacterRef；
17. 地点 alias 命中后输出 canonical location；
18. alias 不进入正式身份字段；
19. alias 与 canonical name 重叠时执行最长不重叠规则；
20. ASCII alias 正确使用词边界；
21. 正则特殊字符 alias 被安全转义；
22. 同一 target 多次命中只输出一次；
23. aliases 输入顺序变化不影响输出；
24. 未提供 aliases 时现有解析行为不变。

CLI 和工作流：

25. validate --aliases 静态成功；
26. build --aliases 成功；
27. aliases target 错误时 staging 不创建；
28. build 产物使用 canonical 人物和地点；
29. manifest aliases_sha256 为真实文件哈希；
30. 未提供 aliases 时 aliases_sha256 为 null；
31. render help 中不存在 --aliases；
32. 独立 render 不读取 aliases；
33. 删除 script、clips、aliases 和 retrieval 后，已有 timeline 仍可独立 rerender；
34. 现有 clip、freeze_frame、placeholder 媒体回归全部通过；
35. 媒体测试不得跳过。

### 18.8 第 3B-2 批 emotion / shot_scale 测试契约

最低覆盖：

模型：

1. ClipAsset 接受合法 emotion；
2. ClipAsset 接受合法 shot_scale；
3. 非法 emotion 拒绝；
4. 非法 shot_scale 拒绝；
5. 两字段省略时为 null；
6. ShotRequirement 相同；
7. ScoreBreakdown 支持 null / Decimal 情绪和景别分数。

parser：

8. happy 关键词提取；
9. sad；
10. angry；
11. fearful；
12. surprised；
13. tense；
14. calm；
15. close_up；
16. medium；
17. wide；
18. 无匹配返回 null；
19. 同段多情绪按固定规则选择；
20. 同段多景别按固定规则选择；
21. emotion 与 aliases 人物匹配可同时工作；
22. shot_scale 与 aliases 地点匹配可同时工作；
23. parser 不使用 fuzzy matching。

评分：

24. requirement emotion null → score null；
25. requirement emotion 有值、asset 相同 → 1；
26. requirement emotion 有值、asset 不同 → 0；
27. requirement emotion 有值、asset null → 0；
28-31. shot_scale 对称测试；
32. 两新维度 inactive 时 active weights 精确恢复旧权重；
33. 只有 emotion active 时权重正确归一化；
34. emotion + shot_scale 都 active 时权重正确；
35. Decimal HALF_UP 继续生效；
36. 不新增 emotion / shot_scale hard gate。

retrieval：

37. 构造两个其他分数相同的候选，emotion exact match 胜出；
38. shot_scale exact match 胜出；
39. emotion mismatch 不直接 hard reject；
40. 高排名规则和现有稳定 tie-break 不变；
41. clip > freeze_frame > placeholder 逻辑不变。

兼容：

42. 旧 demo 不含新 metadata 时 selection 与第 3B-1 一致；
43. 旧 demo source frame ranges 一致；
44. 旧 demo output.mp4 SHA256 与第 3B-1 已验收输出一致；
45. aliases demo 继续通过；
46. freeze demo 继续通过；
47. 独立 render 不根据 emotion / shot_scale 改变输出。

工作流：

48. build 生成带 emotion / shot_scale 的 parsed_script；
49. retrieval_results 保存对应 score；
50. timeline 保存 requirement 和 score metadata；
51. 独立 rerender 成功；
52. 删除 script / clips / aliases / retrieval 后仍可独立 rerender；
53. build 与 rerender 输出一致；
54. 所有既有测试继续通过；
55. 媒体测试零跳过。

### 18.9 第 3B-3 批 selection trace / count_frames / manifest 测试契约

最低覆盖：

selection trace：

1. selected clip trace；
2. freeze fallback 保存；
3. freeze fallback 后找到 clip；
4. 多个 freeze candidate 时只保存第一名；
5. too_short trace；
6. character gate failure；
7. action gate failure；
8. total < 0.55 early stop；
9. exhausted candidates；
10. placeholder final decision；
11. freeze final decision；
12. clip final decision；
13. final_decision 与 timeline 一致；
14. Top 3 与 scanned candidates 独立；
15. 第 5 名才选中时 trace 能记录 rank 1～5；
16. 未扫描候选不进入 trace；
17. repeated build trace 字节一致；
18. trace 无时间 / UUID / path / source text。

count_frames：

19. clip selected source 执行 count_frames；
20. freeze selected source 执行 count_frames；
21. placeholder 不执行；
22. 同 asset 多 shot 只 count 一次；
23. 不同 asset ID 同文件分别作为两个审计身份；
24. asset_id 字节序执行；
25. metadata == counted 成功；
26. metadata != counted 明确失败；
27. mismatch 不创建 staging；
28. mismatch 不执行 render；
29. mismatch 不重新 retrieval；
30. mismatch 不 fallback；
31. freeze mismatch 不修改 source_frame_count；
32. 未选中的素材不 count_frames。

manifest：

33. selected_source_frame_audit 真实值；
34. 全 placeholder 时为 {}；
35. key 按 asset_id 稳定排序；
36. core_artifact_member_sha256 三个真实哈希可复算；
37. member hash 键稳定；
38. core_artifact_sha256 旧定义仍可复算；
39. running manifest member hash 为 null；
40. succeeded manifest 为最终值；
41. render 后失败的 failed manifest 保留已知 audit / member hash；
42. aliases_sha256 语义不变；
43. selected_source_sha256 语义不变；
44. output_sha256 语义不变。

回归：

45. 原 demo selection 不变；
46. 原 demo source ranges 不变；
47. 原 demo output.mp4 SHA256 仍为 `f70d0187d3ff0427f7aaeb55778df492693688d3e8256c76daff2d45efc22a0e`；
48. freeze demo selection / output 行为不变；
49. aliases demo 行为不变；
50. semantic demo selection 不变；
51. semantic demo output.mp4 SHA256 仍为 `239b2b1ce2835750177ae86a80ac8131e357a6e62aff900488f2e7e2faeea414`；
52. independent rerender 仍与 build 输出一致；
53. renderer 不读取 trace / manifest；
54. 所有历史测试继续通过；
55. 媒体测试零跳过。

### 18.10 阶段 B：检索质量集与 30×1000 压力测试契约（已实现并验收）

#### B1 检索质量集

位置：`tests/quality/` 或仓库现有测试结构下等价位置。纯测试/benchmark 数据集，不使用真实动漫素材，产品运行时不得加载该质量集。

每个 case 固定包含：

```text
case_id
ShotRequirement
candidate ProbedClip 列表
expected_selected_asset_id（或 null）
expected_strategy
expected_reason_code
case_tags
简短的 expected rationale
```

质量集不是正式产品 JSON，可以使用 Python fixture / 测试数据对象。

至少 30 个独立 retrieval case（第一版不超过 50 个），覆盖：

```text
人物：
1. 单人物 ID 精确匹配
2. 多人物 recall
3. precision 惩罚
4. 同名不同 ID 不合并
5. name-only fallback
6. requirement 未指定人物

地点：
7. location ID exact
8. location name similarity
9. requirement 无地点

动作：
10. action exact
11. action description 辅助匹配
12. source_text description 辅助匹配
13. 动作明显不相关

时长 / strategy：
14. exact_length
15. center_trim
16. freeze_frame fallback
17. 高排名 freeze 后出现完整 clip
18. too_short
19. placeholder

aliases：
20. 人物 alias 最终 canonical
21. 地点 alias 最终 canonical

emotion：
22. emotion exact 提升正确候选
23. emotion mismatch 不 hard reject
24. asset emotion null 得 0

shot_scale：
25. shot_scale exact
26. shot_scale mismatch 不 hard reject

组合：
27. character + location + action
28. aliases + emotion + shot_scale
29. 高总分但 character gate fail
30. total < 0.55 early stop
```

expected truth 规则：

- 必须人工锁定并显式写出；禁止“运行当前 retriever → 把实际输出自动保存成 expected”的快照做法；
- 每个 case 至少断言 `selected_asset_id`、`selected_strategy`、`reason_code`；
- 有源策略（clip / freeze_frame）还应断言 `source_in_frame`、`source_frame_count`；
- 必要 case 再断言 `selected_global_rank`、`freeze_fallback_asset_id`、`stop_reason`。

质量指标：

```text
selection_accuracy = 完全命中 expected_selected_asset_id 的 case 数 / 全部 case 数
strategy_accuracy
reason_code_accuracy
按 case_tags 聚合的通过/失败数量（例如 character: 6/6、location: 3/3、emotion: 3/3、shot_scale: 2/2、strategy: 6/6）
```

第一版不引入 precision@k、recall@k、NDCG、MRR；Top 3 排名质量以后单独研究。

初始门槛：

```text
selection_accuracy = 100%
strategy_accuracy = 100%
reason_code_accuracy = 100%
```

这里的 100% 只表示“当前人工定义的最小规则质量集全部符合产品契约”，不得宣传成真实世界动漫素材检索准确率。

失败报告至少输出：case_id、expected/actual selected asset、expected/actual strategy、case tags；可附 actual global rank 与 selection trace 摘要；不得输出大量无关 candidate 数据；质量测试不得自动修改 golden expectation。

#### B2 30×1000 压力测试

测试对象是 parser 之后的 retrieval 核心：`retrieve(requirements, probed_clips)` 或当前等价 API。

输入固定：

```text
30 个 ShotRequirement（bench_shot_001 ... bench_shot_030）
1000 个 ProbedClip（bench_clip_0001 ... bench_clip_1000）
总 pair 数 = 30,000
```

- 不创建 1000 个真实 MP4；不执行 ffprobe、ffmpeg、SHA256、count_frames、render、staging、manifest 或 filesystem media pipeline；
- 压力测试使用确定性构造的内存 ShotRequirement / ProbedClip；
- 1000 个 synthetic ProbedClip 使用固定 deterministic generator：优先完全不用 random；若确需伪随机，必须使用本地固定 `random.Random(<固定整数>)`；禁止全局 random、当前时间、UUID、`hash()`；
- 每个 clip 稳定生成 characters、location、action、description、nb_frames、emotion、shot_scale；`resolved_path` 使用不会实际访问的稳定测试 Path，不得访问文件系统；
- 30 个 benchmark requirements 覆盖不同人物数量、地点、action 长度、source_text 长度、target_frames、emotion、shot_scale，但必须全部满足现有模型上限；压力测试不是 schema fuzzing。

性能测量边界：

- 只计时 retrieval 核心；计时前先执行至少 1 次 warm-up，正式至少测 5 次；
- 使用 `time.perf_counter()`，报告 min / median / max；
- 时间数据只能出现在测试报告或 benchmark 输出，不得写入正式核心 JSON；
- 内存可选使用标准库 `tracemalloc`，不新增 psutil；建议软目标 peak < 256 MiB，第一版为报告指标，不作为 CI 硬失败门槛。

性能门槛（当前开发机单进程，宽松、可本地验证，不是长期 SLA）：

```text
median <= 5.0 秒
max <= 8.0 秒
```

如果首次真实 benchmark 明显超过门槛：不改算法，先报告实际 min / median / max、主要 hotspot、是否 debug/coverage 环境，然后停止等待用户决定是否调整门槛或单独授权优化。

determinism 门禁：

- 同一 30×1000 输入至少连续运行 3 次；
- 必须断言所有 shot 的 `selected_asset_id`、`selected_global_rank`、strategy、reason_code、source_in_frame、source_frame_count 完全一致；
- `selection_trace` 序列化结果必须一致；
- 不得把 wall-clock benchmark 数据放进 trace。

候选顺序扰动门禁：

- 将同一 1000 个 ProbedClip 输入列表用固定方式重新排列；
- 最终 selection、rank after deterministic sorting、strategy、reason、source range 必须不变；
- 如果完全相同的排序键出现且只能依赖输入稳定性才能区分，则必须检查现有 asset_id tie-break 是否已消除歧义；不得通过增加新 tie-break 改契约；若发现契约不足，报告并停止。

缓存有效性门禁：

- clip 静态文本预处理（normalization / bigram 等）必须是 O(1000) 次，而不是 30 × 1000 次；
- 通过 monkeypatch / counter 在测试中计数相关 normalization 调用；
- 不得为了测试把 benchmark-only 字段写进产品 JSON。

SequenceMatcher 门禁：

- `SEQUENCE_MATCHER_MAX_CODEPOINTS = 256` 必须保持；
- B2 至少包含一部分 > 256 code point 的 action / description 文本；
- 测试证明这些 pair 不调用 SequenceMatcher；
- 不得调整 256 阈值。

selection trace 规模：

- 报告 30 个 shot 的平均 scanned candidate 数与最大 scanned candidate 数；
- 这不是产品指标，只用于理解 gate early-stop 是否有效；
- 不得人为限制“最多扫描 N 个候选”来改善 benchmark；selection 仍按真实规则运行。

Top 3 门禁：

- `top_k=3` 只影响展示；
- 不得只评分 Top 3、只扫描 Top 3 或预截断候选到 Top 3；
- 1000 个 candidates 必须参与现有全量 scoring / ranking。

真实媒体回归：

- 阶段 B 实现完成后仍必须运行原 demo、freeze demo、aliases demo、semantic demo；
- 已验收 golden hash（`f70d0187d3ff0427f7aaeb55778df492693688d3e8256c76daff2d45efc22a0e` 与 `239b2b1ce2835750177ae86a80ac8131e357a6e62aff900488f2e7e2faeea414`）继续保持；
- 独立 rerender 与 build 输出一致；媒体测试零跳过。

benchmark 与 pytest 的关系：

- 质量集：正常 pytest 硬门禁；
- 30×1000 determinism：正常 pytest 硬门禁；
- 30×1000 精确 wall-clock 性能：不作为普通 CI 的严格跨机器门禁；提供独立 benchmark 入口或 marker（例如 `pytest -m benchmark`）；当前开发机验收时必须真实执行。

禁止引入 benchmark 框架：

```text
pytest-benchmark
asv
pyperf
pandas
numpy
psutil
```

第一版只使用 pytest、`time.perf_counter`、`statistics.median`，可选 `tracemalloc`；依赖保持不变。

质量报告产物（`.tmp/quality-report.json`、`.tmp/retrieval-benchmark.json`）：

- 不属于正式产品 JSON；不进入 manifest、timeline 或 runs managed entries；可以被 .gitignore 忽略；
- 固定至少包含：

```json
{
  "quality": {
    "total_cases": 30,
    "passed_cases": 30,
    "selection_accuracy": 1.0,
    "strategy_accuracy": 1.0,
    "reason_code_accuracy": 1.0,
    "by_tag": {}
  },
  "benchmark": {
    "shots": 30,
    "clips": 1000,
    "pairs": 30000,
    "runs": 5,
    "min_seconds": 0.0,
    "median_seconds": 0.0,
    "max_seconds": 0.0,
    "deterministic": true,
    "average_scanned_candidates": 0.0,
    "max_scanned_candidates": 0
  }
}
```

可选：`peak_python_bytes`。

阶段 B 完成定义：

```text
B1 完成：至少 30 个人工锁定质量 case；全部 expected selection / strategy / reason 通过；输出质量摘要；不修改 retrieval 产品规则。
B2 完成：30 × 1000 = 30,000 pair；全量 scoring；全量 stable sort；正常 gate scan；3 次以上结果确定性一致；输入候选顺序扰动后结果一致；clip 静态预处理不发生 shot×clip 重复；长文本不违规调用 SequenceMatcher；当前机器真实性能数字被记录；媒体回归全部通过。
```

---

## 19. Codex 工作协议和实施批次

### 19.1 通用协议

每次执行：

1. 完整阅读根目录 `AGENTS.md`；
2. 阅读现有 `README.md`、`pyproject.toml` 和本批相关代码；
3. 检查 Git 状态；
4. 给出不超过 8 行的实施计划；
5. 明确当前批次完成门槛；
6. 只完成用户指定批次；
7. 不自动进入下一批次；
8. 修改前搜索现有接口，避免重复模块；
9. 不删除未知文件；
10. 不随意重命名用户文件；
11. 不增加本批未要求的依赖；
12. 优先小函数和真实集成测试；
13. 运行本批规定的命令；
14. 命令不能运行时报告阻塞原因；
15. 不伪造 FFmpeg、安装、测试或 demo 结果；
16. 不把 TODO、`pass`、空函数或跳过测试计为完成；
17. 不写死本机绝对路径；
18. 报告未实现内容和已知限制；
19. 当前批次完成后停止。

### 19.2 批次 0：FFmpeg 可行性门禁

目标：在写产品架构前验证最高风险媒体假设。

实现或执行：

```text
检查 ffmpeg / ffprobe 版本和能力
动态生成受限媒体
生成 clip / freeze 原型 / placeholder 三段
显式 setparams 写 BT.709
逐段验证帧数和颜色
验证真实 extradata SHA256 相同（-show_data_hash sha256）
记录 avg_frame_rate 并验证不进入 concat signature
concat demuxer stream copy
单个 anullsrc 精确样本 AAC
验证最终帧数、音频、start time 和静音
```

禁止：

```text
建立完整项目架构
parser
retrieval
Web
数据库
LLM
```

停止条件：

- 原型真实通过；或
- 明确记录阻塞和失败命令；
- 不得在原型失败时进入第 1 批。

### 19.3 批次 1：Renderer Walking Skeleton

实现：

```text
src 布局
Typer CLI --help / --version
strict Pydantic timeline 模型
原子 JSON
FFmpeg adapter
受限 timeline path 校验
source size / SHA256 校验
clip 策略
纯黑 placeholder
video-only 片段编码
逐段 frame validation
concat signature（排除 avg_frame_rate）
concat stream copy
一次静音 AAC
最终输出验证
独立 anime-remix render
动态 smoke fixture 生成脚本
手写 timeline fixture
真实 renderer smoke MP4
```

禁止：

```text
script parser
clips.json loader
retrieval
build workflow
freeze_frame
配置系统
并发 probe
```

验收：

```bash
python -m pip install -e ".[dev]"
anime-remix --help
anime-remix --version
pytest -q
ruff check .
python tests/fixtures/generate_render_smoke.py \
  --output .tmp/render-smoke-fixture

anime-remix render \
  --timeline .tmp/render-smoke-fixture/timeline.json \
  --output runs/render-smoke.mp4
```

媒体测试零跳过。完成后停止。

### 19.4 批次 2：Planner + Build MVR

实现：

```text
script.md loader
clips.json 1.9 wrapper schema
安全素材路径
serial ffprobe，asset ID 固定顺序
MVR 媒体输入契约
CharacterRef canonical merge
规则段落解析
人物/地点最长不重叠匹配
对白、动作、目标帧
有界文本相似度和缓存
人物 F2、地点、动作、时长评分
Decimal HALF_UP
稳定全量排序
Top 3 只展示
gate 扫描和可渲染性
exact_length / center_trim / no_candidate
选中源 size / SHA256
timeline 编译
preflight 后 sibling staging
marker 和 manifest
parsed_script.json
retrieval_results.json
timeline.json
build 调用已验证 render workflow
成功后 rename 到不存在 target
demo build
修改 timeline 后 rerender
```

禁止：

```text
freeze_frame
aliases.json
情绪/景别评分
build --overwrite
source drift override
LLM
YAML 配置
并发 probe
复杂媒体转换
```

验收：执行第 18.5 节全部命令，媒体测试零跳过。

完成后必须停止，让用户检查 MP4 和 timeline，不得自动进入第 3A 批。

### 19.5 批次 3A：freeze_frame 产品策略与渲染闭环

只有用户明确要求才执行。实现范围严格限于 §1.4、§9.1、§9.6、§11.6、§11.7 和 §12.4 定义的 freeze_frame 契约：

可实现：

```text
TimelineStrategy.FREEZE_FRAME = "freeze_frame"
reason_code 增加 short_source_freeze（保持封闭 Literal）
freeze_frame 候选资格与 clip > freeze_frame > placeholder 扫描
planner 固定源帧字段
独立 tpad 过滤图
tpad 位于目标 fps 后
tpad 进入 required FFmpeg capability
freeze 审计语义写入 retrieval_results.json
build 与独立 render 支持 freeze_frame
```

仍禁止：

```text
aliases.json
情绪/景别评分
speed_adjust
多种 freeze 模式 / 用户可配置阈值
首帧冻结 / 中间帧冻结 / 只冻结单张任意帧
25/30 fps
VFR
旋转
非方形 SAR
BT.601 转换
字幕/字体/配音或音乐
build --overwrite
source drift override
LLM
YAML 配置
并发 probe
复杂媒体转换
```

验收：执行第 18.6 节测试契约与第 20.4 节命令，媒体测试零跳过。

完成后必须停止，让用户检查 freeze_frame 视频，不得自动进入第 3B-1 批。

### 19.6 批次 3B-1：aliases.json 人物和地点别名（已完成并人工验收）

只有用户明确要求才执行。实现范围严格限于 §7.5、§10.2、§10.3、§15.1、§15.4 和 §16 定义的 aliases 契约：

```text
aliases.json 1.9 对象 schema（人物/地点别名）
canonical target 校验（clips.json canonical merge 之后）
alias_key 规范化与冲突检查
validate / build 可选 --aliases
parser 词典扩展与 canonical 输出
manifest aliases_sha256
省略 --aliases 时行为与第 3A 批完全一致
```

仍禁止：

```text
动作别名
情绪/景别别名
模糊匹配 / 拼音匹配 / 同义词模型
Embedding / LLM 自动别名
自动别名推导
多层别名
多文件合并 / 热加载 / 配置目录
selection trace 扩展
count_frames 审计
manifest 其他补强
speed_adjust
VFR
旋转
overwrite / rollback
LLM
字幕和字体
```

验收：执行第 18.7 节测试契约与第 20.5 节命令，媒体测试零跳过。

完成后必须停止，不得自动进入第 3B-2 或 3B-3 批。

### 19.7 批次 3B-2：情绪与景别提取和评分（已实现并人工验收）

只有用户明确要求才执行。实现范围严格限于 §1.6、§7.2、§9.1、§9.2、§9.3、§9.4、§10.7、§11.3、§11.4、§11.5、§11.6 和 §18.8 定义的 emotion / shot_scale 契约：

可实现：

```text
Emotion / ShotScale 封闭枚举（null 表示未指定/未识别）
ClipAsset / ShotRequirement 可选字段（省略等于 null，旧 JSON 继续兼容）
parser 固定词典提取与多命中选择
exact categorical match 评分
六维基础权重与 inactive 归一化（旧四维精确恢复）
timeline 保存 planner metadata 与 score（renderer 不读取）
demo/semantic/ 无版权合成 demo（emotion 区分、shot_scale 区分、双 inactive 回归）
```

仍禁止：

```text
CV 情绪识别 / 人脸表情识别 / 自动镜头景别视觉分类
LLM / Embedding / sentiment model
用户自定义情绪词典
emotion aliases / shot_scale aliases
多情绪、情绪强度、情绪变化轨迹
景别距离矩阵、景别部分匹配
fuzzy matching、拼音、分词
emotion / shot_scale hard gate
selection trace 扩展
count_frames 审计
manifest 其他补强
speed_adjust
VFR
旋转
overwrite / rollback
字幕、配音或音乐
```

验收：执行第 18.8 节测试契约与第 20.6 节命令，媒体测试零跳过。

完成后必须停止，不得自动进入第 3B-3 批。

### 19.8 批次 3B-3：selection trace / count_frames / manifest 补强（已实现并人工验收）

只有用户明确要求才执行。实现范围严格限于 §1.7、§11.7、§11.8、§12.8、§14.2、§15.1、§15.2、§15.4、§16 和 §18.9 定义的契约：

可实现：

```text
retrieval_results.json 每个 shot 增加正式 selection_trace
ScannedCandidateTrace（复用 ScoreBreakdown 量化 score）
candidate decision / content_gate / frame_gate / stop_reason 封闭枚举
freeze_fallback_asset_id 单值审计（不建 fallback stack）
final_decision 与 timeline 一致性
build 对最终选中 unique asset_id 串行 ffprobe -count_frames（asset_id UTF-8 字节序）
metadata_nb_frames == counted_nb_frames 严格门禁
count mismatch 在 staging 前失败（不 reretrieval / fallback / placeholder / 修改 freeze source_frame_count）
manifest selected_source_frame_audit 与 core_artifact_member_sha256 及生命周期
core_artifact_sha256 定义不变；selection_trace 使 retrieval_results.json 字节变化属预期
```

仍禁止：

```text
新检索算法 / 新 scoring 维度 / 权重调整 / 新 hard gate
emotion / shot_scale / aliases 改进
fuzzy matching / 拼音 / LLM / Embedding / reranking model / CV
对全部素材 count_frames / 并发 ffprobe / background audit
新 CLI 参数
source drift override / build overwrite / rollback
dependency lock hash / normalized segment hash / joined_video hash / renderer trace
subtitles / voice / music / Web / DB / Worker / 无关重构
```

验收：执行第 18.9 节测试契约与第 20.7 节命令，媒体测试零跳过。

完成后必须停止，等待用户检查。

### 19.9 阶段 B：检索质量集 + 30×1000 压力测试（已实现并验收）

只有用户明确要求才执行。实现范围严格限于 §1.9、§18.10、§20.8 和 §22 定义的阶段 B 契约：

可实现：

```text
tests/quality/ 检索质量集（>= 30 个人工锁定 case，<= 50）
质量指标与按 case_tags 聚合报告
30×1000 内存压力测试（30 个 requirement × 1000 个 synthetic ProbedClip）
determinism / shuffle / cache / SequenceMatcher / Top 3 门禁
perf_counter + statistics.median benchmark（warm-up + 至少 5 runs）
.tmp/quality-report.json 与 .tmp/retrieval-benchmark.json
pytest marker（例如 -m benchmark）隔离 wall-clock 性能
```

仍禁止：

```text
新产品功能
parser / aliases / emotion / shot_scale / scoring / weights / hard gates / ranking / strategy 优先级修改
selection trace / renderer / FFmpeg / manifest 产品语义修改
为改善 benchmark 数字而优化算法
并发 / multiprocessing / numpy / pandas / psutil / 缓存框架
pytest-benchmark / asv / pyperf
视频生成 API / generate strategy / prompt compiler / reference image / 模型 provider / 云 API / API key
Web / DB / Worker / 无关重构
```

验收：执行 §18.10 测试契约与 §20.8 命令，媒体测试零跳过。阶段 B 已完成并验收，不再作为进行中批次；retrieval 产品规则冻结，性能优化只能作为独立 B3 批授权。

### 19.10 阶段 2 G0：视频生成模型可行性实验（产品契约已定义，代码未开始）

只有用户明确要求，且完成最新模型调研并由用户确认首个实验 provider/model 后才执行。实现范围严格限于 §1.10 与 §20.9：

可实现：

```text
experiments/video-generation/ 实验目录（README.md、requests/、outputs/、probes/、results.json）
最薄的实验脚本或人工调用单个已确认模型
GenerationRequest 实验记录与 results.json
模型原始输出 probe 与 FFmpeg normalization 思路转换实验
Test A / B / C 三个无版权单镜头实验与人工 pass / borderline / fail 记录
reference-image consistency 对照实验
```

仍禁止：

```text
GENERATE strategy / 修改 clip、freeze_frame、placeholder / planner 改动
provider framework / 多镜头自动生成 / character database
LoRA 训练 / fine-tuning / prompt optimizer agent / LLM planner
自动 seed 搜索 / 无限重试 / 自动质量评分模型
lip sync / voice / subtitles / music / Web / DB / Worker / billing
版权动漫角色素材 / 素材抓取逻辑
```

验收：按 §1.10 的 G0 成功定义报告实验结果；不得修改产品 src；G0 通过后才允许规划 G1（Single-Shot Generation Pipeline）。完成后停止。

---

## 20. 可直接复制给 Codex 的分批指令

### 20.1 批次 0

```text
完整阅读根目录 AGENTS.md，只执行第 19.2 节“批次 0：FFmpeg 可行性门禁”，完成后停止。

在当前环境真实检查 ffmpeg、ffprobe、libx264、AAC 和所需过滤器。动态生成受限媒体，验证 24 帧 clip、36 帧 freeze_frame 原型和 12 帧 placeholder；所有片段必须 video-only、BT.709 limited，并通过 nb_read_frames 校验。记录各片段实际 avg_frame_rate（不要求不同），验证真实 extradata SHA256 相同（-show_data_hash sha256）、avg_frame_rate 不进入 concat signature，且不同长度片段在其他签名字段一致时 concat demuxer 仍可 stream copy。用单个 anullsrc 按 72 帧推导的精确样本数只编码一次 AAC，验证最终 72 帧、3 秒音频、start_time=0 和静音。

不要建立完整项目架构，不要实现 parser、retrieval、Web、数据库或 LLM。报告真实命令、结果和失败；原型失败时不得进入批次 1。
```

### 20.2 批次 1

```text
完整阅读根目录 AGENTS.md 和现有代码，只完成第 19.3 节“批次 1：Renderer Walking Skeleton”，完成后停止。

实现 src 布局、Typer CLI、strict Timeline 1.9 模型、原子 JSON、FFmpeg adapter、timeline 相对源路径和 source size/SHA256 校验、clip 与纯黑 placeholder 两种策略、video-only 标准化片段、逐段 nb_read_frames 校验、排除 avg_frame_rate 的 concat signature、concat demuxer stream copy、单个 anullsrc 一次 AAC、最终帧数/音频/start_time/静音验证，以及只读取 timeline 和源文件的 anime-remix render。

先运行 `tests/fixtures/generate_render_smoke.py` 动态生成无版权合成媒体、timeline 和真实指纹，再使用该手写 timeline fixture 生成 `runs/render-smoke.mp4`。不要实现 script parser、clips.json、retrieval、build、freeze_frame、YAML 配置或并发 probe。运行安装、CLI、pytest -q、ruff check . 和真实 render；媒体测试零跳过；不要开始批次 2。
```

### 20.3 批次 2

```text
完整阅读根目录 AGENTS.md 和现有代码，只完成第 19.4 节“批次 2：Planner + Build MVR”，完成后停止。

实现 script.md 和 clips.json 1.9 对象 schema、安全素材路径、串行 ffprobe、受限 24 fps 媒体契约、CharacterRef canonical merge、段落规则解析、人物/地点最长不重叠匹配、对白、动作、整数目标帧、有界文本相似度和缓存、人物 F2、地点/动作/时长评分、Decimal HALF_UP、稳定全量排序、Top 3 只展示、gate 扫描、exact_length/center_trim/no_candidate、选中源 size/SHA256、自包含 timeline、parsed_script.json、retrieval_results.json、sibling staging、marker/manifest，以及调用现有 render workflow 的完整 build。

先运行 `python demo/generate_media.py` 生成无版权合成 demo 素材，再真实运行安装、CLI、pytest -q、ruff check .、validate --probe-media、demo build 和独立 rerender。媒体测试零跳过。不要实现 freeze_frame、aliases、情绪/景别、build --overwrite、source drift override、YAML、LLM、并发 probe或复杂媒体转换。完成后停止，让用户检查 output.mp4 和 timeline.json，不要开始第 3A 批。
```

### 20.4 批次 3A

```text
完整阅读根目录 AGENTS.md，只完成第 19.5 节“批次 3A：freeze_frame 产品策略与渲染闭环”，完成后停止。

按 §9.1 增加 FREEZE_FRAME = "freeze_frame"；按 §9.6 扩展封闭 reason_code（增加 short_source_freeze）并实现 freeze_frame 字段不变量；按 §11.6 实现内容门槛与帧数资格（clip_eligible / freeze_eligible / too_short，MIN_FREEZE_SOURCE_FRAMES = 24）以及 clip > freeze_frame > placeholder 扫描；按 §11.7 写入 planner 固定源帧字段和 retrieval 审计；按 §12.4 实现独立过滤图（tpad 位于目标 fps 后、stop=target_frames 过量补帧、trim=end_frame 精确封口），并把 tpad 加入 required FFmpeg capability。

实现 build 与独立 render 的 freeze_frame 支持，保持独立 render 只读取 timeline 和源文件、source size/SHA256 校验不变、clip 与 placeholder 行为不变。运行第 18.6 节测试契约、全部媒体回归、pytest -q 和 ruff check .，并用新目录（例如 runs/demo-freeze-001）真实 build 与独立 rerender，生成含 freeze_frame item 的 timeline 和 output.mp4。

不要实现 aliases、情绪/景别、speed_adjust、25/30 fps、VFR、旋转、非方形 SAR、BT.601 转换、多种 freeze 模式、用户可配置阈值、首帧/中间帧/单帧冻结、build overwrite/rollback、source drift override、LLM、字幕/字体/配音/音乐或新工作流框架。完成后停止，等待用户检查 freeze_frame 视频。
```

### 20.5 批次 3B-1

```text
完整阅读根目录 AGENTS.md，只完成第 19.6 节“批次 3B-1：aliases.json 人物和地点别名”，完成后停止。

按 §7.5 实现 aliases.json 1.9 对象 schema（extra=forbid、strict、allow_inf_nan=False、default_factory、大小与数量限制）；在 clips.json canonical merge 之后校验人物/地点 target 必须为真实 canonical ID；按 alias_key = NFKC → strip → casefold 实现类别内唯一性与 canonical 冗余/冲突检查；按 §10.2/§10.3 扩展 parser 词典（canonical ID + canonical name + aliases）并保持最长不重叠匹配、词边界、正则转义和 canonical 输出；为 validate/build 增加可选 --aliases（render 不支持、不读取）；manifest 增加 aliases_sha256（未提供为 null，不参与 core_artifact_sha256）；省略 --aliases 时行为与第 3A 批完全一致。

运行第 18.7 节测试契约、全部媒体回归、pytest -q 和 ruff check .。不要实现动作/情绪/景别别名、模糊/拼音/同义词匹配、Embedding、LLM、自动别名、多层别名、多文件合并、热加载、selection trace/count_frames/manifest 其他补强、speed_adjust、VFR、旋转、overwrite/rollback、字幕/配音/音乐或新工作流框架。完成后停止，等待用户检查。
```

### 20.6 批次 3B-2（已实现并人工验收）

```text
完整阅读根目录 AGENTS.md，只完成第 19.7 节“批次 3B-2：情绪与景别提取和评分”，完成后停止。

按 §9.1 增加 Emotion（happy/sad/angry/fearful/surprised/tense/calm）与 ShotScale（close_up/medium/wide）封闭枚举，null 表示未指定/未识别且 null != calm；按 §7.2/§9.2/§9.3 为 ClipAsset 与 ShotRequirement 增加可选 emotion / shot_scale（省略等于 null，旧 clips.json / timeline.json 无此字段时继续兼容）；按 §10.7 用代码内固定词典对 emotion 与 shot_scale 独立、确定性提取（多命中按起点升序 → term 长度降序 → enum 固定顺序 → term 字节序升序选第一项，无命中为 null）；按 §11.3 实现 exact categorical match 评分（requirement 未指定 → score null 且维度 inactive；指定后素材相同 → 1，不同或素材缺失 → 0）；按 §11.4 扩展六维基础权重（character 0.20 / location 0.12 / action 0.36 / duration 0.12 / emotion 0.10 / shot_scale 0.10），两新维度均 inactive 时活跃权重归一化必须精确恢复旧四维 0.25 / 0.15 / 0.45 / 0.15；不新增 emotion / shot_scale hard gate；稳定排序不新增 tie-break；timeline 中 requirement.emotion / requirement.shot_scale 与 score.emotion / score.shot_scale 仅作 planner metadata / score，renderer 不得读取决定 FFmpeg 行为。

运行第 18.8 节测试契约、全部媒体回归、pytest -q 和 ruff check .，并用 demo/semantic/（至少三镜头：emotion 区分、shot_scale 区分、双 inactive 回归；可同时使用 aliases，但不得依赖）真实 build 与独立 rerender，确认旧 demo 的 selection、source frame ranges 和 output.mp4 SHA256 与第 3B-1 已验收结果一致。

不要实现 CV 情绪识别、人脸表情识别、自动镜头景别视觉分类、LLM、Embedding、sentiment model、用户自定义情绪词典、emotion/shot_scale aliases、多情绪/情绪强度/情绪变化轨迹、景别距离矩阵/景别部分匹配、fuzzy matching/拼音/分词、emotion/shot_scale hard gate、selection trace 扩展、count_frames 审计、manifest 进一步补强、speed_adjust、VFR、转场、字幕/配音/音乐、Web/DB 或无关重构。完成后停止，等待用户检查。
```

### 20.7 批次 3B-3（已实现并人工验收）

```text
完整阅读根目录 AGENTS.md，只完成第 19.8 节“批次 3B-3：selection trace / count_frames / manifest 补强”，完成后停止。

按 §11.8 在 retrieval_results.json 每个 shot 增加正式 selection_trace（scanned_candidates / stop_reason / freeze_fallback_asset_id / final_decision），candidate 必须复用 ScoreBreakdown 最终量化 score，content_gate / frame_gate / decision / stop_reason 使用封闭枚举，scanned_candidates 只含实际扫描候选并按 global rank 顺序；按 §12.8 对最终选中 unique asset_id 用 ffprobe -count_frames 串行执行真实计数（asset_id UTF-8 字节序、最多 10 次），严格校验 metadata_nb_frames == counted_nb_frames，mismatch 在 staging 前失败且不 reretrieval / fallback / placeholder / 修改 freeze source_frame_count；按 §15.1 / §15.2 把 count audit 插入 hash_selected_sources 之后、compile timeline 之前；按 §15.4 增加 selected_source_frame_audit（全 placeholder 为 {}）与 core_artifact_member_sha256（running 为 null、succeeded 为最终值、failed 按生命周期保留），core_artifact_sha256 旧定义不变；不新增 CLI 参数；独立 render 不读取 trace / manifest / audit。

运行第 18.9 节测试契约、全部媒体回归、pytest -q 和 ruff check .，并真实 build / 独立 rerender，确认原 demo、freeze demo、aliases demo、semantic demo 的 selection / source ranges / output.mp4 SHA256 与已验收结果一致（f70d0187d3ff0427f7aaeb55778df492693688d3e8256c76daff2d45efc22a0e 与 239b2b1ce2835750177ae86a80ac8131e357a6e62aff900488f2e7e2faeea414）。

不要实现新检索算法、新 scoring 维度、权重调整、新 hard gate、emotion / shot_scale / aliases 改进、fuzzy / 拼音 / LLM / Embedding / reranking / CV、对全部素材 count_frames、并发 ffprobe、background audit、新 CLI 参数、source drift override、overwrite / rollback、dependency lock hash、normalized / joined hash、renderer trace、字幕 / 配音 / 音乐、Web / DB / Worker 或无关重构。完成后停止，等待用户检查。
```

### 20.8 阶段 B（已实现并验收）

```text
完整阅读根目录 AGENTS.md，只完成第 19.9 节“阶段 B：检索质量集 + 30×1000 压力测试”，完成后停止。

按 §18.10 建立 tests/quality/ 检索质量集：至少 30 个人工锁定的 retrieval case（case_id、ShotRequirement、candidate ProbedClip 列表、expected_selected_asset_id / expected_strategy / expected_reason_code、case_tags、expected rationale），覆盖人物/地点/动作/时长与 strategy/aliases/emotion/shot_scale/组合共 30 类，expected truth 必须人工写出，禁止用当前输出快照；计算 selection_accuracy / strategy_accuracy / reason_code_accuracy 与按 tag 聚合，初始门槛全部 100%（只是规则回归集，不代表真实世界准确率）。

按 §18.10 建立 30×1000 内存压力测试：30 个 ShotRequirement（bench_shot_001..030）× 1000 个 synthetic ProbedClip（bench_clip_0001..1000）= 30,000 pair；使用固定 deterministic generator（固定 random.Random(<固定整数>) 或完全不用 random），不创建真实 MP4，不执行 ffprobe / ffmpeg / SHA256 / count_frames / render / staging / manifest / filesystem；只测 retrieval 核心。至少 3 次连续运行断言 selection / rank / strategy / reason / source range / selection_trace 序列化完全一致；固定重排 1000 个候选后结果不变；用 monkeypatch/counter 证明 clip 静态预处理是 O(1000) 而非 30×1000；包含 >256 code point 文本并证明不调用 SequenceMatcher；Top 3 只展示，1000 候选全部参与 scoring/ranking/scan。

性能 benchmark 使用 time.perf_counter + statistics.median：先 warm-up ≥1 次，正式 ≥5 次，报告 min / median / max；当前开发机门槛 median <= 5.0 秒、max <= 8.0 秒（宽松本地门槛，非跨机器 CI 门禁；可选用 tracemalloc 报告 peak_python_bytes，软目标 < 256 MiB）。把 wall-clock 性能放到独立 marker（例如 pytest -m benchmark），质量集与 determinism 是普通 pytest 硬门禁。输出 .tmp/quality-report.json 与 .tmp/retrieval-benchmark.json（不属于正式产品 JSON，不进入 manifest/timeline/runs）。

运行全部历史回归（原 demo、freeze、aliases、semantic），golden hash f70d0187d3ff0427f7aaeb55778df492693688d3e8256c76daff2d45efc22a0e 与 239b2b1ce2835750177ae86a80ac8131e357a6e62aff900488f2e7e2faeea414 必须保持；媒体测试零跳过。

不要修改 parser / aliases / emotion / shot_scale / scoring / weights / hard gates / ranking / strategy 优先级 / selection trace / renderer / FFmpeg / manifest 产品语义；不要为改善 benchmark 数字优化算法；不要引入并发、multiprocessing、numpy、pandas、psutil、pytest-benchmark、asv、pyperf 或缓存框架；不要接入视频生成 API、generate strategy、prompt compiler、reference image、模型 provider、云 API 或 API key。若性能明显超门槛，先报告 min/median/max 与 hotspot 后停止，等待用户决定。完成后停止，等待用户检查。
```

### 20.9 阶段 2 G0

```text
完整阅读根目录 AGENTS.md，只完成第 19.10 节“阶段 2 G0：视频生成模型可行性实验”，完成后停止。

先进行截至执行当天的模型调研：比较当前可获得的视频生成模型/API（程序化 API、text-to-video、image-to-video、reference image / character consistency、输出时长、分辨率、延迟、价格、使用限制、合法使用边界），不得按旧知识锁死 provider；把调研结果交给用户，由用户确认首个实验 provider/model 后才能写任何实验代码。

在 experiments/video-generation/ 下建立实验目录（README.md、requests/、outputs/、probes/、results.json）。使用原创人物与合成参考图，按 §1.10 固定 GenerationRequest（duration_seconds=3、aspect_ratio=16:9），执行 Test A（静态人物+简单动作，medium）、Test B（明显动作，wide）、Test C（近景表情，平静→惊讶）三个无版权单镜头实验。若模型支持 image-to-video / reference image，必须用同一张原创人物参考图分别生成 A/B/C 并人工比较头发、脸型、服装、颜色与主要身份特征。

对每个模型记录 provider、model、generation mode、input type、reference-image support、requested/actual duration、aspect ratio、actual resolution、actual fps、actual frame count、generation latency、success/failure，可记录 estimated cost 但不得假造；对每个生成视频做 10 项人工检查（主体一致、脸、手/肢体、动作、背景漂移、shot_scale、emotion、闪烁、换人、是否适合做 Remix 源），等级 pass / borderline / fail，写进 results.json（不得含 API key、环境变量或敏感信息）。

生成成功后不改产品代码：先手工 ffprobe 原始输出，再尝试用现有 FFmpeg normalization 思路转成受限 MP4（H.264 1280×720 24fps CFR yuv420p SAR 1:1 BT.709 limited、去音频），区分 raw_output_probe 与 normalized_output_probe。

不得实现 GENERATE strategy，不得修改 clip / freeze_frame / placeholder / planner，不得建立 provider framework，不得接 Web / DB / Worker，不得做多镜头自动生成、LoRA、fine-tuning、prompt optimizer、LLM planner、自动 seed、无限重试、自动质量评分、lip sync、voice、subtitles、music、billing 或素材抓取。按 §1.10 成功定义报告：至少一个模型生成三镜头、三输出均可稳定标准化、至少两镜头 pass/borderline、reference image 明显约束身份、无系统性失败、成本/延迟可接受。G0 通过后才允许规划 G1；不通过则报告模型限制并停止。完成后停止，等待用户检查。
```

---

## 21. Codex 完成报告格式

```text
完成内容
- ...

关键设计决定
- ...

修改文件
- ...

实际执行
- `...`

测试结果
- 通过：...
- 失败：...
- 跳过：...（必须说明；媒体测试不允许跳过）

运行环境
- Python：...
- FFmpeg：...
- ffprobe：...

当前完成点
- 第 0 批未达到 / 已达到
- 第 1 批未达到 / 已达到
- 第 2 批 MVR 未达到 / 已达到
- 第 3A 批 freeze_frame 未开始 / 已实现 / 已人工验收
- 第 3B-1 批 aliases 未开始 / 已实现 / 已人工验收
- 第 3B-2 批 emotion / shot_scale 未开始 / 已实现 / 已人工验收
- 第 3B-3 批 selection trace / count_frames / manifest 未开始 / 已实现 / 已人工验收
- 第一阶段 Planner / Renderer 主闭环 未完成 / 完成并冻结
- 阶段 B 检索质量集 + 30×1000 压力测试 未完成 / 完成并冻结
- 第二阶段 G0 视频生成可行性实验 契约未补齐 / 已补齐、代码未开始 / 已执行 / 已验收
- 第二阶段 G1 Single-Shot Generation Pipeline 未授权 / 已授权

生成成果
- MP4：...
- timeline：...

已知限制
- ...

未实现
- ...

下一批次
- 未自动开始
```

---

## 22. README 最低内容

第 2 批 README 至少包含：

```text
项目用途
MVR 边界
系统要求
安装
严格 24 fps 媒体输入契约
clips.json 1.9 格式
script.md 格式
validate
build
render
输出文件
timeline 编辑示例
clip 与 placeholder
源指纹和 drift 失败
video-only 片段和最终一次 AAC
常见错误
测试命令
版权提醒
尚未实现内容
```

README 不得宣称支持：

```text
任意 MP4
VFR
旋转
非方形 SAR
阶段 B quality set / benchmark（阶段 B 代码实现完成前）
视频生成 / GENERATE strategy（正式集成完成前）
speed_adjust
字幕
LLM
Web
```

第 3A 批代码实现完成后，README 必须补充 freeze_frame 的选择条件、timeline 字段示例和独立 rerender 说明。

第 3B-1 批代码实现完成后，README 才可增加：aliases.json 格式、--aliases 使用方法、只支持人物和地点、alias 输出 canonical ID/name、冲突和 target 校验、render 不读取 aliases、aliases 文件不是自动发现的、当前不支持动作/情绪/景别/模糊匹配/LLM。

第 3B-2 批代码实现并验收后，README 才可增加：

```text
emotion / shot_scale 可选素材字段
固定枚举（Emotion / ShotScale）
rule parser 固定关键词
exact categorical scoring
六维基础权重与 inactive 归一化
缺失字段行为（requirement 未指定为 null；素材缺失按 0 分）
无独立 hard gate
renderer 不使用这些字段
当前只是粗粒度规则模式，不是视觉识别或 AI 情绪识别
```

第 3B-3 批代码实现并验收后，README 才可增加：

```text
retrieval_results 包含完整 selection trace
Top 3 与实际 scan trace 的区别
trace 只用于解释，不改变 selection
build 对最终选中源执行 ffprobe count_frames
metadata nb_frames 必须与 counted frames 完全一致
selected_source_frame_audit
core_artifact_member_sha256
独立 render 不读取这些 build 审计数据
```

阶段 B 实现并验收后，README 才可增加：

```text
quality set 只是规则回归集，不代表真实世界准确率
30×1000 retrieval benchmark
benchmark 不包含 FFmpeg
当前机器实测性能
deterministic guarantee
当前第一阶段冻结点
下一阶段为生成模型探索
```

第二阶段 G0 实验产物（`experiments/video-generation/`）不写入 README 产品功能；只有 G1 进入正式集成并验收后，README 才可增加生成流程说明。

---

## 23. MVR 后路线图

只有第 2 批真实通过并经用户检查后，才允许规划后续：

```text
阶段 A-3A：freeze_frame 产品策略与渲染闭环（已实现并人工验收）
阶段 A-3B-1：aliases.json 人物和地点别名（已实现并人工验收）
阶段 A-3B-2：情绪和景别（已实现并人工验收）
阶段 A-3B-3：selection trace / count_frames / manifest 补强（已实现并人工验收）
阶段 B：检索质量集和 30×1,000 压力测试（完成并冻结）
第二阶段 G0：视频生成模型可行性实验（产品契约已补齐，代码实现未开始）
阶段 C：25/30 fps、VFR→CFR、旋转、SAR、SDR 颜色转换
阶段 D：source drift override、overwrite、backup/rollback、依赖哈希锁
阶段 E：可选 LLM parser
阶段 F：Embedding 和人工审查工作台
阶段 G：配音、字幕、音乐
阶段 H：Web、数据库和 Worker
```

第一阶段 Remix Planner / Renderer + retrieval baseline 与阶段 B 已完成并冻结。第二阶段 G0 契约已补齐，尚未选择视频生成模型、尚未调用任何生成 API；G0 通过后才允许规划 G1（Single-Shot Generation Pipeline）。G0 不接入 generate strategy、prompt compiler、模型 provider、云 API 或 API key 的正式集成。

后续不得破坏：

- timeline 人工可编辑；
- 独立离线 render；
- 规则模式；
- video-only 标准化片段；
- 最终一次连续音频；
- 帧数作为视频真值；
- 已有 MVR 测试。

---

## 24. 已验证的可行性事实

在参考环境 FFmpeg 7.1.3 中，已用合成素材验证：

```text
片段帧数：24 + 36 + 12
最终视频帧数：72
最终 AAC：48 kHz、双声道
音频时长：3.000 秒
音视频 start_time：0
静音 max_volume：-91.0 dB
```

参考环境（FFmpeg 7.1.3）中观察到的事实：

```text
合法不同长度片段的 avg_frame_rate 曾分别为：
576/23
864/35
288/11
```

该组数值仅是参考环境事实，不作为所有环境的验收门禁；其他环境可能记录为不同值（例如 FFmpeg 9.0 下三段均为 24/1）。验收只要求记录实际值，并证明 `avg_frame_rate` 不进入 concat signature。

因此：

- `avg_frame_rate` 不得进入 concat signature；
- `nb_read_frames` 是片段和最终视频长度真值；
- stream duration 不得替代帧数；
- `setparams` 必须显式写入颜色 primaries、transfer、space、range 和 progressive；
- `freeze_frame`（第 3A 批）的 `tpad` 必须位于目标 fps 之后（§12.4）。

Codex 仍必须在实际仓库环境重新执行第 0 批，不得只引用本节视为通过。

---

## 25. 最终约束

Codex 的首要任务不是建立完整平台，而是尽快交付一个真实运行、可观察、可修改、可重渲染的纵向闭环。

第 1 个技术成果：

```text
手写 timeline
→ 独立 render
→ 真实 MP4
```

第 1 个产品成果：

```text
用户提供剧本和人工标注素材
→ 系统自动匹配或占位
→ 生成可编辑 timeline.json
→ 生成可播放 output.mp4
→ 修改 timeline 后可再次渲染
```

在第 2 批真实成功之前，不得以以下工作替代核心成果：

```text
复杂评分
性能压测
确定性哈希体系
发布回滚
任意媒体兼容
字幕字体
LLM
Web
Agent 框架
```
