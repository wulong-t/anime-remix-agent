# G1-MK1-R-PREP-L：Remote 单样本支持工具

本目录新增两个薄工具，为已验收的 `G1-MK1-L` harness 提供 Remote 单样本执行
与本地 QA 证据生成能力。它们**不修改** `manual_keyframe_mvp.py`、现有测试、
FFmpeg/产品代码、AniSora 脚本、依赖、schema 或任何已冻结语义。

边界：本批次工具与测试不读取仓库真实媒体，不调用模型，不访问网络，不输出
密钥或环境变量值；所有路径来自显式配置，不做目录递归发现。测试只在 pytest
临时目录生成合成 PNG/MP4。

## 1. `remote_sample.py run`

```bash
python experiments/manual_keyframe_mvp/remote_sample.py run \
  --package <exact-package-dir> \
  --runner-config <runner-config.json> \
  --output <new-output-dir>
```

### 1.1 执行前校验（复用 G1-MK1-L harness）

运行前完整复验：package manifest、九个 package 成员及 hash/size 绑定、
request、inspection、approval、sampling contract、guide hash、
`first_formal_gate.active=true` 与 `anisora_input.txt` 绑定。任何不符直接
失败，不调用 runner。

### 1.2 Runner 配置 `g1-mk1-runner-config-v1`

顶层必须是对象，schema 固定为 `g1-mk1-runner-config-v1`，未知字段拒绝：

```json
{
  "schema_version": "g1-mk1-runner-config-v1",
  "python_executable": "<absolute path or bare command>",
  "anisora_workdir": "<absolute directory>",
  "bf16_runner_script": "<absolute path>",
  "checkpoint_dir": "<absolute directory>",
  "checkpoint_files": [
    {"relative_path": "models/.../file.safetensors", "size_bytes": 123456}
  ],
  "ffmpeg": "<absolute path or bare command>",
  "ffprobe": "<absolute path or bare command>",
  "nvidia_smi": "<absolute path or bare command>",
  "cgroup_memory_current": "<absolute path>",
  "cgroup_memory_peak": "<absolute path>",
  "cgroup_memory_events": "<absolute path>"
}
```

约束：

- `checkpoint_files` 必须为非空列表；`relative_path` 是相对
  `checkpoint_dir` 的 canonical 正斜杠相对路径（拒绝绝对路径、反斜杠、
  `.`/`..`、设备名、symlink/reparse 组件）；`size_bytes` 必须为正整数。
- 目录与 cgroup/runner 脚本路径必须是绝对路径；可执行文件可以是绝对路径
  或裸命令名（按 PATH 解析）。
- 所有路径均为配置数据，工具从不递归枚举。

### 1.3 冻结命令

严格构造以下 argv（只替换配置/输出路径，其余逐字冻结）：

```text
<python> <bf16-runner> --task i2v-14B --size 1280*720
--ckpt_dir <checkpoint-dir> --image <new-runner-output-dir>
--prompt <runtime-input-file> --base_seed 4096 --frame_num 81
--sample_steps 40 --sample_shift 5 --sample_guide_scale 5
--offload_model True
```

运行时输入文件（`runner-output/anisora_input.txt`）固定为：

```text
<package sampling_contract.resolved_prompt>@@<abs-k0>,<abs-k_end>&&0,1
```

只使用 package 绑定的 prompt 与 package 内两幅 guide 的绝对路径指针；
prompt 文本、seed、模型参数与 guide 字节不漂移。已发布的证据目录中保留
`runtime_input.txt` 副本；该文件含绝对 guide 路径（按契约的运行时指针），
但**不包含**任何环境变量值。

### 1.4 Preflight（失败则绝不调用 runner）

- 精确校验配置的 python/runner 脚本/checkpoint 目录/checkpoint 文件大小/
  ffmpeg/ffprobe/nvidia-smi/cgroup 三个文件与全新输出目录；
- Python/CUDA 可见性 probe、`ffmpeg -version`、`ffprobe -version`、
  `nvidia-smi` GPU probe、cgroup memory.current/peak/events 读取；
- 任一失败即 `remote_environment`，发布失败证据并停止。

### 1.5 单次采样与成功门禁

- 最多调用 runner 一次，无内部重试；
- 捕获精确 argv/cwd、合并 runner.log、退出码、运行时长、GPU 采样 CSV、
  cgroup memory 采样 CSV 与 memory.events；
- 有效样本定义：恰好一个 runner 产出的 `0.mp4`，完整可解码，H.264、
  video-only、`1280x704`、`16/1` CFR、恰好 81 个可计数帧；
- 成功后原子发布：`preflight.json`、`result.json`、`raw_shot.mp4`（含
  SHA256）、严格 `g1-mk1-sampling-receipt-v1`、`valid_sample_complete.json`
  （`runner_invocations=1`）、`runtime_input.txt`、`runner.log`、GPU/memory
  采样与 `memory_events.txt`，然后停止；
- 没有有效 raw 绝不发布 success marker 或 sampling receipt；已存在的
  输出目录被拒绝，第二个样本无法在同一输出启动。

失败（preflight 或 sampling technical）同样原子发布完整失败证据：
`preflight.json`（`passed=false`）、`result.json`
（`status=preflight_failed|sampling_technical`）、`runner.log`（若已调用）、
采样/事件文件，无 success marker、无 receipt、无 raw。

### 1.6 修订摘要（R1 Rework）

- R1：Python/CUDA/GPU 为硬门禁。`torch_available` 必须恰为 `true`、
  `cuda_available` 必须恰为 `true`、`device_count` 必须为 >=1 的整数
  （`bool` 不算整数）、`cuda_version` 必须为非空字符串、`nvidia-smi` 必须
  返回至少一行非空 GPU 行；任一不满足即 `remote_environment` 失败且 runner
  零次调用。
- R4：checkpoint 根目录与每个相对路径组件拒绝 symlink/reparse，解析后
  必须仍在解析后的 checkpoint 根内、为普通文件且大小精确匹配。
- R5：runner 返回码必须为 0（非零即使存在 `0.mp4` 也失败）；runner-output
  立即目录必须恰好一个名为 `0.mp4` 的普通非链接 MP4（额外 MP4 即失败）；
  两种情况都不发布 receipt/success marker。
- R6：运行时输入字节先捕获并记录 SHA256，调用后复验文件仍为普通非链接
  文件且字节完全一致；发布的 `runtime_input.txt` 只来自捕获字节，变异即
  失败且无 receipt/marker。
- R8：失败路径的资源证据为 best-effort：保留原始失败，不可用资源记录为
  显式 bounded error，仍原子发布允许的失败证据。

## 2. `qa_evidence.py`

```bash
python experiments/manual_keyframe_mvp/qa_evidence.py \
  --package <exact-package-dir> \
  --raw <exact-raw-shot.mp4> \
  --finalized-run <exact-finalized-run-dir> \
  --output <new-output-dir>
```

### 2.1 复验

- 完整复验 package（同 `remote_sample`）；
- raw 文件、`remote_receipt.json`、`generation_manifest.json`、
  `raw_shot.mp4`、`generated_clip.mp4` 的精确 SHA 绑定；不做目录发现；
- raw 严格满足冻结 raw 契约并完整可解码；`generated_clip.mp4` 为
  video-only H.264 `1280x720` 121 帧。

### 2.2 端点指标（证据，不判 PASS/FAIL）

- 将 `K0`/`K_end` 用与 finalized 媒体相同的确定性
  `scale(decrease) -> pad(1280x720, black)` 画布转换；
- 与 `generated_clip.mp4` 的 frame `0` / `120` 在 RGB 字节域比较；
- 记录 MAE、MSE 与标准 PSNR（`MSE=0` 时以字符串 `"infinite"` 表示，
  JSON 安全）；记录尺寸、帧索引与 artifact hashes；
- 每个端点参考画布（K0 / K_end / raw-start / raw-end）均通过 FFmpeg 执行
  与 finalized 媒体相同的冻结变换（R1 修订；不再使用自定义
  nearest-neighbor 路径）：
  `scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2`
  → `pad=1280:720:(ow-iw)/2:(oh-ih)/2:black` → `setsar=1` →
  `format=yuv420p` →
  `setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog`，
  再经 FFmpeg 以 RGB24 提取；参考图与 `generated_clip.mp4` 解码帧处于同一
  字节域转换路径；
- 指标仅为证据，不自动给出视觉结论。

### 2.3 图表与发布

- `raw_contact_sheet.png`：raw frame `0,10,...,80`，固定 3×3 顺序；
- `endpoint_comparison.png`：固定顺序 `K0, K_end, raw-start, raw-end`
  （2×2）；
- 原子发布 `metrics.json`、两张 PNG、`artifacts.json`（含 hash、尺寸、
  帧/顺序）；`metrics.json` 中 `capability_verdict` 固定为 `null`；
- 无网络、无新依赖、无 capability verdict。

### 2.4 修订摘要（R1 Rework）

- R3：绑定校验后只捕获一次 K0/K_end/raw/generated 字节并写入 probe
  staging，所有 probe/完整解码/抽帧/指标/contact sheet/报告 hash 均只使用
  捕获字节；校验后绝不重开 live 路径。
- R7：`generated_clip.mp4` 必须满足完整 finalized 契约（单个 H.264 video
  流、零 audio、1280x720、恰好 121 个计数帧、24/1 CFR、yuv420p、SAR 1:1、
  BT.709 limited、progressive、chroma-left）并完整解码校验，复用
  `FFmpegToolkit.validate_segment`。

## 3. 测试

```bash
python -m pytest tests/tools/test_manual_keyframe_remote_support.py -q
python -m pytest tests/tools/test_manual_keyframe_mvp.py \
  tests/tools/test_manual_keyframe_remote_support.py -q
python -m ruff check experiments/manual_keyframe_mvp/remote_sample.py \
  experiments/manual_keyframe_mvp/qa_evidence.py \
  tests/fixtures/fake_anisora_runner.py \
  tests/tools/test_manual_keyframe_remote_support.py
```

测试使用 `tests/fixtures/fake_anisora_runner.py`（按 `ANISORA_FAKE_MODE`
合成有效/无效 MP4）与真实 ffmpeg，仅在 pytest 临时目录生成合成媒体。

## 4. 数据出站边界

允许：本任务文本、两份 G1-MK1 契约/计划、`AGENTS.md`、
`WORKER_EXECUTION_CONTRACT.md`、上述精确源码/测试/文档、路径受限的 Git
metadata 与 diff、本批次六个写路径产出的代码/测试/文档/receipt/输出。

禁止：密钥/token/SSH/凭据/环境变量值/用户或全局 Git 配置；所有真实或合成
图片/视频/音频 bytes、base64、缩略图、像素派生数据、模型权重、checkpoint
与正式 package 媒体；未列出的源码、测试、实验产物、目录 metadata 或文件
清单。
