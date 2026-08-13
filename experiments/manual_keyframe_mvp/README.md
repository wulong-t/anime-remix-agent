# G1-MK1-L：Manual Keyframe MVP（Local Deterministic Harness）

本目录实现 `G1-MK1-L`：一个人工关键帧单镜头实验的**本地确定性 harness**。
它只做输入结构/媒体技术检查、精确哈希绑定、两阶段人工批准、Remote 打包、
raw 121 帧标准化、one-item Timeline 1.9 与现有 renderer 闭环。

边界：本次 Worker 的实现与测试不读取仓库中的真实 PNG/MP4/音频，不调用任何
模型，不访问网络，不向任何外部目的地发送数据；测试只在 pytest 临时目录内
生成合成媒体。用户运行 `inspect` / `package` / `finalize` 时，harness 只读取
用户显式传入并授权的文件；它不会自动发现、联网发送或读取未指定媒体。
工具不读取凭据库或环境变量。

## 1. 输入格式

输入根目录必须包含（路径相对 request.json 所在目录）：

```text
request.json
inputs/k0.png
inputs/k_end.png
inputs/k0.provenance.json
inputs/k_end.provenance.json
approval.json            # inspect 生成 pending 模板，由用户填写
```

`request.json` 使用实验 schema `g1-mk1-request-v1`：

```json
{
  "schema_version": "g1-mk1-request-v1",
  "request_id": "g1mk1-head-turn",
  "start_keyframe": "inputs/k0.png",
  "end_keyframe": "inputs/k_end.png",
  "start_provenance": "inputs/k0.provenance.json",
  "end_provenance": "inputs/k_end.provenance.json",
  "start_sha256": "64 lowercase hex",
  "end_sha256": "64 lowercase hex",
  "subject_description": "non-empty text",
  "scene_description": "non-empty text",
  "action": "non-empty simple action",
  "start_state": "non-empty text",
  "end_state": "non-empty text",
  "emotion": "calm",
  "shot_scale": "medium",
  "camera": "fixed",
  "duration_seconds": 5,
  "aspect_ratio": "16:9"
}
```

约束（由 harness 强制，失败即 `input_contract`）：

- 顶层必须是对象，未知字段拒绝；
- `request_id` 匹配 `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`；
- 四个路径必须是相对路径：拒绝 URL、绝对路径、盘符、UNC、设备名、`..`、
  symlink 和根目录逃逸；
- 文本字段去除首尾空白后非空，拒绝换行/控制字符和 AniSora 分隔符 `@@` / `&&`；
- `camera` 严格为 `fixed`，`duration_seconds` 严格为 `5`，
  `aspect_ratio` 严格为 `16:9`；
- `emotion` / `shot_scale` 只允许现有封闭枚举值或 `null`；在枚举检查前先做
  string/null 类型门禁，list/object/number 一律按 `input_contract` 拒绝；
- 两个声明 SHA256 必须与实际文件一致，且 `start_sha256 != end_sha256`。

图片技术合同：

- 两张输入均为单张可解码 PNG、普通文件、每张不超过 25 MiB；
- `K0` 与 `K_end` 像素宽高必须完全一致，范围 `512..4096`；
- 画布与 `16:9` 的相对误差不超过 0.5%；
- 只允许 RGB 或 RGBA；拒绝动画 PNG（acTL）、交错 PNG、未知关键块、
  损坏 CRC/IDAT 和 IEND 之后的多余数据；
- IDAT 只在 IHDR 画布（`512..4096`）与格式门禁通过后解压，且使用有上限的
  增量解压（最多产生 IHDR 推导 raw 大小 + 1 字节）；zlib 流必须正常 EOF、
  无 trailing/unused 数据且解压字节数恰好等于 IHDR 推导值，超出即失败；
- harness 不得静默 crop/resize/pad/调色/修图；不合约直接失败。

首个正式门禁是 run-scoped：`inspection.first_formal_gate.active` 对任何
request 都为 `true`，只要其精确的 `start_keyframe` / `end_keyframe` 路径与
声明 SHA256 匹配被检字节，且两张图片满足上述图片技术合同。sampling
contract 与 package 会携带并绑定该状态，后续 Remote task 仍须单独要求
`active=true`。不再有硬编码的冻结 K0/K_end 媒体身份或禁止 SHA；每次 run 的
路径/哈希绑定、TOCTOU 拒绝、人工批准、固定采样参数与图片合同保持原样。

`provenance` JSON 使用精确顶层 key 集：`asset`、`sha256`、`creation_method`、
`external_inputs`、`named_references`、`rights_basis`、`public_demo_allowed`、
`notes`；未知字段拒绝。`named_references` 必须是精确对象
`artists / studios / series / characters`（均为单行字符串数组）。
`asset` 必须是安全的 canonical 相对路径，且等于该 provenance 对应的
request 关键帧路径；`sha256` 必须与对应关键帧严格一致。
`creation_method` / `rights_basis` 为去首尾空白后非空的单行字符串；
`notes` 为单行字符串（可为空）；数组元素必须为单行字符串、无控制字符。

## 2. 两阶段人工批准

`inspect` 只生成 `pending` 模板，绝不代替用户填写权利声明或视觉评审：

1. 运行 `inspect`，得到 `inspection.json` 与 `approval.json`（pending）；
2. 用户人工检查 `K0` / `K_end`，填写：
   - `rights.*`（必须为 `true` 的权利项、`public_demo_allowed` 布尔值）；
   - `visual_review.*`（`pass | borderline | fail`）；
   - `accept_borderline`（任一维度为 borderline 时必须为 `true`）；
   - `overall = "approved"`、带时区的 RFC 3339 `approved_at`。

`package` 会重新校验 request、provenance、图片、inspection、approval 与精确
哈希；任一维度为 `fail`、borderline 未接受、哈希漂移或文件在批准后变化，
一律拒绝且不发布。

## 2.1 单一不可变 bytes snapshot

`inspect`、`package` 与 `finalize` 都遵循同一证据模型：每个输入/成员文件
的内容**只捕获一次**；所有结构、路径、SHA256、PNG、provenance、inspection、
approval、manifest、sampling contract、receipt 与 raw 校验只消费该次捕获的
bytes；staging/run 只写捕获的 bytes，绝不重新读取 source 路径。

- request 先捕获并解析出安全路径，再捕获被引用文件；guide 声明 hash 与捕获
  的 guide bytes 严格比较；
- 源文件在捕获后变化时，发布物只能使用捕获的原 bytes；源文件在 request
  捕获后、guide 捕获前变化时，guide hash 校验必须失败；
- `package` 的 manifest 头（request/start/end/inspection/approval/sampling
  hashes 与 request_id）与九个成员快照交叉绑定，任何 member 在 manifest
  验证后、首次捕获前被替换都会因 hash/size mismatch 失败；
- `finalize` 对 package root 在 `resolve()` 前拒绝 symlink/junction/reparse
  point，member 组件继续执行相同的 link/reparse 检查；
- 布尔字段使用真正 bool 门禁，JSON `0/1` 不得冒充布尔值；frozen
  sampling/receipt 的 int/float/bool/list 做递归 strict type/value 比较，
  `true == 1`、`5 == 5.0` 之类的 Python 相等语义不能绕过。

## 3. 命令

```bash
python experiments/manual_keyframe_mvp/manual_keyframe_mvp.py \
  inspect --request <request.json> --output <new-inspection-dir>

python experiments/manual_keyframe_mvp/manual_keyframe_mvp.py \
  package --request <request.json> --inspection <inspection.json> \
  --approval <approval.json> --output <new-package-dir>

python experiments/manual_keyframe_mvp/manual_keyframe_mvp.py \
  finalize --package <package-dir> --raw <raw_shot.mp4> \
  --remote-receipt <sampling_receipt.json> --output <new-run-dir>
```

所有输出目录都必须是新目录；已存在一律拒绝，不实现 overwrite。
`inspect` / `package` / `finalize` 均采用同父目录 staging + 原子 rename，
任一写入或发布失败都不会留下 target，也不留 staging 残留。

## 4. 目录结构

### inspection 目录

```text
inspection.json     # 结构/媒体技术检查、哈希、first-gate 状态
approval.json       # pending 模板，用户填写
```

### package 目录（Remote 输入）

```text
package_manifest.json
sampling_contract.json
anisora_input.txt               # prompt@@inputs/k0.png,inputs/k_end.png&&0,1
request.json
inspection.json
approval.json
inputs/k0.png
inputs/k_end.png
inputs/k0.provenance.json
inputs/k_end.provenance.json
```

所有路径为相对路径，不含密钥、环境变量或绝对路径。
`package_manifest.files` 必须**恰好**包含九个成员，不多不少：
`request.json`、`inputs/k0.png`、`inputs/k_end.png`、
`inputs/k0.provenance.json`、`inputs/k_end.provenance.json`、
`inspection.json`、`approval.json`、`sampling_contract.json`、
`anisora_input.txt`；每个 key 必须是 canonical POSIX 相对路径（拒绝
URL、绝对路径、盘符、UNC、设备名、反斜线、`.`/`..`、symlink/reparse-point
逃逸），成员必须是普通文件且 hash/size 与 manifest 严格一致；
`package_manifest.json` 不得自引用。manifest 顶层使用 exact key 集
（`schema_version`、`request_id`、`request_sha256`、`start_sha256`、
`end_sha256`、`inspection_sha256`、`approval_sha256`、
`sampling_contract_sha256`、`created_at`、`inputs`、`files`），unknown/missing
拒绝；`created_at` 必须为带时区的 RFC 3339；`inputs` 严格为
`start_keyframe/end_keyframe` 两个 key；每个 `files` record 严格为
`{sha256, size_bytes}`，size 为非 bool 非负 int。

### run 目录（Local finalize 输出）

```text
.anime-remix-run
generation_manifest.json
request.json / approval.json / package_manifest.json / remote_receipt.json
raw_shot.mp4
generated_clip.mp4
timeline.json
clips.json
render.log
output.mp4
normalized/                         # 现有 renderer 的中间产物
```

## 5. 冻结的 sampling / normalization 契约

Remote sampling 契约固定在 `sampling_contract.json`：

```text
provider  = IndexTeam / official Index-AniSora
model     = AniSora V3.1 (Wan 14B)
task      = i2v-14B
dtype     = bfloat16 runtime shim
size      = 1280*720；观测 raw canvas = 1280x704
guides    = 0, 1（K0 -> K_end）
raw       = 81 帧 @ 16fps
seed      = 4096；steps = 40；shift = 5；guide_scale = 5
offload   = true；aesthetic = 5.5；motion = 3.0
valid content samples = 恰好一个
```

`finalize` 严格校验 remote receipt 与上述冻结参数一致，并且：

- raw 文件 SHA256 与 receipt 严格一致；
- raw 恰好一个 H.264 视频流、无音频、`1280x704`、`16/1` CFR、可计数 81 帧；
  创建 run staging 前还会执行完整视频解码检查，任何解码错误即
  `media_normalization` 失败；
- 标准化使用冻结过滤顺序
  `scale(decrease) -> pad(1280x720,black) -> setsar=1 -> format=yuv420p
   -> setparams(BT.709 limited/progressive) -> fps=24,start_time=0,round=near
   -> trim=end_frame=121 -> setpts=N/(24*TB)`；
- `generated_clip.mp4` 严格为 H.264 high/level 3.1、1280x720、24/1 CFR、
  121 帧、yuv420p、SAR 1:1、BT.709 limited、progressive、chroma left、
  video-only；帧数不允许 ±1；
- one-item Timeline 1.9：`strategy=clip`、`source_in_frame=0`、
  `source_frame_count=target_frames=121`、`reason_code=exact_length`；
- 发布严格现有 schema 的 `clips.json`（schema 1.9、恰好一个
  Generated ClipAsset，见第 10 节）；
- 调用现有 `render_timeline`，最终 `output.mp4` 必须为 121 帧 H.264 +
  121/24 秒的 48 kHz 双声道 AAC 静音（样本数 242000）。

`sampling_contract.json` 至少包含并绑定：`request_id`、`request_sha256`、
`start_sha256`、`end_sha256`、`guide_files`、`guide_sha256`、
`guide_positions`、`frozen_parameters`、`prompt_template`、
`resolved_prompt`、`input_line`、两条 retry rule 与 `first_formal_gate`
状态；schema 使用 exact key 集，unknown/missing 拒绝，每个值做 strict
JSON type/value 比较；`anisora_input.txt` 必须严格等于 `input_line + "\n"`。

`g1-mk1-sampling-receipt-v1` 使用固定 allowlist schema，未知字段拒绝，且严格
绑定：`request_id`、`request_sha256`、`package_manifest_sha256`、
`sampling_contract_sha256`、`start_sha256`、`end_sha256`、`raw_sha256`、
`status=success` 与全部冻结 sampling 参数。Remote 治理 Worker Receipt/日志是
另一份证据，不合并进最小 sampling receipt。`finalize` 先证明 receipt 的
`package_manifest_sha256` 等于实际 packaged manifest，再接受 raw。

最终 AAC 样本数来自真实测量：probe 必须记录 audio stream 的 `time_base` 与
`duration_ts`，只有严格为 `1/48000` 与 `242000` 时才写入
`generation_manifest.total_audio_samples`；不先注入常量再声称 probe。

## 6. 边界与禁止项

- 本次 Worker 实现/测试不读取仓库真实媒体；用户运行时会读取其显式传入并
  授权的文件。工具不会自动发现、联网发送或读取未指定媒体；
- 不修改 Timeline schema/enum、planner、retriever、parser、scoring 或现有
  renderer 语义；不新增产品 CLI、provider abstraction、Web/DB/依赖；
- 不 merge/push/rebase/reset/force；不自动批准、不自动生成 `K_end`、
  不调用模型、不访问 Remote；
- 不把密钥、token、SSH、环境变量值写入任何输出；调用者不得把密钥或敏感
  认证信息填入任何输入文件（工具不声称能从合法自由文本中识别任意藏匿的
  密钥，只要求调用者不放入）。

## 7. 错误分类与 stop rule

CLI 失败输出 `ERROR <layer>: <message>`，退出码 2，分类：

```text
input_contract         JSON/路径/哈希/PNG/画布不合约
rights_blocked         provenance 缺失或冲突
approval_blocked       人工批准缺失、拒绝、borderline 未接受或哈希漂移
media_normalization    raw probe 或 121 帧标准化失败
renderer_interface     Timeline 1.9 或现有 renderer/最终媒体门禁失败
evidence_incomplete    manifest/contract/receipt/文件/哈希证据不完整、
                       跨 package 绑定失败、package root link/reparse、
                       或任何验证后替换（TOCTOU）
```

Stop rule：

- 没有用户批准的 `K_end`、精确哈希批准或媒体出站授权前，不得启动 Remote；
- Local harness 未通过则 `REWORK`，不启动 Remote；
- 获得一个有效内容样本后立即停止；内容失败不得换 prompt/seed/参数/关键帧；
- 技术失败只能在第一个有效样本出现前按冻结参数修复重试；
- 不自动进入产品化、`G1-MK1-R` 或任何后续批次。

## 8. 测试

```bash
python -m pytest tests/tools/test_manual_keyframe_mvp.py -q
python -m pytest tests/integration/test_generated_source_normalization.py -q
python -m ruff check src tests experiments/manual_keyframe_mvp
```

## 9. G1-MK2-L 产品化更新（不改变 G1-MK1 已验收语义）

本批次把用户接受的 G1-MK1-R 实验中的两条运行级经验产品化，不新增
generate strategy，不接触 Remote：

### 9.1 `sample_steps` 作为 package 级采样参数

`package` 新增可选 `--sample-steps INTEGER`，默认 `40`；只接受真实整数
`1..100`，bool/零/负数/>100 在验证层拒绝。`sampling_contract.json` 仍为
schema `g1-mk1-sampling-contract-v1`，其
`frozen_parameters.sample_steps` 记录所选值，并继续由 package manifest SHA
绑定。

```bash
python experiments/manual_keyframe_mvp/manual_keyframe_mvp.py \
  package --request <request.json> --inspection <inspection.json> \
  --approval <approval.json> --output <new-package-dir> --sample-steps 10
```

`remote_sample`、receipt 验证、`finalize`、generation manifest 与 QA 均
读取/比较 packaged contract 中的 `sample_steps`；Remote argv 的
`--sample_steps` 只来自已验证 contract，不再硬编码 `40`。steps-10 的合法
package + receipt 可用现有工具直接 finalize 并产出 QA，无需 run-local
shim。

### 9.2 持久化 raw recovery 旁路产物

对输出路径 `<parent>/<name>`，确定性旁路为
`<parent>/<name>.raw-recovery.mp4` 与 `<parent>/<name>.raw-recovery.json`：

- 调用 runner 前若 output 或任一 recovery 路径已存在即拒绝，绝不覆盖；
- runner 返回后、任何 exit-code/probe/decode 等后续验证可失败之前，若
  立即输出目录包含恰好一个名为 `0.mp4` 的普通非链接文件，则原子复制其
  bytes 到 recovery MP4，并写入含 SHA256、size、request/package 绑定、
  `sample_steps` 与 `validation_status: unverified` 的 manifest；
- recovery 旁路永不删除；正式 raw 验证全部通过后，manifest 原子更新为
  `validation_status: valid`；
- 成功仍发布正式 `raw_shot.mp4`、receipt 与 valid marker，且 recovery 与
  正式 raw 的 SHA256 必须一致；
- preflight/runner 失败且从未产生精确普通 `0.mp4` 时不创建 recovery。

### 9.3 状态记录（如实保留历史）

G1-MK1-R 单样本实验为 Technical PASS，用户已明确接受播放结果作为成功；
Chief 的 contact-sheet 评估更保守，此判断保留不抹除。G1-MK2-L 只做本地
产品化，不自动进入下一阶段。

## 10. G1-MK3-L Generated ClipAsset handoff（不改变已验收语义）

本批次补齐最小的本地产品边界：成功 `finalize` 必须原子发布严格现有
schema 的 `clips.json`，QA 验证其与 timeline、generation manifest 和实际
normalized 媒体的绑定。仍是本地 only，不接触 Remote、不采样、不新增
generate strategy。

`clips.json` 契约：

- `schema_version` 严格为 `"1.9"`，恰好一个 `ClipAsset`；
- asset `id` 直接使用已验证的 `request_id`（request 正则与 domain
  `ID_PATTERN` 完全一致，确定性且合法），且必须等于
  `timeline.items[0].source_asset_id`；
- `path` 固定为 `generated_clip.mp4`；
- `characters` 为 `[]`，`location_id` / `location_name` 为 `null`；
- `action`、`emotion`、`shot_scale` 只来自 request；description 是
  subject / scene / `start_state -> end_state` 的确定性拼接
  （`generated_clip_description`），不推断身份或地点元数据；
- timeline 继续 `strategy=clip`、schema 1.9、121 帧、`source_path` 为
  `generated_clip.mp4`，仍使用现有 renderer。

`generation_manifest.json`：schema 标识符保持
`g1-mk1-generation-manifest-v1`（明确 additive 决策——只新增 `clips`
绑定记录 `{path, sha256}`；QA 同时校验 `timeline` 绑定哈希，不弱化任何
既有校验）。

QA（`qa_evidence`）按固定清单捕获精确的 `clips.json` 与 `timeline.json`
字节，不做目录发现，并验证：

- `clips.json` 能作为严格 `ClipsDocument` 解析（schema 1.9、单素材、
  id/path/characters/location/action/emotion/shot_scale/description
  形状与 request 映射）；
- generation manifest 中 `clips` / `timeline` 的 path + SHA256 与实际
  捕获字节严格一致；
- timeline `source_asset_id` / `source_path` 与 clips asset 完全匹配，
  `source_sha256` / `source_size_bytes` 与捕获的 `generated_clip.mp4`
  字节直接一致；
- normalized 媒体路径 / 哈希关系（`generated_clip.mp4` 与 manifest
  `normalized` 绑定）一致。

失败保持原子：任何校验或写入失败都不会发布部分 run 或部分 clips
文档；默认 `sample_steps=40` 与 steps-10 路径保持有效。

## 11. G1-MK4-L session_agent（薄状态协调器，不接触 Remote）

`session_agent.py` 是一个实验/学习入口，不加入产品 CLI 或 pyproject。它
只做最薄的阶段协调：复用已 PASS 的 `manual_keyframe_mvp`
（inspect/package/finalize）与 `qa_evidence`（QA），管理一次未来手动镜头
会话：

```text
init -> awaiting_approval -> awaiting_remote -> finalize+QA -> complete
```

```bash
python experiments/manual_keyframe_mvp/session_agent.py \
  init --request <exact request.json> --workspace <new dir> \
       [--sample-steps 1..100]            # 默认 40
python experiments/manual_keyframe_mvp/session_agent.py \
  status --workspace <exact dir>
python experiments/manual_keyframe_mvp/session_agent.py \
  advance --workspace <exact dir> \
          [--remote-output <exact successful remote dir>]
```

会话只保存最小真值（`<workspace>/session.json`，exact-key schema
`g1-mk4-manual-shot-session-v1`）：

```json
{
  "schema_version": "g1-mk4-manual-shot-session-v1",
  "request": {"path": "<canonical path>", "id": "<request_id>", "sha256": "..."},
  "sample_steps": 40,
  "phase": "awaiting_approval | awaiting_remote | complete",
  "package_manifest_sha256": null | "64 lowercase hex",
  "completion": null | {
    "generation_manifest_sha256": "...",
    "qa_metrics_sha256": "...",
    "qa_artifacts_sha256": "..."
  }
}
```

不保存时间戳、`next_action` 副本、remote-output 记录、artifact 图或
描述；单一机器可读 `next_action` 由 phase 在 status 响应中派生。status
校验 exact request、存在时的 exact package manifest，以及 completion
记录的 generation-manifest/QA 哈希；不做目录枚举。schema/path/hash
drift 一律拒绝，不猜测、不修复。

可恢复性：

- `init` 在目标 workspace 的同级临时目录运行 `cmd_inspect` 并写入
  session，再整体 rename 到新 workspace；任何失败只清理该已验证临时
  目录，目标 workspace 保持不存在；
- `awaiting_approval` 的 `advance` 在 workspace 内临时目录调用现有
  `cmd_package`，经 `remote_sample.validate_package` 验证后再发布
  `package/` 并更新 session；session 写入失败会回滚本次新建的
  `package/`，保持旧 phase、无最终 `package/`；
- `awaiting_remote` 带 `--remote-output` 时是一次复合过渡：先对精确临时
  输出目录运行现有 `cmd_finalize` 与 `cmd_qa`，两者都成功后才发布
  `finalized/` 与 `qa/` 并原子转入 `complete`；finalize/QA/rename/session
  任一失败只清理本次新建的临时/最终路径，保持 `awaiting_remote`，同一
  remote 输出可原样重试；
- `complete` 的 `advance` 是幂等只读 status，不再调用 finalize/QA。

本工具不调用 SSH/SCP/Remote/AniSora/FFmpeg 或 shell；媒体工作只经由
既有 PASS 函数，测试全部为合成数据。

## 12. G1-MK5-L generation_queue（placeholder → 手动生成队列）

`generation_queue.py` 是最小的确定性 Planner→generation handoff：读取一个明确
传入的现有 `timeline.json`，一次性捕获字节并用现有 `Timeline` Pydantic 模型严格
校验同一捕获对象，然后原子发布唯一的 `generation_queue.json`（schema
`g1-mk5-generation-queue-v1`），只列出 strategy 为 `placeholder` 的镜头，供既有
手动关键帧会话开始前由人类补齐 K0/K_end 输入。不修改产品默认与 Timeline，不读源
媒体、不渲染、不启动生成会话、不接触 Remote。

```bash
python experiments/manual_keyframe_mvp/generation_queue.py plan \
  --timeline <exact timeline.json> --output <new directory>
```

队列顶层 exact keys：

```text
schema_version = g1-mk5-generation-queue-v1
timeline_sha256              # 捕获的 timeline 字节 SHA256
timeline_schema_version = 1.9
total_timeline_items
pending_count
items
next_action
```

每个 item 只使用既有 Timeline/requirement 事实（shot_id / order / target_frames /
source_text / action / emotion / shot_scale / characters / location_id /
location_name），`status = needs_manual_keyframes`，并固定携带
`required_human_inputs`：

```text
subject_description
scene_description
start_state
end_state
k0_png
k_end_png
k0_provenance
k_end_provenance
rights_and_visual_approval
```

`next_action` 是派生对象：有 pending 时为 `provide_manual_keyframes` +
第一个 queue shot_id，否则为 `none` + null；不含描述、时间戳、命令、媒体路径或
重复审计图。输出走同父 staging + 原子 rename；拒绝已存在输出、symlink/reparse、
无效 timeline，捕获/校验边界内单次读取；失败只清理协调者自有 staging。

边界：本地 text-only，不改 src/parser/retriever/timeline schema/enum/renderer/
placeholder 行为/session agent 或既有 PASS 工具；不新增依赖与抽象；不 push/merge。
测试见 `tests/tools/test_manual_keyframe_generation_queue.py`。

## 13. G1-MK6-L generation_bridge（queue → session → 解析 Timeline 闭环）

`generation_bridge.py` 关闭本地可选闭环：一个 queue 中的 placeholder 镜头 →
人工填写的 manual request → 既有 `session_agent` 会话 → 完成的 generated clip
→ 原 Timeline 的严格解析副本，供既有 renderer 直接渲染。`start` 校验精确
timeline/queue 绑定并选择唯一 queue item，要求 `request_id == shot_id` 且
action/emotion/shot_scale 与 queue item 完全一致，然后在 timeline 目录下新建
直接子目录 workspace 并调用 `session_agent.cmd_init`（输出 phase 为
`awaiting_approval`）；`resolve` 要求会话 phase `complete`，只读取固定
`finalized/` 路径，把 `generated_clip.mp4` 字节与
`generation_manifest.json.normalized` 及严格单素材 `clips.json` 绑定，再将
原 placeholder item 按 121 帧源替换为：

```text
target_frames == 121 -> clip in=0 count=121 reason_code=exact_length
target_frames <  121 -> clip count=target_frames in=(121-target_frames)//2
                        reason_code=center_trim
target_frames >  121 -> freeze_frame in=0 count=121
                        reason_code=short_source_freeze
```

只替换目标 item（`source_asset_id=shot_id`、POSIX 相对
`<workspace>/finalized/generated_clip.mp4`、字节 SHA/size、score 保持 null），
保留其余 item、render profile 与顺序；严格校验新 Timeline 后原子发布到原
timeline 目录内的新 JSON，拒绝已存在/目录外/覆盖原 timeline/symlink 输出与任何
绑定漂移。不运行媒体工具、不自动批准、不改 queue、不重试。

```bash
python experiments/manual_keyframe_mvp/generation_bridge.py start \
  --timeline <timeline.json> --queue <generation_queue.json> \
  --shot-id <shot> --request <request.json> --workspace <new child dir> \
  [--sample-steps 1..100]

python experiments/manual_keyframe_mvp/generation_bridge.py resolve \
  --timeline <timeline.json> --queue <generation_queue.json> \
  --shot-id <shot> --workspace <completed session> --output <new timeline.json>
```

边界：本地 text-only，不改 src/parser/retriever/timeline schema/enum/renderer/
queue/session agent 或既有 PASS 工具；不新增 schema/enum/依赖与抽象；不
push/merge。测试见 `tests/tools/test_manual_keyframe_generation_bridge.py`。
