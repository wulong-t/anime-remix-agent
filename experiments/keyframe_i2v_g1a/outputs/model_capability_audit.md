# G1-A Model Capability Audit：官方 Multi-Keyframe Temporal Guidance

AniSora fixed commit: `6cdce3a17548d7ff0f2e05978469f134da25e68e`
Audit date: 2026-08-08
Scope: read-only; no model download, no GPU sampling.

## 结论（TL;DR）

```text
README 截图中的多关键帧能力  = AniSora V1.0（CogVideoX-5B，AniSora-K 关键帧插帧）
对应 checkpoint             = IndexTeam/Index-anisora 5B (SAT 格式, 31.34GB) + CogVideoX_VAE_T5 (9.97GB)
对应 inference script       = anisoraV1_infer/fastercache_sample_cogvideox_sp.py
是否真正支持首/中/末帧        = 代码中存在 latent-concat 实现（len=1/2/3 分支），
                              但当前 checkout 没有任何可运行的官方调用入口
Guide Frames 输入格式         = engine API 的 image_path 列表（设计上）；CLI prompt 文件路径不可用
是否需要 mask                = 否（latent 拼接，零填充插值；不是 anisora_anymask 同一条实现）
本机是否已有 checkpoint       = 否（HF cache 为空；models/ 仅有 anymask）
```

## 1. 能力归属

`README_CN.md` 项目指南明确：

```text
AniSora V1.0（anisoraV1_infer）
  - 基于 CogVideoX-5B 基础模型训练
  - 支持局部区域控制、时间控制（首帧/尾帧/关键帧插帧、多帧引导）
```

README “时间控制示例”表格（首帧/中间帧/末帧/视频）展示的正是该能力；
评测表另列出 `AniSora-K`（关键帧插帧）与 `AniSora-I`（帧插值平均）。
AniSora V2/V3/AnyMask 的 README 均未宣称多关键帧引导。

## 2. 官方实现位置与语义

文件：`anisoraV1_infer/fastercache_sample_cogvideox_sp.py`

核心逻辑位于 `sampling_main`（engine 分支，`extra_args` 传入）：

```text
len(image_path)==1: [first, zeros(T-1)]                    first at latent t=0
len(image_path)==2: [first, zeros(T-2), mid]               second at latent t=T-1 (首+末)
len(image_path)==3: [first, zeros((T-3)//2), mid,
                     zeros(T-3-((T-3)//2)), last]          first/mid/last at 0, (T-3)//2+1, T-1
```

默认 `sampling_num_frames=13`（latent）→ `mid` 位于 latent t=6（49 输出帧的中间）。
条件为 VAE latent 沿时间维拼接（`concat`），未知区间用 zeros 填充，
**不使用 mask**，与 `anisora_anymask` 的 spatial mask 路径不是同一条实现。

## 3. 为什么当前 checkout 不可直接运行

1. **CLI prompt 文件路径只能单图**：
   `text, image_path = text.split("@@")` 后执行 `image_path=[image_path]`，
   `len(image_path)` 恒为 1，多图分支（len==2/3）在 CLI 路径不可达。
2. **Engine API 损坏**：
   `__init__.py`（CVModel）构造 `VideoSysEngine(Args(num_gpus))`，
   而 `Args.__init__` 里 `self.pipeline_cls = child` 引用了模块级未定义变量 `child`，
   构造即 NameError。
3. **脚本 `__main__` 同样损坏**：
   `engine = VideoSysEngine(Args(num_gpus=8)); engine.generate(extra_args={})`
   —— 同一 `child` NameError，且未按 README 处理 `--base`。
4. `videosys/pipelines/cogvideox/pipeline_cogvideox.py` 是通用 diffusers
   CogVideoX pipeline（text2video/first-frame 生成，无多图参数），
   不是 AniSora V1 multi-keyframe 路径，且不加载 SAT 格式 5B checkpoint。

因此：README 展示的能力 = AniSora V1.0 关键帧插帧；
当前 checkout 中 **implementation exists but no runnable official entry point**。
按 G1-A 契约第 2 节：不自行猜接口、不修补官方代码、停止 GPU 实验。

## 4. 输入格式 / 时间位置（设计语义，仅代码审计）

```text
Guide Frames: image files（PNG/JPG），通过 engine extra_args["image_path"] 传 list
时间位置:     latent 索引；3 帧时 = 0 / (T-3)//2+1 / T-1
需要 mask:    否
```

## 5. 模型与硬件需求

```text
checkpoint:  IndexTeam/Index-anisora
             5B/1000/mp_rank_00_model_states.pt  = 31.34 GB
             CogVideoX_VAE_T5/                    =  9.97 GB
             （t5-v1_1-xxl_new 8.87GB + videokl_ch16_long_20w.pt 1.1GB）
合计下载:    ≈ 41.3 GB
本机磁盘:    /root/autodl-tmp 剩余 ≈ 65 GB（可容纳，但余量 23GB）
GPU:         NVIDIA vGPU-48GB（空闲）
cgroup RAM:  90 GiB
官方参考:    4090 级建议 640×1088（fastercache_sample_5b4090.yaml，latent 136×80）；
            1280×720 仅 A100 级推荐，README 标注 2×4090 解码 OOM
```

注：由于官方入口不可用，未下载任何 checkpoint；
VRAM/RAM 可行性是基于官方 README 的估算，未实测。
