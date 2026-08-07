# G0-L：最小视频分层实验（Character + Background）

## 目的

验证一条最小可行链路：

```
source.mp4
→ 单人物 Temporal Mask
→ Character Layer
→ 可用 Background Plate
→ 确定性重新合成
→ preview_composite.mp4
```

本实验**不接入产品**，不定义 ShotAsset / LayerAsset，不修改
`src/anime_remix/`、`tests/`、根 `pyproject.toml`、根 `uv.lock`。
全部代码与产物位于本目录。

## 环境

- GPU：NVIDIA vGPU 48GB（本实验峰值约 2.9GB）
- Python：3.12（复用宿主 torch 2.8.0+cu128）
- 包管理：uv，依赖锁定在 `pyproject.toml` + `uv.lock`
- 不污染根项目环境：`.venv/` 在本目录内

首次安装：

```bash
uv venv --system-site-packages .venv --python python3
uv sync --no-install-package torch --no-install-package torchvision
```

运行：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  .venv/bin/python run_experiment.py
```

## 输入

`input/source.mp4`

| 属性 | 值 |
|---|---|
| 分辨率 | 1918 × 1078 |
| 编码 | h264 |
| 名义 fps | 30 |
| 实际平均 fps | 30.17 |
| 帧数 | 112 |
| 时长 | 3.712 s |

## 方法

### 1. Temporal Mask

- 模型：SAM2.1 Hiera Base+（`facebook/sam2.1-hiera-base-plus`）
- 第 0 帧用 SAM2 Automatic Mask Generator 选择居中的单人物 mask，
  再通过 SAM2 Video Predictor 向整段视频传播。
- 后处理：9×9 闭运算、5×5 开运算、保留人物 bbox 内的主要连通域、
  3 帧时间中值平滑。
- 底部黄色前景按 G0-L 规则归入 non-character。

结果：112 帧全部有 mask，无空帧；
mask 面积比例 mean/min/max = 0.2452 / 0.2444 / 0.2515；
人物 bbox 基本稳定（x 约 575→1315，y 0→950 附近，仅小幅抖动）。

### 2. Character Layer

`outputs/character_rgb.mp4`：`CharacterRGB = Source × Mask`，黑底。

### 3. Background Plate

`outputs/background.png`：

- 76.0% 像素：在至少 5 帧中未被人物遮挡，使用跨帧逐像素中值恢复（真实像素）。
- 24.0% 像素：整段视频始终被人物遮挡，使用 OpenCV Telea inpainting 补全
  （3 轮、半径 5、膨胀扩展）。
- 补全区是估计值，不声称是原始隐藏背景。

### 4. 重合成

`outputs/preview_composite.mp4`：

```
Final = Background × (1 - SoftMask) + Source × SoftMask
```

SoftMask = 硬 mask 的 Gaussian blur（σ=1.2px），仅边缘约 2-3px 羽化，
避免硬边锯齿。人物内部仍严格保留 Source。

## 产物

- `outputs/character_mask.mp4`：逐帧人物 mask（白=人物，黑=非人物）
- `outputs/character_rgb.mp4`：黑底人物层
- `outputs/background.png`：背景 plate
- `outputs/preview_composite.mp4`：确定性重合成
- `outputs/qa_contact.png`：5 帧 × 5 列 QA 拼图
- `outputs/report.json`：完整数值报告
- `outputs/debug_mask_*.png`：mask 证据帧

## 客观指标

- mask 平均占比：24.52%（范围 24.44%–25.15%）
- 羽化带占比：0.72%
- 人物硬 mask 内 composite vs source MAE：0.33
- 严格 mask 外 composite vs background MAE：0.0036（可视为严格来自 background）
- 三个输出视频：1918×1078、30fps、112 帧、完整可解码

## 结论

链路成立，结果判定：**pass（人物边缘为 borderline 可接受）**。

限制：SAM2 编译后处理扩展不可用（`_C`），mask 由 numpy/OpenCV 后处理；
背景中央约 24% 像素为 inpaint 估计。

## 输入媒体

`input/source.mp4` 不随实验提交（媒体素材不纳入仓库），运行前本地放入 `input/` 即可。
