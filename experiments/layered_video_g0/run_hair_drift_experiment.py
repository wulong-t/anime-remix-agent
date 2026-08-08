#!/usr/bin/env python
"""Run one G0-L2 hair-drift experiment with a fixed non-prompt configuration.

Only the prompt changes between runs. All sampling/condition parameters are
hard-fixed to the previous G0-L2 baseline.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
STAGE = WORK / "g0_l2"
ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
WRAPPER = ANISORA_ROOT / "scripts" / "run-anymask-spa.py"
SOURCE = STAGE / "source.mp4"
COND_MASK = STAGE / "source_mask.mp4"
CKPT = ANISORA_ROOT / "models" / "anymask"

PROMPTS = {
    "E0": (
        "Stable anime style, a single original anime girl standing in a fixed composition, "
        "same character as the first frame with identical hairstyle, face, clothing and colors. "
        "Only subtle natural motion: slow blinking, slight head movement, small facial expression change, "
        "gentle breathing. Camera is completely fixed, background must remain exactly unchanged. "
        "No text, no watermark, no extra objects, no large body movement."
    ),
    "E1": (
        "Stable anime style. Keep exactly the same girl as in the first frame. "
        "She has short light golden-brown hair with warm honey-brown tones. "
        "Her hair color must remain exactly the same light golden-brown color in every frame. "
        "Do not change the hair to gray, silver, white, black, or desaturated colors. "
        "Preserve exactly the same hairstyle, face, eyes, clothing, skin tone and character identity as the first frame. "
        "Only subtle natural motion: slow blinking, very slight head movement, small facial expression changes and gentle breathing. "
        "Fixed camera and fixed composition. No new objects, no text, no watermark, no large body movement."
    ),
    "C1": (
        "稳定的日系二维动画风格。始终保持第一帧中的同一个女孩，人物身份不得改变。"
        "女孩留着浅金棕色的短发，发色带有温暖的蜂蜜金棕色调。"
        "所有帧中的头发颜色必须始终保持与第一帧完全一致的浅金棕色，"
        "不要变成灰色、银色、白色、黑色或低饱和度颜色。"
        "保持与第一帧完全一致的发型、脸型、眼睛、服装、肤色和人物主要身份特征。"
        "只允许非常轻微而自然的动作：缓慢眨眼、非常轻微的头部运动、小幅表情变化和轻微呼吸。"
        "摄影机完全固定，构图完全固定。不要新增物体，不要出现文字，不要水印，不要大幅身体动作。"
    ),
    "C2": (
        "日系二维动画。一名浅金棕色短发的少女保持第一帧中的人物外观和身份，"
        "浅金棕色头发、发型、脸型、眼睛、服装和肤色在整个视频中保持一致。"
        "少女自然地轻轻眨眼，头部有非常小的动作，表情发生轻微变化，并有自然呼吸。"
        "固定机位，固定构图，背景保持稳定，人物运动幅度很小。"
    ),
}

OUT_NAMES = {
    "E0": "hair_test_e0_baseline_en.mp4",
    "E1": "hair_test_e1_explicit_en.mp4",
    "C1": "hair_test_c1_explicit_zh.mp4",
    "C2": "hair_test_c2_simple_zh.mp4",
}


def release_checkpoint_cache() -> None:
    """Drop page cache held by checkpoint files via fadvise(DONTNEED).

    The cgroup has a 90 GiB CPU RAM hard limit. After one AnyMask run, the
    ~47 GB checkpoint files remain in page cache (~46 GB), so a later run can
    approach the limit at VAE-decode time and be SIGKILLed (exit 137). This
    releases only kernel page cache; no source or model files are modified.
    """
    freed = 0
    for item in CKPT.iterdir():
        if item.is_file():
            fd = os.open(str(item), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                freed += item.stat().st_size
            finally:
                os.close(fd)
    print(f"[env] released checkpoint page cache: {freed / 1024**3:.2f} GiB", flush=True)


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w") as f:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5,
                )
                if out.returncode == 0:
                    f.write(out.stdout)
                    f.flush()
            except Exception:
                pass
            time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(PROMPTS))
    args = ap.parse_args()
    name = args.experiment
    prompt = PROMPTS[name]
    out_name = OUT_NAMES[name]
    run_dir = WORK / "hair_test" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(f"{prompt}@@{SOURCE}\n", encoding="utf-8")
    log_path = OUTPUTS / f"hair_drift_{name}_anymask.log"
    gpu_path = OUTPUTS / f"hair_drift_{name}_gpu.csv"

    env = os.environ.copy()
    env.update({
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HOME": str(ANISORA_ROOT / "cache" / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(ANISORA_ROOT / "cache" / "huggingface" / "hub"),
        "TORCH_HOME": str(ANISORA_ROOT / "cache" / "torch"),
        "TMPDIR": str(ANISORA_ROOT / "tmp"),
        "UV_LINK_MODE": "copy",
    })

    release_checkpoint_cache()
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_path, stop), daemon=True)
    monitor.start()
    start = time.time()
    cmd = [
        "/root/.local/bin/uv", "run", "--python", "/usr/bin/python3.10", "--no-python-downloads",
        "python", str(WRAPPER),
        "--task", "i2v-14B",
        "--size", "832*480",
        "--ckpt_dir", str(CKPT),
        "--base_seed", "4096",
        "--sample_steps", "8",
        "--sample_shift", "3",
        "--sample_guide_scale", "1",
        "--offload_model", "True",
        "--t5_cpu",
        "--ulysses_size", "1",
        "--ring_size", "1",
        "--prompt", str(prompt_path),
        "--image", str(out_dir),
    ]
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=INDEX_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = proc.returncode
    end = time.time()
    stop.set()
    monitor.join(timeout=5)
    runtime = round(end - start, 2)

    produced = out_dir / "0_ALL.mp4"
    ok = code == 0 and produced.exists()
    if ok:
        shutil.copyfile(produced, OUTPUTS / out_name)
    print(f"[hair-{name}] exit={code} runtime_s={runtime} output={OUTPUTS / out_name if ok else 'MISSING'}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\nhair_experiment={name} exit_code={code} runtime_seconds={runtime}\n")
    if code != 0:
        raise SystemExit(f"experiment {name} failed with exit code {code}")


if __name__ == "__main__":
    main()
