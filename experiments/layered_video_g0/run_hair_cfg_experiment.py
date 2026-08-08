#!/usr/bin/env python
"""G0-L2 CFG single-variable experiment.

Fixed: C2 simple Chinese prompt, source, condition mask, checkpoint,
base_seed=4096, sample_steps=8, sample_shift=3, resolution 832*480,
runtime BF16 shim, negative prompt, offload_model=True, t5_cpu=True,
ulysses_size=1, ring_size=1.

Only sample_guide_scale changes.
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
CKPT = ANISORA_ROOT / "models" / "anymask"

C2_PROMPT = (
    "日系二维动画。一名浅金棕色短发的少女保持第一帧中的人物外观和身份，"
    "浅金棕色头发、发型、脸型、眼睛、服装和肤色在整个视频中保持一致。"
    "少女自然地轻轻眨眼，头部有非常小的动作，表情发生轻微变化，并有自然呼吸。"
    "固定机位，固定构图，背景保持稳定，人物运动幅度很小。"
)


def mem_bytes(path: str) -> int:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return -1


def release_checkpoint_cache() -> None:
    before = mem_bytes("/sys/fs/cgroup/memory.current")
    max_mem = mem_bytes("/sys/fs/cgroup/memory.max")
    freed = 0
    for item in CKPT.iterdir():
        if item.is_file():
            fd = os.open(str(item), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                freed += item.stat().st_size
            finally:
                os.close(fd)
    after = mem_bytes("/sys/fs/cgroup/memory.current")
    print(
        f"[env] memory.current before={before/1024**3:.2f} GiB "
        f"after={after/1024**3:.2f} GiB max={max_mem/1024**3:.2f} GiB "
        f"released={freed/1024**3:.2f} GiB",
        flush=True,
    )


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
    ap.add_argument("--guide-scale", type=float, required=True, choices=[2.0, 3.0, 5.0])
    args = ap.parse_args()
    scale = args.guide_scale
    tag = f"g{int(scale)}"
    out_name = f"hair_cfg_{tag}.mp4"
    run_dir = WORK / "hair_cfg" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(f"{C2_PROMPT}@@{SOURCE}\n", encoding="utf-8")
    log_path = OUTPUTS / f"hair_cfg_{tag}_anymask.log"
    gpu_path = OUTPUTS / f"hair_cfg_{tag}_gpu.csv"

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
        "--sample_guide_scale", str(scale),
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
    print(f"[cfg-{tag}] exit={code} runtime_s={runtime} output={OUTPUTS / out_name if ok else 'MISSING'}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\nhair_cfg={tag} guide_scale={scale} exit_code={code} runtime_seconds={runtime}\n")
    if code != 0:
        raise SystemExit(f"cfg {tag} failed with exit code {code}")


if __name__ == "__main__":
    main()
