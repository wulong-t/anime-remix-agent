#!/usr/bin/env python
"""G0-L4: Temporal Visual Anchoring minimal experiment.

Only variable: which sampled condition-mask frames are full-frame (all-1)
instead of the baseline spatial mask (white=background keep, black=character).

B: one middle anchor  (mask/source frame 55 -> sampled j=28 -> output 28..31)
C: periodic ~1s anchors (mask/source frames 31, 62, 94 -> output 16/32/48)

Everything else is identical to the accepted CFG2 baseline
(hair_cfg_g2.mp4): C2 prompt, seed 4096, steps 8, shift 3, guide 2,
checkpoint, BF16 runtime shim, negative prompt, offload_model=True,
t5_cpu=True, ulysses_size=1, ring_size=1.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
WRAPPER = ANISORA_ROOT / "scripts" / "run-anymask-spa.py"
CKPT = ANISORA_ROOT / "models" / "anymask"

SOURCE = WORK / "g0_l2" / "source.mp4"
BASE_MASK = WORK / "g0_l2" / "source_mask.mp4"

# Audit-derived anchor frames (mask video frame domain == source frame domain).
# SAMPLED_INDICES[j] = source frame sampled as model frame j; output frame j.
SAMPLED_INDICES = [
    0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37,
    39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74,
    76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110,
]

B_ANCHOR_FRAME = 55          # source/mask frame, sampled j=28, output 28..31
C_ANCHOR_FRAMES = [31, 62, 94]  # source/mask frames, sampled j=16/32/48

C2_PROMPT = (
    "日系二维动画。一名浅金棕色短发的少女保持第一帧中的人物外观和身份，"
    "浅金棕色头发、发型、脸型、眼睛、服装和肤色在整个视频中保持一致。"
    "少女自然地轻轻眨眼，头部有非常小的动作，表情发生轻微变化，并有自然呼吸。"
    "固定机位，固定构图，背景保持稳定，人物运动幅度很小。"
)

STAGES = {
    "B": {
        "work_dir": WORK / "g0_l4" / "B_single_middle",
        "out_name": "g0_l4_single_middle_anchor.mp4",
        "log_name": "g0_l4_B_anymask.log",
        "gpu_name": "g0_l4_B_gpu.csv",
        "anchor_frames": [B_ANCHOR_FRAME],
    },
    "C": {
        "work_dir": WORK / "g0_l4" / "C_periodic",
        "out_name": "g0_l4_periodic_anchor.mp4",
        "log_name": "g0_l4_C_anymask.log",
        "gpu_name": "g0_l4_C_gpu.csv",
        "anchor_frames": C_ANCHOR_FRAMES,
    },
}


def mem_bytes(path: str) -> int:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return -1


def release_checkpoint_cache() -> dict:
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
    info = {
        "memory_current_before_gib": round(before / 1024**3, 2),
        "memory_current_after_gib": round(after / 1024**3, 2),
        "memory_max_gib": round(max_mem / 1024**3, 2),
        "checkpoint_bytes_released": freed,
    }
    print(
        f"[env] memory.current before={before/1024**3:.2f} GiB "
        f"after={after/1024**3:.2f} GiB max={max_mem/1024**3:.2f} GiB "
        f"released={freed/1024**3:.2f} GiB",
        flush=True,
    )
    return info


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w") as f:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    f.write(out.stdout)
                    f.flush()
            except Exception:
                pass
            time.sleep(1)


def read_gpu_peak(path: Path) -> tuple[float, float]:
    peak_mem = 0.0
    peak_util = 0.0
    if not path.exists():
        return peak_mem, peak_util
    with path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].replace(".", "", 1).isdigit():
                peak_mem = max(peak_mem, float(parts[0]))
                peak_util = max(peak_util, float(parts[1]))
    return peak_mem, peak_util


def decode_gray(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(out) if out else None


def decode_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def write_mask_video(path: Path, frames: np.ndarray, fps: float = 30.0) -> None:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    n, h, w = frames.shape
    rgb = np.repeat(frames[:, :, :, None], 3, axis=3)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "0",
        "-pix_fmt", "yuvj420p", "-r", str(fps),
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    _, err = proc.communicate(input=rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='replace')}")


def make_anchor_qa(modified_masks: dict[str, np.ndarray]) -> None:
    """QA before generation: Source / Baseline mask / Modified mask."""
    source = decode_rgb(SOURCE)
    base = decode_gray(BASE_MASK)
    rows = [
        ("Baseline normal (j=8)", 15, "B"),
        ("B middle anchor (j=28)", B_ANCHOR_FRAME, "B"),
        ("C anchor ~1s (j=16)", 31, "C"),
        ("C anchor ~2s (j=32)", 62, "C"),
        ("C anchor ~3s (j=48)", 94, "C"),
    ]
    cols = ["Source", "Baseline condition mask", "Modified condition mask"]
    cell_w, cell_h = 360, 203
    label_h, title_h, pad = 24, 30, 4
    canvas = Image.new(
        "RGB",
        (
            len(cols) * cell_w + pad * (len(cols) + 1),
            title_h + len(rows) * (cell_h + label_h) + pad * (len(rows) + 1),
        ),
        (16, 16, 20),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.load_default(size=18)
        label_font = ImageFont.load_default(size=14)
    except TypeError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text(
        (pad + 4, 6),
        "G0-L4 anchor mask QA: Source / Baseline condition mask / Modified condition mask (white=keep)",
        fill=(240, 240, 240),
        font=title_font,
    )
    for row, (label, frame_s, stage) in enumerate(rows):
        mod = modified_masks[stage]
        src_img = Image.fromarray(np.ascontiguousarray(source[frame_s])).resize(
            (cell_w, cell_h), Image.LANCZOS
        )
        base_img = Image.fromarray(np.repeat(base[frame_s][:, :, None], 3, axis=2)).resize(
            (cell_w, cell_h), Image.LANCZOS
        )
        mod_img = Image.fromarray(np.repeat(mod[frame_s][:, :, None], 3, axis=2)).resize(
            (cell_w, cell_h), Image.LANCZOS
        )
        images = [src_img, base_img, mod_img]
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, img in enumerate(images):
            x0 = pad + col * cell_w
            canvas.paste(img, (x0, y0 + label_h))
            wr = float((np.asarray(images[col].convert("L")) > 127).mean())
            draw.text(
                (x0 + 4, y0 + 2),
                f"{label} {cols[col]} white={wr:.2%}",
                fill=(220, 220, 220),
                font=label_font,
            )
    out = OUTPUTS / "g0_l4_anchor_masks_qa.png"
    canvas.save(out)
    print("[qa] anchor mask QA ->", out)


def build_stages(skip_sampling: bool) -> dict[str, dict]:
    base = decode_gray(BASE_MASK)
    modified: dict[str, np.ndarray] = {}
    for tag, cfg in STAGES.items():
        stage_dir = cfg["work_dir"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        src_copy = stage_dir / "source.mp4"
        mask_path = stage_dir / "source_mask.mp4"
        if not src_copy.exists() or src_copy.stat().st_size != SOURCE.stat().st_size:
            shutil.copyfile(SOURCE, src_copy)
        m = base.copy()
        for f in cfg["anchor_frames"]:
            m[f] = 255
        write_mask_video(mask_path, m)
        modified[tag] = m
        (stage_dir / "prompt.txt").write_text(
            f"{C2_PROMPT}@@{src_copy}\n", encoding="utf-8"
        )
        wr = float((m > 127).mean())
        print(
            f"[stage-{tag}] mask {mask_path} frames={m.shape} "
            f"anchors={cfg['anchor_frames']} white_ratio={wr:.4f} "
            f"anchor_white_ratios={[round(float((m[f] > 127).mean()), 4) for f in cfg['anchor_frames']]}"
        )
    make_anchor_qa(modified)
    if skip_sampling:
        print("[run] --skip-sampling: masks + QA done, no AnyMask run.")
    return modified


def run_stage(tag: str, cfg: dict) -> dict:
    stage_dir = cfg["work_dir"]
    out_dir = stage_dir / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS / cfg["log_name"]
    gpu_path = OUTPUTS / cfg["gpu_name"]
    prompt_path = stage_dir / "prompt.txt"

    env = os.environ.copy()
    env.update(
        {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HOME": str(ANISORA_ROOT / "cache" / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(ANISORA_ROOT / "cache" / "huggingface" / "hub"),
            "TORCH_HOME": str(ANISORA_ROOT / "cache" / "torch"),
            "TMPDIR": str(ANISORA_ROOT / "tmp"),
            "UV_LINK_MODE": "copy",
        }
    )

    mem_info = release_checkpoint_cache()
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_path, stop), daemon=True)
    monitor.start()
    start = time.time()
    cmd = [
        "/root/.local/bin/uv", "run", "--python", "/usr/bin/python3.10",
        "--no-python-downloads", "python", str(WRAPPER),
        "--task", "i2v-14B",
        "--size", "832*480",
        "--ckpt_dir", str(CKPT),
        "--base_seed", "4096",
        "--sample_steps", "8",
        "--sample_shift", "3",
        "--sample_guide_scale", "2",
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
    runtime = round(time.time() - start, 2)
    stop.set()
    monitor.join(timeout=5)
    peak_mem, peak_util = read_gpu_peak(gpu_path)

    produced = out_dir / "0_ALL.mp4"
    ok = code == 0 and produced.exists()
    if ok:
        shutil.copyfile(produced, OUTPUTS / cfg["out_name"])
    print(
        f"[run-{tag}] exit={code} runtime_s={runtime} "
        f"peak_vram_mib={peak_mem} peak_util={peak_util} "
        f"output={OUTPUTS / cfg['out_name'] if ok else 'MISSING'}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng0_l4={tag} exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_mem} peak_util={peak_util} "
            f"memory_before_gib={mem_info['memory_current_before_gib']} "
            f"memory_after_gib={mem_info['memory_current_after_gib']} "
            f"memory_max_gib={mem_info['memory_max_gib']} "
            f"anchor_frames={cfg['anchor_frames']}\n"
        )
    if code != 0:
        raise SystemExit(f"g0_l4 {tag} failed with exit code {code}")
    if not ok:
        raise SystemExit(f"g0_l4 {tag} output missing")
    return {
        "tag": tag,
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": peak_mem,
        "peak_util": peak_util,
        "memory": mem_info,
        "output": str(OUTPUTS / cfg["out_name"]),
        "anchor_frames": cfg["anchor_frames"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sampling", action="store_true")
    args = ap.parse_args()
    build_stages(args.skip_sampling)
    if args.skip_sampling:
        return
    results = {}
    for tag, cfg in STAGES.items():
        results[tag] = run_stage(tag, cfg)
    print(json_summary(results))


def json_summary(results: dict) -> str:
    import json

    return json.dumps(results, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
