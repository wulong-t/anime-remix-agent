#!/usr/bin/env python
"""G0-L6: Full-Condition Editability Gate (single generation).

Full-condition mask: every frame all-white -> binary_mask=1 for all T*H*W,
so Img_list_new == Img_list (full source video RGB as VAE condition).
Single edit task: light golden-brown hair -> deep red / wine red, with all
other variables identical to the accepted CFG2 baseline.
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
STAGE = WORK / "g0_l6" / "full_condition"
SOURCE = WORK / "g0_l2" / "source.mp4"

ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
WRAPPER = ANISORA_ROOT / "scripts" / "run-anymask-spa.py"
CKPT = ANISORA_ROOT / "models" / "anymask"

EDIT_PROMPT = (
    "日系二维动画。保持原视频中的同一个女孩，人物身份、脸型、眼睛、发型、"
    "肤色、服装、身体动作、姿势、背景、摄影机和构图都保持与原视频一致。"
    "只修改一个属性：将女孩原本浅金棕色的头发改变为明显的深红色、酒红色头发。"
    "整个视频中所有头发区域都应稳定保持深红色，包括刘海、两侧头发、外围头发和后发。"
    "除此之外不要改变任何人物特征、物体、动作、背景或画面结构。"
)


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


def make_mask_qa(mask: np.ndarray) -> None:
    source = decode_rgb(SOURCE)
    rows = []
    for fi in [0, 28, 56, 84, 111]:
        mask_img = np.repeat(mask[fi][:, :, None], 3, axis=2)
        rows.append(
            (
                f"source F{fi}",
                [
                    cv2.resize(source[fi], (832, 464), interpolation=cv2.INTER_AREA),
                    cv2.resize(mask_img, (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(source[fi], (832, 464), interpolation=cv2.INTER_AREA),
                ],
            )
        )
    cols = ["Source", "Full Condition Mask (all white)", "Conditioned RGB (=Source when mask=1)"]
    cell_w, cell_h = 340, 191
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
        label_font = ImageFont.load_default(size=13)
    except TypeError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text(
        (pad + 4, 6),
        "G0-L6 full-condition mask QA: Source / Full Condition Mask / Conditioned RGB",
        fill=(240, 240, 240),
        font=title_font,
    )
    for row, (label, imgs) in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        draw.text((pad + 4, y0 + 1), label, fill=(230, 230, 230), font=label_font)
        for col, img in enumerate(imgs):
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 1), cols[col], fill=(220, 220, 220), font=label_font)
    out = OUTPUTS / "g0_l6_full_condition_mask_qa.png"
    canvas.save(out)
    print("[qa] full-condition mask QA ->", out)


def build_stage(skip_sampling: bool) -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    src_copy = STAGE / "source.mp4"
    if not src_copy.exists() or src_copy.stat().st_size != SOURCE.stat().st_size:
        shutil.copyfile(SOURCE, src_copy)
    mask_path = STAGE / "source_mask.mp4"
    mask = np.full((112, 1078, 1918), 255, dtype=np.uint8)
    write_mask_video(mask_path, mask)
    (STAGE / "prompt.txt").write_text(f"{EDIT_PROMPT}@@{src_copy}\n", encoding="utf-8")
    decoded = decode_gray(mask_path)
    info = {
        "mask_frame_count": int(len(decoded)),
        "mask_shape": list(decoded.shape),
        "white_ratio": round(float((decoded > 127).mean()), 6),
        "samples": {
            fi: {
                "min": int(decoded[fi].min()),
                "max": int(decoded[fi].max()),
                "white_ratio": round(float((decoded[fi] > 127).mean()), 6),
            }
            for fi in [0, 28, 56, 84, 111]
        },
    }
    print("[mask]", json_dumps(info))
    make_mask_qa(decoded)
    assert len(decoded) == 112, "mask frame count != source frame count"
    assert decoded.min() == 255 and decoded.max() == 255, "mask not all-white"
    if skip_sampling:
        print("[run] --skip-sampling: mask + QA done, no AnyMask run.")
    return info


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


def run_generation() -> dict:
    out_dir = STAGE / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS / "g0_l6_anymask.log"
    gpu_path = OUTPUTS / "g0_l6_gpu.csv"
    prompt_path = STAGE / "prompt.txt"

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
    out_name = "g0_l6_full_condition_red_hair.mp4"
    if ok:
        shutil.copyfile(produced, OUTPUTS / out_name)
    print(
        f"[run] exit={code} runtime_s={runtime} peak_vram_mib={peak_mem} "
        f"peak_util={peak_util} output={OUTPUTS / out_name if ok else 'MISSING'}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng0_l6 exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_mem} peak_util={peak_util} "
            f"memory_before_gib={mem_info['memory_current_before_gib']} "
            f"memory_after_gib={mem_info['memory_current_after_gib']} "
            f"memory_max_gib={mem_info['memory_max_gib']}\n"
        )
    if code != 0 or not ok:
        raise SystemExit(f"g0_l6 generation failed exit={code} ok={ok}")
    return {
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": peak_mem,
        "peak_util": peak_util,
        "memory": mem_info,
        "output_path": str(OUTPUTS / out_name),
        "condition_mode": "full_video",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sampling", action="store_true")
    args = ap.parse_args()
    build_stage(args.skip_sampling)
    if not args.skip_sampling:
        run_generation()


if __name__ == "__main__":
    main()
