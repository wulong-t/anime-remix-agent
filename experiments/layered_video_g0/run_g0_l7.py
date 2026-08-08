#!/usr/bin/env python
"""G0-L7: Target/Reference Decoupling minimal experiment (single generation).

Only change vs the whole-character masked baseline:
  Img_list_new = Img_list.to(device) * binary_mask   (baseline, zero-fill)
  Img_list_new = Img_list.to(device)                 (G0-L7 decoupled)

binary_mask / msk are byte-identical to the G0-L2/CFG2 baseline
(background=1 known, character=0 unknown/target, first frame forced 1 by code).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
STAGE = WORK / "g0_l7" / "decoupled"
RUNTIME = WORK / "g0_l7" / "runtime"
SOURCE = WORK / "g0_l2" / "source.mp4"
BASE_MASK = WORK / "g0_l2" / "source_mask.mp4"

ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
CKPT = ANISORA_ROOT / "models" / "anymask"

EDIT_PROMPT = (
    "日系二维动画。保持原视频中的同一个女孩，人物身份、脸型、眼睛、发型、"
    "肤色、服装、身体动作、姿势、背景、摄影机和构图都保持与原视频一致。"
    "只修改一个属性：将女孩原本浅金棕色的头发改变为明显的深红色、酒红色头发。"
    "整个视频中所有头发区域都应稳定保持深红色，包括刘海、两侧头发、外围头发和后发。"
    "除此之外不要改变任何人物特征、物体、动作、背景或画面结构。"
)

DRY_FRAME = 28  # non-first model/output frame (source frame 55)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_video(path: Path, start: int = 0, end: int = 112, ss: float = 3.5):
    vr = VideoReader(uri=str(path), height=-1, width=-1)
    num_frames = int(ss * 16 + 1)
    indices = np.arange(start, end, (end - start) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = (frames - 0.5) / 0.5
    frames = frames.permute(0, 3, 1, 2)
    return frames, indices


def inverse_normalize(x: torch.Tensor) -> np.ndarray:
    rgb = ((x * 0.5 + 0.5) * 255.0).clamp(0, 255).permute(0, 2, 3, 1)
    return rgb.numpy().astype(np.uint8)


def dry_diagnostic() -> dict:
    """Reproduce binary_mask / msk / Img_list_new for DRY_FRAME and verify decoupling."""
    print("[dry] decoupling diagnostic before diffusion")
    Img_list, indices = read_video(SOURCE, 0, 112, 3.5)
    Img_mask_list, mask_indices = read_video(BASE_MASK, 0, 112, 3.5)
    assert (indices == mask_indices).all()
    assert indices[DRY_FRAME] == 55
    binary_mask = torch.where(
        Img_mask_list > 0, torch.ones_like(Img_mask_list), torch.zeros_like(Img_mask_list)
    )
    binary_mask[0, :, :, :] = 1
    # model-domain spatial resize (same as pipeline)
    Img_list_r = F.interpolate(Img_list, size=(464, 832), mode="nearest")
    binary_mask_r = F.interpolate(binary_mask, size=(464, 832), mode="nearest")
    baseline_new = Img_list_r * binary_mask_r
    decoupled_new = Img_list_r  # G0-L7 change

    person = binary_mask_r[DRY_FRAME] == 0
    person_orig = Img_list_r[DRY_FRAME][person]
    person_baseline = baseline_new[DRY_FRAME][person]
    person_decoupled = decoupled_new[DRY_FRAME][person]
    ok = bool(
        (person_baseline == 0).all()
        and torch.allclose(person_decoupled, person_orig, atol=0.0)
        and not (person_decoupled == 0).all()
    )
    info = {
        "frame": DRY_FRAME,
        "source_frame_index": int(indices[DRY_FRAME]),
        "binary_mask_person_zero_ratio": round(float((binary_mask_r[DRY_FRAME, 0] == 0).float().mean()), 4),
        "msk_semantics": "background=1 known; character=0 unknown/target (first frame forced 1 by code)",
        "baseline_person_value_min_max_mean": [
            float(person_baseline.min()),
            float(person_baseline.max()),
            float(person_baseline.mean()),
        ],
        "decoupled_person_value_min_max_mean": [
            float(person_decoupled.min()),
            float(person_decoupled.max()),
            float(person_decoupled.mean()),
        ],
        "original_person_value_min_max_mean": [
            float(person_orig.min()),
            float(person_orig.max()),
            float(person_orig.mean()),
        ],
        "decoupling_holds": bool(ok),
    }
    print("[dry]", json.dumps(info, ensure_ascii=False, indent=2))

    # QA: Source / Binary Mask / Baseline Masked RGB / G0-L7 Full Reference RGB
    source_frames = cv2.VideoCapture(str(SOURCE))
    src_list = []
    while True:
        okf, f = source_frames.read()
        if not okf:
            break
        src_list.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    source_frames.release()
    src = np.stack(src_list)
    rows = [
        (
            f"j={DRY_FRAME} source F{indices[DRY_FRAME]}",
            [
                cv2.resize(src[indices[DRY_FRAME]], (832, 464), interpolation=cv2.INTER_AREA),
                np.repeat((binary_mask_r[DRY_FRAME, 0].numpy() * 255)[:, :, None].astype(np.uint8), 3, axis=2),
                inverse_normalize(baseline_new)[DRY_FRAME],
                inverse_normalize(decoupled_new)[DRY_FRAME],
            ],
        )
    ]
    make_qa_sheet(
        OUTPUTS / "g0_l7_condition_decoupling_qa.png",
        "G0-L7 dry diagnostic: Source / Binary Mask (person=0 target) / Baseline Masked RGB / Decoupled Full Reference RGB",
        rows,
        ["Source", "Binary Mask", "Baseline Masked RGB (zero-fill)", "G0-L7 Full Reference RGB"],
        cell_w=340,
        cell_h=191,
    )
    if not ok:
        raise SystemExit("decoupling dry diagnostic FAILED: stop before GPU")
    return info


def make_qa_sheet(path: Path, title: str, rows: list[tuple[str, list[np.ndarray]]], cols: list[str], cell_w: int = 300, cell_h: int = 180) -> None:
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
    draw.text((pad + 4, 6), title, fill=(240, 240, 240), font=title_font)
    for row, (label, imgs) in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        draw.text((pad + 4, y0 + 1), label, fill=(230, 230, 230), font=label_font)
        for col, img in enumerate(imgs):
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 1), cols[col], fill=(220, 220, 220), font=label_font)
    canvas.save(path)
    print("[qa]", path)


def build_runtime() -> Path:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX_DIR / "generate-pi-i2v-any-mask1_spa.py", RUNTIME / "generate-pi-i2v-any-mask1_spa.py")
    shutil.copytree(
        INDEX_DIR / "wan",
        RUNTIME / "wan",
        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
    )
    p = RUNTIME / "wan" / "image2video_any_mask1_spa.py"
    s = p.read_text(encoding="utf-8")
    old = "        Img_list_new = Img_list.to(self.device) * binary_mask  \n"
    new = "        Img_list_new = Img_list.to(self.device)  \n"
    assert s.count(old) == 1, f"pattern count={s.count(old)}"
    p.write_text(s.replace(old, new), encoding="utf-8")
    (RUNTIME / "run_decoupled.py").write_text(
        """#!/usr/bin/env python
import os
import runpy
import sys

import torch

torch.set_default_dtype(torch.bfloat16)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
sys.argv = ["generate-pi-i2v-any-mask1_spa.py"] + sys.argv[1:]
runpy.run_path(
    os.path.join(os.getcwd(), "generate-pi-i2v-any-mask1_spa.py"),
    run_name="__main__",
)
""",
        encoding="utf-8",
    )
    return RUNTIME


def run_generation() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    src_copy = STAGE / "source.mp4"
    if not src_copy.exists() or src_copy.stat().st_size != SOURCE.stat().st_size:
        shutil.copyfile(SOURCE, src_copy)
    mask_copy = STAGE / "source_mask.mp4"
    if not mask_copy.exists() or mask_copy.stat().st_size != BASE_MASK.stat().st_size:
        shutil.copyfile(BASE_MASK, mask_copy)
    assert sha256(mask_copy) == sha256(BASE_MASK), "mask differs from whole-character baseline"
    (STAGE / "prompt.txt").write_text(f"{EDIT_PROMPT}@@{src_copy}\n", encoding="utf-8")
    out_dir = STAGE / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = build_runtime()

    log_path = OUTPUTS / "g0_l7_anymask.log"
    gpu_path = OUTPUTS / "g0_l7_gpu.csv"
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
        str(ANISORA_ROOT / ".venv" / "bin" / "python"), str(rt / "run_decoupled.py"),
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
        "--prompt", str(STAGE / "prompt.txt"),
        "--image", str(out_dir),
    ]
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=rt, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = proc.returncode
    runtime = round(time.time() - start, 2)
    stop.set()
    monitor.join(timeout=5)
    peak_mem, peak_util = read_gpu_peak(gpu_path)
    produced = out_dir / "0_ALL.mp4"
    ok = code == 0 and produced.exists()
    out_name = "g0_l7_target_reference_decoupled.mp4"
    if ok:
        shutil.copyfile(produced, OUTPUTS / out_name)
    print(
        f"[run] exit={code} runtime_s={runtime} peak_vram_mib={peak_mem} "
        f"peak_util={peak_util} output={OUTPUTS / out_name if ok else 'MISSING'}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng0_l7 exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_mem} peak_util={peak_util} "
            f"memory_before_gib={mem_info['memory_current_before_gib']} "
            f"memory_after_gib={mem_info['memory_current_after_gib']} "
            f"memory_max_gib={mem_info['memory_max_gib']}\n"
        )
    if code != 0 or not ok:
        raise SystemExit(f"g0_l7 generation failed exit={code} ok={ok}")
    return {
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": peak_mem,
        "peak_util": peak_util,
        "memory": mem_info,
        "output_path": str(OUTPUTS / out_name),
        "condition_mode": "target_mask_unchanged + full_source_reference",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-only", action="store_true")
    args = ap.parse_args()
    audit = {
        "source_path": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "mask_path": str(BASE_MASK),
        "mask_sha256": sha256(BASE_MASK),
    }
    print("[audit]", json.dumps(audit, ensure_ascii=False, indent=2))
    dry = dry_diagnostic()
    if args.dry_only:
        print("[run] --dry-only: decoupling verified, no GPU generation.")
        return
    run_generation()


if __name__ == "__main__":
    main()
