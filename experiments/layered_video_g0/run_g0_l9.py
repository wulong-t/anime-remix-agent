#!/usr/bin/env python
"""G0-L9: Hair-only Minimal Editable Region Gate.

Stages:
  1. hair_character_mask.mp4: SAM2.1 video propagation from a frame-0 hair
     seed (character mask + hair color + head zone, face/neck excluded).
  2. Hair mask QA gate (visual sheet + programmatic checks). Stop on failure.
  3. AnyMask condition mask: Hair=0 unknown, everything else=1 known.
  4. Diffusion-precondition QA (only hair zero-filled).
  5. Single AnyMask run (official runner, no source modification).
"""

from __future__ import annotations

import argparse
import json
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
L9 = WORK / "g0_l9"
STAGE = L9 / "hair_only"
FRAMES_DIR = L9 / "sam2_frames"
SOURCE = WORK / "g0_l2" / "source.mp4"
MASKS_NPY = WORK / "masks.npy"

ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
WRAPPER = ANISORA_ROOT / "scripts" / "run-anymask-spa.py"
CKPT = ANISORA_ROOT / "models" / "anymask"
MODEL_ID = "facebook/sam2.1-hiera-base-plus"

# Source-domain geometry (1918x1078)
FACE_BOX = (960, 190, 1180, 460)
NECK_BOX = (1000, 440, 1180, 530)
HEAD_ZONE_Y = 700
HAIR_BOX = (560, 0, 1360, 700)

EDIT_PROMPT = (
    "日系二维动画。保持原视频中的同一个女孩，人物身份、脸型、眼睛、发型、"
    "肤色、颈部、服装、身体动作、姿势、背景、摄影机和构图全部保持与原视频一致。"
    "女孩的头发从第一帧原本的浅金棕色开始，在视频开始后的短时间内自然变化为"
    "明显的深红色、酒红色，随后在剩余视频中稳定保持酒红色。"
    "只改变头发颜色，不改变头发形状、长度、发型和运动。"
    "刘海、头顶头发、两侧头发、外围头发、后发和发梢都应统一变化为酒红色。"
    "脸、眼睛、皮肤、颈部、服装、身体、动作、背景和其他物体不得改变。"
)


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


def write_gray_video(path: Path, frames: np.ndarray, fps: float = 30.0) -> None:
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
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='replace')}")


def write_frames_dir(frames: np.ndarray) -> Path:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            str(FRAMES_DIR / f"{i:06d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    return FRAMES_DIR


def build_hair_seed(frame0: np.ndarray, char0: np.ndarray) -> np.ndarray:
    """Hair seed on frame 0: character x hair color x head zone, face/neck excluded."""
    hsv = cv2.cvtColor(frame0, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, w = frame0.shape[:2]
    hair_color = (hsv[..., 0] >= 4) & (hsv[..., 0] <= 28) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 80)
    zone = np.zeros((h, w), dtype=bool)
    zone[:HEAD_ZONE_Y, :] = True
    fx0, fy0, fx1, fy1 = FACE_BOX
    nx0, ny0, nx1, ny1 = NECK_BOX
    face = np.zeros((h, w), dtype=bool)
    face[fy0:fy1, fx0:fx1] = True
    neck = np.zeros((h, w), dtype=bool)
    neck[ny0:ny1, nx0:nx1] = True
    seed = char0 & hair_color & zone & ~face & ~neck
    seed = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    if n > 1:
        best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        seed = (labels == best).astype(np.uint8) * 255
    return (seed > 127).astype(np.uint8) * 255


def run_sam2(frames: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
    import torch
    from sam2.build_sam import build_sam2_video_predictor_hf

    write_frames_dir(frames)
    predictor = build_sam2_video_predictor_hf(MODEL_ID, device="cuda")
    state = predictor.init_state(
        video_path=str(FRAMES_DIR),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=initial_mask > 127)
    raw = {}
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(state, start_frame_idx=0):
        m = video_res_masks[0]
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        if m.dtype != bool:
            m = m > 0.0
        raw[int(frame_idx)] = np.squeeze(np.ascontiguousarray(m)) > 0
    del predictor, state
    torch.cuda.empty_cache()
    return np.stack([raw[i] for i in range(len(frames))])


def postprocess_hair(raw: np.ndarray) -> np.ndarray:
    """Largest connected component + temporal 3-frame median."""
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    proc = []
    for m in raw:
        m = (m.astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            m = (labels == best).astype(np.uint8) * 255
        else:
            m = np.zeros_like(m)
        proc.append(m)
    out = []
    for i in range(len(proc)):
        lo, hi = max(0, i - 1), min(len(proc) - 1, i + 1)
        out.append(np.median(np.stack(proc[lo:hi + 1]), axis=0).astype(np.uint8))
    return np.stack(out)


def mask_metrics(hair: np.ndarray, char: np.ndarray) -> dict:
    h_bool = hair > 127
    c_bool = char > 0 if char.dtype == bool else char > 127
    area = h_bool.reshape(len(h_bool), -1).mean(axis=1)
    fx0, fy0, fx1, fy1 = FACE_BOX
    nx0, ny0, nx1, ny1 = NECK_BOX
    face_zone = np.zeros_like(h_bool[0])
    face_zone[fy0:fy1, fx0:fx1] = True
    neck_zone = np.zeros_like(h_bool[0])
    neck_zone[ny0:ny1, nx0:nx1] = True
    outside = np.stack([h & ~c for h, c in zip(h_bool, c_bool)])
    boxes = []
    for m in h_bool:
        ys, xs = np.where(m)
        if len(xs):
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    b = np.array(boxes) if boxes else None
    return {
        "mean_area_ratio": round(float(area.mean()), 4),
        "area_min_max": [round(float(area.min()), 4), round(float(area.max()), 4)],
        "area_std": round(float(area.std()), 4),
        "hair_character_ratio": round(float(h_bool.mean() / max(c_bool.mean(), 1e-9)), 4),
        "outside_character_ratio": round(float(outside.sum() / max(h_bool.sum(), 1)), 4),
        "face_overlap_ratio": round(float((h_bool & face_zone[None]).sum() / max(h_bool.sum(), 1)), 4),
        "neck_overlap_ratio": round(float((h_bool & neck_zone[None]).sum() / max(h_bool.sum(), 1)), 4),
        "empty_frames": int((area == 0).sum()),
        "bbox_mean": b.mean(axis=0).round(1).tolist() if b is not None else None,
        "bbox_min": b.min(axis=0).tolist() if b is not None else None,
        "bbox_max": b.max(axis=0).tolist() if b is not None else None,
    }


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


def hair_mask_qa(frames: np.ndarray, char: np.ndarray, hair: np.ndarray) -> None:
    sample = [0, 28, 56, 84, 111]
    rows = []
    for fi in sample:
        overlay = frames[fi].copy()
        hm = hair[fi] > 127
        overlay[hm] = (overlay[hm].astype(np.int16) * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        rows.append(
            (
                f"source F{fi}",
                [
                    cv2.resize(frames[fi], (832, 464), interpolation=cv2.INTER_AREA),
                    cv2.resize(np.repeat((char[fi] * 255)[:, :, None].astype(np.uint8), 3, axis=2), (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(np.repeat((hair[fi] * 255)[:, :, None].astype(np.uint8), 3, axis=2), (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(overlay, (832, 464), interpolation=cv2.INTER_AREA),
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l9_hair_mask_qa.png",
        "G0-L9 hair mask QA: Source / Character mask / Hair mask / Hair overlay (red)",
        rows,
        ["Source", "Character Mask", "Hair Mask", "Hair Overlay"],
        cell_w=340,
        cell_h=191,
    )


def build_mask() -> dict:
    print("[mask] building hair mask with SAM2")
    frames = decode_rgb(SOURCE)
    char = np.load(MASKS_NPY) > 127
    seed = build_hair_seed(frames[0], char[0])
    seed_ratio = float((seed > 127).mean())
    print(f"[mask] seed ratio={seed_ratio:.4f}")
    if not (0.01 <= seed_ratio <= 0.25):
        raise SystemExit(f"seed area out of expected range: {seed_ratio:.4f}")
    raw = run_sam2(frames, seed)
    hair = postprocess_hair(raw)
    write_gray_video(OUTPUTS / "hair_character_mask.mp4", hair)
    metrics = mask_metrics(hair, char)
    print("[mask]", json.dumps(metrics, ensure_ascii=False, indent=2))
    hair_mask_qa(frames, char, hair)
    gate = {
        "outside_character_ok": metrics["outside_character_ratio"] < 0.02,
        "face_overlap_ok": metrics["face_overlap_ratio"] < 0.08,
        "neck_overlap_ok": metrics["neck_overlap_ratio"] < 0.05,
        "no_empty_frames": metrics["empty_frames"] == 0,
        "area_in_range": 0.02 <= metrics["mean_area_ratio"] <= 0.25,
        "temporal_stable": metrics["area_std"] / max(metrics["mean_area_ratio"], 1e-9) < 0.3,
    }
    gate["passed"] = bool(all(gate.values()))
    print("[mask-gate]", json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("MASK_GATE_FAILED: stop before AnyMask")
    return {"metrics": metrics, "gate": gate}


def build_condition() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    src_copy = STAGE / "source.mp4"
    if not src_copy.exists() or src_copy.stat().st_size != SOURCE.stat().st_size:
        shutil.copyfile(SOURCE, src_copy)
    hair = decode_rgb(OUTPUTS / "hair_character_mask.mp4")[..., 0]
    cond = 255 - hair  # hair=0 unknown, everything else=1 known
    write_gray_video(STAGE / "source_mask.mp4", cond)
    (STAGE / "prompt.txt").write_text(f"{EDIT_PROMPT}@@{src_copy}\n", encoding="utf-8")
    print("[condition] hair-only mask written:", STAGE / "source_mask.mp4")


def condition_qa() -> None:
    """Replicate preprocessing up to Img_list_new for early/mid/late frames."""
    src = decode_rgb(SOURCE)
    mask = decode_rgb(STAGE / "source_mask.mp4")[..., 0]
    indices = np.arange(0, len(src), len(src) / 57).astype(int)
    frames = []
    rows = []
    for j in [8, 28, 48]:
        imgs = src[indices[j]].astype(np.float32) / 255.0
        imgs = (imgs - 0.5) / 0.5
        msk = mask[indices[j]].astype(np.float32) / 255.0
        msk = (msk - 0.5) / 0.5
        binary = (msk > 0).astype(np.float32)
        cond_rgb = np.clip((imgs * binary[:, :, None] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        frames.append((j, indices[j], binary))
        rows.append(
            (
                f"j={j} source F{indices[j]}",
                [
                    cv2.resize(src[indices[j]], (832, 464), interpolation=cv2.INTER_AREA),
                    cv2.resize(np.repeat((mask[indices[j]] * 255)[:, :, None].astype(np.uint8), 3, axis=2), (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(np.repeat((binary * 255)[:, :, None].astype(np.uint8), 3, axis=2), (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(cond_rgb, (832, 464), interpolation=cv2.INTER_NEAREST),
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l9_condition_qa.png",
        "G0-L9 condition QA: Source / Hair Mask / AnyMask Condition (hair=0) / Masked RGB Condition",
        rows,
        ["Source", "Hair Mask", "Condition Mask", "Masked RGB (only hair zero-filled)"],
        cell_w=340,
        cell_h=191,
    )
    # programmatic: only hair region is zero; face/neck/background keep RGB
    hair = decode_rgb(OUTPUTS / "hair_character_mask.mp4")[..., 0] > 127
    for j, si, binary in frames:
        cond = np.zeros((1078, 1918, 3), dtype=np.float32)
        imgs = (src[si].astype(np.float32) / 255.0 - 0.5) / 0.5
        cond = imgs * binary[:, :, None]
        unknown = binary == 0
        assert np.allclose(cond[unknown], 0.0, atol=1e-6), f"j={j} non-hair zeroed!"
        known = binary == 1
        assert np.allclose(cond[known], imgs[known], atol=1e-6), f"j={j} known RGB altered!"
        print(f"[condition-qa] j={j}: unknown_ratio={float(unknown.mean()):.4f} "
              f"hair_zero_fill_ok=True known_rgb_ok=True")


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
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
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


def run_generation() -> dict:
    out_dir = STAGE / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS / "g0_l9_anymask.log"
    gpu_path = OUTPUTS / "g0_l9_gpu.csv"
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
        "--prompt", str(STAGE / "prompt.txt"),
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
    out_name = "g0_l9_hair_only_red_edit.mp4"
    if ok:
        shutil.copyfile(produced, OUTPUTS / out_name)
    print(
        f"[run] exit={code} runtime_s={runtime} peak_vram_mib={peak_mem} "
        f"peak_util={peak_util} output={OUTPUTS / out_name if ok else 'MISSING'}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng0_l9 exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_mem} peak_util={peak_util} "
            f"memory_before_gib={mem_info['memory_current_before_gib']} "
            f"memory_after_gib={mem_info['memory_current_after_gib']} "
            f"memory_max_gib={mem_info['memory_max_gib']}\n"
        )
    if code != 0 or not ok:
        raise SystemExit(f"g0_l9 generation failed exit={code} ok={ok}")
    return {
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": peak_mem,
        "peak_util": peak_util,
        "memory": mem_info,
        "output_path": str(OUTPUTS / out_name),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-only", action="store_true")
    ap.add_argument("--no-generation", action="store_true")
    args = ap.parse_args()
    L9.mkdir(parents=True, exist_ok=True)
    build_mask()
    if args.mask_only:
        print("[run] --mask-only: mask gate passed, stopping.")
        return
    build_condition()
    condition_qa()
    if args.no_generation:
        print("[run] --no-generation: condition QA passed, stopping.")
        return
    run_generation()


if __name__ == "__main__":
    main()
