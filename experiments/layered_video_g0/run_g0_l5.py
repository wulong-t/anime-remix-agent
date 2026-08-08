#!/usr/bin/env python
"""G0-L5: Masked Condition Fill Leakage diagnostic.

Stages:
  A: source.mp4 integrity (ffprobe/sha256/sampled frames, no generation)
  C: reproduce the exact VAE-pre-condition tensors (read_video, mask read,
     normalization, binary threshold, nearest resize) and stop before VAE.
  D: compare the condition fill color (inverse-normalized tensor 0) with the
     CFG2 raw gray artifact inside the person mask.
  Optional black-fill diagnostic: one AnyMask run whose only change is the
     masked unknown RGB fill (tensor 0 -> tensor -1, i.e. RGB black), executed
     from a private copied runtime under work/g0_l5/runtime (AniSora tracked
     files untouched).
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
L5 = WORK / "g0_l5"
SOURCE = WORK / "g0_l2" / "source.mp4"
BASE_MASK = WORK / "g0_l2" / "source_mask.mp4"
MASKS_NPY = WORK / "masks.npy"
CFG2_RAW = OUTPUTS / "hair_cfg_g2.mp4"
DYN_MASK_PATH = OUTPUTS / "composite_mask.mp4"

ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")
INDEX_DIR = ANISORA_ROOT / "Index-anisora" / "anisora_anymask"
CKPT = ANISORA_ROOT / "models" / "anymask"
WRAPPER = ANISORA_ROOT / "scripts" / "run-anymask-spa.py"

SRC_W, SRC_H = 1918, 1078
RAW_W, RAW_H = 832, 464
HAIR_ROI_SRC = (700, 50, 1200, 220)
NECK_ROI_SRC = (820, 300, 1180, 430)
FRONT_HAIR_RAW = (303, 21, 520, 60)
OUTER_HAIR_RAW = [(303, 21, 360, 94), (460, 21, 520, 94)]
NECK_RAW = (370, 130, 490, 210)
LOWER_HAIR_RAW = (340, 90, 480, 180)
REF_RGB = np.array([206.9, 165.0, 117.9], dtype=np.float32)
GRAY_SAT_THRESHOLD = 40.0
GRAY_MIN_VALUE = 60.0
NEAR_FILL_DIST = 25.0  # pre-registered L2 distance on 0-255 RGB
FILL_RGB = np.array([127.5, 127.5, 127.5], dtype=np.float32)
SAMPLE_J = [8, 28, 48]  # non-anchor model/output frame indices for previews
SOURCE_SAMPLE_FRAMES = [0, 28, 56, 84, 111]

C2_PROMPT = (
    "日系二维动画。一名浅金棕色短发的少女保持第一帧中的人物外观和身份，"
    "浅金棕色头发、发型、脸型、眼睛、服装和肤色在整个视频中保持一致。"
    "少女自然地轻轻眨眼，头部有非常小的动作，表情发生轻微变化，并有自然呼吸。"
    "固定机位，固定构图，背景保持稳定，人物运动幅度很小。"
)


# ---------------------------------------------------------------------------
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


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    info = json.loads(subprocess.check_output(cmd, text=True))
    stream = info["streams"][0]
    fmt = info["format"]

    def fps(s: str) -> float | None:
        try:
            num, den = s.split("/")
            return float(num) / float(den) if float(den) != 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    decodable = (
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        ).returncode
        == 0
    )
    return {
        "path": str(path),
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0/1")),
        "r_frame_rate": fps(stream.get("r_frame_rate", "0/1")),
        "frame_count": int(stream.get("nb_read_frames") or 0),
        "duration": float(fmt.get("duration", 0.0)),
        "decodable": decodable,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_video(path: Path, start: int = 0, end: int = 120, ss: float = 4.0):
    """Exact replica of the official read_video (fixed commit)."""
    vr = VideoReader(uri=str(path), height=-1, width=-1)
    actual_fps = vr.get_avg_fps()
    num_frames = int(ss * 16 + 1)
    indices = np.arange(start, end, (end - start) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = (frames - 0.5) / 0.5
    frames = frames.permute(0, 3, 1, 2)
    return frames, 16, indices, actual_fps


def model_dims() -> tuple[int, int]:
    """Exact replica of the h,w computation in image2video_any_mask1_spa.py."""
    vr = VideoReader(uri=str(SOURCE), height=-1, width=-1)
    h0, w0 = vr[0].shape[0], vr[0].shape[1]
    aspect_ratio = h0 / w0
    max_area = 832 * 480
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2]
    )
    return int(lat_h * vae_stride[1]), int(lat_w * vae_stride[2])


def inverse_normalize(x: torch.Tensor) -> np.ndarray:
    """Inverse of read_video normalization: x'=(x/255-0.5)/0.5 -> x=(x'*0.5+0.5)*255."""
    rgb = ((x * 0.5 + 0.5) * 255.0).clamp(0, 255).permute(0, 2, 3, 1)
    return rgb.numpy().astype(np.uint8)


def gray_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < GRAY_SAT_THRESHOLD) & (hsv[..., 2] > GRAY_MIN_VALUE)


def median_rgb(px: np.ndarray) -> list[float] | None:
    if len(px) == 0:
        return None
    return np.median(px, axis=0).round(1).tolist()


def near_fill_mask(rgb: np.ndarray) -> np.ndarray:
    d = np.sqrt(((rgb.astype(np.float32) - FILL_RGB[None, None, :]) ** 2).sum(axis=2))
    return d <= NEAR_FILL_DIST


# ---------------------------------------------------------------------------
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


def crop_resize(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
def stage_a() -> dict:
    """Source integrity audit (no generation)."""
    print("[stage-A] source integrity audit")
    info = {
        "path": str(SOURCE),
        "sha256": sha256(SOURCE),
        "ffprobe": probe(SOURCE),
    }
    frames = decode_rgb(SOURCE)
    masks = np.load(MASKS_NPY)  # (112,1078,1918) uint8 white=character
    char = masks > 127
    rows = []
    per_frame = {}
    midgray_all = []
    for fi in SOURCE_SAMPLE_FRAMES:
        f = frames[fi]
        x0, y0, x1, y1 = HAIR_ROI_SRC
        hair = f[y0:y1, x0:x1]
        hair_char = char[fi, y0:y1, x0:x1]
        hair_px = hair[hair_char].astype(np.float32)
        nx0, ny0, nx1, ny1 = NECK_ROI_SRC
        neck = f[ny0:ny1, nx0:nx1]
        neck_char = char[fi, ny0:ny1, nx0:nx1]
        neck_px = neck[neck_char].astype(np.float32)
        hsv = cv2.cvtColor(hair, cv2.COLOR_RGB2HSV)[hair_char].astype(np.float32)
        hair_rgb = hair_px.mean(axis=0) if len(hair_px) else np.zeros(3, np.float32)
        hair_sat = float(hsv[:, 1].mean()) if len(hsv) else 0.0
        hair_dist = float(np.linalg.norm(hair_rgb - REF_RGB)) if len(hair_px) else float("nan")
        neck_sat = float(cv2.cvtColor(neck, cv2.COLOR_RGB2HSV)[neck_char].astype(np.float32)[:, 1].mean()) if neck_char.any() else 0.0
        midgray = (np.abs(f.astype(np.int16) - 128) <= 10).all(axis=2) & char[fi]
        midgray_ratio = float(midgray.mean())
        midgray_all.append(midgray_ratio)
        per_frame[fi] = {
            "hair_mean_rgb": hair_rgb.round(1).tolist(),
            "hair_saturation": round(hair_sat, 1),
            "hair_distance_from_reference": round(hair_dist, 2),
            "neck_mean_rgb": neck_px.mean(axis=0).round(1).tolist() if len(neck_px) else None,
            "neck_saturation": round(neck_sat, 1),
            "neck_character_coverage": round(float(neck_char.mean()), 3),
            "midgray_ratio_inside_character": round(midgray_ratio, 4),
        }
        rows.append(
            (
                f"source F{fi}",
                [
                    cv2.resize(f, (RAW_W, RAW_H), interpolation=cv2.INTER_AREA),
                    cv2.resize(crop_resize(f, HAIR_ROI_SRC), (300, 180), interpolation=cv2.INTER_AREA),
                    cv2.resize(crop_resize(f, NECK_ROI_SRC), (300, 180), interpolation=cv2.INTER_AREA),
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l5_source_frames_qa.png",
        "G0-L5 Stage A: source.mp4 sampled frames (full / hair ROI / neck ROI)",
        rows,
        ["Source full", "Hair ROI (700,50,1200,220)", "Neck ROI (820,300,1180,430)"],
        cell_w=340,
        cell_h=191,
    )
    hair_sats = [per_frame[f]["hair_saturation"] for f in SOURCE_SAMPLE_FRAMES]
    neck_sats = [per_frame[f]["neck_saturation"] for f in SOURCE_SAMPLE_FRAMES]
    info["per_frame"] = per_frame
    info["source_colors_normal"] = bool(min(hair_sats) >= 70 and min(neck_sats) >= 40)
    info["source_gray_artifact_present"] = bool(max(midgray_all) >= 0.03)
    print("[stage-A]", json.dumps(info, ensure_ascii=False, indent=2))
    return info


def stage_c() -> dict:
    """Reproduce exact VAE-pre-condition tensors."""
    print("[stage-C] reproduce VAE-pre-condition tensors")
    Img_list, _, indices, actual_fps = read_video(SOURCE, 0, 112, 3.5)
    Img_mask_list, _, mask_indices, mask_fps = read_video(BASE_MASK, 0, 112, 3.5)
    binary_mask = torch.where(
        Img_mask_list > 0, torch.ones_like(Img_mask_list), torch.zeros_like(Img_mask_list)
    )
    binary_mask[0, :, :, :] = 1
    h, w = model_dims()
    Img_list_r = F.interpolate(Img_list, size=(h, w), mode="nearest")
    binary_mask_r = F.interpolate(binary_mask, size=(h, w), mode="nearest")
    Img_list_new = Img_list_r * binary_mask_r
    assert tuple(Img_list_r.shape[2:]) == (h, w), (Img_list_r.shape, h, w)
    assert tuple(binary_mask_r.shape[2:]) == (h, w)
    assert len(indices) == 57 and indices[28] == 55
    assert (indices == mask_indices).all()

    out = {
        "input_numeric_range": [float(Img_list.min()), float(Img_list.max())],
        "input_mean": round(float(Img_list.mean()), 5),
        "normalization": "uint8 /255 -> [0,1]; (x-0.5)/0.5 -> [-1,1]",
        "model_domain_hw": [h, w],
        "indices": indices.tolist(),
        "mask_fps": float(mask_fps),
        "source_fps": float(actual_fps),
        "binary_mask_semantics": "1=keep source RGB (background); 0=unknown (character/DynamicRegion)",
        "frames": {},
    }
    for j in SAMPLE_J:
        person = binary_mask_r[j] == 0  # [3,h,w], all channels identical
        known = ~person
        orig_px = Img_list_r[j][person]
        new_px = Img_list_new[j][person]
        known_px = Img_list_new[j][known]
        unknown_exactly_zero = bool((Img_list_new[j][person] == 0.0).all())
        out["frames"][str(j)] = {
            "source_frame_index": int(indices[j]),
            "person_pixel_count": int(person[0].sum()),
            "Img_list_person_min": round(float(orig_px.min()), 5),
            "Img_list_person_max": round(float(orig_px.max()), 5),
            "Img_list_person_mean": round(float(orig_px.mean()), 5),
            "Img_list_person_median": round(float(orig_px.median()), 5),
            "Img_list_new_person_min": round(float(new_px.min()), 5),
            "Img_list_new_person_max": round(float(new_px.max()), 5),
            "Img_list_new_person_mean": round(float(new_px.mean()), 5),
            "Img_list_new_person_median": round(float(new_px.median()), 5),
            "Img_list_new_known_min": round(float(known_px.min()), 5),
            "Img_list_new_known_max": round(float(known_px.max()), 5),
            "Img_list_new_known_mean": round(float(known_px.mean()), 5),
            "unknown_exactly_tensor_zero": unknown_exactly_zero,
        }
    out["unknown_fill_tensor_value"] = 0.0
    fill_rgb = inverse_normalize(torch.zeros(1, 3, 1, 1))
    out["unknown_fill_rgb_uint8"] = fill_rgb[0, 0, 0].astype(int).tolist()
    out["unknown_fill_rgb_exact"] = [127.5, 127.5, 127.5]
    out["zero_inverse_midgray"] = bool(
        all(abs(float(v) - 127.5) <= 0.5 for v in out["unknown_fill_rgb_uint8"])
    )

    # Preview: Source / Condition Mask / Masked Condition inverse / CFG2 Raw
    raw = decode_rgb(CFG2_RAW)
    source_frames = decode_rgb(SOURCE)
    cond_inv = inverse_normalize(Img_list_new)
    rows = []
    for j in SAMPLE_J:
        si = int(indices[j])
        mask_gray = np.repeat((binary_mask_r[j, 0].numpy() * 255)[:, :, None].astype(np.uint8), 3, axis=2)
        rows.append(
            (
                f"j={j} source F{si}",
                [
                    cv2.resize(source_frames[si], (w, h), interpolation=cv2.INTER_AREA),
                    mask_gray,
                    cond_inv[j],
                    raw[j],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l5_masked_condition_preview.png",
        "G0-L5 Stage C: Source / Condition Mask / Masked Condition inverse-normalized / CFG2 Raw",
        rows,
        ["Source Frame", "Condition Mask (white=keep)", "Img_list_new inverse RGB", "CFG2 Raw Output"],
        cell_w=340,
        cell_h=191,
    )
    print("[stage-C]", json.dumps(out, ensure_ascii=False, indent=2))
    return out


def stage_d() -> dict:
    """Compare condition fill color with CFG2 raw gray artifact inside person."""
    print("[stage-D] fill color vs raw gray artifact")
    raw = decode_rgb(CFG2_RAW)
    dyn = decode_rgb(DYN_MASK_PATH)[..., 0] > 127
    assert len(raw) == len(dyn) == 56
    out = {"condition_fill_rgb": FILL_RGB.tolist(), "frames": {}}

    def roi_metrics(rgb: np.ndarray, person: np.ndarray, box: tuple[int, int, int, int]):
        x0, y0, x1, y1 = box
        p = person[y0:y1, x0:x1]
        reg = rgb[y0:y1, x0:x1]
        g = gray_mask(reg) & p
        nf = near_fill_mask(reg) & p
        gray_px = reg[g].astype(np.float32)
        return {
            "character_coverage": round(float(p.mean()), 3),
            "gray_pixel_count": int(g.sum()),
            "gray_median_rgb": median_rgb(gray_px),
            "gray_mean_rgb": gray_px.mean(axis=0).round(1).tolist() if len(gray_px) else None,
            "near_fill_ratio": round(float(nf.sum() / max(p.sum(), 1)), 4),
            "fill_color_distance": (
                round(float(np.linalg.norm(np.median(gray_px, axis=0) - FILL_RGB)), 2)
                if len(gray_px)
                else None
            ),
        }

    for j in SAMPLE_J:
        rgb = raw[j]
        person = dyn[j]
        g = gray_mask(rgb) & person
        nf = near_fill_mask(rgb) & person
        nf_out = near_fill_mask(rgb) & ~person
        gray_px = rgb[g].astype(np.float32)
        out["frames"][str(j)] = {
            "raw_gray_region_median_rgb": median_rgb(gray_px),
            "raw_gray_region_mean_rgb": gray_px.mean(axis=0).round(1).tolist() if len(gray_px) else None,
            "raw_gray_pixel_count": int(g.sum()),
            "fill_color_distance": (
                round(float(np.linalg.norm(np.median(gray_px, axis=0) - FILL_RGB)), 2)
                if len(gray_px)
                else None
            ),
            "pixels_near_fill_ratio_person": round(float(nf.sum() / max(person.sum(), 1)), 4),
            "pixels_near_fill_ratio_outside": round(float(nf_out.sum() / max((~person).sum(), 1)), 4),
            "near_fill_spatial_specificity": (
                round((nf.sum() / max(person.sum(), 1)) / max(nf_out.sum() / max((~person).sum(), 1), 1e-9), 2)
            ),
            "roi_front_hair": roi_metrics(rgb, person, FRONT_HAIR_RAW),
            "roi_outer_hair_left": roi_metrics(rgb, person, OUTER_HAIR_RAW[0]),
            "roi_outer_hair_right": roi_metrics(rgb, person, OUTER_HAIR_RAW[1]),
            "roi_neck": roi_metrics(rgb, person, NECK_RAW),
            "roi_lower_hair": roi_metrics(rgb, person, LOWER_HAIR_RAW),
        }

    # Similarity QA: Source / Masked Condition / CFG2 Raw / near-fill highlight / heatmap
    Img_list, _, indices, _ = read_video(SOURCE, 0, 112, 3.5)
    Img_mask_list, _, _, _ = read_video(BASE_MASK, 0, 112, 3.5)
    binary_mask = torch.where(Img_mask_list > 0, torch.ones_like(Img_mask_list), torch.zeros_like(Img_mask_list))
    binary_mask[0, :, :, :] = 1
    h, w = model_dims()
    Img_list_new = F.interpolate(Img_list, size=(h, w), mode="nearest") * F.interpolate(binary_mask, size=(h, w), mode="nearest")
    cond_inv = inverse_normalize(Img_list_new)
    source_frames = decode_rgb(SOURCE)
    rows = []
    for j in SAMPLE_J:
        si = int(indices[j])
        rgb = raw[j]
        person = dyn[j]
        nf = near_fill_mask(rgb) & person
        highlight = rgb.copy()
        highlight[nf] = (highlight[nf] * 0.45 + np.array([255, 0, 0]) * 0.55).astype(np.uint8)
        dist = np.sqrt(((rgb.astype(np.float32) - FILL_RGB[None, None, :]) ** 2).sum(axis=2)).clip(0, 80)
        heat = np.zeros_like(rgb)
        d8 = np.zeros((h, w), np.uint8)
        d8[person] = dist[person].astype(np.uint8)
        jet = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        heat[person] = jet[person]
        heat[~person] = 40
        rows.append(
            (
                f"j={j} source F{si}",
                [
                    cv2.resize(source_frames[si], (w, h), interpolation=cv2.INTER_AREA),
                    cond_inv[j],
                    rgb,
                    highlight,
                    heat,
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l5_fill_similarity_qa.png",
        "G0-L5 Stage D: fill-color similarity (near-fill <=25 RGB inside person in red; heatmap=dist to fill 0-80)",
        rows,
        ["Source", "Masked Condition", "CFG2 Raw", "Near-fill highlight", "Fill-distance heatmap"],
        cell_w=300,
        cell_h=169,
    )
    print("[stage-D]", json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
def decide(a: dict, c: dict, d: dict) -> dict:
    """Pre-registered stop gates / support criteria."""
    reasons = []
    if not a["source_colors_normal"]:
        return {"result": "source_issue", "reasons": ["source colors abnormal at sampled frames"]}
    if not all(c["frames"][str(j)]["unknown_exactly_tensor_zero"] for j in SAMPLE_J):
        reasons.append("Img_list_new unknown region is not exactly tensor 0")
    if not c["zero_inverse_midgray"]:
        reasons.append("inverse of tensor 0 is not mid-gray")
    if any(d["frames"][str(j)]["fill_color_distance"] is None or d["frames"][str(j)]["fill_color_distance"] > 25 for j in SAMPLE_J):
        reasons.append("raw gray median color distance to fill > 25")
    if any(d["frames"][str(j)]["pixels_near_fill_ratio_person"] < 0.20 for j in SAMPLE_J):
        reasons.append("near-fill pixel ratio inside person < 0.20")
    weak_spatial = any(
        d["frames"][str(j)]["near_fill_spatial_specificity"] < 3.0 for j in SAMPLE_J
    )
    if reasons:
        return {
            "result": "model_generation",
            "reasons": reasons,
            "zero_fill_leakage_supported": False,
        }
    if weak_spatial:
        return {
            "result": "mixed",
            "reasons": ["fill distance/ratio pass but spatial specificity < 3x at some frames"],
            "zero_fill_leakage_supported": True,
        }
    return {
        "result": "fill_leakage",
        "reasons": ["source normal; unknown region exact tensor 0; inverse mid-gray; raw gray close to fill; strong spatial specificity"],
        "zero_fill_leakage_supported": True,
    }


# ---------------------------------------------------------------------------
def build_black_fill_runtime() -> Path:
    rt = L5 / "runtime"
    if rt.exists():
        shutil.rmtree(rt)
    rt.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX_DIR / "generate-pi-i2v-any-mask1_spa.py", rt / "generate-pi-i2v-any-mask1_spa.py")
    shutil.copytree(
        INDEX_DIR / "wan",
        rt / "wan",
        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
    )
    p = rt / "wan" / "image2video_any_mask1_spa.py"
    s = p.read_text(encoding="utf-8")
    old = "        Img_list_new = Img_list.to(self.device) * binary_mask  \n"
    new = "        Img_list_new = Img_list.to(self.device) * binary_mask + (-1.0) * (1.0 - binary_mask)  \n"
    assert s.count(old) == 1, f"pattern count={s.count(old)}"
    p.write_text(s.replace(old, new), encoding="utf-8")
    (rt / "run_black_fill.py").write_text(
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
    return rt


def run_black_fill() -> dict:
    stage_dir = L5 / "D_black_fill"
    stage_dir.mkdir(parents=True, exist_ok=True)
    src_copy = stage_dir / "source.mp4"
    if not src_copy.exists() or src_copy.stat().st_size != SOURCE.stat().st_size:
        shutil.copyfile(SOURCE, src_copy)
    mask_copy = stage_dir / "source_mask.mp4"
    if not mask_copy.exists() or mask_copy.stat().st_size != BASE_MASK.stat().st_size:
        shutil.copyfile(BASE_MASK, mask_copy)
    (stage_dir / "prompt.txt").write_text(f"{C2_PROMPT}@@{src_copy}\n", encoding="utf-8")
    out_dir = stage_dir / "anymask_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    rt = build_black_fill_runtime()
    log_path = OUTPUTS / "g0_l5_black_fill_anymask.log"
    gpu_path = OUTPUTS / "g0_l5_black_fill_gpu.csv"
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
        str(ANISORA_ROOT / ".venv" / "bin" / "python"), str(rt / "run_black_fill.py"),
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
        "--prompt", str(stage_dir / "prompt.txt"),
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
    if ok:
        shutil.copyfile(produced, OUTPUTS / "g0_l5_black_fill.mp4")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng0_l5_black_fill exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_mem} peak_util={peak_util} "
            f"memory_before_gib={mem_info['memory_current_before_gib']} "
            f"memory_after_gib={mem_info['memory_current_after_gib']} "
            f"memory_max_gib={mem_info['memory_max_gib']}\n"
        )
    print(
        f"[black-fill] exit={code} runtime_s={runtime} peak_vram_mib={peak_mem} "
        f"output={OUTPUTS / 'g0_l5_black_fill.mp4' if ok else 'MISSING'}"
    )
    if code != 0 or not ok:
        raise SystemExit(f"black-fill failed exit={code} ok={ok}")
    return {
        "executed": True,
        "fill_rgb": [0.0, 0.0, 0.0],
        "runtime_seconds": runtime,
        "peak_vram_mib": peak_mem,
        "exit_code": code,
        "memory": mem_info,
        "output_path": str(OUTPUTS / "g0_l5_black_fill.mp4"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-black-fill", action="store_true", help="stop after Stage D decision")
    args = ap.parse_args()
    L5.mkdir(parents=True, exist_ok=True)
    a = stage_a()
    c = stage_c()
    d = stage_d()
    decision = decide(a, c, d)
    print("[decision]", json.dumps(decision, ensure_ascii=False, indent=2))
    if decision["result"] == "source_issue":
        print("STOP: source issue; no further stages.")
        return
    if not decision["zero_fill_leakage_supported"]:
        print("STOP: zero-fill leakage hypothesis NOT SUPPORTED; no new generation.")
        return
    if args.no_black_fill:
        print("--no-black-fill: diagnostic generation skipped.")
        return
    run_black_fill()


if __name__ == "__main__":
    main()
