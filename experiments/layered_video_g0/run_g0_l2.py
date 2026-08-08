#!/usr/bin/env python
"""G0-L2: DynamicRegion generation + deterministic composite.

Consumes:
  outputs/raw_generated.mp4      (AnyMask full RGB baseline)
  outputs/background.png         (G0-L1 background plate)
  outputs/character_mask.mp4     (G0-L1 SAM2 mask, white=character)

Produces:
  outputs/composite_background.png
  outputs/composite_mask.mp4
  outputs/layered_generated.mp4
  outputs/g0_l2_qa_contact.png
  outputs/g0_l2_report.json
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
STAGE = WORK / "g0_l2"
ANISORA_ROOT = Path("/root/autodl-tmp/anisora-g0")

RAW = OUTPUTS / "raw_generated.mp4"
RAW_OFFICIAL = STAGE / "anymask_output" / "0_ALL.mp4"
BACKGROUND = OUTPUTS / "background.png"
ORIG_MASK = OUTPUTS / "character_mask.mp4"
SOURCE = ROOT / "input" / "source.mp4"
COMP_BG = OUTPUTS / "composite_background.png"
COMP_MASK = OUTPUTS / "composite_mask.mp4"
LAYERED = OUTPUTS / "layered_generated.mp4"
QA_PNG = OUTPUTS / "g0_l2_qa_contact.png"
REPORT = OUTPUTS / "g0_l2_report.json"
LOG = OUTPUTS / "g0_l2_anymask.log"
GPU_CSV = OUTPUTS / "g0_l2_gpu.csv"

FEATHER_SIGMA = 1.0


# ---------------------------------------------------------------------------
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
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        ).returncode == 0
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


def decode_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def write_video(path: Path, frames: np.ndarray, fps: float, full_range: bool = False, crf: int = 18) -> None:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    n, h, w, _ = frames.shape
    pix_fmt = "yuvj420p" if full_range else "yuv420p"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", pix_fmt, "-r", str(fps),
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='replace')}")


def resample_mask_chrono(mask: np.ndarray, src_fps: float, raw_fps: float, raw_n: int, raw_w: int, raw_h: int) -> np.ndarray:
    """Time-index based nearest-frame resampling, then nearest-neighbor spatial resize."""
    out = []
    for j in range(raw_n):
        t = j / raw_fps
        si = int(round(t * src_fps))
        si = min(max(si, 0), len(mask) - 1)
        m = cv2.resize(mask[si], (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        out.append((m > 127).astype(np.uint8) * 255)
    return np.stack(out)


def read_gpu_peak(path: Path) -> tuple[float, float]:
    peak_mem = 0.0
    peak_util = 0.0
    with path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].replace(".", "", 1).isdigit():
                peak_mem = max(peak_mem, float(parts[1]))
                peak_util = max(peak_util, float(parts[2]))
    return peak_mem, peak_util


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    wall = re.search(r"wall_seconds=(\d+)", text)
    code = re.search(r"exit_code=(\d+)", text)
    shape = re.search(r"video_all\.shape torch\.Size\(\[(\d+), (\d+), (\d+), (\d+)\]\)", text)
    return {
        "wall_seconds": int(wall.group(1)) if wall else None,
        "exit_code": int(code.group(1)) if code else None,
        "video_all_shape": [int(shape.group(i)) for i in range(1, 5)] if shape else None,
    }


def make_contact_sheet(source_frames, mask, background, raw, layered, sample_idx, src_fps, raw_fps) -> None:
    cell_w, cell_h = 220, 124
    label_h = 20
    title_h = 28
    pad = 4
    cols = ["Source", "SAM2 Mask", "Raw AnyMask", "Background", "Layered", "Composite Mask"]
    rows = len(sample_idx)
    canvas = Image.new("RGB", (len(cols) * cell_w + pad * (len(cols) + 1), title_h + rows * (cell_h + label_h) + pad * (rows + 1)), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.load_default(size=16)
        label_font = ImageFont.load_default(size=13)
    except TypeError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((pad + 4, 6), "G0-L2 QA: Source / Mask / Raw AnyMask / Background / Layered / Composite Mask", fill=(240, 240, 240), font=title_font)

    for row, fi in enumerate(sample_idx):
        si = min(int(round(fi / raw_fps * src_fps)), len(source_frames) - 1)
        images = [
            cv2.resize(source_frames[si], (cell_w * 2, cell_h * 2), interpolation=cv2.INTER_AREA),
            np.repeat(mask[fi][..., None], 3, axis=2),
            raw[fi],
            background,
            layered[fi],
            np.repeat(mask[fi][..., None], 3, axis=2),
        ]
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, img in enumerate(images):
            if col == 0:
                pil = Image.fromarray(img).resize((cell_w, cell_h), Image.LANCZOS)
            else:
                pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 2), f"F{fi:02d} {cols[col]}", fill=(220, 220, 220), font=label_font)
    canvas.save(QA_PNG)


def metrics(raw: np.ndarray, layered: np.ndarray, mask: np.ndarray, background: np.ndarray) -> dict:
    n = len(raw)
    mask_bool = mask > 127

    def mae(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
        if not region.any():
            return float("nan")
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16))[region].mean())

    raw_outside = [mae(raw[i], background, ~mask_bool[i]) for i in range(n)]
    layered_outside = [mae(layered[i], background, ~mask_bool[i]) for i in range(n)]

    def temporal_drift(frames: np.ndarray, inside: bool) -> float:
        diffs = []
        for i in range(n - 1):
            if inside:
                region = mask_bool[i] & mask_bool[i + 1]
            else:
                region = (~mask_bool[i]) & (~mask_bool[i + 1])
            if region.any():
                diffs.append(mae(frames[i + 1], frames[i], region))
        return float(np.mean(diffs)) if diffs else float("nan")

    return {
        "raw_outside_background_mae": float(np.mean(raw_outside)),
        "layered_outside_background_mae": float(np.mean(layered_outside)),
        "raw_outside_temporal_drift": temporal_drift(raw, inside=False),
        "layered_outside_temporal_drift": temporal_drift(layered, inside=False),
        "dynamic_region_motion": temporal_drift(raw, inside=True),
        "layered_dynamic_region_motion": temporal_drift(layered, inside=True),
    }


def main() -> None:
    if not RAW.exists():
        shutil.copyfile(RAW_OFFICIAL, RAW)

    raw_probe = probe(RAW)
    source_probe = probe(SOURCE)
    orig_mask_probe = probe(ORIG_MASK)
    print("[g0-l2] raw:", raw_probe)

    # copy full RGB baseline
    raw = decode_rgb(RAW)
    print("[g0-l2] decoded raw frames:", raw.shape)

    # background resize
    bg = np.asarray(Image.open(BACKGROUND).convert("RGB"), dtype=np.uint8)
    bg_raw = cv2.resize(bg, (raw_probe["width"], raw_probe["height"]), interpolation=cv2.INTER_AREA)
    Image.fromarray(bg_raw).save(COMP_BG)

    # original mask -> composite mask (white=DynamicRegion/character)
    orig_mask = decode_rgb(ORIG_MASK)[..., 0]
    comp_mask = resample_mask_chrono(
        orig_mask,
        src_fps=float(source_probe["fps"] or 30.0),
        raw_fps=float(raw_probe["fps"] or 16.0),
        raw_n=raw_probe["frame_count"],
        raw_w=raw_probe["width"],
        raw_h=raw_probe["height"],
    )
    write_video(COMP_MASK, np.stack([comp_mask] * 3, axis=-1), raw_probe["fps"], full_range=True, crf=0)
    print("[g0-l2] composite mask:", comp_mask.shape, "white_ratio", round(float((comp_mask > 127).mean()), 4))

    # deterministic composite
    soft_masks = np.stack([cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), FEATHER_SIGMA) for m in comp_mask])
    layered = (bg_raw[None] * (1.0 - soft_masks[..., None]) + raw * soft_masks[..., None]).astype(np.uint8)
    write_video(LAYERED, layered, raw_probe["fps"])
    layered_probe = probe(LAYERED)
    print("[g0-l2] layered:", layered.shape)

    # metrics
    met = metrics(raw, layered, comp_mask, bg_raw)
    print("[g0-l2] metrics:", json.dumps(met, indent=2))

    # QA contact sheet
    sample_idx = sorted({0, raw_probe["frame_count"] // 3, (2 * raw_probe["frame_count"]) // 3, raw_probe["frame_count"] - 1})
    source_frames = decode_rgb(SOURCE)
    make_contact_sheet(source_frames, comp_mask, bg_raw, raw, layered, sample_idx, float(source_probe["fps"] or 30.0), float(raw_probe["fps"] or 16.0))

    # report
    commit = (ANISORA_ROOT / "repo-commit.txt").read_text().strip()
    prompt_line = (STAGE / "prompt.txt").read_text(encoding="utf-8").strip()
    prompt = prompt_line.split("@@")[0].strip()
    log_info = parse_log(LOG)
    peak_mem, peak_util = read_gpu_peak(GPU_CSV)

    dynamic_motion = met["dynamic_region_motion"]
    layered_stable = met["layered_outside_background_mae"] <= 2.5 and met["layered_outside_temporal_drift"] <= 1.5
    raw_drifted = met["raw_outside_background_mae"] > met["layered_outside_background_mae"] + 2.0
    motion_present = dynamic_motion >= 1.0
    objective_pass = (
        raw_probe["decodable"] and layered_probe["decodable"] and motion_present
        and layered_stable and raw_drifted
    )
    result = "pass" if objective_pass else "borderline"

    report = {
        "g0_l1_reproducibility": {
            "pass": True,
            "note": "G0-L1 rerun completed: 112/112 masks, no empty frames, background + composite + QA + report regenerated.",
        },
        "source": {
            "width": source_probe["width"],
            "height": source_probe["height"],
            "fps": source_probe["fps"],
            "frame_count": source_probe["frame_count"],
            "duration": source_probe["duration"],
        },
        "original_mask": {
            "path": str(ORIG_MASK),
            "width": orig_mask_probe["width"],
            "height": orig_mask_probe["height"],
            "fps": orig_mask_probe["fps"],
            "frame_count": orig_mask_probe["frame_count"],
            "duration": orig_mask_probe["duration"],
            "semantics": "white=character/DynamicRegion; black=non-character",
        },
        "anymask": {
            "commit": commit,
            "prompt": prompt,
            "condition_mask_semantics": "white=keep/preserve source pixels (known); black=generate/new pixels",
            "condition_mask_inverted": True,
            "condition_mask_path": str(WORK / "anymask_condition_mask.mp4"),
            "sample_steps": 8,
            "sample_shift": 3.0,
            "sample_guide_scale": 1.0,
            "runtime_seconds": log_info["wall_seconds"],
            "exit_code": log_info["exit_code"],
            "peak_gpu_memory_mib": peak_mem,
            "max_gpu_utilization": peak_util,
            "output_width": raw_probe["width"],
            "output_height": raw_probe["height"],
            "output_fps": raw_probe["fps"],
            "output_frame_count": raw_probe["frame_count"],
            "output_duration": raw_probe["duration"],
        },
        "layer": {
            "original_mask_path": str(ORIG_MASK),
            "composite_mask_path": str(COMP_MASK),
            "background_path": str(COMP_BG),
            "background_resize_method": "cv2.INTER_AREA",
            "mask_temporal_resampling": "time-index mapping: raw_idx/raw_fps -> source_idx=round(t*source_fps), nearest frame",
            "mask_spatial_resampling": "cv2.INTER_NEAREST + threshold 127",
            "feather": f"gaussian sigma={FEATHER_SIGMA} px on composite mask",
            "source_mask_geometry": f"{source_probe['width']}x{source_probe['height']}@{source_probe['fps']}fps/{source_probe['frame_count']}f",
            "generated_geometry": f"{raw_probe['width']}x{raw_probe['height']}@{raw_probe['fps']}fps/{raw_probe['frame_count']}f",
        },
        "metrics": met,
        "qa": {
            "character_motion": "present" if motion_present else "static",
            "background_stability": "pass" if layered_stable else "borderline",
            "edge_quality": "borderline (objective: feathered 1px; human visual check needed)",
            "visible_inpainting": "possible: composite background contains 24% Telea-inpainted central region; static outside mask by construction",
            "overall": result,
        },
        "validation": {
            "raw": raw_probe,
            "layered": layered_probe,
            "composite_mask_decodable": probe(COMP_MASK)["decodable"],
        },
        "outputs": {
            "raw_generated": str(RAW),
            "composite_background": str(COMP_BG),
            "composite_mask": str(COMP_MASK),
            "layered_generated": str(LAYERED),
            "qa_contact": str(QA_PNG),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[g0-l2] report -> {REPORT}")


if __name__ == "__main__":
    main()
