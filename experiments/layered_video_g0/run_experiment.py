#!/usr/bin/env python
"""G0-L minimal two-layer video decomposition experiment.

Pipeline:
  source.mp4
    -> per-frame Character/DynamicRegion temporal mask (SAM 2.1)
    -> Character RGB layer (Source x Mask, black background)
    -> Background Plate (cross-frame median recovery + cv2.inpaint for
       never-visible pixels)
    -> deterministic recomposite: Final = Background*(1-Mask) + Source*Mask
    -> preview_composite.mp4, QA contact sheet, report.json

Scope: this experiment deliberately avoids product integration. It only
writes under experiments/layered_video_g0/ and never touches src/anime_remix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# HF direct access is blocked in this environment; use the mirror by default.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "source.mp4"
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
FRAMES_DIR = WORK / "frames"

MODEL_ID = "facebook/sam2.1-hiera-base-plus"
OUTPUT_FPS = 30.0
MIN_VISIBLE_FRAMES = 5
SAMPLE_FRAMES = [0, 28, 56, 84, 111]


# ---------------------------------------------------------------------------
# media helpers
# ---------------------------------------------------------------------------
def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    info = json.loads(subprocess.check_output(cmd, text=True))
    stream = info["streams"][0]
    fmt = info["format"]

    def parse_fps(s: str) -> float | None:
        try:
            num, den = s.split("/")
            return float(num) / float(den) if float(den) != 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "r_frame_rate": parse_fps(stream.get("r_frame_rate", "0/1")),
        "avg_frame_rate": parse_fps(stream.get("avg_frame_rate", "0/1")),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "duration": float(fmt["duration"]),
    }


def decode_frames(path: Path, width: int, height: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    if frames.size == 0:
        raise RuntimeError(f"could not decode any frames from {path}")
    return frames


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    n, h, w, _ = frames.shape
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    _, err = proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='replace')}")


def probe_output(path: Path) -> dict:
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

    def parse_fps(s: str) -> float | None:
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
        "fps": parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0/1")),
        "frame_count": int(stream.get("nb_read_frames") or 0),
        "duration": float(fmt.get("duration", 0.0)),
        "decodable": decodable,
    }


def write_frames_dir(frames: np.ndarray) -> Path:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(
            str(FRAMES_DIR / f"{i:06d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    return FRAMES_DIR


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
def select_initial_mask(first_frame: np.ndarray) -> np.ndarray:
    """Run SAM 2 automatic mask generation on frame 0 and pick the central
    character-sized mask. Falls back to a central-point mask if none matches."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    model = build_sam2_hf(MODEL_ID, device="cuda")
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=800,
    )
    anns = generator.generate(first_frame)
    del generator, model
    torch.cuda.empty_cache()

    h, w, _ = first_frame.shape
    candidates = []
    for ann in anns:
        mask = ann["segmentation"]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        area = float(mask.mean())
        bx, by, bw, bh = ann["bbox"]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        central = (0.30 * w < cx < 0.70 * w) and (0.30 * h < cy < 0.70 * h)
        if not central:
            continue
        if not (0.01 < area < 0.65):
            continue
        candidates.append((area, ann["predicted_iou"], ann["stability_score"], mask))
    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1] * c[2]), reverse=True)
        return np.ascontiguousarray(candidates[0][3], dtype=bool)
    # conservative fallback: central point prompt region
    y, x = int(h * 0.5), int(w * 0.5)
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y - 80) : y + 80, max(0, x - 60) : x + 60] = True
    return mask


def run_sam2_video(frames: np.ndarray) -> list[np.ndarray]:
    from sam2.build_sam import build_sam2_video_predictor_hf

    write_frames_dir(frames)
    initial_mask = select_initial_mask(frames[0])
    print(f"[seg] initial mask ratio: {initial_mask.mean():.4f}")

    predictor = build_sam2_video_predictor_hf(MODEL_ID, device="cuda")
    inference_state = predictor.init_state(
        video_path=str(FRAMES_DIR),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )
    predictor.add_new_mask(inference_state, frame_idx=0, obj_id=1, mask=initial_mask)

    raw_masks: dict[int, np.ndarray] = {}
    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
        inference_state, start_frame_idx=0
    ):
        pred_mask = video_res_masks[0]
        if hasattr(pred_mask, "detach"):
            pred_mask = pred_mask.detach().cpu().numpy()
        if pred_mask.dtype != bool:
            pred_mask = pred_mask > 0.0
        pred_mask = np.squeeze(pred_mask)
        raw_masks[int(frame_idx)] = np.ascontiguousarray(pred_mask)
    del predictor, inference_state
    torch.cuda.empty_cache()

    if len(raw_masks) != len(frames):
        raise RuntimeError(
            f"SAM2 produced {len(raw_masks)} masks for {len(frames)} frames"
        )
    return [raw_masks[i] for i in range(len(frames))]


def postprocess_masks(raw: list[np.ndarray]) -> np.ndarray:
    """Morphological cleanup + largest-character component + 3-frame median."""
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    processed = []
    prev: np.ndarray | None = None
    for m in raw:
        m = np.asarray(m, dtype=bool).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_kernel)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            if prev is not None:
                m = prev.copy()
            else:
                m = np.zeros_like(m)
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas)) + 1
            lx, ly, lw, lh, la = stats[largest]
            keep = np.zeros_like(labels, dtype=bool)
            for i in range(1, n):
                ix, iy, iw, ih, ia = stats[i]
                # keep significant components whose centroid is inside the
                # largest component's bounding box (single-character scope)
                if ia >= max(200, 0.005 * la) and (
                    lx <= cents[i][0] <= lx + lw and ly <= cents[i][1] <= ly + lh
                ):
                    keep |= labels == i
            m = (keep.astype(np.uint8)) * 255
            if m.sum() == 0:
                m = (labels == largest).astype(np.uint8) * 255
        processed.append(m)
        prev = m

    # 3-frame temporal median (simple, explainable smoothing)
    smoothed = []
    for i in range(len(processed)):
        lo, hi = max(0, i - 1), min(len(processed) - 1, i + 1)
        stack = np.stack([processed[j] for j in range(lo, hi + 1)], axis=0)
        smoothed.append(np.median(stack, axis=0).astype(np.uint8))
    return np.stack(smoothed, axis=0)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------
def make_character_layer(frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.where(masks[..., None] > 0, frames, 0).astype(np.uint8)


def make_background(
    frames: np.ndarray, masks: np.ndarray, min_visible: int = MIN_VISIBLE_FRAMES
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n, h, w, _ = frames.shape
    visible_counts = np.sum(masks == 0, axis=0)
    hole = visible_counts < min_visible
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    block = 96
    for y0 in range(0, h, block):
        y1 = min(y0 + block, h)
        vals = np.stack(
            [
                np.where(
                    mask[y0:y1, :, None] == 0,
                    frame[y0:y1].astype(np.float32),
                    np.nan,
                )
                for mask, frame in zip(masks, frames)
            ],
            axis=0,
        )
        med = np.nanmedian(vals, axis=0)
        bg[y0:y1] = np.nan_to_num(med, nan=0.0).astype(np.uint8)

    visible_ratio = float(1.0 - hole.mean())
    inpainted_ratio = float(hole.mean())

    # deterministic cv2.inpaint (Telea) for never-visible pixels; run a few
    # passes with an expanding hole mask to propagate texture into large holes.
    hole_u8 = (hole.astype(np.uint8)) * 255
    inpaint_mask = hole_u8.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(3):
        bg = cv2.inpaint(bg, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
    return bg, hole, visible_ratio, inpainted_ratio


def make_composite(
    frames: np.ndarray, masks: np.ndarray, background: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Feathered deterministic recomposite.

    soft = Gaussian-blurred hard mask (sigma ~1.2 px). Inside the mask the
    source is kept; outside it the background is kept; only a ~2-3 px band is
    feathered to hide hard-mask aliasing.
    """
    composites = []
    soft_masks = []
    for i in range(len(frames)):
        hard = (masks[i].astype(np.float32) / 255.0)
        soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=1.2)
        comp = (
            background * (1.0 - soft[..., None])
            + frames[i] * soft[..., None]
        ).astype(np.uint8)
        composites.append(comp)
        soft_masks.append(soft)
    return np.stack(composites, axis=0), np.stack(soft_masks, axis=0)


# ---------------------------------------------------------------------------
# QA / report
# ---------------------------------------------------------------------------
def make_contact_sheet(
    frames: np.ndarray,
    masks: np.ndarray,
    background: np.ndarray,
    character: np.ndarray,
    composites: np.ndarray,
    sample_frames: list[int],
) -> None:
    cell_w, cell_h = 320, 180
    label_h = 24
    title_h = 34
    pad = 6
    cols = ["Original", "Mask", "Background", "Character", "Composite"]
    rows = len(sample_frames)
    canvas = Image.new(
        "RGB", (len(cols) * cell_w + pad * (len(cols) + 1), title_h + rows * (cell_h + label_h) + pad * (rows + 1)), (18, 18, 22)
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.load_default(size=20)
        label_font = ImageFont.load_default(size=16)
    except TypeError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((pad + 4, 8), "G0-L preview QA: Original / Mask / Background / Character / Composite", fill=(240, 240, 240), font=title_font)

    for row, fi in enumerate(sample_frames):
        images = [
            frames[fi],
            np.repeat(masks[fi][..., None], 3, axis=2),
            background,
            character[fi],
            composites[fi],
        ]
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, img in enumerate(images):
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 6, y0 + 2), f"F{fi:03d} {cols[col]}", fill=(220, 220, 220), font=label_font)
    canvas.save(OUTPUTS / "qa_contact.png")


def compute_metrics(
    frames: np.ndarray,
    masks: np.ndarray,
    background: np.ndarray,
    character: np.ndarray,
    composites: np.ndarray,
    soft_masks: np.ndarray,
    hole: np.ndarray,
) -> dict:
    mask_ratios = (masks > 0).mean(axis=(1, 2))
    hard = masks > 0
    edge = (soft_masks > 0.02) & (soft_masks < 0.98)

    # inside the hard mask (excluding feathered edge): composite should equal source
    core = np.stack([cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0 for m in hard], axis=0)
    if core.any():
        diff_inside = float(np.abs(composites.astype(np.int16) - frames.astype(np.int16))[core].mean())
    else:
        diff_inside = None

    # strictly outside hard mask and outside feather band: composite should equal background
    strict_out = ~hard & ~edge
    if strict_out.any():
        diff_outside = float(np.abs(composites.astype(np.int16) - background[None].astype(np.int16))[strict_out].mean())
    else:
        diff_outside = None

    return {
        "mask_average_ratio": float(mask_ratios.mean()),
        "mask_ratio_min": float(mask_ratios.min()),
        "mask_ratio_max": float(mask_ratios.max()),
        "edge_feather_ratio": float(edge.mean()),
        "composite_vs_source_inside_hard_mask_mae": diff_inside,
        "composite_vs_background_strict_outside_mae": diff_outside,
        "background_visible_pixel_ratio": float(1.0 - hole.mean()),
        "background_inpainted_pixel_ratio": float(hole.mean()),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print("[probe] probing source.mp4 ...")
    probe = probe_video(INPUT)
    print(" ", json.dumps(probe, ensure_ascii=False))

    frames = decode_frames(INPUT, probe["width"], probe["height"])
    actual_fps = len(frames) / probe["duration"]
    print(f"[probe] decoded frames: {len(frames)}  actual_fps: {actual_fps:.4f}")
    if probe["nb_frames"] and len(frames) != probe["nb_frames"]:
        print(f"[probe] WARNING: decoded count {len(frames)} != ffprobe nb_frames {probe['nb_frames']}")

    print("[seg] running SAM2.1 video tracking ...")
    raw_masks = run_sam2_video(frames)
    masks = postprocess_masks(raw_masks)
    np.save(WORK / "masks.npy", masks)
    _mask_ratios = (masks > 0).mean(axis=(1, 2))
    print(f"[seg] per-frame mask ratio mean={_mask_ratios.mean():.4f} min={_mask_ratios.min():.4f} max={_mask_ratios.max():.4f} empty_frames={int((_mask_ratios == 0).sum())}")
    print(f"[seg] mask mean ratio {masks.mean()/255:.4f}  min {masks.reshape(len(masks),-1).min(axis=1).mean()/255:.4f}")

    mask_video = np.stack([masks] * 3, axis=-1)
    mask_path = OUTPUTS / "character_mask.mp4"
    print("[out] writing character_mask.mp4 ...")
    write_video(mask_path, mask_video, OUTPUT_FPS)

    print("[bg] recovering background plate ...")
    background, hole, visible_ratio, inpainted_ratio = make_background(frames, masks)
    bg_path = OUTPUTS / "background.png"
    Image.fromarray(background).save(bg_path)
    print(f"[bg] visible pixels {visible_ratio:.4f}, inpainted {inpainted_ratio:.4f}")

    print("[layer] writing character_rgb.mp4 ...")
    character = make_character_layer(frames, masks)
    char_path = OUTPUTS / "character_rgb.mp4"
    write_video(char_path, character, OUTPUT_FPS)

    print("[comp] writing preview_composite.mp4 ...")
    composites, soft_masks = make_composite(frames, masks, background)
    comp_path = OUTPUTS / "preview_composite.mp4"
    write_video(comp_path, composites, OUTPUT_FPS)

    print("[qa] building contact sheet and metrics ...")
    make_contact_sheet(frames, masks, background, character, composites, SAMPLE_FRAMES)
    metrics = compute_metrics(frames, masks, background, character, composites, soft_masks, hole)

    # debug PNG evidence
    for fi in SAMPLE_FRAMES:
        Image.fromarray(mask_video[fi]).save(OUTPUTS / f"debug_mask_{fi:03d}.png")

    print("[validate] probing outputs ...")
    validations = {
        "mask": probe_output(mask_path),
        "character": probe_output(char_path),
        "composite": probe_output(comp_path),
        "input": probe,
    }

    report = {
        "experiment": "G0-L minimal two-layer video decomposition",
        "input": {
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe["r_frame_rate"],
            "frame_count": len(frames),
            "duration": probe["duration"],
            "codec": probe["codec"],
        },
        "segmentation": {
            "model": f"SAM2.1 {MODEL_ID} (video mask propagation, frame-0 mask prompt)",
            "device": "cuda:0",
            "success": True,
            "average_mask_ratio": metrics["mask_average_ratio"],
            "mask_ratio_min": metrics["mask_ratio_min"],
            "mask_ratio_max": metrics["mask_ratio_max"],
            "postprocessing": {
                "morphological_close_kernel": "9x9 ellipse",
                "morphological_open_kernel": "5x5 ellipse",
                "largest_character_component": True,
                "temporal_smoothing": "3-frame median",
                "composite_feather": "gaussian sigma=1.2 px on hard mask",
            },
            "known_issues": [
                "SAM2 compiled post-processing (_C) unavailable; binary masks still generated and post-processed in numpy/OpenCV.",
                "Bottom yellow foreground is intentionally classified as non-character per G0-L scope.",
            ],
        },
        "background": {
            "method": "per-pixel temporal median of frames where the character mask is absent, then cv2.inpaint (Telea) for never-visible pixels",
            "used_cross_frame_recovery": True,
            "used_inpainting": True,
            "visible_pixel_ratio": metrics["background_visible_pixel_ratio"],
            "inpainted_pixel_ratio": metrics["background_inpainted_pixel_ratio"],
            "limitation": "Pixels hidden in every frame are inpainted estimates, not recovered true background.",
        },
        "outputs": {
            "mask_path": str(mask_path),
            "character_path": str(char_path),
            "background_path": str(bg_path),
            "composite_path": str(comp_path),
            "qa_contact_path": str(OUTPUTS / "qa_contact.png"),
        },
        "metrics": metrics,
        "validation": {
            "output_decodable": all(v["decodable"] for v in validations.values() if isinstance(v, dict) and "decodable" in v),
            "output_frame_count": {
                "mask": validations["mask"]["frame_count"],
                "character": validations["character"]["frame_count"],
                "composite": validations["composite"]["frame_count"],
            },
            "output_fps": {
                "mask": validations["mask"]["fps"],
                "character": validations["character"]["fps"],
                "composite": validations["composite"]["fps"],
            },
            "output_duration": {
                "mask": validations["mask"]["duration"],
                "character": validations["character"]["duration"],
                "composite": validations["composite"]["duration"],
            },
            "input_frames_decoded": len(frames),
        },
        "runtime_seconds": round(time.time() - t0, 2),
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }
    report_path = OUTPUTS / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] report -> {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
