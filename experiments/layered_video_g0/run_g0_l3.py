#!/usr/bin/env python
"""G0-L3: Generated Content Mask minimal diagnostic.

Hypothesis: the gray 'shell/hood' around the CFG2 character comes from using
the source editable mask directly as the final composite mask. This script:
  1) aligns source mask to CFG2 raw geometry,
  2) runs SAM2 on the CFG2 raw to get the actually-generated character mask,
  3) compares masks and gray-artifact overlap,
  4) produces old (source-mask) and new (generated-mask) composites,
  5) writes QA sheets and g0_l3_report.json.

No AnyMask regeneration is performed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from run_experiment import MODEL_ID, postprocess_masks, select_initial_mask
from run_g0_l2 import probe

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
L3 = WORK / "g0_l3"
FRAMES_DIR = L3 / "frames"

RAW = OUTPUTS / "hair_cfg_g2.mp4"
SOURCE_MASK = OUTPUTS / "character_mask.mp4"
BG = OUTPUTS / "composite_background.png"
SOURCE_VIDEO = ROOT / "input" / "source.mp4"
GEN_MASK_MP4 = OUTPUTS / "generated_character_mask.mp4"
OLD_COMP = OUTPUTS / "layered_cfg2_source_mask.mp4"
NEW_COMP = OUTPUTS / "layered_cfg2_generated_mask.mp4"
MASK_QA = OUTPUTS / "g0_l3_mask_comparison_qa.png"
COMP_QA = OUTPUTS / "g0_l3_composite_qa.png"
REPORT = OUTPUTS / "g0_l3_report.json"

FEATHER_SIGMA = 1.0
GRAY_SAT_THRESHOLD = 40.0
GRAY_MIN_VALUE = 60.0
CROP = (170, 0, 660, 270)
MASK_FRAMES = [0, 14, 28, 42, 55]
COMP_FRAMES = [14, 28, 42, 55]


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


def write_video(path: Path, frames: np.ndarray, fps: float, full_range: bool = False, crf: int = 18) -> None:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    n, h, w, _ = frames.shape
    pix = "yuvj420p" if full_range else "yuv420p"
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", pix, "-r", str(fps), str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err.decode(errors='replace')}")


def align_source_mask(mask: np.ndarray, src_fps: float, raw_fps: float, raw_n: int, raw_w: int, raw_h: int) -> np.ndarray:
    out = []
    for j in range(raw_n):
        t = j / raw_fps
        si = min(max(int(round(t * src_fps)), 0), len(mask) - 1)
        m = cv2.resize(mask[si], (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)
        out.append((m > 127).astype(np.uint8) * 255)
    return np.stack(out)


def run_sam2(raw: np.ndarray) -> np.ndarray:
    from sam2.build_sam import build_sam2_video_predictor_hf

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(raw):
        cv2.imwrite(str(FRAMES_DIR / f"{i:06d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    initial = select_initial_mask(raw[0])
    print("[sam2] initial mask ratio:", round(float(initial.mean()), 4))
    predictor = build_sam2_video_predictor_hf(MODEL_ID, device="cuda")
    state = predictor.init_state(video_path=str(FRAMES_DIR), offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=initial)
    raw_masks = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0):
        m = masks[0]
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        if m.dtype != bool:
            m = m > 0.0
        raw_masks[int(frame_idx)] = np.squeeze(np.ascontiguousarray(m))
    del predictor, state
    torch.cuda.empty_cache()
    processed = postprocess_masks([raw_masks[i] for i in range(len(raw))])
    return processed


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def gray_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < GRAY_SAT_THRESHOLD) & (hsv[..., 2] > GRAY_MIN_VALUE)


def crop(img: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = CROP
    return img[y0:y1, x0:x1]


def overlay_masks(raw_frame, source_mask, gen_mask):
    img = raw_frame.copy()
    src_only = source_mask.astype(bool) & ~gen_mask.astype(bool)
    gen_only = gen_mask.astype(bool) & ~source_mask.astype(bool)
    img[src_only] = (img[src_only] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    img[gen_only] = (img[gen_only] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    return img


def make_qa_sheet(rows, cols, images_by_cell, path, title):
    cell_w, cell_h = 300, 180
    label_h, title_h, pad = 24, 30, 4
    canvas = Image.new("RGB", (len(cols) * cell_w + pad * (len(cols) + 1), title_h + len(rows) * (cell_h + label_h) + pad * (len(rows) + 1)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        tf = ImageFont.load_default(size=18); lf = ImageFont.load_default(size=14)
    except TypeError:
        tf = ImageFont.load_default(); lf = ImageFont.load_default()
    draw.text((pad + 4, 6), title, fill=(240, 240, 240), font=tf)
    for row, fi in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, label in enumerate(cols):
            img = images_by_cell[(fi, col)]
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 2), f"F{fi:02d} {label}", fill=(220, 220, 220), font=lf)
    canvas.save(path)


def main():
    L3.mkdir(parents=True, exist_ok=True)
    raw = decode_rgb(RAW)
    src_mask_orig = decode_rgb(SOURCE_MASK)[..., 0]
    bg = np.asarray(Image.open(BG).convert("RGB"), dtype=np.uint8)
    src_probe = probe(SOURCE_VIDEO)
    raw_probe = probe(RAW)
    print("[g0-l3] raw:", raw_probe)

    src_mask = align_source_mask(src_mask_orig, float(src_probe["fps"] or 30.0), float(raw_probe["fps"] or 16.0), raw_probe["frame_count"], raw_probe["width"], raw_probe["height"])
    np.save(L3 / "source_mask.npy", src_mask)
    print("[g0-l3] source mask aligned:", src_mask.shape, "white_ratio", round(float((src_mask > 127).mean()), 4))

    # gray artifact zero-generation diagnostic
    gray_all = np.stack([gray_mask(f) for f in raw])
    src_bool = src_mask > 127
    gray_inside_src = float((gray_all & src_bool).mean())
    gray_outside_src = float((gray_all & ~src_bool).mean())
    print(f"[g0-l3] gray inside source ratio={gray_inside_src:.4f} outside={gray_outside_src:.4f}")

    # SAM2 generated mask
    gen_mask = run_sam2(raw)
    np.save(L3 / "generated_mask.npy", gen_mask)
    write_video(GEN_MASK_MP4, np.stack([gen_mask] * 3, axis=-1), raw_probe["fps"], full_range=True, crf=0)
    print("[g0-l3] generated mask:", gen_mask.shape, "white_ratio", round(float((gen_mask > 127).mean()), 4))

    gen_bool = gen_mask > 127
    inter = src_bool & gen_bool
    union = src_bool | gen_bool
    iou = (inter.sum(axis=(1, 2)) / np.maximum(union.sum(axis=(1, 2)), 1)).mean()
    src_only = src_bool & ~gen_bool
    gen_only = gen_bool & ~src_bool
    src_only_ratio = float(src_only.mean())
    gen_only_ratio = float(gen_only.mean())
    gray_in_src_only = float((gray_all & src_only).mean())
    gray_in_gen_only = float((gray_all & gen_only).mean())
    gray_overlap_src_only = float((gray_all & src_only).sum() / max(gray_all.sum(), 1))
    print(f"[g0-l3] iou={iou:.4f} src_only={src_only_ratio:.4f} gen_only={gen_only_ratio:.4f} gray_src_only={gray_in_src_only:.4f} gray_gen_only={gray_in_gen_only:.4f}")

    # composites
    soft_src = np.stack([cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), FEATHER_SIGMA) for m in src_mask])
    soft_gen = np.stack([cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), FEATHER_SIGMA) for m in gen_mask])
    old_comp = (raw * soft_src[..., None] + bg[None] * (1.0 - soft_src[..., None])).astype(np.uint8)
    new_comp = (raw * soft_gen[..., None] + bg[None] * (1.0 - soft_gen[..., None])).astype(np.uint8)
    write_video(OLD_COMP, old_comp, raw_probe["fps"])
    write_video(NEW_COMP, new_comp, raw_probe["fps"])
    new_probe = probe(NEW_COMP)

    gray_old_all = np.stack([gray_mask(f) for f in old_comp])
    gray_new_all = np.stack([gray_mask(f) for f in new_comp])
    gray_old_inside = float((gray_old_all & src_bool).sum() / max(src_bool.sum(), 1))
    gray_new_inside = float((gray_new_all & gen_bool).sum() / max(gen_bool.sum(), 1))
    old_outside = ~src_bool
    new_outside = ~gen_bool
    old_bg_mae = float(np.abs(old_comp.astype(np.int16) - bg[None].astype(np.int16))[old_outside].mean())
    new_bg_mae = float(np.abs(new_comp.astype(np.int16) - bg[None].astype(np.int16))[new_outside].mean())
    print(f"[g0-l3] old gray_inside={gray_old_inside:.4f} bg_mae={old_bg_mae:.4f}; new gray_inside={gray_new_inside:.4f} bg_mae={new_bg_mae:.4f}")

    # bbox trajectories
    src_boxes = [mask_bbox(m) for m in src_mask]
    gen_boxes = [mask_bbox(m) for m in gen_mask]
    src_boxes = [b for b in src_boxes if b]
    gen_boxes = [b for b in gen_boxes if b]

    # QA sheets
    src_frames = decode_rgb(SOURCE_VIDEO)
    mask_cells = {}
    for fi in MASK_FRAMES:
        si = min(int(round(fi / raw_probe["fps"] * src_probe["fps"])), len(src_frames) - 1)
        mask_cells[(fi, 0)] = crop(raw[fi])
        mask_cells[(fi, 1)] = np.repeat(crop(src_mask[fi])[..., None], 3, axis=2)
        mask_cells[(fi, 2)] = np.repeat(crop(gen_mask[fi])[..., None], 3, axis=2)
        mask_cells[(fi, 3)] = np.repeat((crop(src_only[fi].astype(np.uint8)) * 255)[..., None], 3, axis=2)
        mask_cells[(fi, 4)] = crop(overlay_masks(raw[fi], src_mask[fi], gen_mask[fi]))
    make_qa_sheet(MASK_FRAMES, ["Raw CFG2", "Source Mask", "Generated Mask", "Source-only Area", "Overlay"], mask_cells, MASK_QA, "G0-L3 Mask comparison: gray shell vs source-only region")

    comp_cells = {}
    for fi in COMP_FRAMES:
        si = min(int(round(fi / raw_probe["fps"] * src_probe["fps"])), len(src_frames) - 1)
        ref = cv2.resize(src_frames[si], (raw_probe["width"], raw_probe["height"]), interpolation=cv2.INTER_AREA)
        comp_cells[(fi, 0)] = crop(raw[fi])
        comp_cells[(fi, 1)] = crop(old_comp[fi])
        comp_cells[(fi, 2)] = crop(new_comp[fi])
        comp_cells[(fi, 3)] = crop(ref)
    make_qa_sheet(COMP_FRAMES, ["Raw CFG2", "Old Source-mask Composite", "New Generated-mask Composite", "Reference Source"], comp_cells, COMP_QA, "G0-L3 Composite A/B: gray shell removal vs hair cutting")

    # result
    gray_reduced = gray_new_inside < gray_old_inside * 0.5
    hair_cut_risk = src_only_ratio > 0.12
    if gray_reduced and not hair_cut_risk:
        result = "pass"
    elif gray_reduced and hair_cut_risk:
        result = "borderline"
    elif not gray_reduced:
        result = "fail"
    else:
        result = "inconclusive"

    report = {
        "input": {
            "cfg2_path": str(RAW),
            "sha256": "1e0e9390ed5f495f53949688c613c46ba3a5ec0f55a98c4411be21b7b530d570",
            "provenance": {
                "prompt": "C2 simple Chinese",
                "guide_scale": 2.0,
                "seed": 4096,
                "steps": 8,
                "shift": 3.0,
                "checkpoint": "/root/autodl-tmp/anisora-g0/models/anymask",
            },
            "ffprobe": raw_probe,
        },
        "source_mask": {
            "mean_area_ratio": float((src_mask > 127).mean()),
            "bbox_mean": np.mean(src_boxes, axis=0).round(1).tolist() if src_boxes else None,
            "bbox_min": np.min(src_boxes, axis=0).tolist() if src_boxes else None,
            "bbox_max": np.max(src_boxes, axis=0).tolist() if src_boxes else None,
        },
        "generated_mask": {
            "mean_area_ratio": float((gen_mask > 127).mean()),
            "bbox_mean": np.mean(gen_boxes, axis=0).round(1).tolist() if gen_boxes else None,
            "bbox_min": np.min(gen_boxes, axis=0).tolist() if gen_boxes else None,
            "bbox_max": np.max(gen_boxes, axis=0).tolist() if gen_boxes else None,
            "empty_frames": int((gen_mask > 127).sum(axis=(1, 2)) == 0).sum() if False else int(((gen_mask > 127).sum(axis=(1, 2)) == 0).sum()),
        },
        "comparison": {
            "mean_iou": round(float(iou), 4),
            "source_only_ratio": round(src_only_ratio, 4),
            "generated_only_ratio": round(gen_only_ratio, 4),
            "gray_inside_source_ratio": round(gray_inside_src, 4),
            "gray_outside_source_ratio": round(gray_outside_src, 4),
            "gray_inside_source_only_ratio": round(gray_in_src_only, 4),
            "gray_inside_generated_only_ratio": round(gray_in_gen_only, 4),
            "gray_overlap_with_source_only": round(gray_overlap_src_only, 4),
        },
        "old_composite": {
            "gray_artifact_ratio": round(gray_old_inside, 4),
            "background_mae": round(old_bg_mae, 4),
        },
        "new_composite": {
            "gray_artifact_ratio": round(gray_new_inside, 4),
            "background_mae": round(new_bg_mae, 4),
            "frame_count": new_probe["frame_count"],
            "fps": new_probe["fps"],
            "decodable": new_probe["decodable"],
        },
        "result": result,
        "conclusion": {
            "output_mask_should_be_independent": result in ("pass", "borderline"),
            "evidence": [
                f"gray inside source mask={gray_inside_src:.4f}, outside={gray_outside_src:.4f}",
                f"gray overlap with source-only={gray_overlap_src_only:.4f}",
                f"old gray inside={gray_old_inside:.4f} -> new gray inside={gray_new_inside:.4f}",
                f"source_only_ratio={src_only_ratio:.4f}, generated_only_ratio={gen_only_ratio:.4f}",
                f"mean IoU={iou:.4f}",
            ],
            "limitations": [
                "gray detection is a simple HSV threshold (sat<40, value>60), not a visual grading",
                "SAM2 generated mask quality is not manually verified; QA sheets provided",
                "single CFG2 sample, single seed",
            ],
        },
        "outputs": {
            "generated_mask": str(GEN_MASK_MP4),
            "old_composite": str(OLD_COMP),
            "new_composite": str(NEW_COMP),
            "mask_qa": str(MASK_QA),
            "composite_qa": str(COMP_QA),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[g0-l3] report ->", REPORT)


if __name__ == "__main__":
    main()
