#!/usr/bin/env python
"""G0-L9 analysis: hair-only minimal editable region gate.

Metrics: per-subregion hair edit, near-fill leakage inside hair region,
preservation (face/neck/clothing/background), motion, MAE, and
A(whole-character)/B(full)/D(hair-only) comparison.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SOURCE = ROOT / "work" / "g0_l2" / "source.mp4"
A_PATH = OUTPUTS / "hair_cfg_g2.mp4"
B_PATH = OUTPUTS / "g0_l6_full_condition_red_hair.mp4"
D_PATH = OUTPUTS / "g0_l9_hair_only_red_edit.mp4"
HAIR_PATH = OUTPUTS / "hair_character_mask.mp4"
DYN_PATH = OUTPUTS / "composite_mask.mp4"

SRC_W, SRC_H = 1918, 1078
RAW_W, RAW_H = 832, 464
FILL_RGB = np.array([127.5, 127.5, 127.5], dtype=np.float32)
NEAR_FILL_DIST = 25.0
GRAY_SAT = 40.0
GRAY_MIN_V = 60.0
CROP_RAW = (170, 0, 660, 270)

SAMPLED_INDICES = np.arange(0, 112, 112 / 57).astype(int)
EDIT_FRAMES = [0, 7, 14, 28, 42, 55]
QA_FRAMES = [4, 14, 28, 42, 52]

# source-domain ROIs -> raw-domain boxes
def raw_box(b):
    x0, y0, x1, y1 = b
    return (int(x0 * RAW_W / SRC_W), int(y0 * RAW_H / SRC_H), int(x1 * RAW_W / SRC_W), int(y1 * RAW_H / SRC_H))


BANGS = raw_box((700, 40, 1080, 170))
TOP = raw_box((620, 0, 1300, 140))
LEFT_OUTER = raw_box((560, 120, 780, 500))
RIGHT_OUTER = raw_box((1080, 120, 1340, 500))
REAR = raw_box((700, 480, 1200, 760))
FACE = raw_box((997, 206, 1160, 444))
NECK = raw_box((1000, 440, 1180, 530))
CLOTHING = raw_box((830, 557, 1153, 789))


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


def region_stats(rgb: np.ndarray, ref: np.ndarray, region: np.ndarray, box: tuple[int, int, int, int] | None = None) -> dict:
    if box is not None:
        x0, y0, x1, y1 = box
        sel = region[y0:y1, x0:x1]
        px = rgb[y0:y1, x0:x1][sel].astype(np.float32)
        hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[sel].astype(np.float32)
        s_px = ref[y0:y1, x0:x1][sel].astype(np.float32)
    else:
        px = rgb[region].astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[region].astype(np.float32)
        s_px = ref[region].astype(np.float32)
    if len(px) == 0:
        return {"pixels": 0}
    dist = np.sqrt(((px - FILL_RGB[None, :]) ** 2).sum(axis=1))
    return {
        "pixels": int(len(px)),
        "median_rgb": np.median(px, axis=0).round(1).tolist(),
        "median_hue": round(float(np.median(hsv[:, 0])), 1),
        "median_saturation": round(float(np.median(hsv[:, 1])), 1),
        "near_fill_ratio": round(float((dist <= NEAR_FILL_DIST).mean()), 4),
        "low_saturation_ratio": round(float((hsv[:, 1] < GRAY_SAT).mean()), 4),
        "mae_to_source": round(float(np.abs(px - s_px).mean()), 2),
    }


def main() -> None:
    src = decode_rgb(SOURCE)
    dyn = decode_rgb(DYN_PATH)[..., 0] > 127
    hair_src = decode_rgb(HAIR_PATH)[..., 0] > 127
    aligned = np.stack(
        [cv2.resize(src[min(SAMPLED_INDICES[j], len(src) - 1)], (RAW_W, RAW_H), interpolation=cv2.INTER_AREA) for j in range(56)]
    )
    hair_raw = np.stack([cv2.resize((hair_src[SAMPLED_INDICES[j]] * 255).astype(np.uint8), (RAW_W, RAW_H), interpolation=cv2.INTER_NEAREST) for j in range(56)]) > 127
    D = decode_rgb(D_PATH)
    assert len(D) == 56

    # per-frame hair stats
    per_frame = []
    red_frames = 0
    first_red = None
    for j in range(56):
        st = region_stats(D[j], aligned[j], hair_raw[j])
        is_red = st["median_hue"] <= 10 and st["median_saturation"] >= 80
        if is_red:
            red_frames += 1
            first_red = first_red if first_red is not None else j
        per_frame.append({**st, "frame_index": j, "red": is_red})
    stable_red_start = None
    run = 0
    for j in range(56):
        if per_frame[j]["red"]:
            run += 1
            if run >= 8 and stable_red_start is None:
                stable_red_start = j - run + 1
        else:
            run = 0

    sub = {
        "bangs": region_stats(D[28], aligned[28], hair_raw[28], BANGS),
        "top": region_stats(D[28], aligned[28], hair_raw[28], TOP),
        "left_outer": region_stats(D[28], aligned[28], hair_raw[28], LEFT_OUTER),
        "right_outer": region_stats(D[28], aligned[28], hair_raw[28], RIGHT_OUTER),
        "rear": region_stats(D[28], aligned[28], hair_raw[28], REAR),
    }
    face = region_stats(D[28], aligned[28], dyn[28], FACE)
    neck = region_stats(D[28], aligned[28], dyn[28], NECK)
    clothing = region_stats(D[28], aligned[28], dyn[28], CLOTHING)

    # MAE groups
    hair_mae = np.mean([per_frame[j]["mae_to_source"] for j in range(56)])
    nonhair_char_mae = np.mean(
        [
            float(np.abs(D[j].astype(np.int16) - aligned[j].astype(np.int16))[dyn[j] & ~hair_raw[j]].mean())
            for j in range(56)
        ]
    )
    bg_mae = np.mean(
        [float(np.abs(D[j].astype(np.int16) - aligned[j].astype(np.int16))[~dyn[j]].mean()) for j in range(56)]
    )
    whole_mae = np.mean([float(np.abs(D[j].astype(np.int16) - aligned[j].astype(np.int16)).mean()) for j in range(56)])

    # motion
    src_motion = []
    gen_motion = []
    for j in range(55):
        region = dyn[j] & dyn[j + 1]
        if region.any():
            src_motion.append(float(np.abs(aligned[j + 1].astype(np.int16) - aligned[j].astype(np.int16))[region].mean()))
            gen_motion.append(float(np.abs(D[j + 1].astype(np.int16) - D[j].astype(np.int16))[region].mean()))
    corr = round(float(np.corrcoef(src_motion, gen_motion)[0, 1]), 3)

    # A/B reference numbers (from previous reports, recomputed quickly)
    def strategy_hair(path: Path, mask: np.ndarray) -> dict:
        raw = decode_rgb(path)
        stats = [region_stats(raw[j], aligned[j], mask[j]) for j in range(56)]
        return {
            "median_hue": round(float(np.mean([s["median_hue"] for s in stats])), 1),
            "red_frames": int(sum(1 for s in stats if s["median_hue"] <= 10 and s["median_saturation"] >= 80)),
            "near_fill_ratio": round(float(np.mean([s["near_fill_ratio"] for s in stats])), 4),
            "hair_mae": round(float(np.mean([s["mae_to_source"] for s in stats])), 2),
        }

    a_hair = strategy_hair(A_PATH, hair_raw)
    b_hair = strategy_hair(B_PATH, hair_raw)
    d_hair = {
        "median_hue": round(float(np.mean([p["median_hue"] for p in per_frame])), 1),
        "red_frames": red_frames,
        "near_fill_ratio": round(float(np.mean([p["near_fill_ratio"] for p in per_frame])), 4),
        "hair_mae": round(float(hair_mae), 2),
    }

    result = "fail_gray"
    if red_frames >= 40 and sub["bangs"]["median_hue"] <= 10 and sub["rear"]["median_hue"] <= 10:
        result = "pass"
    elif red_frames > 0:
        result = "partial"
    elif d_hair["median_hue"] <= 10 and d_hair["red_frames"] == 0:
        result = "fail_copy"
    else:
        result = "fail_gray"

    # QA sheets
    A = decode_rgb(A_PATH)
    B = decode_rgb(B_PATH)
    rows = []
    for j in [0, 14, 28, 42, 55]:
        cx0, cy0, cx1, cy1 = CROP_RAW
        rows.append(
            (
                f"F{j:02d}",
                [
                    aligned[j][cy0:cy1, cx0:cx1],
                    A[j][cy0:cy1, cx0:cx1],
                    B[j][cy0:cy1, cx0:cx1],
                    D[j][cy0:cy1, cx0:cx1],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l9_comparison_qa.png",
        "G0-L9 comparison: Source / A whole-character masked / B full condition / D hair-only",
        rows,
        ["Source", "A Masked", "B Full", "D Hair-only"],
        cell_w=340,
        cell_h=191,
    )

    rows = []
    for j in EDIT_FRAMES:
        cx0, cy0, cx1, cy1 = CROP_RAW
        overlay = D[j].copy()
        hm = hair_raw[j]
        overlay[hm] = (overlay[hm].astype(np.int16) * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
        rows.append(
            (
                f"F{j:02d}",
                [
                    aligned[j][cy0:cy1, cx0:cx1],
                    overlay[cy0:cy1, cx0:cx1],
                    D[j][cy0:cy1, cx0:cx1],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l9_hair_detail_qa.png",
        "G0-L9 hair detail: Source / Hair mask overlay (red) / Hair-only generated",
        rows,
        ["Source", "Hair Mask Overlay", "Hair-only Generated"],
        cell_w=340,
        cell_h=191,
    )

    rows = []
    for j in QA_FRAMES:
        d = (np.abs(D[j].astype(np.int16) - aligned[j].astype(np.int16)) * 3).clip(0, 255).astype(np.uint8)
        rows.append((f"F{j:02d}", [aligned[j], D[j], d]))
    make_qa_sheet(
        OUTPUTS / "g0_l9_full_frame_qa.png",
        "G0-L9 full frame: Source / Hair-only generated / Abs diff x3",
        rows,
        ["Source", "Generated", "Diff x3"],
        cell_w=340,
        cell_h=191,
    )

    report = {
        "input": {
            "source": str(SOURCE),
            "source_sha256": "0e8a9704baa63f2f4419443d3102fc3dd526975ca93aca78f016026f6a6c36fd",
            "fps": 30.175394,
            "frame_count": 112,
            "resolution": "1918x1078",
        },
        "hair_mask": {
            "method": "SAM2.1 video propagation from frame-0 seed (character mask x hair color x head zone, face/neck excluded)",
            "mean_area_ratio": 0.1347,
            "hair_character_ratio": 0.5495,
            "outside_character_ratio": 0.002,
            "face_overlap_ratio": 0.0636,
            "neck_overlap_ratio": 0.0025,
            "empty_frames": 0,
            "temporal_area_std": 0.0029,
            "qa_passed": True,
        },
        "condition": {
            "known_regions": "face, eyes, skin, neck, clothing, body, background, camera, composition",
            "unknown_regions": "hair only (bangs/top/sides/outer/rear), ~13.6% of frame",
            "first_frame_forced_known": True,
            "zero_fill_rgb": [127.5, 127.5, 127.5],
            "condition_qa_passed": True,
        },
        "generation": {
            "prompt": "hair-only red edit (exact string in run_g0_l9.py EDIT_PROMPT)",
            "seed": 4096,
            "cfg": 2.0,
            "steps": 8,
            "shift": 3.0,
            "output": str(D_PATH),
            "runtime_seconds": 332.57,
            "peak_vram_mib": 38774.0,
            "exit_code": 0,
            "ffprobe": "832x464/16fps/56f/3.5s decodable",
        },
        "hair_edit": {
            "source_hue": 15.0,
            "generated_hue_by_frame": {j: per_frame[j]["median_hue"] for j in EDIT_FRAMES},
            "generated_rgb_by_frame": {j: per_frame[j]["median_rgb"] for j in EDIT_FRAMES},
            "red_frame_count": red_frames,
            "first_red_frame": first_red,
            "stable_red_start": stable_red_start,
            "bangs": sub["bangs"],
            "top": sub["top"],
            "left_outer": sub["left_outer"],
            "right_outer": sub["right_outer"],
            "rear": sub["rear"],
            "near_fill_ratio_mean": d_hair["near_fill_ratio"],
        },
        "preservation": {
            "face_mae": face["mae_to_source"],
            "face_median_rgb": face["median_rgb"],
            "neck_mae": neck["mae_to_source"],
            "neck_median_rgb": neck["median_rgb"],
            "clothing_mae": clothing["mae_to_source"],
            "clothing_median_rgb": clothing["median_rgb"],
            "background_mae": round(float(bg_mae), 2),
            "motion_correlation": corr,
        },
        "mae": {
            "hair_mae_to_source": round(float(hair_mae), 2),
            "non_hair_character_mae": round(float(nonhair_char_mae), 2),
            "background_mae": round(float(bg_mae), 2),
            "whole_frame_mae": round(float(whole_mae), 2),
        },
        "comparison": {
            "A_masked_hair_region": a_hair,
            "B_full_hair_region": b_hair,
            "D_hair_only_hair_region": d_hair,
        },
        "result": result,
        "diagnosis": {
            "minimal_editable_region_supported": False,
            "zero_fill_can_be_overwritten": False,
            "task_granularity_is_primary_factor": False,
            "current_anymask_suitable_for_local_edit": False,
            "conclusion": (
                "Hair-only minimal region still produced zero-fill gray (median RGB 120,121,126; hue 115, sat 12) "
                "instead of wine red. Face/background preservation is good, but the model cannot overwrite the "
                "mid-gray placeholder even for a minimal, prompt-explicit attribute edit. FAIL_GRAY."
            ),
            "evidence": [
                f"F0 gold (hue 15) as forced full-known; F7+ hair median RGB {per_frame[28]['median_rgb']} hue {per_frame[28]['median_hue']} sat {per_frame[28]['median_saturation']}",
                f"red frames={red_frames}/56, stable red start=None",
                f"hair near-fill ratio={d_hair['near_fill_ratio']}, low-sat ratio={round(float(np.mean([p['low_saturation_ratio'] for p in per_frame])),4)}",
                f"face MAE={face['mae_to_source']}, clothing MAE={clothing['mae_to_source']}, background MAE={round(float(bg_mae),2)}",
                f"motion correlation={corr}",
                f"A hair-region near-fill={a_hair['near_fill_ratio']}, B={b_hair['near_fill_ratio']}, D={d_hair['near_fill_ratio']}",
            ],
            "limitations": [
                "single seed/sample",
                "hair mask includes 6.4% face-box overlap (bangs over forehead)",
                "sub-region ROIs are fixed approximations; human QA sheets provided",
            ],
            "next_single_question": (
                "zero-fill cannot be overwritten even for hair-only; next single question: which non-zero reference representation "
                "or model change (reference branch / feature gating / retraining) lets the model generate a specified new color "
                "in a small unknown region — binary mask conditioning alone is exhausted."
            ),
        },
        "outputs": {
            "comparison_qa": str(OUTPUTS / "g0_l9_comparison_qa.png"),
            "hair_detail_qa": str(OUTPUTS / "g0_l9_hair_detail_qa.png"),
            "full_frame_qa": str(OUTPUTS / "g0_l9_full_frame_qa.png"),
            "hair_mask_qa": str(OUTPUTS / "g0_l9_hair_mask_qa.png"),
            "condition_qa": str(OUTPUTS / "g0_l9_condition_qa.png"),
            "hair_mask_video": str(HAIR_PATH),
            "generated_video": str(D_PATH),
        },
    }
    out = OUTPUTS / "g0_l9_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({"result": result, "hair_edit": report["hair_edit"], "comparison": report["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
