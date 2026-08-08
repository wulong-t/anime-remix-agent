#!/usr/bin/env python
"""G0-L6 analysis: source-copy vs editability vs preservation.

Time alignment: generated frame j (56f @16fps) <-> source frame indices[j]
(112f @30fps), where indices are the exact read_video sampling indices.
All MAE metrics compare generated raw output with the aligned source frame
resized to 832x464 (INTER_AREA), inside the fixed DynamicRegion mask.
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
GEN = OUTPUTS / "g0_l6_full_condition_red_hair.mp4"
DYN_PATH = OUTPUTS / "composite_mask.mp4"

SRC_W, SRC_H = 1918, 1078
RAW_W, RAW_H = 832, 464
HAIR_ROI_SRC = (700, 50, 1200, 220)
OUTER_HAIR_RAW = [(303, 21, 360, 94), (460, 21, 520, 94)]
FRONT_HAIR_RAW = (303, 21, 520, 60)
FACE_RAW = (370, 85, 480, 165)
CLOTHING_RAW = (360, 240, 500, 340)
CROP_RAW = (170, 0, 660, 270)
SRC_HAIR_RGB = np.array([206.9, 165.0, 117.9], dtype=np.float32)
TARGET_RED_RGB = np.array([139.0, 0.0, 0.0], dtype=np.float32)
GRAY_SAT_THRESHOLD = 40.0
GRAY_MIN_VALUE = 60.0

SAMPLED_INDICES = [
    0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37,
    39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74,
    76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110,
]
EDIT_FRAMES = [0, 7, 14, 28, 42, 55]
QA_FRAMES = [4, 14, 28, 42, 52]
TEMPORAL_RANGE = range(24, 33)


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


def probe(path: Path) -> dict:
    import subprocess

    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = info["streams"][0]
    fmt = info["format"]
    num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else None
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "frame_count": int(stream.get("nb_read_frames") or 0),
        "duration": float(fmt.get("duration", 0.0)),
    }


def gray_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < GRAY_SAT_THRESHOLD) & (hsv[..., 2] > GRAY_MIN_VALUE)


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


def roi_stats(rgb: np.ndarray, ref: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    x0, y0, x1, y1 = box
    p = mask[y0:y1, x0:x1]
    if not p.any():
        return {"coverage": 0.0}
    g = rgb[y0:y1, x0:x1][p].astype(np.float32)
    s = ref[y0:y1, x0:x1][p].astype(np.float32)
    hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[p].astype(np.float32)
    return {
        "coverage": round(float(p.mean()), 3),
        "median_rgb": np.median(g, axis=0).round(1).tolist(),
        "mean_rgb": g.mean(axis=0).round(1).tolist(),
        "median_hue": round(float(np.median(hsv[:, 0])), 1),
        "median_saturation": round(float(np.median(hsv[:, 1])), 1),
        "redness_r_minus_gb2": round(float(np.median(g[:, 0] - (g[:, 1] + g[:, 2]) / 2.0)), 1),
        "mae_to_source": round(float(np.abs(g - s).mean()), 2),
        "color_distance_to_source_hair": round(float(np.linalg.norm(np.median(g, axis=0) - SRC_HAIR_RGB)), 1),
        "color_distance_to_target_red": round(float(np.linalg.norm(np.median(g, axis=0) - TARGET_RED_RGB)), 1),
    }


def main() -> None:
    src = decode_rgb(SOURCE)
    gen = decode_rgb(GEN)
    dyn = decode_rgb(DYN_PATH)[..., 0] > 127
    assert gen.shape[0] == 56 and dyn.shape[0] == 56
    aligned = np.stack(
        [cv2.resize(src[min(SAMPLED_INDICES[j], len(src) - 1)], (RAW_W, RAW_H), interpolation=cv2.INTER_AREA) for j in range(56)]
    )

    per_frame = []
    for j in range(56):
        d = np.abs(gen[j].astype(np.int16) - aligned[j].astype(np.int16))
        p = dyn[j]
        hair = roi_stats(gen[j], aligned[j], p, HAIR_ROI_SRC_SCALED())
        per_frame.append(
            {
                "frame_index": j,
                "source_frame_index": SAMPLED_INDICES[j],
                "whole_frame_mae": round(float(d.mean()), 2),
                "character_mae": round(float(d[p].mean()), 2),
                "background_mae": round(float(d[~p].mean()), 2),
                "hair": hair,
                "gray_artifact_ratio_person": round(float((gray_mask(gen[j]) & p).sum() / max(p.sum(), 1)), 4),
            }
        )

    src_motion = []
    gen_motion = []
    for j in range(55):
        region = dyn[j] & dyn[j + 1]
        if not region.any():
            continue
        src_motion.append(float(np.abs(aligned[j + 1].astype(np.int16) - aligned[j].astype(np.int16))[region].mean()))
        gen_motion.append(float(np.abs(gen[j + 1].astype(np.int16) - gen[j].astype(np.int16))[region].mean()))

    def mean_std(x):
        return [round(float(np.mean(x)), 2), round(float(np.std(x)), 2)] if x else [None, None]

    whole_mae_mean = float(np.mean([f["whole_frame_mae"] for f in per_frame]))
    char_mae_mean = float(np.mean([f["character_mae"] for f in per_frame]))
    bg_mae_mean = float(np.mean([f["background_mae"] for f in per_frame]))
    hair_mae_mean = float(np.mean([f["hair"]["mae_to_source"] for f in per_frame]))
    mean_hue = float(np.mean([f["hair"]["median_hue"] for f in per_frame]))
    mean_sat = float(np.mean([f["hair"]["median_saturation"] for f in per_frame]))
    hair_change = float(np.linalg.norm(np.mean([f["hair"]["median_rgb"] for f in per_frame], axis=0) - SRC_HAIR_RGB))
    red_frames = sum(1 for f in per_frame if f["hair"]["median_hue"] <= 10.0 and f["hair"]["median_saturation"] >= 80)
    gray_mean = float(np.mean([f["gray_artifact_ratio_person"] for f in per_frame]))

    face = roi_stats(gen[28], aligned[28], dyn[28], FACE_RAW)
    clothing = roi_stats(gen[28], aligned[28], dyn[28], CLOTHING_RAW)
    outer_l = roi_stats(gen[28], aligned[28], dyn[28], OUTER_HAIR_RAW[0])
    outer_r = roi_stats(gen[28], aligned[28], dyn[28], OUTER_HAIR_RAW[1])
    front = roi_stats(gen[28], aligned[28], dyn[28], FRONT_HAIR_RAW)

    edit_frames = {j: per_frame[j]["hair"] for j in EDIT_FRAMES}
    hair_edit_success = bool(mean_hue <= 10.0 and mean_sat >= 80 and hair_change >= 40)
    copy_likelihood = bool(whole_mae_mean < 8.0 and hair_change < 25.0)
    preservation_ok = bool(
        bg_mae_mean <= 10.0
        and face["mae_to_source"] <= 15.0
        and clothing["mae_to_source"] <= 15.0
        and mean_std(gen_motion)[0] is not None
    )

    if hair_edit_success and preservation_ok:
        result = "pass"
        reason = "hair edited to red while preservation metrics stay within bounds"
    elif hair_edit_success and not preservation_ok:
        result = "fail_generation"
        reason = "hair edited but preservation metrics out of bounds (face/clothing/background/motion changed)"
    elif copy_likelihood and not hair_edit_success:
        result = "fail_copy"
        reason = "output closely matches source and hair was not edited; full condition lacks editability"
    else:
        result = "borderline"
        reason = "partial evidence; needs human QA confirmation"

    # ---------- QA: identity/edit ----------
    rows = []
    for j in [0, 14, 28, 42, 55]:
        cx0, cy0, cx1, cy1 = CROP_RAW
        rows.append(
            (
                f"F{j:02d}",
                [aligned[j][cy0:cy1, cx0:cx1], gen[j][cy0:cy1, cx0:cx1]],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l6_identity_edit_qa.png",
        "G0-L6 identity/edit QA: Source vs Full-Condition Generated (head/upper-body crop)",
        rows,
        ["Source", "Full Condition Generated"],
        cell_w=340,
        cell_h=191,
    )

    # ---------- QA: whole frame ----------
    rows = []
    for j in QA_FRAMES:
        d = np.abs(gen[j].astype(np.int16) - aligned[j].astype(np.int16)).clip(0, 255).astype(np.uint8)
        diff = (d * 3).clip(0, 255).astype(np.uint8)
        rows.append((f"F{j:02d}", [aligned[j], gen[j], diff]))
    make_qa_sheet(
        OUTPUTS / "g0_l6_full_frame_qa.png",
        "G0-L6 whole-frame QA: Source / Generated / Abs-diff (x3)",
        rows,
        ["Source", "Generated", "Abs Diff x3"],
        cell_w=340,
        cell_h=191,
    )

    # ---------- QA: temporal ----------
    cols = [f"F{j:02d}" for j in TEMPORAL_RANGE]
    rows = []
    for label, arr in [
        ("Source", aligned),
        ("Generated", gen),
        ("AbsDiff", None),
    ]:
        imgs = []
        for j in TEMPORAL_RANGE:
            if arr is not None:
                imgs.append(arr[j])
            else:
                d = np.abs(gen[j].astype(np.int16) - aligned[j].astype(np.int16)).clip(0, 255).astype(np.uint8)
                imgs.append((d * 3).clip(0, 255).astype(np.uint8))
        rows.append((label, imgs))
    make_qa_sheet(
        OUTPUTS / "g0_l6_temporal_qa.png",
        "G0-L6 temporal QA: F24-F32 Source / Generated / AbsDiff",
        rows,
        cols,
        cell_w=160,
        cell_h=90,
    )

    run_info = parse_run_info()
    report = {
        "input": {
            "source_path": str(SOURCE),
            "source_sha256": "0e8a9704baa63f2f4419443d3102fc3dd526975ca93aca78f016026f6a6c36fd",
            "fps": 30.175394,
            "frame_count": 112,
            "duration": 3.711979,
            "width": 1918,
            "height": 1078,
        },
        "generation": {
            "prompt": "full_condition_red_hair_edit (exact string in run_g0_l6.py EDIT_PROMPT)",
            "seed": 4096,
            "steps": 8,
            "shift": 3.0,
            "cfg": 2.0,
            "condition_mode": "full_video",
            "mask_semantics": "all frames all-white; binary_mask=1 for all T*H*W; Img_list_new==Img_list",
            "output_path": str(GEN),
            **run_info,
        },
        "edit": {
            "target": "hair_color: light golden-brown -> deep red/wine red",
            "source_color": SRC_HAIR_RGB.tolist(),
            "target_color": TARGET_RED_RGB.tolist(),
            "mean_generated_hair_rgb": np.mean([f["hair"]["median_rgb"] for f in per_frame], axis=0).round(1).tolist(),
            "mean_hue": round(mean_hue, 1),
            "mean_saturation": round(mean_sat, 1),
            "hair_change_from_source": round(hair_change, 1),
            "red_frames_ratio": round(red_frames / 56.0, 4),
            "edit_success": hair_edit_success,
            "temporal_consistency": round(float(np.std([f["hair"]["median_hue"] for f in per_frame])), 2),
            "outer_hair_success": bool(outer_l["median_hue"] <= 10.0 and outer_r["median_hue"] <= 10.0),
            "front_hair": front,
            "outer_hair_left": outer_l,
            "outer_hair_right": outer_r,
            "sample_frames": edit_frames,
        },
        "preservation": {
            "face_mae": face["mae_to_source"],
            "face_median_rgb": face.get("median_rgb"),
            "clothing_mae": clothing["mae_to_source"],
            "clothing_median_rgb": clothing.get("median_rgb"),
            "skin_region": FACE_RAW,
            "motion_source_mean_std": mean_std(src_motion),
            "motion_generated_mean_std": mean_std(gen_motion),
            "motion_preserved": round(float(np.corrcoef(src_motion, gen_motion)[0, 1]), 3) if len(src_motion) == len(gen_motion) and len(src_motion) > 2 else None,
            "background_mae_mean": round(bg_mae_mean, 2),
            "camera": "fixed (background MAE proxy)",
            "gray_artifact_ratio_person_mean": round(gray_mean, 4),
        },
        "source_copy": {
            "whole_frame_mae": round(whole_mae_mean, 2),
            "character_mae": round(char_mae_mean, 2),
            "background_mae": round(bg_mae_mean, 2),
            "hair_roi_mae": round(hair_mae_mean, 2),
            "copy_likelihood": copy_likelihood,
            "per_frame": per_frame,
        },
        "result": result,
        "decision": {
            "full_condition_suitable_for_mvp": result == "pass" or result == "borderline",
            "evidence": [
                f"hair mean RGB={np.mean([f['hair']['median_rgb'] for f in per_frame], axis=0).round(1).tolist()} hue={round(mean_hue,1)} (source hue 15; red requires <=10)",
                f"hair change from source={round(hair_change,1)}; red frames={red_frames}/56",
                f"whole MAE={round(whole_mae_mean,2)} char={round(char_mae_mean,2)} bg={round(bg_mae_mean,2)}",
                f"motion src={mean_std(src_motion)} gen={mean_std(gen_motion)}",
                f"gray artifact ratio in person={round(gray_mean,4)}",
                reason,
            ],
            "limitations": [
                "single seed, single prompt, single sample",
                "objective metrics only; human QA images provided",
                "hair ROI is source-anchored; generated silhouette drift could bias ROI",
            ],
            "next_single_question": (
                "如何设计介于 Full（preservation 强、editability 不足）与 Masked（editability 高、zero-fill leakage）之间的 conditioning representation？"
                if result == "fail_copy"
                else "在保留 preservation 的前提下提升 editability 的单一 conditioning 变量是什么？"
            ),
        },
        "outputs": {
            "identity_edit_qa": str(OUTPUTS / "g0_l6_identity_edit_qa.png"),
            "full_frame_qa": str(OUTPUTS / "g0_l6_full_frame_qa.png"),
            "temporal_qa": str(OUTPUTS / "g0_l6_temporal_qa.png"),
            "mask_qa": str(OUTPUTS / "g0_l6_full_condition_mask_qa.png"),
            "generated_video": str(GEN),
        },
    }
    out = OUTPUTS / "g0_l6_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({k: report[k] for k in ["edit", "preservation", "source_copy", "result"]}, ensure_ascii=False, indent=2))


def HAIR_ROI_SRC_SCALED():
    x0 = int(HAIR_ROI_SRC[0] * RAW_W / SRC_W)
    y0 = int(HAIR_ROI_SRC[1] * RAW_H / SRC_H)
    x1 = int(HAIR_ROI_SRC[2] * RAW_W / SRC_W)
    y1 = int(HAIR_ROI_SRC[3] * RAW_H / SRC_H)
    return x0, y0, x1, y1


def parse_run_info() -> dict:
    log = OUTPUTS / "g0_l6_anymask.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(
        r"g0_l6 exit_code=(\d+) runtime_seconds=([\d.]+) peak_vram_mib=([\d.]+) "
        r"peak_util=([\d.]+) memory_before_gib=([\d.]+) memory_after_gib=([\d.]+) memory_max_gib=([\d.]+)",
        text,
    )
    if not m:
        return {}
    return {
        "exit_code": int(m.group(1)),
        "runtime_seconds": float(m.group(2)),
        "peak_vram_mib": float(m.group(3)),
        "peak_util": float(m.group(4)),
        "memory_before_gib": float(m.group(5)),
        "memory_after_gib": float(m.group(6)),
        "memory_max_gib": float(m.group(7)),
    }


if __name__ == "__main__":
    main()
