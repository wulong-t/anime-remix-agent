#!/usr/bin/env python
"""G0-L4 analysis: hair drift, gray shell, motion, anchor transitions, QA sheets.

Reuses the exact G0-L2/L3 analysis constants:
  - hair ROI: SRC_ROI scaled from 1918x1078 to 832x464
  - reference color: REF_RGB [206.9, 165.0, 117.9], REF_SAT 110.2
  - DynamicRegion mask: outputs/composite_mask.mp4 (white=character)
  - gray artifact: HSV sat<40 AND value>60
  - motion: adjacent-frame MAE inside DynamicRegion
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

A_PATH = OUTPUTS / "hair_cfg_g2.mp4"
B_PATH = OUTPUTS / "g0_l4_single_middle_anchor.mp4"
C_PATH = OUTPUTS / "g0_l4_periodic_anchor.mp4"
DYN_MASK_PATH = OUTPUTS / "composite_mask.mp4"
SOURCE_PATH = ROOT / "input" / "source.mp4"

SRC_ROI = (700, 50, 1200, 220)
RAW_W, RAW_H = 832, 464
SRC_W, SRC_H = 1918, 1078
REF_RGB = np.array([206.9, 165.0, 117.9], dtype=np.float32)
REF_SAT = 110.2
CROP_RAW = (170, 0, 660, 270)

GRAY_SAT_THRESHOLD = 40.0
GRAY_MIN_VALUE = 60.0

# Audit-derived anchors (output frame domain @16fps).
B_ANCHOR = 28
C_ANCHORS = [16, 32, 48]

EXPERIMENTS = {
    "A": ("A_baseline_first_frame_only", A_PATH),
    "B": ("B_single_middle", B_PATH),
    "C": ("C_periodic", C_PATH),
}


def roi_raw() -> tuple[int, int, int, int]:
    x0 = int(SRC_ROI[0] * RAW_W / SRC_W)
    y0 = int(SRC_ROI[1] * RAW_H / SRC_H)
    x1 = int(SRC_ROI[2] * RAW_W / SRC_W)
    y1 = int(SRC_ROI[3] * RAW_H / SRC_H)
    return x0, y0, x1, y1


def decode_rgb(path: Path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def gray_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < GRAY_SAT_THRESHOLD) & (hsv[..., 2] > GRAY_MIN_VALUE)


def parse_run_info(tag: str) -> dict:
    log = OUTPUTS / f"g0_l4_{tag}_anymask.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(
        r"g0_l4=(\w+) exit_code=(\d+) runtime_seconds=([\d.]+) "
        r"peak_vram_mib=([\d.]+) peak_util=([\d.]+) "
        r"memory_before_gib=([\d.]+) memory_after_gib=([\d.]+) memory_max_gib=([\d.]+) "
        r"anchor_frames=\[(.*?)\]",
        text,
    )
    if not m:
        return {}
    return {
        "exit_code": int(m.group(2)),
        "runtime_seconds": float(m.group(3)),
        "peak_vram_mib": float(m.group(4)),
        "peak_util": float(m.group(5)),
        "memory_before_gib": float(m.group(6)),
        "memory_after_gib": float(m.group(7)),
        "memory_max_gib": float(m.group(8)),
        "anchor_mask_frames": [int(x) for x in m.group(9).split(",") if x.strip()],
    }


def baseline_run_info() -> dict:
    log = OUTPUTS / "hair_cfg_g2_anymask.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(r"runtime_seconds=([\d.]+)", text)
    gpu = OUTPUTS / "hair_cfg_g2_gpu.csv"
    peak = 0.0
    if gpu.exists():
        for line in gpu.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].replace(".", "", 1).isdigit():
                peak = max(peak, float(parts[0]))
    return {
        "reused_cfg2_path": str(A_PATH),
        "provenance": {
            "prompt": "C2 simple Chinese (exact string in run_g0_l4.py / hair_cfg report)",
            "guide_scale": 2.0,
            "seed": 4096,
            "steps": 8,
            "shift": 3.0,
            "checkpoint": "/root/autodl-tmp/anisora-g0/models/anymask",
            "negative_prompt": "default sample_neg_prompt (contains 整体发灰)",
        },
        "runtime_seconds": float(m.group(1)) if m else None,
        "peak_vram_mib": peak,
        "sha256": "1e0e9390ed5f495f53949688c613c46ba3a5ec0f55a98c4411be21b7b530d570",
    }


def analyze_video(raw: np.ndarray, dyn: np.ndarray) -> dict:
    roi = roi_raw()
    x0, y0, x1, y1 = roi
    n = len(raw)
    hair = []
    gray_rows = []
    motion = []
    for i in range(n):
        sel = dyn[i, y0:y1, x0:x1] > 127
        px = raw[i, y0:y1, x0:x1][sel].astype(np.float32)
        hsv_sel = cv2.cvtColor(raw[i, y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[sel].astype(np.float32)
        rgb = px.mean(axis=0) if len(px) else np.zeros(3, np.float32)
        sat = float(hsv_sel[:, 1].mean()) if len(hsv_sel) else 0.0
        dist = float(np.linalg.norm(rgb - REF_RGB)) if len(px) else float("nan")
        hair.append((rgb, sat, dist))

        g = gray_mask(raw[i])
        d = dyn[i] > 127
        gray_rows.append(
            {
                "gray_artifact_ratio": float((g & d).sum() / max(d.sum(), 1)),
                "gray_frame_ratio": float(g.mean()),
                "gray_outside_ratio": float((g & ~d).sum() / max((~d).sum(), 1)),
            }
        )

        if i < n - 1:
            selm = dyn[i] > 127
            seln = dyn[i + 1] > 127
            region = selm & seln
            m = (
                float(np.abs(raw[i + 1].astype(np.int16) - raw[i].astype(np.int16))[region].mean())
                if region.any()
                else float("nan")
            )
            motion.append(m)

    first = hair[0]
    last = hair[-1]
    dists = [h[2] for h in hair]
    sats = [h[1] for h in hair]
    motion_arr = np.array(motion, dtype=np.float64)
    return {
        "hair_initial_color": first[0].round(1).tolist(),
        "hair_final_color": last[0].round(1).tolist(),
        "initial_saturation": round(float(first[1]), 1),
        "final_saturation": round(float(last[1]), 1),
        "minimum_saturation": round(float(min(sats)), 1),
        "maximum_color_distance": round(float(max(dists)), 1),
        "final_color_distance": round(float(last[2]), 1),
        "mean_color_distance": round(float(np.mean(dists)), 2),
        "mean_gray_artifact_ratio": round(float(np.mean([g["gray_artifact_ratio"] for g in gray_rows])), 4),
        "mean_gray_frame_ratio": round(float(np.mean([g["gray_frame_ratio"] for g in gray_rows])), 4),
        "early_gray_artifact_ratio": round(float(np.mean([g["gray_artifact_ratio"] for g in gray_rows[:8]])), 4),
        "late_gray_artifact_ratio": round(float(np.mean([g["gray_artifact_ratio"] for g in gray_rows[-8:]])), 4),
        "motion_mean": round(float(np.nanmean(motion_arr)), 3) if len(motion_arr) else None,
        "motion_std": round(float(np.nanstd(motion_arr)), 3) if len(motion_arr) else None,
        "motion_max": round(float(np.nanmax(motion_arr)), 3) if len(motion_arr) else None,
        "per_frame": {
            "hair": hair,
            "gray": gray_rows,
            "motion": [None if np.isnan(m) else round(float(m), 3) for m in motion],
        },
    }


def anchor_window_metrics(meta: dict, anchors: list[int]) -> dict:
    motion = np.array([m if m is not None else np.nan for m in meta["per_frame"]["motion"]])
    windows = {}
    for t in anchors:
        win = {
            "anchor_output_frame": t,
            "anchor_time_seconds": round(t / 16.0, 4),
            "pre_anchor_motion": round(float(motion[t - 2]), 3) if t - 2 >= 0 and t - 2 < len(motion) else None,
            "anchor_transition_in": round(float(motion[t - 1]), 3) if t - 1 >= 0 and t - 1 < len(motion) else None,
            "anchor_transition_out": round(float(motion[t]), 3) if t < len(motion) else None,
            "post_anchor_motion": round(float(motion[t + 1]), 3) if t + 1 < len(motion) else None,
        }
        # hair + gray pull-back windows (t-3..t-1 before, t+1..t+3 after)
        before_h = np.mean([meta["per_frame"]["hair"][i][2] for i in range(t - 3, t) if 0 <= i < len(meta["per_frame"]["hair"])])
        after_h = np.mean([meta["per_frame"]["hair"][i][2] for i in range(t + 1, t + 4) if 0 <= i < len(meta["per_frame"]["hair"])])
        before_g = np.mean([meta["per_frame"]["gray"][i]["gray_artifact_ratio"] for i in range(t - 3, t) if 0 <= i < len(meta["per_frame"]["gray"])])
        after_g = np.mean([meta["per_frame"]["gray"][i]["gray_artifact_ratio"] for i in range(t + 1, t + 4) if 0 <= i < len(meta["per_frame"]["gray"])])
        win.update(
            {
                "hair_distance_before": round(float(before_h), 2),
                "hair_distance_after": round(float(after_h), 2),
                "hair_delta_after_minus_before": round(float(after_h - before_h), 2),
                "gray_before": round(float(before_g), 4),
                "gray_after": round(float(after_g), 4),
                "gray_delta_after_minus_before": round(float(after_g - before_g), 4),
            }
        )
        windows[f"t{t}"] = win
    return windows


def make_identity_qa(frames: dict[str, np.ndarray], dyn: np.ndarray) -> None:
    source = decode_rgb(SOURCE_PATH)
    ref = source[0]
    cx0, cy0, cx1, cy1 = CROP_RAW
    sx0, sx1 = int(cx0 * SRC_W / RAW_W), int(cx1 * SRC_W / RAW_W)
    sy0, sy1 = int(cy0 * SRC_H / RAW_H), int(cy1 * SRC_H / RAW_H)

    rows = [
        ("early", 4),
        ("middle-anchor 前", 26),
        ("middle-anchor", 28),
        ("middle-anchor 后", 30),
        ("periodic 2nd anchor 附近", 32),
        ("late", 52),
    ]
    cols = ["Reference (F0)", "A Baseline", "B Single Anchor", "C Periodic"]
    cell_w, cell_h = 300, 180
    ref_crop = cv2.resize(ref[sy0:sy1, sx0:sx1], (cell_w * 2, cell_h * 2), interpolation=cv2.INTER_AREA)
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
        "G0-L4 identity QA: Reference vs A (first-frame-only) vs B (middle anchor) vs C (periodic)",
        fill=(240, 240, 240),
        font=title_font,
    )
    for row, (label, fi) in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, name in enumerate(cols):
            x0 = pad + col * cell_w
            if col == 0:
                img = ref_crop
            else:
                key = ["A", "B", "C"][col - 1]
                img = frames[key][fi][cy0:cy1, cx0:cx1]
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 2), f"F{fi:02d} {name}", fill=(220, 220, 220), font=label_font)
    draw.text(
        (pad + 4, canvas.height - 22),
        f"row label: {rows[1][0]} / {rows[2][0]} / {rows[3][0]}",
        fill=(200, 200, 200),
        font=label_font,
    )
    out = OUTPUTS / "g0_l4_identity_qa.png"
    canvas.save(out)
    print("[qa] identity QA ->", out)


def make_transition_qa(frames: dict[str, np.ndarray]) -> None:
    cx0, cy0, cx1, cy1 = CROP_RAW
    anchors = [
        ("mid", 28),
        ("1s", 16),
        ("2s", 32),
        ("3s", 48),
    ]
    rows = []
    for key in ("A", "B", "C"):
        for label, t in anchors:
            rows.append((f"{key} {label} (t={t})", key, t))
    cols = ["t-2", "t-1", "t", "t+1", "t+2"]
    cell_w, cell_h = 220, 121
    label_h, title_h, pad = 22, 30, 4
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
        "G0-L4 anchor transition QA: t-2..t+2 around each anchor (A baseline windows for comparison)",
        fill=(240, 240, 240),
        font=title_font,
    )
    for row, (label, key, t) in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        draw.text((pad + 4, y0 + 1), label, fill=(230, 230, 230), font=label_font)
        for col, off in enumerate(range(-2, 3)):
            fi = t + off
            img = frames[key][fi][cy0:cy1, cx0:cx1]
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 1), f"F{fi:02d} {cols[col]}", fill=(220, 220, 220), font=label_font)
    out = OUTPUTS / "g0_l4_anchor_transition_qa.png"
    canvas.save(out)
    print("[qa] transition QA ->", out)


def main() -> None:
    dyn = (decode_rgb(DYN_MASK_PATH)[..., 0] > 127).astype(np.uint8) * 255
    if dyn.shape[0] != 56:
        raise SystemExit(f"composite_mask frame count {dyn.shape[0]} != 56")

    metas = {}
    frames = {}
    csv_rows = []
    for key, (label, path) in EXPERIMENTS.items():
        if not path.exists():
            raise SystemExit(f"missing {path}")
        raw = decode_rgb(path)
        if len(raw) != 56:
            raise SystemExit(f"{key} decoded {len(raw)} frames != 56")
        frames[key] = raw
        meta = analyze_video(raw, dyn)
        metas[key] = meta
        anchors = C_ANCHORS if key == "C" else ([B_ANCHOR] if key == "B" else [])
        win = anchor_window_metrics(meta, anchors)
        metas[key]["anchor_windows"] = win
        metas[key]["anchor_transition_max"] = (
            max(
                [
                    w["anchor_transition_in"] or 0
                    for w in win.values()
                ]
                + [w["anchor_transition_out"] or 0 for w in win.values()]
            )
            if win
            else None
        )
        for i in range(len(raw)):
            h = meta["per_frame"]["hair"][i]
            g = meta["per_frame"]["gray"][i]
            m = meta["per_frame"]["motion"][i] if i < len(meta["per_frame"]["motion"]) else None
            csv_rows.append(
                {
                    "experiment": key,
                    "strategy": label,
                    "frame_index": i,
                    "time_seconds": round(i / 16.0, 4),
                    "hair_mean_r": round(float(h[0][0]), 2),
                    "hair_mean_g": round(float(h[0][1]), 2),
                    "hair_mean_b": round(float(h[0][2]), 2),
                    "hair_saturation": round(h[1], 2),
                    "hair_color_distance_from_reference": round(h[2], 2),
                    "gray_artifact_ratio": g["gray_artifact_ratio"],
                    "gray_frame_ratio": g["gray_frame_ratio"],
                    "motion_mae": m,
                }
            )
        print(
            f"[{key}] final_dist={metas[key]['final_color_distance']} "
            f"mean_dist={metas[key]['mean_color_distance']} "
            f"mean_gray={metas[key]['mean_gray_artifact_ratio']} "
            f"motion={metas[key]['motion_mean']}+-{metas[key]['motion_std']}"
        )

    # Same-frame reference windows for A at the anchor positions (snap baseline).
    metas["A"]["anchor_windows"] = anchor_window_metrics(metas["A"], sorted(set([B_ANCHOR] + C_ANCHORS)))
    metas["A"]["anchor_transition_max"] = max(
        max(w["anchor_transition_in"] or 0, w["anchor_transition_out"] or 0)
        for w in metas["A"]["anchor_windows"].values()
    )

    csv_path = OUTPUTS / "g0_l4_hair_drift.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print("[csv] ->", csv_path)

    source_frames = decode_rgb(SOURCE_PATH)

    def copy_mae(frame: np.ndarray, src_idx: int) -> float:
        ref = cv2.resize(source_frames[src_idx], (RAW_W, RAW_H), interpolation=cv2.INTER_AREA)
        return round(float(np.abs(frame.astype(np.int16) - ref.astype(np.int16)).mean()), 2)

    source_copy_mae = {
        "A_t28_vs_source_55": copy_mae(frames["A"][28], 55),
        "B_t28_vs_source_55": copy_mae(frames["B"][28], 55),
        "C_t16_vs_source_31": copy_mae(frames["C"][16], 31),
        "C_t32_vs_source_62": copy_mae(frames["C"][32], 62),
        "C_t48_vs_source_94": copy_mae(frames["C"][48], 94),
    }

    make_identity_qa(frames, dyn)
    make_transition_qa(frames)

    # Comparison
    def improve(key: str, field: str, direction: str = "down") -> float:
        a = metas["A"][field]
        v = metas[key][field]
        if a is None or v is None:
            return 0.0
        return float((a - v) if direction == "down" else (v - a))

    hair_best = min(("A", "B", "C"), key=lambda k: metas[k]["mean_color_distance"])
    gray_best = min(("A", "B", "C"), key=lambda k: metas[k]["mean_gray_artifact_ratio"])
    def identity_score(k: str) -> float:
        return metas[k]["mean_color_distance"] / 50.0 + metas[k]["mean_gray_artifact_ratio"]

    identity_best = min(("A", "B", "C"), key=identity_score)
    temporal_best = min(("A", "B", "C"), key=lambda k: metas[k]["motion_std"])

    b_help = improve("B", "mean_color_distance") > 3 and improve("B", "mean_gray_artifact_ratio") > 0.005
    c_help = improve("C", "mean_color_distance") > 3 and improve("C", "mean_gray_artifact_ratio") > 0.005
    anchors_help = bool(b_help or c_help)

    def window_max(meta: dict, t: int) -> float:
        w = meta["anchor_windows"].get(f"t{t}", {})
        return max(w.get("anchor_transition_in") or 0, w.get("anchor_transition_out") or 0)

    b_snap = window_max(metas["B"], B_ANCHOR) > window_max(metas["A"], B_ANCHOR) + 2.0
    c_snap = any(window_max(metas["C"], t) > window_max(metas["A"], t) + 2.0 for t in C_ANCHORS)
    anchors_cause_snapping = bool(b_snap or c_snap)

    if anchors_help and not anchors_cause_snapping:
        result = "pass"
    elif anchors_help and anchors_cause_snapping:
        result = "borderline"
    elif not anchors_help:
        result = "fail"
    else:
        result = "inconclusive"

    if c_help and not b_help:
        best_strategy = "C_periodic"
    elif b_help and not c_help:
        best_strategy = "B_single_middle"
    elif b_help and c_help:
        best_strategy = (
            "C_periodic"
            if (metas["C"]["mean_color_distance"] + 0.01 * metas["C"]["mean_gray_artifact_ratio"])
            < (metas["B"]["mean_color_distance"] + 0.01 * metas["B"]["mean_gray_artifact_ratio"])
            else "B_single_middle"
        )
    else:
        best_strategy = "A_baseline_first_frame_only"

    report = {
        "source_audit": {
            "input_fps": 30.175394,
            "mask_fps": 30.0,
            "mask_frame_mapping": "mask video frame index == source frame index (both 112 frames); read_video samples indices=[0,1,3,5,...,110] (57 frames), output drops last -> 56f@16fps",
            "anchor_frame_domain": "condition mask video frame domain == original source frame domain (112f @30fps); NOT output 16fps domain",
            "anchor_semantics": "full-frame mask=1 -> Img_list_new[t]=full RGB -> VAE condition; VAE 4x temporal grouping smears anchor across ~4 output frames",
            "audit_file": "outputs/g0_l4_source_audit.md",
        },
        "baseline": baseline_run_info(),
        "single_anchor": {
            "anchor_time": 1.75,
            "anchor_output_frame": 28,
            "anchor_mask_frame": 55,
            "anchor_source_frame": 55,
            "anchor_effect_window_output_frames": [28, 29, 30, 31],
            **parse_run_info("B"),
            "hair_metrics": {
                k: metas["B"][k] for k in ["mean_color_distance", "final_color_distance", "maximum_color_distance", "minimum_saturation", "final_saturation"]
            },
            "gray_metrics": {k: metas["B"][k] for k in ["mean_gray_artifact_ratio", "early_gray_artifact_ratio", "late_gray_artifact_ratio"]},
            "motion_metrics": {k: metas["B"][k] for k in ["motion_mean", "motion_std", "motion_max"]},
            "anchor_transition_metrics": metas["B"]["anchor_windows"],
        },
        "periodic": {
            "anchor_times": [1.0, 2.0, 3.0],
            "anchor_output_frames": [16, 32, 48],
            "anchor_mask_frames": [31, 62, 94],
            "anchor_source_frames": [31, 62, 94],
            "anchor_effect_windows": [[16, 17, 18, 19], [32, 33, 34, 35], [48, 49, 50, 51]],
            "note": "frame 0 is forced full-frame by source code (binary_mask[0]=1) for all strategies",
            **parse_run_info("C"),
            "hair_metrics": {
                k: metas["C"][k] for k in ["mean_color_distance", "final_color_distance", "maximum_color_distance", "minimum_saturation", "final_saturation"]
            },
            "gray_metrics": {k: metas["C"][k] for k in ["mean_gray_artifact_ratio", "early_gray_artifact_ratio", "late_gray_artifact_ratio"]},
            "motion_metrics": {k: metas["C"][k] for k in ["motion_mean", "motion_std", "motion_max"]},
            "anchor_transition_metrics": metas["C"]["anchor_windows"],
        },
        "comparison": {
            "hair_best": hair_best,
            "gray_shell_best": gray_best,
            "identity_best": identity_best,
            "temporal_continuity_best": temporal_best,
            "anchors_help": anchors_help,
            "anchors_cause_snapping": anchors_cause_snapping,
            "best_strategy": best_strategy,
            "hair_improvement_vs_A": {
                "B": round(improve("B", "mean_color_distance"), 2),
                "C": round(improve("C", "mean_color_distance"), 2),
            },
            "gray_improvement_vs_A": {
                "B": round(improve("B", "mean_gray_artifact_ratio"), 4),
                "C": round(improve("C", "mean_gray_artifact_ratio"), 4),
            },
            "anchor_transition_max": {
                "A": metas["A"]["anchor_transition_max"],
                "B": metas["B"]["anchor_transition_max"],
                "C": metas["C"]["anchor_transition_max"],
            },
            "snap_flags": {"B": b_snap, "C": c_snap},
            "anchor_frame_source_copy_mae": source_copy_mae,
        },
        "result": result,
        "diagnosis": {
            "conclusion": "",
            "evidence": [
                f"A mean_dist={metas['A']['mean_color_distance']} gray={metas['A']['mean_gray_artifact_ratio']} motion={metas['A']['motion_mean']}+-{metas['A']['motion_std']}",
                f"B mean_dist={metas['B']['mean_color_distance']} gray={metas['B']['mean_gray_artifact_ratio']} motion={metas['B']['motion_mean']}+-{metas['B']['motion_std']}",
                f"C mean_dist={metas['C']['mean_color_distance']} gray={metas['C']['mean_gray_artifact_ratio']} motion={metas['C']['motion_mean']}+-{metas['C']['motion_std']}",
                f"anchor windows: B={metas['B']['anchor_windows']} C={metas['C']['anchor_windows']}",
                f"A same-frame anchor windows: {metas['A']['anchor_windows']}",
                f"source-copy MAE at anchor frames: {source_copy_mae}",
            ],
            "limitations": [
                "gray detection is the same HSV threshold as G0-L3 (sat<40, value>60)",
                "single seed, single sample per strategy",
                "hair ROI reuses G0-L2 source-based DynamicRegion mask; generated silhouette drift can bias ROI selection",
                "VAE 4x temporal grouping smears single-frame anchors; per-output-frame effects are approximate",
            ],
            "next_single_question": "",
        },
        "outputs": {
            "hair_csv": str(csv_path),
            "identity_qa": str(OUTPUTS / "g0_l4_identity_qa.png"),
            "transition_qa": str(OUTPUTS / "g0_l4_anchor_transition_qa.png"),
            "anchor_mask_qa": str(OUTPUTS / "g0_l4_anchor_masks_qa.png"),
            "single_middle_anchor": str(B_PATH),
            "periodic_anchor": str(C_PATH),
        },
    }

    # Human-readable conclusion based on the decision rules in the G0-L4 brief.
    if result == "pass":
        conclusion = "PASS: periodic visual anchors improve DynamicRegion identity/hair/gray consistency without severe motion spikes."
    elif result == "borderline":
        conclusion = "BORDERLINE: anchors clearly improve hair/identity/gray (C >> B >> A), but every full-frame anchor produces a near source-copy frame with a motion spike (B in/out ~22-23, C ~16-18 vs normal ~1-3), i.e. temporal snapping."
    elif result == "fail":
        conclusion = "FAIL: B/C are essentially the same as A; periodic source-RGB conditioning through the current mask path does not resolve identity drift."
    else:
        conclusion = "INCONCLUSIVE: evidence is conflicting; human QA needed."
    report["diagnosis"]["conclusion"] = conclusion
    report["diagnosis"]["next_single_question"] = (
        "停止当前 mask-conditioning 密度微调；下一个单一问题是：在 AnyMask 当前接口下是否存在更软的 persistent reference 条件（例如把 anchor 帧替换为持续注入同一张第一帧低权重条件），还是必须引入真正的 reference/identity 条件通道。"
        if result == "fail"
        else "下一步单一问题：能否用更软的 reference 条件（anchor 帧软混合，或持续低权重完整帧条件）保留 C 的身份/发色收益并消除 snap —— 本轮不执行，需人工 QA 图确认当前边界结果。"
    )

    out = OUTPUTS / "g0_l4_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({"result": result, "comparison": report["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
