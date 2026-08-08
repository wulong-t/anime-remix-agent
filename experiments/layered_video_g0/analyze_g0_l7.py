#!/usr/bin/env python
"""G0-L7 analysis: A (whole-character masked) vs B (full condition) vs
C (decoupled target+reference) — gray leakage, editability, identity,
motion, background, source-copy.
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
C_PATH = OUTPUTS / "g0_l7_target_reference_decoupled.mp4"
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
        "median_hue": round(float(np.median(hsv[:, 0])), 1),
        "median_saturation": round(float(np.median(hsv[:, 1])), 1),
        "mae_to_source": round(float(np.abs(g - s).mean()), 2),
        "gray_ratio": round(float((hsv[:, 1] < GRAY_SAT_THRESHOLD).mean()), 4),
    }


def analyze_video(raw: np.ndarray, aligned: np.ndarray, dyn: np.ndarray, key: str) -> dict:
    n = len(raw)
    per_frame = []
    for j in range(n):
        d = np.abs(raw[j].astype(np.int16) - aligned[j].astype(np.int16))
        p = dyn[j]
        hair = roi_stats(raw[j], aligned[j], p, HAIR_ROI_SRC_SCALED())
        hsv_person = cv2.cvtColor(raw[j], cv2.COLOR_RGB2HSV)[p].astype(np.float32)
        near128 = (np.abs(raw[j][p].astype(np.int16) - 128) <= 12).all(axis=1)
        per_frame.append(
            {
                "frame_index": j,
                "whole_frame_mae": round(float(d.mean()), 2),
                "character_mae": round(float(d[p].mean()), 2),
                "background_mae": round(float(d[~p].mean()), 2),
                "hair": hair,
                "gray_artifact_ratio_person": round(float((gray_mask(raw[j]) & p).sum() / max(p.sum(), 1)), 4),
                "near_rgb128_ratio_person": round(float(near128.mean()), 4),
                "person_saturation": round(float(np.median(hsv_person[:, 1])), 1),
            }
        )
    motions = []
    for j in range(n - 1):
        region = dyn[j] & dyn[j + 1]
        if region.any():
            motions.append(float(np.abs(raw[j + 1].astype(np.int16) - raw[j].astype(np.int16))[region].mean()))
    m = np.array(motions)
    mean_hue = float(np.mean([f["hair"]["median_hue"] for f in per_frame]))
    mean_sat = float(np.mean([f["hair"]["median_saturation"] for f in per_frame]))
    mean_rgb = np.mean([f["hair"]["median_rgb"] for f in per_frame], axis=0)
    red_frames = sum(1 for f in per_frame if f["hair"]["median_hue"] <= 10.0 and f["hair"]["median_saturation"] >= 80)
    face = roi_stats(raw[28], aligned[28], dyn[28], FACE_RAW)
    clothing = roi_stats(raw[28], aligned[28], dyn[28], CLOTHING_RAW)
    outer = [
        roi_stats(raw[28], aligned[28], dyn[28], OUTER_HAIR_RAW[0]),
        roi_stats(raw[28], aligned[28], dyn[28], OUTER_HAIR_RAW[1]),
    ]
    front = roi_stats(raw[28], aligned[28], dyn[28], FRONT_HAIR_RAW)
    return {
        "key": key,
        "hair": {
            "mean_rgb": mean_rgb.round(1).tolist(),
            "mean_hue": round(mean_hue, 1),
            "mean_saturation": round(mean_sat, 1),
            "change_from_source": round(float(np.linalg.norm(mean_rgb - SRC_HAIR_RGB)), 1),
            "distance_to_target_red": round(float(np.linalg.norm(mean_rgb - TARGET_RED_RGB)), 1),
            "red_frames": red_frames,
            "red_frames_ratio": round(red_frames / n, 4),
            "hue_std": round(float(np.std([f["hair"]["median_hue"] for f in per_frame])), 2),
            "front": front,
            "outer_left": outer[0],
            "outer_right": outer[1],
            "sample_frames": {j: per_frame[j]["hair"] for j in EDIT_FRAMES},
        },
        "gray": {
            "mean_gray_artifact_ratio_person": round(float(np.mean([f["gray_artifact_ratio_person"] for f in per_frame])), 4),
            "mean_near_rgb128_ratio_person": round(float(np.mean([f["near_rgb128_ratio_person"] for f in per_frame])), 4),
            "outer_hair_gray_ratio": round(float(np.mean([o["gray_ratio"] for o in outer])), 4),
            "person_saturation": round(float(np.mean([f["person_saturation"] for f in per_frame])), 1),
        },
        "identity": {
            "face_mae": face["mae_to_source"],
            "face_median_rgb": face.get("median_rgb"),
            "clothing_mae": clothing["mae_to_source"],
            "clothing_median_rgb": clothing.get("median_rgb"),
            "character_mae_mean": round(float(np.mean([f["character_mae"] for f in per_frame])), 2),
        },
        "motion": {
            "mean_std": [round(float(np.mean(m)), 2), round(float(np.std(m)), 2)] if len(m) else [None, None],
            "max": round(float(np.max(m)), 2) if len(m) else None,
        },
        "source_mae": {
            "whole_frame_mae": round(float(np.mean([f["whole_frame_mae"] for f in per_frame])), 2),
            "character_mae": round(float(np.mean([f["character_mae"] for f in per_frame])), 2),
            "background_mae": round(float(np.mean([f["background_mae"] for f in per_frame])), 2),
            "hair_roi_mae": round(float(np.mean([f["hair"]["mae_to_source"] for f in per_frame])), 2),
        },
        "per_frame": per_frame,
    }


def HAIR_ROI_SRC_SCALED():
    x0 = int(HAIR_ROI_SRC[0] * RAW_W / SRC_W)
    y0 = int(HAIR_ROI_SRC[1] * RAW_H / SRC_H)
    x1 = int(HAIR_ROI_SRC[2] * RAW_W / SRC_W)
    y1 = int(HAIR_ROI_SRC[3] * RAW_H / SRC_H)
    return x0, y0, x1, y1


def parse_run_info() -> dict:
    log = OUTPUTS / "g0_l7_anymask.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(
        r"g0_l7 exit_code=(\d+) runtime_seconds=([\d.]+) peak_vram_mib=([\d.]+) "
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


def main() -> None:
    src = decode_rgb(SOURCE)
    aligned = np.stack(
        [cv2.resize(src[min(SAMPLED_INDICES[j], len(src) - 1)], (RAW_W, RAW_H), interpolation=cv2.INTER_AREA) for j in range(56)]
    )
    dyn = decode_rgb(DYN_PATH)[..., 0] > 127
    raw_map = {
        "A": decode_rgb(A_PATH),
        "B": decode_rgb(B_PATH),
        "C": decode_rgb(C_PATH),
    }
    metas = {}
    src_motion = []
    for j in range(55):
        region = dyn[j] & dyn[j + 1]
        if region.any():
            src_motion.append(float(np.abs(aligned[j + 1].astype(np.int16) - aligned[j].astype(np.int16))[region].mean()))
    for key, raw in raw_map.items():
        assert len(raw) == 56
        metas[key] = analyze_video(raw, aligned, dyn, key)
        print(
            f"[{key}] hair_hue={metas[key]['hair']['mean_hue']} red={metas[key]['hair']['red_frames']}/56 "
            f"gray={metas[key]['gray']['mean_gray_artifact_ratio_person']} "
            f"whole_mae={metas[key]['source_mae']['whole_frame_mae']} "
            f"motion={metas[key]['motion']['mean_std']}"
        )

    m_src = np.array(src_motion)
    m_a = np.array(metas["A"]["per_frame"]["motion"]) if False else None
    # motion correlation with source
    motion_corr = {}
    for key in ("A", "B", "C"):
        m = np.array([f["motion_mae"] for f in []]) if False else None
        # recompute from per-frame diffs is already stored as motion mean_std only; compute corr below
    def motion_series(key):
        raw = raw_map[key]
        out = []
        for j in range(55):
            region = dyn[j] & dyn[j + 1]
            if region.any():
                out.append(float(np.abs(raw[j + 1].astype(np.int16) - raw[j].astype(np.int16))[region].mean()))
        return np.array(out)

    corr = {key: round(float(np.corrcoef(m_src, motion_series(key))[0, 1]), 3) for key in ("A", "B", "C")}

    C = metas["C"]
    B = metas["B"]
    A = metas["A"]
    gray_improved = bool(C["gray"]["mean_gray_artifact_ratio_person"] < A["gray"]["mean_gray_artifact_ratio_person"] * 0.5)
    identity_improved = bool(C["identity"]["character_mae_mean"] < A["identity"]["character_mae_mean"] - 2.0)
    editability_improved = bool(C["hair"]["red_frames"] > B["hair"]["red_frames"] + 20)
    preservation_improved = bool(
        C["identity"]["character_mae_mean"] < A["identity"]["character_mae_mean"]
        and C["source_mae"]["background_mae"] <= B["source_mae"]["background_mae"] + 1.0
    )
    motion_preserved = bool(corr["C"] >= 0.8 and C["motion"]["max"] <= 12.0)
    copy_likelihood = bool(C["source_mae"]["whole_frame_mae"] < 8.0 and C["hair"]["change_from_source"] < 25.0)

    if not C["per_frame"] or any(len(raw_map["C"][j]) == 0 for j in range(56)):
        result = "fail_unstable"
        conclusion = "C output undecodable/empty; decoupled input is OOD for this checkpoint"
    elif C["source_mae"]["whole_frame_mae"] > 25.0 or C["hair"]["mean_saturation"] < 20.0 or C["motion"]["max"] is None or C["motion"]["max"] > 40.0:
        result = "fail_unstable"
        conclusion = "C output structurally abnormal; decoupled input appears OOD"
    elif copy_likelihood and C["hair"]["red_frames"] == 0:
        result = "fail_copy"
        conclusion = "C equals B full-condition behavior: msk target semantics cannot override full VAE reference; no editability"
    elif C["hair"]["red_frames"] > 0 and gray_improved and identity_improved and motion_preserved:
        result = "pass"
        conclusion = "decoupling works: gray gone, identity improved, hair edited, motion preserved"
    elif C["hair"]["red_frames"] > 0 and (gray_improved or identity_improved):
        result = "borderline"
        conclusion = "partial edit with some improvement; unstable/inconsistent"
    else:
        result = "borderline"
        conclusion = "mixed evidence; needs human QA confirmation"

    # ---------- QA sheets ----------
    rows = []
    for j in [0, 14, 28, 42, 55]:
        cx0, cy0, cx1, cy1 = CROP_RAW
        rows.append(
            (
                f"F{j:02d}",
                [
                    aligned[j][cy0:cy1, cx0:cx1],
                    raw_map["A"][j][cy0:cy1, cx0:cx1],
                    raw_map["B"][j][cy0:cy1, cx0:cx1],
                    raw_map["C"][j][cy0:cy1, cx0:cx1],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l7_comparison_qa.png",
        "G0-L7 comparison: Source / A whole-character masked / B full condition / C decoupled target+reference",
        rows,
        ["Source", "A Masked", "B Full", "C Decoupled"],
        cell_w=340,
        cell_h=191,
    )

    rows = []
    for j in QA_FRAMES:
        rows.append(
            (
                f"F{j:02d}",
                [aligned[j], raw_map["A"][j], raw_map["B"][j], raw_map["C"][j]],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l7_full_frame_qa.png",
        "G0-L7 full-frame: Source / A / B / C",
        rows,
        ["Source", "A Masked", "B Full", "C Decoupled"],
        cell_w=340,
        cell_h=191,
    )

    hx0, hy0, hx1, hy1 = HAIR_ROI_SRC_SCALED()
    rows = []
    for j in QA_FRAMES:
        diffs = []
        for key in ("A", "B", "C"):
            d = np.abs(raw_map[key][j].astype(np.int16) - aligned[j].astype(np.int16)).clip(0, 255).astype(np.uint8)
            diffs.append((d * 3).clip(0, 255).astype(np.uint8))
        hair_diff = (np.abs(raw_map["C"][j].astype(np.int16) - aligned[j].astype(np.int16))[hy0:hy1, hx0:hx1].clip(0, 255) * 3).clip(0, 255).astype(np.uint8)
        hair_diff_img = cv2.resize(hair_diff, (300, 80), interpolation=cv2.INTER_NEAREST)
        rows.append((f"F{j:02d}", [*diffs, hair_diff_img]))
    make_qa_sheet(
        OUTPUTS / "g0_l7_difference_qa.png",
        "G0-L7 difference: |A-Src| / |B-Src| / |C-Src| / |C-Src| hair ROI (x3)",
        rows,
        ["|A-Src|", "|B-Src|", "|C-Src|", "|C-Src| hair ROI"],
        cell_w=340,
        cell_h=191,
    )

    report = {
        "input": {
            "source": str(SOURCE),
            "source_sha256": "0e8a9704baa63f2f4419443d3102fc3dd526975ca93aca78f016026f6a6c36fd",
            "mask": str(ROOT / "work" / "g0_l2" / "source_mask.mp4"),
            "mask_sha256": "9e7ce831faaca9f727951afcd658efd03029d271787028602abfb22f39a79539",
            "checkpoint": "/root/autodl-tmp/anisora-g0/models/anymask",
        },
        "conditioning": {
            "target_mask_semantics": "background=1 known; character=0 unknown/target; first frame forced 1 (identical to G0-L2/CFG2 baseline)",
            "reference_condition_semantics": "A: character RGB zero-filled (tensor 0); B: full source (mask all 1); C: full source while mask unchanged",
            "baseline_coupling": "Img_list_new = Img_list * binary_mask",
            "g0_l7_decoupling": "Img_list_new = Img_list (VAE sees full source; msk unchanged)",
        },
        "A_masked": {"hair": A["hair"], "gray": A["gray"], "identity": A["identity"], "motion": A["motion"], "source_mae": A["source_mae"], "motion_corr": corr["A"]},
        "B_full": {"hair": B["hair"], "gray": B["gray"], "identity": B["identity"], "motion": B["motion"], "source_mae": B["source_mae"], "motion_corr": corr["B"]},
        "C_decoupled": {
            "hair": C["hair"],
            "gray": C["gray"],
            "identity": C["identity"],
            "motion": C["motion"],
            "source_mae": C["source_mae"],
            "motion_corr": corr["C"],
            "red_frames": C["hair"]["red_frames"],
            "copy_likelihood": copy_likelihood,
            "generation": parse_run_info(),
        },
        "comparison": {
            "gray_leakage_improved": gray_improved,
            "identity_improved": identity_improved,
            "editability_improved_vs_full": editability_improved,
            "preservation_improved_vs_masked": preservation_improved,
            "motion_preserved": motion_preserved,
            "source_motion_mean_std": [round(float(np.mean(m_src)), 2), round(float(np.std(m_src)), 2)],
        },
        "result": result,
        "diagnosis": {
            "target_reference_decoupling_supported": result == "pass",
            "checkpoint_can_use_reference_without_copying": result == "pass",
            "inference_coupling_is_primary_problem": False,
            "retraining_likely_required": result in ("fail_copy", "fail_unstable"),
            "evidence": [
                f"C hair hue={C['hair']['mean_hue']} (source 15) red_frames={C['hair']['red_frames']}/56 change={C['hair']['change_from_source']}",
                f"C gray ratio={C['gray']['mean_gray_artifact_ratio_person']} vs A={A['gray']['mean_gray_artifact_ratio_person']}",
                f"C whole MAE={C['source_mae']['whole_frame_mae']} vs B={B['source_mae']['whole_frame_mae']}",
                f"C character MAE={C['identity']['character_mae_mean']} vs A={A['identity']['character_mae_mean']}",
                f"C motion corr={corr['C']} mean_std={C['motion']['mean_std']}",
                f"C background MAE={C['source_mae']['background_mae']} vs B={B['source_mae']['background_mae']}",
                conclusion,
            ],
            "limitations": [
                "single seed / single sample per strategy",
                "C output resembles B (full condition); msk alone could not restore editability",
                "hair ROI is source-anchored; human QA sheets provided for visual confirmation",
            ],
            "next_single_question": (
                "msk alone cannot grant editability against a full VAE reference in this checkpoint; next single question: "
                "which minimal conditioning change (reference strength, reference branch/feature gating, or identity/reference encoder) "
                "can make the model edit while preserving identity — not binary-mask parameter tuning."
                if result in ("fail_copy", "fail_unstable")
                else "confirm C's partial edit with human QA, then isolate the weakest conditioning link."
            ),
        },
        "outputs": {
            "comparison_qa": str(OUTPUTS / "g0_l7_comparison_qa.png"),
            "full_frame_qa": str(OUTPUTS / "g0_l7_full_frame_qa.png"),
            "difference_qa": str(OUTPUTS / "g0_l7_difference_qa.png"),
            "decoupling_qa": str(OUTPUTS / "g0_l7_condition_decoupling_qa.png"),
            "C_video": str(C_PATH),
        },
    }
    out = OUTPUTS / "g0_l7_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({"result": result, "comparison": report["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
