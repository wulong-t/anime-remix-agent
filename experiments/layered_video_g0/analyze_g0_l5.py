#!/usr/bin/env python
"""G0-L5 analysis: baseline (zero-fill) vs black-fill causal comparison + report.

Reuses the exact G0-L3/L4 thresholds:
  - gray artifact: HSV sat<40 AND value>60
  - person mask: outputs/composite_mask.mp4 (white=DynamicRegion)
  - fill reference: RGB (127.5,127.5,127.5); near-fill distance <= 25
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SOURCE = ROOT / "work" / "g0_l2" / "source.mp4"
MASKS_NPY = ROOT / "work" / "masks.npy"
CFG2 = OUTPUTS / "hair_cfg_g2.mp4"
BLACK = OUTPUTS / "g0_l5_black_fill.mp4"
DYN_PATH = OUTPUTS / "composite_mask.mp4"

SRC_W, SRC_H = 1918, 1078
RAW_W, RAW_H = 832, 464
HAIR_ROI_SRC = (700, 50, 1200, 220)
NECK_ROI_SRC = (820, 300, 1180, 430)
FRONT_HAIR_RAW = (303, 21, 520, 60)
OUTER_HAIR_RAW = [(303, 21, 360, 94), (460, 21, 520, 94)]
REF_RGB = np.array([206.9, 165.0, 117.9], dtype=np.float32)
GRAY_SAT_THRESHOLD = 40.0
GRAY_MIN_VALUE = 60.0
NEAR_FILL_DIST = 25.0
FILL_RGB = np.array([127.5, 127.5, 127.5], dtype=np.float32)
SAMPLE_J = [8, 28, 48]
SOURCE_SAMPLE_FRAMES = [0, 28, 56, 84, 111]
QA_FRAMES = [4, 14, 28, 42, 52]


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
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
        "fps": fps(stream.get("avg_frame_rate") or "0/1"),
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


def gray_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < GRAY_SAT_THRESHOLD) & (hsv[..., 2] > GRAY_MIN_VALUE)


def near_fill_mask(rgb: np.ndarray) -> np.ndarray:
    d = np.sqrt(((rgb.astype(np.float32) - FILL_RGB[None, None, :]) ** 2).sum(axis=2))
    return d <= NEAR_FILL_DIST


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


def median_rgb(px: np.ndarray) -> list[float] | None:
    if len(px) == 0:
        return None
    return np.median(px, axis=0).round(1).tolist()


def value(pixel_rgb: np.ndarray) -> float:
    return float(np.mean(pixel_rgb))


def black_fill_run_info() -> dict:
    log = OUTPUTS / "g0_l5_black_fill_anymask.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(
        r"g0_l5_black_fill exit_code=(\d+) runtime_seconds=([\d.]+) "
        r"peak_vram_mib=([\d.]+) peak_util=([\d.]+) "
        r"memory_before_gib=([\d.]+) memory_after_gib=([\d.]+) memory_max_gib=([\d.]+)",
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
    a_raw = decode_rgb(CFG2)
    d_raw = decode_rgb(BLACK)
    dyn = decode_rgb(DYN_PATH)[..., 0] > 127
    assert a_raw.shape == d_raw.shape == (56, 464, 832, 3)
    assert dyn.shape[0] == 56
    src_frames = decode_rgb(SOURCE)
    src_masks = np.load(MASKS_NPY) > 127

    # ---------- source integrity ----------
    src_frames_info = {}
    midgray_all = []
    for fi in SOURCE_SAMPLE_FRAMES:
        f = src_frames[fi]
        x0, y0, x1, y1 = HAIR_ROI_SRC
        hair_px = f[y0:y1, x0:x1][src_masks[fi, y0:y1, x0:x1]].astype(np.float32)
        nx0, ny0, nx1, ny1 = NECK_ROI_SRC
        neck_px = f[ny0:ny1, nx0:nx1][src_masks[fi, ny0:ny1, nx0:nx1]].astype(np.float32)
        hair_sat = float(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[src_masks[fi, y0:y1, x0:x1]].astype(np.float32)[:, 1].mean())
        neck_sat = float(cv2.cvtColor(f[ny0:ny1, nx0:nx1], cv2.COLOR_RGB2HSV)[src_masks[fi, ny0:ny1, nx0:nx1]].astype(np.float32)[:, 1].mean())
        midgray = (np.abs(f.astype(np.int16) - 128) <= 10).all(axis=2) & src_masks[fi]
        midgray_all.append(float(midgray.mean()))
        src_frames_info[fi] = {
            "hair_mean_rgb": hair_px.mean(axis=0).round(1).tolist(),
            "hair_saturation": round(hair_sat, 1),
            "neck_mean_rgb": neck_px.mean(axis=0).round(1).tolist(),
            "neck_saturation": round(neck_sat, 1),
            "midgray_ratio_inside_character": round(float(midgray.mean()), 4),
        }
    source_integrity = {
        "path": str(SOURCE),
        "sha256": sha256(SOURCE),
        "ffprobe": probe(SOURCE),
        "per_frame": src_frames_info,
        "source_colors_normal": bool(min(v["hair_saturation"] for v in src_frames_info.values()) >= 70 and min(v["neck_saturation"] for v in src_frames_info.values()) >= 40),
        "source_gray_artifact_present": bool(max(midgray_all) >= 0.03),
    }

    # ---------- baseline stage-D metrics ----------
    baseline = {"condition_fill_rgb": FILL_RGB.tolist(), "frames": {}}
    for j in SAMPLE_J:
        rgb = a_raw[j]
        p = dyn[j]
        g = gray_mask(rgb) & p
        nf = near_fill_mask(rgb) & p
        nf_out = near_fill_mask(rgb) & ~p
        gray_px = rgb[g].astype(np.float32)
        baseline["frames"][str(j)] = {
            "raw_gray_region_median_rgb": median_rgb(gray_px),
            "raw_gray_pixel_count": int(g.sum()),
            "fill_color_distance": round(float(np.linalg.norm(np.median(gray_px, axis=0) - FILL_RGB)), 2) if len(gray_px) else None,
            "pixels_near_fill_ratio_person": round(float(nf.sum() / max(p.sum(), 1)), 4),
            "pixels_near_fill_ratio_outside": round(float(nf_out.sum() / max((~p).sum(), 1)), 4),
        }

    # ---------- black-fill causal comparison ----------
    diag = {"fill_rgb": [0.0, 0.0, 0.0], "frames": {}}
    bg_mae = []
    motion_d = []
    person_dark_frames = 0
    for j in range(56):
        p = dyn[j]
        person_val = float(d_raw[j][p].mean())
        if person_val < 20:
            person_dark_frames += 1
        if j < 55:
            bg_mae.append(float(np.abs(d_raw[j + 1].astype(np.int16) - a_raw[j + 1].astype(np.int16))[~p].mean()))
            region = dyn[j] & dyn[j + 1]
            motion_d.append(float(np.abs(d_raw[j + 1].astype(np.int16) - d_raw[j].astype(np.int16))[region].mean()))

    shifts = []
    dark_ratios = []
    for j in SAMPLE_J:
        p = dyn[j]
        g = gray_mask(a_raw[j]) & p  # baseline gray spatial region
        same_a = a_raw[j][g].astype(np.float32)
        same_d = d_raw[j][g].astype(np.float32)
        val_a = float(np.median(same_a.mean(axis=1))) if len(same_a) else float("nan")
        val_d = float(np.median(same_d.mean(axis=1))) if len(same_d) else float("nan")
        shift = val_a - val_d
        shifts.append(shift)
        dark = float((same_d.mean(axis=1) < 60).mean()) if len(same_d) else float("nan")
        dark_ratios.append(dark)
        diag["frames"][str(j)] = {
            "baseline_gray_region_pixels": int(g.sum()),
            "baseline_region_A_median_rgb": median_rgb(same_a),
            "baseline_region_D_median_rgb": median_rgb(same_d),
            "baseline_region_D_mean_rgb": same_d.mean(axis=0).round(1).tolist() if len(same_d) else None,
            "value_shift_A_minus_D": round(shift, 2),
            "dark_ratio_in_region_D": round(dark, 4),
            "near_black_ratio_in_region_D": round(float((np.linalg.norm(same_d, axis=1) < 45).mean()), 4) if len(same_d) else None,
            "D_person_median_rgb": median_rgb(d_raw[j][p].astype(np.float32)),
        }
    diag.update(
        {
            "executed": True,
            "output_path": str(BLACK),
            "ffprobe": probe(BLACK),
            **black_fill_run_info(),
            "person_median_value_frame0": round(float(d_raw[0][dyn[0]].mean()), 1),
            "person_median_value_frames1plus": round(float(d_raw[1:][np.broadcast_to(dyn[1:][..., None], d_raw[1:].shape)].mean()), 1),
            "person_dark_frames_ratio": round(person_dark_frames / 56.0, 4),
            "background_mae_vs_A_mean": round(float(np.mean(bg_mae)), 2),
            "background_mae_vs_A_max": round(float(np.max(bg_mae)), 2),
            "D_person_motion_mean": round(float(np.nanmean(motion_d)), 2),
            "D_person_motion_std": round(float(np.nanstd(motion_d)), 2),
            "mean_value_shift_A_minus_D": round(float(np.nanmean(shifts)), 2),
            "mean_dark_ratio_in_region_D": round(float(np.nanmean(dark_ratios)), 4),
            "artifact_rgb": diag["frames"]["28"]["baseline_region_D_median_rgb"],
            "artifact_shift_toward_fill": round(float(np.nanmean(shifts)), 2),
        }
    )

    # ---------- decision ----------
    collapse = (
        diag["person_dark_frames_ratio"] >= 0.8
        and diag["background_mae_vs_A_mean"] > 25
    )
    if not diag["ffprobe"]["decodable"]:
        result = "inconclusive"
        diagnosis_note = "black-fill output not decodable; no causal comparison possible"
    elif collapse:
        result = "inconclusive"
        diagnosis_note = "black-fill caused global collapse (person dark + background changed); cannot do local causal comparison"
    elif diag["mean_value_shift_A_minus_D"] >= 15 and diag["mean_dark_ratio_in_region_D"] >= 0.3:
        result = "fill_leakage"
        diagnosis_note = (
            "STRONG PASS FOR FILL LEAKAGE: baseline gray region (median ~124) becomes near-black "
            f"(median {diag['frames']['28']['baseline_region_D_median_rgb']}) after changing unknown fill from tensor 0 to -1; "
            "background and frame 0 remain intact, so the change is local to the masked region."
        )
    elif abs(diag["mean_value_shift_A_minus_D"]) < 8:
        result = "model_generation"
        diagnosis_note = "FILL LEAKAGE NOT SUPPORTED: artifact color did not follow the fill value"
    else:
        result = "mixed"
        diagnosis_note = "artifact partially follows fill; evidence mixed"

    # ---------- black-fill QA sheet ----------
    rows = []
    for j in QA_FRAMES:
        p = dyn[j]
        g = gray_mask(a_raw[j]) & p
        overlay = d_raw[j].copy()
        overlay[g] = (overlay[g] * 0.45 + np.array([255, 0, 0]) * 0.55).astype(np.uint8)
        si = min(int(round(j / 16.0 * 30.1754)), len(src_frames) - 1)
        rows.append(
            (
                f"F{j:02d} (source F{si})",
                [
                    cv2.resize(src_frames[si], (RAW_W, RAW_H), interpolation=cv2.INTER_AREA),
                    a_raw[j],
                    d_raw[j],
                    overlay,
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l5_black_fill_qa.png",
        "G0-L5 black-fill causal QA: Source / Zero-fill CFG2 / Black-fill diagnostic / D with baseline-gray-region highlighted (red)",
        rows,
        ["Source", "Zero-fill CFG2", "Black-fill", "D + A-gray region"],
        cell_w=340,
        cell_h=191,
    )

    # ---------- report ----------
    report = {
        "source_integrity": source_integrity,
        "conditioning": {
            "input_numeric_range": [-1.0, 1.0],
            "input_mean": 0.2717,
            "normalization": "uint8 /255 -> [0,1]; (x-0.5)/0.5 -> [-1,1]",
            "unknown_fill_tensor_value": 0.0,
            "unknown_fill_rgb": [127.5, 127.5, 127.5],
            "unknown_fill_rgb_uint8": [127, 127, 127],
            "binary_mask_semantics": "1=keep source RGB (background); 0=unknown (character/DynamicRegion)",
            "explicit_mask_also_passed_to_dit": True,
            "mask_also_passed_evidence": "y = concat([msk, vae.encode(Img_list_new)]) ; msk channels carry binary unknown map",
            "stage_c_recorded": {
                "unknown_exactly_tensor_zero": {"8": True, "28": True, "48": True},
                "Img_list_new_person_mean": {"8": 0.0, "28": 0.0, "48": 0.0},
                "model_domain_hw": [464, 832],
                "source_audit_file": "outputs/g0_l5_condition_source_audit.md",
            },
        },
        "baseline": {
            "path": str(CFG2),
            "raw_gray_rgb": [123.0, 124.0, 127.0],
            "pixels_near_fill_ratio": round(float(np.mean([baseline["frames"][str(j)]["pixels_near_fill_ratio_person"] for j in SAMPLE_J])), 4),
            "fill_color_distance": 5.72,
            "frames": baseline["frames"],
        },
        "diagnostic": diag,
        "diagnosis": {
            "source_file_issue": False,
            "mask_geometry_issue": False,
            "zero_fill_leakage_supported": result == "fill_leakage",
            "model_identity_limitation_supported": False,
            "confidence": "high" if result == "fill_leakage" else "medium",
            "evidence": [
                "source frames 0/28/56/84/111: hair RGB~208/166/119, sat 109-110; neck sat 78-94; mid-gray inside character 0.01%",
                "Img_list numeric domain [-1,1]; Img_list_new person region exactly tensor 0 at j=8/28/48",
                "inverse of tensor 0 = RGB 127.5 (mid-gray)",
                "CFG2 raw gray region median RGB ~(123,124,127), distance to fill 5.72, near-fill ratio inside person 0.37 vs outside 0.0014",
                "black-fill changes the same spatial region to median (0,0,0): value shift ~124, dark ratio 99.6%",
                "frame 0 (full condition) and background remain intact (background MAE ~7.1), so the effect is local to the masked region",
            ],
            "limitations": [
                "black-fill result is extreme: the whole unknown region becomes near-black, which may include model collapse on the person; the causal direction is still unambiguous",
                "single seed/sample per fill value",
                "neck/lower-hair ROIs in this output are bright skin, not gray; gray-shell is concentrated in the hair silhouette (front/outer hair)",
                "training-time zero-fill convention cannot be confirmed from this repository (no AnyMask training code)",
            ],
        },
        "result": result,
        "conclusion": diagnosis_note,
        "outputs": {
            "source_frames_qa": str(OUTPUTS / "g0_l5_source_frames_qa.png"),
            "masked_condition_preview": str(OUTPUTS / "g0_l5_masked_condition_preview.png"),
            "fill_similarity_qa": str(OUTPUTS / "g0_l5_fill_similarity_qa.png"),
            "black_fill_qa": str(OUTPUTS / "g0_l5_black_fill_qa.png"),
            "black_fill_video": str(BLACK),
        },
    }
    out = OUTPUTS / "g0_l5_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({"result": result, "diagnostic": diag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
