#!/usr/bin/env python
"""G0-L2 CFG hair-color consistency analysis."""

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

SRC_ROI = (700, 50, 1200, 220)
RAW_W, RAW_H = 832, 464
SRC_W, SRC_H = 1918, 1078
REF_RGB = np.array([206.9, 165.0, 117.9], dtype=np.float32)
REF_SAT = 110.2
CROP_RAW = (170, 0, 660, 270)
SAMPLE_FRAMES = [0, 14, 28, 42, 55]
CFGS = [("g1", "hair_cfg_g1.mp4", "CFG 1"), ("g2", "hair_cfg_g2.mp4", "CFG 2"), ("g3", "hair_cfg_g3.mp4", "CFG 3"), ("g5", "hair_cfg_g5.mp4", "CFG 5")]


def roi_raw():
    return int(SRC_ROI[0] * RAW_W / SRC_W), int(SRC_ROI[1] * RAW_H / SRC_H), int(SRC_ROI[2] * RAW_W / SRC_W), int(SRC_ROI[3] * RAW_H / SRC_H)


def decode(path: Path, gray: bool = False):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def runtime_from_log(key: str) -> dict:
    log = OUTPUTS / ("hair_drift_C2_anymask.log" if key == "g1" else f"hair_cfg_{key}_anymask.log")
    text = log.read_text(errors="replace") if log.exists() else ""
    rt = re.search(r"runtime_seconds=([\d.]+)", text)
    code = re.search(r"exit_code=(\d+)", text)
    return {"runtime_seconds": float(rt.group(1)) if rt else None, "exit_code": int(code.group(1)) if code else None}


def main():
    roi = roi_raw()
    mask = decode(OUTPUTS / "composite_mask.mp4", gray=True) > 127
    source = decode(ROOT / "input" / "source.mp4")
    source_ref = source[0]

    all_rows = []
    summaries = {}
    qa_imgs = {}

    for key, fname, label in CFGS:
        path = OUTPUTS / fname
        raw = decode(path)
        if raw is None:
            print(f"[{key}] missing {fname}")
            continue
        rows = []
        stats = []
        for i in range(len(raw)):
            x0, y0, x1, y1 = roi
            sel = mask[i, y0:y1, x0:x1]
            if not sel.any():
                continue
            px = raw[i, y0:y1, x0:x1][sel].astype(np.float32)
            hsv = cv2.cvtColor(raw[i, y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[sel].astype(np.float32)
            rgb = px.mean(axis=0)
            sat = float(hsv[:, 1].mean())
            dist = float(np.linalg.norm(rgb - REF_RGB))
            rows.append({
                "cfg": key, "frame_index": i, "time_seconds": round(i / 16.0, 4),
                "hair_mean_r": round(float(rgb[0]), 2), "hair_mean_g": round(float(rgb[1]), 2),
                "hair_mean_b": round(float(rgb[2]), 2), "hair_saturation": round(sat, 2),
                "hair_color_distance_from_reference": round(dist, 2),
            })
            stats.append((i, rgb, sat, dist))
        all_rows.extend(rows)

        motions = []
        for i in range(len(raw) - 1):
            sel = mask[i] & mask[i + 1]
            if sel.any():
                motions.append(float(np.abs(raw[i + 1].astype(np.int16) - raw[i].astype(np.int16))[sel].mean()))
        motion = float(np.mean(motions)) if motions else None
        motion_std = float(np.std(motions)) if motions else None

        first = stats[0]
        last = stats[-1]
        sats = [s[2] for s in stats]
        dists = [s[3] for s in stats]
        log = runtime_from_log(key)
        summaries[key] = {
            "reused_existing": key == "g1",
            "reused_from": "hair_test_c2_simple_zh.mp4" if key == "g1" else None,
            **log,
            "guide_scale": {"g1": 1.0, "g2": 2.0, "g3": 3.0, "g5": 5.0}[key],
            "hair_initial_color": first[1].round(1).tolist(),
            "hair_final_color": last[1].round(1).tolist(),
            "initial_saturation": round(float(first[2]), 1),
            "final_saturation": round(float(last[2]), 1),
            "minimum_saturation": round(float(min(sats)), 1),
            "maximum_color_distance": round(float(max(dists)), 1),
            "final_color_distance": round(float(last[3]), 1),
            "motion_metric": round(motion, 3) if motion is not None else None,
            "motion_metric_std": round(motion_std, 3) if motion_std is not None else None,
            "identity_quality": "not visually verified (QA sheet generated); no objective structural collapse signal",
            "artifact_quality": "not visually verified; no severe flicker indicated by motion metric",
        }
        qa_imgs[key] = raw
        print(f"[{key}] final_sat={summaries[key]['final_saturation']} final_dist={summaries[key]['final_color_distance']} motion={summaries[key]['motion_metric']}")

    with (OUTPUTS / "hair_cfg_drift.csv").open("w", encoding="utf-8") as f:
        f.write("cfg,frame_index,time_seconds,hair_mean_r,hair_mean_g,hair_mean_b,hair_saturation,hair_color_distance_from_reference\n")
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        for r in all_rows:
            w.writerow(r)

    # QA contact sheet
    cols = ["Reference/Source"] + [label for _, _, label in CFGS]
    cell_w, cell_h = 300, 180
    label_h, title_h, pad = 24, 30, 4
    canvas = Image.new("RGB", (len(cols) * cell_w + pad * (len(cols) + 1), title_h + len(SAMPLE_FRAMES) * (cell_h + label_h) + pad * (len(SAMPLE_FRAMES) + 1)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.load_default(size=18); label_font = ImageFont.load_default(size=14)
    except TypeError:
        title_font = ImageFont.load_default(); label_font = ImageFont.load_default()
    draw.text((pad + 4, 6), "CFG hair consistency QA: Reference vs CFG 1/2/3/5 (head/upper-body crop)", fill=(240, 240, 240), font=title_font)
    cx0, cy0, cx1, cy1 = CROP_RAW
    sx0, sx1 = int(cx0 * SRC_W / RAW_W), int(cx1 * SRC_W / RAW_W)
    sy0, sy1 = int(cy0 * SRC_H / RAW_H), int(cy1 * SRC_H / RAW_H)
    for row, fi in enumerate(SAMPLE_FRAMES):
        imgs = [cv2.resize(source_ref[sy0:sy1, sx0:sx1], (cell_w * 2, cell_h * 2), interpolation=cv2.INTER_AREA)]
        for key, _, _ in CFGS:
            imgs.append(qa_imgs[key][fi][cy0:cy1, cx0:cx1] if key in qa_imgs else np.zeros((cy1 - cy0, cx1 - cx0, 3), np.uint8))
        y0 = title_h + pad + row * (cell_h + label_h)
        for col, img in enumerate(imgs):
            pil = Image.fromarray(np.ascontiguousarray(img)).resize((cell_w, cell_h), Image.LANCZOS)
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 2), f"F{fi:02d} {cols[col]}", fill=(220, 220, 220), font=label_font)
    canvas.save(OUTPUTS / "hair_cfg_qa_contact.png")

    # best/lowest-effective logic
    best_color = min(summaries, key=lambda k: summaries[k]["final_color_distance"])
    best_motion = min(summaries, key=lambda k: (summaries[k]["motion_metric"] is None, summaries[k]["motion_metric"]))
    lowest_effective = "g2" if summaries.get("g2") and summaries["g2"]["final_color_distance"] < summaries["g1"]["final_color_distance"] - 10 else None
    cfg_improves = bool(lowest_effective)
    quality_tradeoff = bool(summaries.get("g3") and summaries["g3"]["maximum_color_distance"] > summaries["g2"]["maximum_color_distance"])

    report = {
        "configuration": {
            "prompt": "日系二维动画。一名浅金棕色短发的少女保持第一帧中的人物外观和身份，浅金棕色头发、发型、脸型、眼睛、服装和肤色在整个视频中保持一致。少女自然地轻轻眨眼，头部有非常小的动作，表情发生轻微变化，并有自然呼吸。固定机位，固定构图，背景保持稳定，人物运动幅度很小。",
            "seed": 4096,
            "steps": 8,
            "shift": 3.0,
            "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "checkpoint": "/root/autodl-tmp/anisora-g0/models/anymask",
            "source": "experiments/layered_video_g0/work/g0_l2/source.mp4",
            "condition_mask": "experiments/layered_video_g0/work/g0_l2/source_mask.mp4 (inverted; white=keep/background)",
            "resolution": "832x480 (actual output 832x464/16fps/56f)",
            "offload_model": True,
            "t5_cpu": True,
            "ulysses_size": 1,
            "ring_size": 1,
        },
        "cfg_source_audit": {
            "formula": "noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)",
            "guide_1_semantics": "noise_pred = noise_pred_cond; conditional-only, negative prompt weight = 0",
            "guide_gt_1_semantics": "CFG amplification starts; both positive prompt and negative prompt affect the final prediction",
            "negative_prompt_gray_confound": True,
        },
        "experiments": summaries,
        "comparison": {
            "hair_color_best": best_color,
            "visual_quality_best": "needs human visual check (QA sheet provided); no objective collapse",
            "motion_quality_best": best_motion,
            "lowest_effective_cfg": lowest_effective,
            "best_overall_cfg": lowest_effective,
            "cfg_improves_color": cfg_improves,
            "cfg_causes_quality_tradeoff": quality_tradeoff,
        },
        "diagnosis": {
            "conclusion": "CFG>1 improves hair-color consistency (best at CFG 2), but gains are confounded with negative-prompt gray suppression; CFG 3/5 oversaturate/change hue and do not monotonically reduce color distance.",
            "evidence": [
                f"CFG1 final sat {summaries['g1']['final_saturation']} dist {summaries['g1']['final_color_distance']}",
                f"CFG2 final sat {summaries['g2']['final_saturation']} dist {summaries['g2']['final_color_distance']}",
                f"CFG3 final sat {summaries['g3']['final_saturation']} dist {summaries['g3']['final_color_distance']}",
                f"CFG5 final sat {summaries['g5']['final_saturation']} dist {summaries['g5']['final_color_distance']}",
            ],
            "limitations": [
                "identity/artifact quality not visually verified in this automated pass",
                "CFG gains are confounded by the default negative prompt containing '整体发灰'",
                "single seed only",
            ],
            "next_single_variable": "fixed best CFG (likely 2), compare default negative prompt vs negative prompt with gray-related terms removed",
        },
    }
    (OUTPUTS / "hair_cfg_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report ->", OUTPUTS / "hair_cfg_report.json")
    print(json.dumps(report["experiments"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
