#!/usr/bin/env python
"""G1-B post-run analysis: fidelity, composition, motion, identity, background."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
GEN = OUTPUTS / "g1b_new_shot.mp4"
NEW = WORK / "keyframes" / "new_start.png"
BG = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/outputs/background.png")
MASK = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/work/masks.npy")
RUN_RESULT = OUTPUTS / "g1b_run_result.json"
PREP_META = WORK / "prep_meta.json"

SOURCE_FRAME = 0
DX = 192  # 10% of 1918, from prep


def decode_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def probe(path: Path) -> dict:
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
    s = info["streams"][0]
    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
    return {
        "codec": s.get("codec_name"),
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "fps": float(num) / float(den) if float(den) else None,
        "frame_count": int(s.get("nb_read_frames") or 0),
        "duration": float(info["format"].get("duration", 0.0)),
    }


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_qa_sheet(path: Path, title: str, rows, cols, cell_w=220, cell_h=124) -> None:
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
    draw.text((pad + 4, 6), title, fill=(240, 240, 240), font=_font(18))
    for row, (label, imgs) in enumerate(rows):
        y0 = title_h + pad + row * (cell_h + label_h)
        draw.text((pad + 4, y0 + 1), label, fill=(230, 230, 230), font=_font(13))
        for col, img in enumerate(imgs):
            pil = Image.fromarray(np.ascontiguousarray(img)).resize(
                (cell_w, cell_h), Image.LANCZOS
            )
            x0 = pad + col * cell_w
            canvas.paste(pil, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 1), cols[col], fill=(220, 220, 220), font=_font(13))
    canvas.save(path)
    print("[qa]", path)


def largest_component(mask: np.ndarray, min_frac=0.03) -> np.ndarray:
    m = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=bool)
    area = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(area)) + 1
    if area[idx - 1] < mask.size * min_frac:
        return np.zeros_like(mask, dtype=bool)
    return labels == idx


def main() -> None:
    gen = decode_rgb(GEN)
    H, W = gen.shape[1], gen.shape[2]
    p = probe(GEN)
    print("[probe]", json.dumps(p))

    new_pil = Image.open(NEW).convert("RGB")
    new_resized = cv2.resize(np.asarray(new_pil), (W, H), interpolation=cv2.INTER_AREA)
    bg_pil = Image.open(BG).convert("RGB")
    bg_resized = cv2.resize(np.asarray(bg_pil), (W, H), interpolation=cv2.INTER_AREA)

    mask_full = (np.load(MASK)[SOURCE_FRAME].astype(np.float32) / 255.0)
    mask_resized = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_AREA)
    dx_out = int(round(W * DX / 1918.0))
    shifted_alpha = np.zeros((H, W), dtype=np.float32)
    shifted_alpha[:, dx_out:] = mask_resized[:, : W - dx_out]

    # ---- fidelity: generated F0 vs new_start ----
    a = gen[0].astype(np.float32)
    b = new_resized.astype(np.float32)
    d = np.abs(a - b)
    mae = float(d.mean())
    psnr = float(10 * np.log10(255**2 / max(mae**2, 1e-6)))
    fidelity = {
        "first_frame_mae": round(mae, 2),
        "first_frame_psnr": round(psnr, 2),
        "per_channel_mae": d.mean(axis=(0, 1)).round(2).tolist(),
        "generated_frame": 0,
        "reference": str(NEW),
        "resize_method": "INTER_AREA",
    }
    print("[fidelity]", json.dumps(fidelity, indent=2))

    # ---- composition preservation ----
    expected_box = (
        int((576 + DX) * W / 1918) - 70,
        0,
        int((1324 + DX) * W / 1918) + 70,
        int(947 * H / 1078) + 40,
    )
    expected_cx = float((953.226 + DX) * W / 1918)
    centroids = []
    bboxes = []
    char_masks = []
    for fr in gen:
        diff = np.abs(fr.astype(np.int16) - bg_resized.astype(np.int16)).mean(axis=2)
        cand = diff > 30
        x0, y0, x1, y1 = expected_box
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)
        cand_restricted = np.zeros_like(cand)
        cand_restricted[y0:y1, x0:x1] = cand[y0:y1, x0:x1]
        comp = largest_component(cand_restricted)
        char_masks.append(comp)
        ys, xs = np.where(comp)
        if len(xs) == 0:
            centroids.append(None)
            bboxes.append(None)
        else:
            centroids.append((float(xs.mean()), float(ys.mean())))
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    valid_c = [c for c in centroids if c is not None]
    valid_b = [b for b in bboxes if b is not None]
    cx_traj = [c[0] for c in valid_c]
    cy_traj = [c[1] for c in valid_c]
    comp = {
        "expected_new_center_x": round(expected_cx, 2),
        "original_center_x_scaled": round(953.226 * W / 1918, 2),
        "centroid_x_mean": round(float(np.mean(cx_traj)), 2),
        "centroid_x_std": round(float(np.std(cx_traj)), 2),
        "centroid_x_min": round(float(np.min(cx_traj)), 2),
        "centroid_x_max": round(float(np.max(cx_traj)), 2),
        "centroid_y_mean": round(float(np.mean(cy_traj)), 2),
        "drift_px": round(float(np.mean(cx_traj) - expected_cx), 2),
        "bbox_frame0": bboxes[0],
        "bbox_frame40": bboxes[40],
        "bbox_frame80": bboxes[80],
        "tracked_frames": len(valid_c),
    }
    print("[composition]", json.dumps(comp, indent=2))

    # ---- motion: adjacent MAE (full + head ROI) ----
    adj = []
    head_adj = []
    head_box = (int(500 * W / 1280), 0, int(1050 * W / 1280), int(320 * H / 704))
    for j in range(len(gen) - 1):
        f0 = gen[j].astype(np.int16)
        f1 = gen[j + 1].astype(np.int16)
        adj.append(float(np.abs(f0 - f1).mean()))
        x0, y0, x1, y1 = head_box
        head_adj.append(float(np.abs(f0[y0:y1, x0:x1] - f1[y0:y1, x0:x1]).mean()))
    adj = np.array(adj)
    head_adj = np.array(head_adj)

    # optical flow in head ROI (horizontal component = head-turn evidence)
    flow_dx = []
    flow_mag = []
    for j in range(len(gen) - 1):
        g0 = cv2.cvtColor(gen[j], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(gen[j + 1], cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = head_box
        fl = cv2.calcOpticalFlowFarneback(
            g0[y0:y1, x0:x1], g1[y0:y1, x0:x1], None,
            0.5, 3, 15, 3, 5, 1.2, 0,
        )
        flow_dx.append(float(fl[..., 0].mean()))
        flow_mag.append(float(np.sqrt((fl**2).sum(axis=2)).mean()))

    freeze_run = 0
    max_freeze = 0
    for v in adj:
        if v < 0.5:
            freeze_run += 1
            max_freeze = max(max_freeze, freeze_run)
        else:
            freeze_run = 0
    motion = {
        "adjacent_frame_mae_mean": round(float(adj.mean()), 3),
        "adjacent_frame_mae_std": round(float(adj.std()), 3),
        "adjacent_frame_mae_max": round(float(adj.max()), 3),
        "argmax_transition": int(adj.argmax()),
        "head_roi_mae_mean": round(float(head_adj.mean()), 3),
        "head_roi_mae_max": round(float(head_adj.max()), 3),
        "head_flow_dx_mean": round(float(np.mean(flow_dx)), 4),
        "head_flow_mag_mean": round(float(np.mean(flow_mag)), 4),
        "max_freeze_run_frames": max_freeze,
        "snap_detected": bool(float(adj.max()) > 20.0),
        "freeze_detected": bool(max_freeze >= 8),
        "g1a2_adjacent_mean_reference": 0.956,
    }
    print("[motion]", json.dumps(motion, indent=2))

    # ---- identity ROIs (source ROIs + 192px x-translation, scaled to output) ----
    def scale_shifted_box(b):
        x0, y0, x1, y1 = b
        return (
            int((x0 + DX) * W / 1918),
            int(y0 * H / 1078),
            int((x1 + DX) * W / 1918),
            int(y1 * H / 1078),
        )

    HAIR = scale_shifted_box((700, 50, 1200, 220))
    FACE = scale_shifted_box((997, 206, 1160, 444))
    CLOTH = scale_shifted_box((700, 480, 1200, 900))

    def roi_median(frame, box):
        x0, y0, x1, y1 = box
        return np.median(frame[y0:y1, x0:x1].reshape(-1, 3), axis=0)

    hair_colors = [roi_median(gen[j], HAIR) for j in range(len(gen))]
    face_colors = [roi_median(gen[j], FACE) for j in range(len(gen))]
    cloth_colors = [roi_median(gen[j], CLOTH) for j in range(len(gen))]

    def stats(name, colors):
        return {
            "mean_rgb": np.mean(colors, axis=0).round(1).tolist(),
            "std_rgb": np.std(colors, axis=0).round(1).tolist(),
            "frame0": colors[0].round(1).tolist(),
            "frame40": colors[40].round(1).tolist(),
            "frame80": colors[80].round(1).tolist(),
            "roi": name,
        }

    identity = {
        "hair": stats("hair", hair_colors),
        "face": stats("face", face_colors),
        "clothing": stats("clothing", cloth_colors),
        "hair_color_note": "light golden-brown expected; compare mean RGB",
    }
    print("[identity]", json.dumps(identity, indent=2))

    # ---- background temporal drift ----
    bg_mask = shifted_alpha <= 0.5
    bg_adj = []
    bg_vs_f0 = []
    for j in range(len(gen) - 1):
        d1 = np.abs(gen[j + 1].astype(np.int16) - gen[j].astype(np.int16))
        bg_adj.append(float(d1[bg_mask].mean()))
        d2 = np.abs(gen[j + 1].astype(np.int16) - gen[0].astype(np.int16))
        bg_vs_f0.append(float(d2[bg_mask].mean()))
    background = {
        "temporal_drift_mean": round(float(np.mean(bg_adj)), 3),
        "temporal_drift_max": round(float(np.max(bg_adj)), 3),
        "drift_vs_frame0_mean": round(float(np.mean(bg_vs_f0)), 3),
        "drift_vs_frame0_max": round(float(np.max(bg_vs_f0)), 3),
    }
    print("[background]", json.dumps(background, indent=2))

    # ---- head-turn / blink heuristics ----
    fx0, fy0, fx1, fy1 = FACE
    skin = []
    for fr in gen:
        r = fr[fy0:fy1, fx0:fx1, 0].astype(np.int16)
        g = fr[fy0:fy1, fx0:fx1, 1].astype(np.int16)
        b = fr[fy0:fy1, fx0:fx1, 2].astype(np.int16)
        m = (r > 70) & (r > g + 5) & (g > b) & (r - b > 20)
        ys, xs = np.where(m)
        skin.append((float(xs.mean()) if len(xs) else float("nan")))
    skin = np.array(skin)
    valid_skin = skin[~np.isnan(skin)]
    head_turn = {
        "face_skin_centroid_x_first": round(float(skin[0]), 2),
        "face_skin_centroid_x_last": round(float(skin[-1]), 2),
        "face_skin_centroid_x_delta": round(float(skin[-1] - skin[0]), 2),
        "face_skin_centroid_x_mean": round(float(np.nanmean(skin)), 2),
        "note": "positive delta = skin centroid moves right within face ROI",
    }
    print("[head_turn]", json.dumps(head_turn, indent=2))

    # blink heuristic: edge energy in eye band
    ey0 = fy0 + int((fy1 - fy0) * 0.35)
    ey1 = fy0 + int((fy1 - fy0) * 0.62)
    ex0 = fx0 + int((fx1 - fx0) * 0.15)
    ex1 = fx1 - int((fx1 - fx0) * 0.15)
    energies = []
    for fr in gen:
        crop = cv2.cvtColor(fr[ey0:ey1, ex0:ex1], cv2.COLOR_RGB2GRAY)
        energies.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()))
    energies = np.array(energies)
    e_mean, e_std = float(energies.mean()), float(energies.std())
    dips = [int(j) for j in range(2, len(energies) - 2) if energies[j] < e_mean - 0.5 * e_std]
    blink = {
        "eye_edge_energy_mean": round(e_mean, 2),
        "eye_edge_energy_std": round(e_std, 2),
        "dip_frames_below_0.5std": dips,
        "possible_blink": bool(dips),
        "caveat": "heuristic only; human QA sheet required",
    }
    print("[blink]", json.dumps(blink, indent=2))

    # ---- QA sheet: head/upper-body crops ----
    samples = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    crop_box = (int(360 * W / 1280), 0, int(1180 * W / 1280), int(600 * H / 704))
    x0, y0, x1, y1 = crop_box
    rows = [("head/upper body", [gen[j][y0:y1, x0:x1] for j in samples])]
    make_qa_sheet(
        OUTPUTS / "g1b_head_motion_qa.png",
        "G1-B head motion: F0/F10/F20/F30/F40/F50/F60/F70/F80 (head/upper-body crop)",
        rows,
        [f"F{j}" for j in samples],
        cell_w=240,
        cell_h=180,
    )

    # ---- decision ----
    prep = json.loads(PREP_META.read_text())
    run = json.loads(RUN_RESULT.read_text())
    keyframe_ok = (
        prep["composition"]["character_rgb_mae"] == 0.0
        and prep["composition"]["background_outside_mae"] < 1.0
        and not prep["composition"]["clipped"]
        and 0.08 <= prep["composition"]["delta_x_ratio"] <= 0.12
    )
    fidelity_ok = mae < 12.0
    comp_ok = abs(comp["drift_px"]) < 50.0
    identity_ok = (
        max(identity["hair"]["std_rgb"]) < 20.0
        and max(identity["face"]["std_rgb"]) < 20.0
        and max(identity["clothing"]["std_rgb"]) < 20.0
    )
    bg_ok = background["temporal_drift_mean"] < 5.0
    temporal_ok = not (motion["snap_detected"] or motion["freeze_detected"])
    motion_visible = motion["adjacent_frame_mae_mean"] > 0.956 or motion["head_roi_mae_mean"] > 1.5
    head_turn_verified = head_turn["face_skin_centroid_x_delta"] > 2.0 or motion["head_flow_dx_mean"] > 0.5

    if not keyframe_ok:
        result = "fail_keyframe"
    elif not fidelity_ok:
        result = "fail_identity"
    elif not comp_ok:
        result = "fail_composition"
    elif not identity_ok:
        result = "fail_identity"
    elif not temporal_ok:
        result = "fail_composition"
    elif not motion_visible:
        result = "fail_static"
    elif not bg_ok:
        result = "fail_composition"
    elif not head_turn_verified:
        result = "borderline"
    else:
        result = "pass"

    report = {
        "input_assets": {
            "character_source": prep["source_frame"]["path"],
            "character_frame": prep["source_frame"]["frame_index"],
            "background": prep["background"]["path"],
            "mask": prep["mask"]["path"],
        },
        "composition": {
            "original_center_x": prep["composition"]["original_character_center_x"],
            "new_center_x": prep["composition"]["new_character_center_x"],
            "delta_x": prep["composition"]["delta_x_pixels"],
            "scale": prep["composition"]["scale"],
            "deterministic": prep["composition"]["deterministic"],
        },
        "keyframe": {
            "path": prep["new_keyframe"]["path"],
            "qa_passed": keyframe_ok,
        },
        "generation": {
            "model": "AniSora V3.1 (Wan 14B)",
            "checkpoint": "/root/autodl-tmp/anisora-v3-g1/models/V3.1",
            "prompt": prep["prompt"],
            "tid": [0],
            "seed": 4096,
            "steps": 40,
            "cfg": 5.0,
            "motion_score": 3.0,
            "runtime_seconds": run["runtime_seconds"],
            "peak_vram_mib": run["peak_vram_mib"],
            "peak_ram_gib": run["peak_ram_gib"],
            "output": str(GEN),
        },
        "fidelity": fidelity,
        "composition_preservation": {
            "centroid_trajectory": {
                "x_mean": comp["centroid_x_mean"],
                "x_std": comp["centroid_x_std"],
                "x_min": comp["centroid_x_min"],
                "x_max": comp["centroid_x_max"],
                "y_mean": comp["centroid_y_mean"],
            },
            "drift": comp["drift_px"],
        },
        "motion": {
            "adjacent_frame_mae": {
                "mean": motion["adjacent_frame_mae_mean"],
                "max": motion["adjacent_frame_mae_max"],
            },
            "head_roi_mae_mean": motion["head_roi_mae_mean"],
            "head_flow_dx_mean": motion["head_flow_dx_mean"],
            "visible_head_turn": head_turn,
            "head_turn_verified_rightward": head_turn_verified,
            "blink": blink,
            "freeze": {"freeze_detected": motion["freeze_detected"], "max_run": motion["max_freeze_run_frames"]},
            "snap": motion["snap_detected"],
        },
        "identity": identity,
        "background": background,
        "result": result,
        "decision": {
            "asset_to_new_keyframe_to_video_supported": result in ("pass", "borderline"),
            "suitable_for_mvp": result == "pass",
            "limitations": [
                "single seed/sample",
                "objective thresholds plus heuristic head-turn/blink; human QA sheet required for final visual verdict",
            ],
            "next_single_question": "If PASS/BORDERLINE: G1-C (Image Edit -> New Keyframe -> Video); if FAIL_STATIC: investigate single-guide motion generation separately.",
        },
        "outputs": {
            "new_shot": str(GEN),
            "head_motion_qa": str(OUTPUTS / "g1b_head_motion_qa.png"),
            "keyframe_qa": str(OUTPUTS / "g1b_new_keyframe_qa.png"),
            "character_asset_qa": str(OUTPUTS / "g1b_character_asset_qa.png"),
        },
    }
    (OUTPUTS / "g1b_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[report] ->", OUTPUTS / "g1b_report.json")
    print(json.dumps({"result": result, "fidelity": fidelity, "motion": motion, "head_turn": head_turn, "blink": blink}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
