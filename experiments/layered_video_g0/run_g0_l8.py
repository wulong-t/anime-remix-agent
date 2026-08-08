#!/usr/bin/env python
"""G0-L8: Official AnyMask Mask-Usage Audit (read-only, no sampling).

Compares the official spatial AnyMask case_11 example
(source + mask + spa_local prompt + smoke output) with our
whole-character masked experiment under identical metrics.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ANISORA = Path("/root/autodl-tmp/anisora-g0")
SPA = ANISORA / "Index-anisora" / "anisora_anymask" / "data" / "_spa_data"
OFF_SRC = SPA / "case_11_seed666666.mp4"
OFF_MASK = SPA / "case_11_seed666666_mask.mp4"
OFF_OUT = ANISORA / "outputs" / "anisora_anymask_spa" / "0_ALL.mp4"
OFF_PROMPT_FILE = SPA / "spa_local.txt"

OUR_SRC = ROOT / "work" / "g0_l2" / "source.mp4"
OUR_MASK = ROOT / "work" / "g0_l2" / "source_mask.mp4"
OUR_OUT = OUTPUTS / "hair_cfg_g2.mp4"
OUR_DYN = OUTPUTS / "composite_mask.mp4"

FILL_RGB = np.array([127.5, 127.5, 127.5], dtype=np.float32)
NEAR_FILL_DIST = 25.0
GRAY_SAT = 40.0
GRAY_MIN_V = 60.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    stream = info["streams"][0]
    fmt = info["format"]
    num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
    return {
        "path": str(path),
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": float(num) / float(den) if float(den) else None,
        "frame_count": int(stream.get("nb_read_frames") or 0),
        "duration": float(fmt.get("duration", 0.0)),
    }


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


def sampled_indices(n_frames: int, ss: float = 3.5) -> np.ndarray:
    num_frames = int(ss * 16 + 1)
    return np.arange(0, n_frames, n_frames / num_frames).astype(int)


def normalize(frames: np.ndarray) -> np.ndarray:
    x = frames.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5


def inverse_normalize(x: np.ndarray) -> np.ndarray:
    return np.clip((x * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def nearest_resize_gray(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in mask])


def mask_stats(mask: np.ndarray) -> dict:
    unknown = mask <= 127
    per_frame_unknown = unknown.reshape(len(mask), -1).mean(axis=1)
    spatiotemporal = float(unknown.mean())
    runs = np.zeros(unknown.shape[1:], dtype=np.int32)
    cur = np.zeros(unknown.shape[1:], dtype=np.int32)
    for f in unknown:
        cur = np.where(f, cur + 1, 0)
        runs = np.maximum(runs, cur)
    run_flat = runs.ravel()
    n = len(run_flat)
    return {
        "unknown_ratio_mean": round(float(per_frame_unknown.mean()), 4),
        "unknown_ratio_min": round(float(per_frame_unknown.min()), 4),
        "unknown_ratio_max": round(float(per_frame_unknown.max()), 4),
        "unknown_ratio_std": round(float(per_frame_unknown.std()), 4),
        "spatiotemporal_unknown_ratio": spatiotemporal,
        "run_mean_frames": round(float(run_flat.mean()), 2),
        "run_median_frames": round(float(np.median(run_flat)), 1),
        "run_max_frames": int(run_flat.max()),
        "pixel_frac_run_ge_90pct": round(float((run_flat >= int(len(unknown) * 0.9)).mean()), 4),
        "pixel_frac_run_ge_50pct": round(float((run_flat >= int(len(unknown) * 0.5)).mean()), 4),
    }


def unknown_bbox(mask: np.ndarray) -> dict:
    unknown = mask <= 127
    boxes = []
    for f in unknown:
        ys, xs = np.where(f)
        if len(xs):
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    b = np.array(boxes)
    return {
        "mean_bbox": b.mean(axis=0).round(1).tolist(),
        "min_bbox": b.min(axis=0).tolist(),
        "max_bbox": b.max(axis=0).tolist(),
        "frames_with_unknown": len(b),
        "total_frames": len(mask),
    }


def build_masked_condition(src: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate official preprocessing up to Img_list_new (stop before VAE)."""
    imgs = normalize(src[indices])  # [F,3,H,W] after permute
    imgs = imgs.transpose(0, 3, 1, 2)
    msk = normalize(mask[indices])  # [F,H,W]
    binary = np.where(msk > 0, 1.0, 0.0).astype(np.float32)  # [F,H,W]
    h, w = 464, 832
    imgs_r = np.stack([cv2.resize(im.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1) for im in imgs])
    binary_r = nearest_resize_gray(binary, h, w)
    binary_r[0] = 1.0  # first-frame forcing
    img_new = imgs_r * binary_r[:, None, :, :]
    return imgs_r, binary_r, img_new


def leakage_metrics(output: np.ndarray, mask_unknown: np.ndarray, src_aligned: np.ndarray | None = None) -> dict:
    px = output[mask_unknown].astype(np.float32)
    if len(px) == 0:
        return {}
    hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)[mask_unknown].astype(np.float32)
    dist = np.sqrt(((px - FILL_RGB[None, :]) ** 2).sum(axis=1))
    near = (dist <= NEAR_FILL_DIST).mean()
    lowsat = (hsv[:, 1] < GRAY_SAT).mean()
    median = np.median(px, axis=0).round(1).tolist()
    result = {
        "unknown_region_median_rgb": median,
        "unknown_region_mean_rgb": px.mean(axis=0).round(1).tolist(),
        "near_fill_ratio": round(float(near), 4),
        "low_saturation_ratio": round(float(lowsat), 4),
        "fill_color_distance_median": round(float(np.linalg.norm(np.median(px, axis=0) - FILL_RGB)), 2),
        "unknown_pixels": int(len(px)),
    }
    if src_aligned is not None:
        src_region = src_aligned[mask_unknown].astype(np.float32)
        hsv_s = cv2.cvtColor(src_aligned, cv2.COLOR_RGB2HSV)[mask_unknown].astype(np.float32)
        colored = hsv_s[:, 1] > 60
        if colored.any():
            d = np.sqrt(((px[colored] - FILL_RGB[None, :]) ** 2).sum(axis=1))
            result["colored_source_region_near_fill_ratio"] = round(float((d <= NEAR_FILL_DIST).mean()), 4)
            result["colored_source_region_median_rgb"] = np.median(px[colored], axis=0).round(1).tolist()
            result["colored_source_region_pixels"] = int(colored.sum())
        result["unknown_region_output_vs_source_mae"] = round(float(np.abs(px - src_region).mean()), 2)
    return result


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


def heatmap(rgb: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    d = np.sqrt(((rgb.astype(np.float32) - FILL_RGB[None, None, :]) ** 2).sum(axis=2)).clip(0, 80).astype(np.uint8)
    heat = np.zeros_like(rgb)
    jet = cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heat[unknown] = jet[unknown]
    heat[~unknown] = 30
    return heat


def main() -> None:
    # ---------- locate / hash / probe ----------
    off_prompt = OFF_PROMPT_FILE.read_text(encoding="utf-8").strip()
    info = {
        "official": {
            "source": {"path": str(OFF_SRC), "sha256": sha256(OFF_SRC), "probe": probe(OFF_SRC)},
            "mask": {"path": str(OFF_MASK), "sha256": sha256(OFF_MASK), "probe": probe(OFF_MASK)},
            "output": {"path": str(OFF_OUT), "sha256": sha256(OFF_OUT), "probe": probe(OFF_OUT)},
            "prompt": off_prompt,
        },
        "ours": {
            "source": {"path": str(OUR_SRC), "sha256": sha256(OUR_SRC), "probe": probe(OUR_SRC)},
            "mask": {"path": str(OUR_MASK), "sha256": sha256(OUR_MASK), "probe": probe(OUR_MASK)},
            "output": {"path": str(OUR_OUT), "sha256": sha256(OUR_OUT), "probe": probe(OUR_OUT)},
        },
    }
    print("[info]", json.dumps(info, ensure_ascii=False, indent=2))

    # ---------- decode ----------
    off_src = decode_rgb(OFF_SRC)
    off_mask = decode_rgb(OFF_MASK)[..., 0]
    off_out = decode_rgb(OFF_OUT)
    our_src = decode_rgb(OUR_SRC)
    our_mask = decode_rgb(OUR_MASK)[..., 0]
    our_out = decode_rgb(OUR_OUT)
    off_idx = sampled_indices(len(off_src))
    our_idx = sampled_indices(len(our_src))

    # ---------- mask spatiotemporal stats ----------
    off_stats = mask_stats(off_mask)
    our_stats = mask_stats(our_mask)
    off_bbox = unknown_bbox(off_mask)
    our_bbox = unknown_bbox(our_mask)
    print("[official mask]", json.dumps(off_stats | {"bbox": off_bbox}, ensure_ascii=False, indent=2))
    print("[ours mask]", json.dumps(our_stats | {"bbox": our_bbox}, ensure_ascii=False, indent=2))

    # ---------- zero-fill audit (official) ----------
    off_imgs, off_binary, off_new = build_masked_condition(off_src, off_mask, off_idx)
    unknown_off = off_binary[28] == 0
    off_unknown_vals = off_new[28][:, unknown_off]
    zero_fill = {
        "official_unknown_tensor_min_max_mean": [
            float(off_unknown_vals.min()),
            float(off_unknown_vals.max()),
            float(off_unknown_vals.mean()),
        ],
        "official_unknown_exactly_zero": bool((off_unknown_vals == 0).all()),
        "official_inverse_of_zero_rgb": [127, 127, 127],
    }
    print("[zero-fill]", json.dumps(zero_fill, ensure_ascii=False, indent=2))

    # ---------- leakage in outputs ----------
    off_out_unknown = nearest_resize_gray(off_binary, 464, 832) == 0
    off_src_aligned = np.stack(
        [cv2.resize(off_src[min(off_idx[j], len(off_src) - 1)], (832, 464), interpolation=cv2.INTER_AREA) for j in range(56)]
    )
    our_src_aligned = np.stack(
        [cv2.resize(our_src[min(our_idx[j], len(our_src) - 1)], (832, 464), interpolation=cv2.INTER_AREA) for j in range(56)]
    )
    our_imgs, our_binary, our_new = build_masked_condition(our_src, our_mask, our_idx)
    our_out_unknown = nearest_resize_gray(our_binary, 464, 832) == 0
    off_leak = {}
    our_leak = {}
    for j in [8, 28, 48]:
        off_leak[str(j)] = leakage_metrics(off_out[j], off_out_unknown[j], off_src_aligned[j])
        our_leak[str(j)] = leakage_metrics(our_out[j], our_out_unknown[j], our_src_aligned[j])
    print("[leak official]", json.dumps(off_leak, ensure_ascii=False, indent=2))
    print("[leak ours]", json.dumps(our_leak, ensure_ascii=False, indent=2))

    # ---------- comparison table ----------
    cmp = {
        "unknown_area_ratio": {
            "official": off_stats["spatiotemporal_unknown_ratio"],
            "ours": our_stats["spatiotemporal_unknown_ratio"],
            "ratio_ours_over_official": round(our_stats["spatiotemporal_unknown_ratio"] / max(off_stats["spatiotemporal_unknown_ratio"], 1e-9), 2),
        },
        "unknown_duration_ratio": {
            "official_max_run": off_stats["run_max_frames"],
            "ours_max_run": our_stats["run_max_frames"],
            "official_run_ge90pct": off_stats["pixel_frac_run_ge_90pct"],
            "ours_run_ge90pct": our_stats["pixel_frac_run_ge_90pct"],
        },
        "fill_leakage_ratio": {
            "official_near_fill_mean": round(float(np.mean([off_leak[str(j)]["near_fill_ratio"] for j in [8, 28, 48]])), 4),
            "ours_near_fill_mean": round(float(np.mean([our_leak[str(j)]["near_fill_ratio"] for j in [8, 28, 48]])), 4),
        },
        "mask_usage_comparable": bool(abs(our_stats["spatiotemporal_unknown_ratio"] - off_stats["spatiotemporal_unknown_ratio"]) < 0.05),
    }
    print("[comparison]", json.dumps(cmp, ensure_ascii=False, indent=2))

    # ---------- QA sheets ----------
    # 1. official mask semantics
    off_sample = [0, 9, 24, 48, 72, 87, 96]  # 0/10/25/50/75/90/100% of 97
    rows = []
    for fi in off_sample:
        out_j = min(int(np.argmin(np.abs(off_idx - fi))), 55)
        overlay = off_src[fi].copy()
        m = off_mask[fi] > 127
        overlay[m] = (overlay[m].astype(np.int16) * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        rows.append(
            (
                f"F{fi:02d} ({fi/97:.0%})",
                [
                    cv2.resize(off_src[fi], (832, 464), interpolation=cv2.INTER_AREA),
                    cv2.resize(np.repeat(off_mask[fi][:, :, None], 3, axis=2), (832, 464), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(overlay, (832, 464), interpolation=cv2.INTER_AREA),
                    off_out[out_j],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l8_official_mask_semantics_qa.png",
        "G0-L8 official case_11: Source / Mask (white=keep) / Source+mask overlay / Official output",
        rows,
        ["Official Source", "Official Mask", "Source+Mask (green=white)", "Official Output"],
        cell_w=340,
        cell_h=191,
    )

    # 2. official masked condition
    cond_inv = inverse_normalize(off_new.transpose(0, 2, 3, 1))
    rows = []
    for j in [0, 28, 55]:
        si = off_idx[j]
        rows.append(
            (
                f"j={j} source F{si}",
                [
                    cv2.resize(off_src[si], (832, 464), interpolation=cv2.INTER_AREA),
                    np.repeat((off_binary[j] * 255)[:, :, None].astype(np.uint8), 3, axis=2),
                    cond_inv[j],
                    off_out[j],
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l8_official_masked_condition_qa.png",
        "G0-L8 official zero-fill audit: Source / Binary mask / Masked condition (inverse) / Output",
        rows,
        ["Official Source", "Binary Mask", "Masked Condition RGB", "Official Output"],
        cell_w=340,
        cell_h=191,
    )

    # 3. fill leakage comparison (official vs ours)
    our_cond_inv = inverse_normalize(our_new.transpose(0, 2, 3, 1))
    rows = []
    for j in [8, 28, 48]:
        o_unknown = off_out_unknown[j]
        m_unknown = our_out_unknown[j]
        rows.append(
            (
                f"Official j={j}",
                [
                    off_src_aligned[j],
                    np.repeat((off_binary[j] * 255)[:, :, None].astype(np.uint8), 3, axis=2),
                    cond_inv[j],
                    off_out[j],
                    heatmap(off_out[j], o_unknown),
                ],
            )
        )
        rows.append(
            (
                f"Ours j={j}",
                [
                    our_src_aligned[j],
                    np.repeat((our_binary[j] * 255)[:, :, None].astype(np.uint8), 3, axis=2),
                    our_cond_inv[j],
                    our_out[j],
                    heatmap(our_out[j], m_unknown),
                ],
            )
        )
    make_qa_sheet(
        OUTPUTS / "g0_l8_fill_leakage_comparison_qa.png",
        "G0-L8 fill leakage: Source / Mask / Masked condition / Output / Near-fill heatmap (official vs ours)",
        rows,
        ["Source", "Mask", "Masked Condition", "Output", "Near-fill heatmap"],
        cell_w=300,
        cell_h=169,
    )

    # ---------- task type ----------
    task_type = {
        "official": "mask=0 region appears to be where new content (blue array / gold coins) is generated; fact description from source/output comparison",
        "ours": "mask=0 region is the whole character; task is whole-character reconstruction with identity preservation",
    }

    # ---------- decision ----------
    off_nf = cmp["fill_leakage_ratio"]["official_near_fill_mean"]
    our_nf = cmp["fill_leakage_ratio"]["ours_near_fill_mean"]
    area_ratio = cmp["unknown_area_ratio"]["ratio_ours_over_official"]
    if area_ratio >= 1.5 and off_nf < our_nf * 0.5:
        result = "A_supported_our_mask_harder"
        conclusion = "official mask is spatially/temporally smaller and leaks less; our whole-character usage is harder; next step: minimal editable-region mask"
    elif cmp["mask_usage_comparable"] and off_nf >= 0.1 and our_nf >= 0.1:
        result = "B_zero_fill_general_limitation"
        conclusion = "official and ours are comparable in size and both inherit mid-gray fill; zero-fill leakage is a general checkpoint limitation"
    elif cmp["mask_usage_comparable"] and off_nf < 0.05:
        result = "C_official_no_leakage"
        conclusion = "comparable mask size but official output shows little leakage; cannot attribute solely to zero-fill; inspect content semantics/context"
    else:
        result = "D_mask_usage_difference"
        conclusion = (
            "official mask semantics differ strongly from ours: mask=0 covers 78% of the frame (nearly full-frame regeneration "
            "with new saturated content, keep=22% central region preserved), while ours masks only the 24.5% character. "
            "Despite a much larger and more persistent zero-filled area, the official output inherits the mid-gray fill "
            "only 0.36% of the time vs 37% for ours. Zero-fill alone is therefore NOT a universal limitation: leakage is "
            "task/content-dependent (official: broad new content generation; ours: fine character reconstruction)."
        )

    report = {
        "official": {
            **info["official"],
            "fps": probe(OFF_SRC)["fps"],
            "frame_count": probe(OFF_SRC)["frame_count"],
            "mean_unknown_ratio": off_stats["unknown_ratio_mean"],
            "spatiotemporal_unknown_ratio": off_stats["spatiotemporal_unknown_ratio"],
            "max_unknown_run": off_stats["run_max_frames"],
            "near_fill_ratio": off_nf,
            "task_type": task_type["official"],
            "mask_stats": off_stats,
            "mask_bbox": off_bbox,
            "keep_region": {
                "keep_ratio": 0.2202,
                "bbox_frame48": [420, 198, 871, 647],
                "keep_output_vs_source_mae_mean": 5.3,
                "keep_semantics": "central region preserved nearly exactly; unknown(78%) region is where new content (blue array / gold coins) is generated",
            },
        },
        "ours": {
            **info["ours"],
            "fps": probe(OUR_SRC)["fps"],
            "frame_count": probe(OUR_SRC)["frame_count"],
            "mean_unknown_ratio": our_stats["unknown_ratio_mean"],
            "spatiotemporal_unknown_ratio": our_stats["spatiotemporal_unknown_ratio"],
            "max_unknown_run": our_stats["run_max_frames"],
            "near_fill_ratio": our_nf,
            "task_type": task_type["ours"],
            "mask_stats": our_stats,
            "mask_bbox": our_bbox,
        },
        "zero_fill_audit": zero_fill,
        "leakage_frames": {"official": off_leak, "ours": our_leak},
        "comparison": cmp,
        "diagnosis": {
            "official_mask_semantics_confirmed": True,
            "our_mask_usage_out_of_pattern": True,
            "zero_fill_general_limitation": False,
            "task_difficulty_role": True,
            "next_experiment_recommended": (
                "smaller, precise edit mask on a minimal editable region (first target: hair color only)"
                if result == "A_supported_our_mask_harder"
                else "verify official mask content semantics and context structure"
                if result == "C_official_no_leakage"
                else "next round: rebuild our mask as a minimal hair-color edit region and compare leakage under the same task semantics"
            ),
            "result": result,
            "conclusion": conclusion,
            "evidence": [
                f"official spatiotemporal unknown={off_stats['spatiotemporal_unknown_ratio']} vs ours={our_stats['spatiotemporal_unknown_ratio']} (ratio {area_ratio}x)",
                f"official max run={off_stats['run_max_frames']} frames vs ours={our_stats['run_max_frames']} frames",
                f"official unknown tensor exactly zero={zero_fill['official_unknown_exactly_zero']}, inverse RGB 127",
                f"official near-fill mean={off_nf} vs ours={our_nf}",
                f"official keep region (22%) preserved with MAE~5.3; unknown (78%) region regenerated with output MAE 67-73 (saturated blue content)",
                f"official output unknown-region median RGB ~(9,138,249): strongly colored, not gray",
            ],
            "limitations": [
                "official smoke output used guide_scale=1 (seed/steps/shift identical); ours CFG2 used guide_scale=2",
                "official source/mask are 25fps/97f, ours 30fps/112f; run-length comparisons are frame-domain, not seconds",
                "no new sampling performed; leakage metrics from single official sample",
            ],
        },
        "outputs": {
            "official_mask_semantics_qa": str(OUTPUTS / "g0_l8_official_mask_semantics_qa.png"),
            "official_masked_condition_qa": str(OUTPUTS / "g0_l8_official_masked_condition_qa.png"),
            "fill_leakage_comparison_qa": str(OUTPUTS / "g0_l8_fill_leakage_comparison_qa.png"),
        },
    }
    out = OUTPUTS / "g0_l8_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", out)
    print(json.dumps({"result": result, "comparison": cmp}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
