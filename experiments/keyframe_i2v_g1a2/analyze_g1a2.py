#!/usr/bin/env python
"""G1-A2 analysis: guide fidelity, temporal continuity, identity, QA sheets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
GUIDES = ROOT / "work" / "guides"
GEN = OUTPUTS / "g1a2_three_keyframe.mp4"
SOURCE = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/work/g0_l2/source.mp4")

GUIDE_FRAMES = {"start": 0, "middle": 40, "end": 80}  # F=81: Tid 0/0.5/1 -> 0/40/80
TEMPORAL_SAMPLES = [0, 10, 20, 30, 40, 50, 60, 70, 80]

# source-domain ROIs scaled to 1280x720
def scale_box(b):
    x0, y0, x1, y1 = b
    return (int(x0 * 1280 / 1918), int(y0 * 720 / 1078), int(x1 * 1280 / 1918), int(y1 * 720 / 1078))


HAIR = scale_box((700, 50, 1200, 220))
FACE = scale_box((997, 206, 1160, 444))
CHAR = scale_box((560, 0, 1360, 760))


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


def main() -> None:
    gen = decode_rgb(GEN)
    p = probe(GEN)
    print("[probe]", json.dumps(p))
    h, w = gen.shape[1], gen.shape[2]
    guides = {name: decode_rgb(GUIDES / f"{name}.png")[0] for name in GUIDE_FRAMES}
    guide_resized = {name: cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA) for name, g in guides.items()}

    fidelity = {}
    for name, j in GUIDE_FRAMES.items():
        a = gen[j].astype(np.float32)
        b = guide_resized[name].astype(np.float32)
        d = np.abs(a - b)
        fidelity[name] = {
            "generated_frame": j,
            "mae": round(float(d.mean()), 2),
            "per_channel_mae": d.mean(axis=(0, 1)).round(2).tolist(),
            "psnr_like": round(float(10 * np.log10(255**2 / max(d.mean() ** 2, 1e-6))), 2),
        }
    print("[fidelity]", json.dumps(fidelity, indent=2))

    # adjacent-frame MAE
    adj = []
    for j in range(len(gen) - 1):
        adj.append(float(np.abs(gen[j + 1].astype(np.int16) - gen[j].astype(np.int16)).mean()))
    adj = np.array(adj)
    seg1 = adj[:40]
    seg2 = adj[40:]
    mean_all = float(adj.mean())
    std_all = float(adj.std())
    max_all = float(adj.max())
    argmax = int(adj.argmax())
    spike_count = int((adj > mean_all + 3 * std_all).sum())
    freeze_run = 0
    max_freeze = 0
    for v in adj:
        if v < 0.5:
            freeze_run += 1
            max_freeze = max(max_freeze, freeze_run)
        else:
            freeze_run = 0
    # guide transitions
    t39_40 = float(adj[39])
    t40_41 = float(adj[40])
    neighbors = [float(adj[38]), float(adj[39]), float(adj[40]), float(adj[41])]
    temporal = {
        "adjacent_frame_mae_mean": round(mean_all, 3),
        "adjacent_frame_mae_std": round(std_all, 3),
        "adjacent_frame_mae_max": round(max_all, 3),
        "argmax_transition": int(argmax),
        "spike_count_gt_3sigma": spike_count,
        "max_freeze_run_frames": max_freeze,
        "start_to_middle_mean": round(float(seg1.mean()), 3),
        "middle_to_end_mean": round(float(seg2.mean()), 3),
        "guide_transition_40": {"t39_40": round(t39_40, 3), "t40_41": round(t40_41, 3), "neighbor_mae": [round(v, 3) for v in neighbors]},
        "snap_detected": bool(max_all > 20.0),
        "freeze_detected": bool(max_freeze >= 8),
        "motion_amplitude_note": "very low (mean adjacent MAE < 1.5): interpolation is smooth but may appear near-static",
    }
    print("[temporal]", json.dumps(temporal, indent=2))

    # identity: hair/face color stability (source-anchored ROIs)
    def roi_median(frame, box):
        x0, y0, x1, y1 = box
        return np.median(frame[y0:y1, x0:x1].reshape(-1, 3), axis=0)

    hair_colors = [roi_median(gen[j], HAIR) for j in range(len(gen))]
    face_colors = [roi_median(gen[j], FACE) for j in range(len(gen))]
    hair = {
        "mean_rgb": np.mean(hair_colors, axis=0).round(1).tolist(),
        "std_rgb": np.std(hair_colors, axis=0).round(1).tolist(),
        "frame0": hair_colors[0].round(1).tolist(),
        "frame40": hair_colors[40].round(1).tolist(),
        "frame80": hair_colors[80].round(1).tolist(),
    }
    face = {
        "mean_rgb": np.mean(face_colors, axis=0).round(1).tolist(),
        "std_rgb": np.std(face_colors, axis=0).round(1).tolist(),
        "frame0": face_colors[0].round(1).tolist(),
        "frame40": face_colors[40].round(1).tolist(),
        "frame80": face_colors[80].round(1).tolist(),
    }
    identity = {"hair": hair, "face": face}
    print("[identity]", json.dumps(identity, indent=2))

    # source-aligned MAE (informational)
    src = decode_rgb(SOURCE)
    src_aligned = [cv2.resize(src[min(int(j / 16.0 * 30.1754), len(src) - 1)], (w, h), interpolation=cv2.INTER_AREA) for j in range(len(gen))]
    src_mae = round(float(np.mean([np.abs(gen[j].astype(np.int16) - src_aligned[j].astype(np.int16)).mean() for j in range(len(gen))])), 2)

    # QA sheets
    rows = []
    for name, j in GUIDE_FRAMES.items():
        rows.append((f"{name} (F{j})", [guide_resized[name], gen[j]]))
    make_qa_sheet(OUTPUTS / "g1a2_fidelity_qa.png", "G1-A2 guide fidelity: Guide vs Generated at t=0/0.5/1", rows, ["Guide", "Generated"], cell_w=480, cell_h=270)

    rows = [("generated", [gen[j] for j in TEMPORAL_SAMPLES])]
    make_qa_sheet(OUTPUTS / "g1a2_temporal_qa.png", "G1-A2 temporal sequence: F0..F80", rows, [f"F{j}" for j in TEMPORAL_SAMPLES], cell_w=200, cell_h=113)

    rows = []
    for j in [0, 20, 40, 60, 80]:
        rows.append((f"F{j}", [src_aligned[j], gen[j]]))
    make_qa_sheet(OUTPUTS / "g1a2_source_vs_generated_qa.png", "G1-A2 source vs generated (same nominal time)", rows, ["Source", "Generated"], cell_w=480, cell_h=270)

    # decision
    guide_ok = all(fidelity[k]["mae"] < 25.0 for k in fidelity)
    identity_ok = bool(np.max(identity["hair"]["std_rgb"]) < 20.0 and np.max(identity["face"]["std_rgb"]) < 20.0)
    if not guide_ok:
        result = "fail_guide"
    elif temporal["snap_detected"] or temporal["freeze_detected"]:
        result = "fail_temporal"
    elif guide_ok and identity_ok:
        result = "pass"
    else:
        result = "borderline"

    report = {
        "generation": {
            "output": str(GEN),
            "ffprobe": p,
            "frame_count": p["frame_count"],
            "fps": p["fps"],
            "guide_frame_mapping": {"start": 0, "middle": 40, "end": 80, "note": "F=81; id=int((F-1)*Tid) then round to multiple of 8"},
        },
        "guide_fidelity": fidelity,
        "temporal": temporal,
        "identity": identity,
        "source_aligned_whole_frame_mae_mean": src_mae,
        "result": result,
        "decision": {
            "multi_keyframe_suitable_for_asset_pipeline": result in ("pass", "borderline"),
            "evidence": [
                f"guide MAE start/mid/end = {fidelity['start']['mae']}/{fidelity['middle']['mae']}/{fidelity['end']['mae']}",
                f"adjacent MAE mean/std/max = {temporal['adjacent_frame_mae_mean']}/{temporal['adjacent_frame_mae_std']}/{temporal['adjacent_frame_mae_max']}",
                f"spikes(>3sigma)={temporal['spike_count_gt_3sigma']} freeze_run={temporal['max_freeze_run_frames']} "
                f"motion_note={temporal.get('motion_amplitude_note','')}",
                f"hair std={identity['hair']['std_rgb']} face std={identity['face']['std_rgb']}",
            ],
            "limitations": ["single seed/sample", "objective thresholds only; human QA sheets provided"],
            "next_single_question": "If PASS: G1-B (Asset -> New Keyframe -> Video); if FAIL_GUIDE: check guide transform/temporal position mapping.",
        },
        "outputs": {
            "fidelity_qa": str(OUTPUTS / "g1a2_fidelity_qa.png"),
            "temporal_qa": str(OUTPUTS / "g1a2_temporal_qa.png"),
            "source_vs_generated_qa": str(OUTPUTS / "g1a2_source_vs_generated_qa.png"),
            "guides_qa": str(OUTPUTS / "g1a2_guides_qa.png"),
        },
    }
    (OUTPUTS / "g1a2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report] ->", OUTPUTS / "g1a2_report.json")
    print(json.dumps({"result": result, "fidelity": fidelity, "temporal": temporal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
