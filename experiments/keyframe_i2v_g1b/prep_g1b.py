#!/usr/bin/env python
"""G1-B asset/keyframe construction (deterministic, no image-editing model)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
ASSETS = WORK / "assets"
KEYFRAMES = WORK / "keyframes"
OUTPUTS = ROOT / "outputs"

SOURCE = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/work/g0_l2/source.mp4")
BACKGROUND = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/outputs/background.png")
MASK = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/work/masks.npy")

FRAME_IDX = 0
DX_FRACTION = 0.10

PROMPT = (
    "A 2D anime girl with light golden-brown hair stands in the same fixed scene. "
    "The camera is completely still. She keeps her body and pose stable while slowly "
    "turning her head to her right by about 20 to 30 degrees. She naturally blinks once "
    "during the motion, with a very slight natural secondary motion in her hair. Her "
    "identity, face, hairstyle, hair color, clothing, body proportions, background and "
    "visual style remain completely consistent with the first frame. No camera movement, "
    "no new objects, no text. aesthetic score: 5.5. motion score: 3.0. "
    "There is no text in the video."
)


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
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_qa_sheet(path: Path, title: str, rows, cols, cell_w=420, cell_h=236) -> None:
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


def checker(rgb, cell=16, n=8):
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if ((x // cell) + (y // cell)) % 2 == 0:
                out[y, x] = (200, 200, 200)
            else:
                out[y, x] = (90, 90, 90)
    return out


def main() -> None:
    for d in (ASSETS, KEYFRAMES, OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)

    frames = decode_rgb(SOURCE)
    fps = 30.175394
    frame = frames[FRAME_IDX]
    H, W = frame.shape[:2]
    masks = np.load(MASK)
    mask = masks[FRAME_IDX].astype(np.float32) / 255.0  # 0 / 0.5 / 1

    # Character RGBA: source RGB + mask alpha (0/127/255 -> 0/0.5/1).
    char_rgb = frame.copy()
    rgba = np.dstack([char_rgb, np.round(mask * 255.0).astype(np.uint8)])
    rgba_path = ASSETS / "character_rgba.png"
    Image.fromarray(rgba, "RGBA").save(rgba_path)

    # Translation right by 10% of frame width.
    dx = int(round(W * DX_FRACTION))
    shifted_alpha = np.zeros_like(mask)
    shifted_rgb = np.zeros_like(char_rgb, dtype=np.float32)
    shifted_alpha[:, dx:] = mask[:, : W - dx]
    shifted_rgb[:, dx:] = char_rgb[:, : W - dx].astype(np.float32)

    bg = np.asarray(Image.open(BACKGROUND).convert("RGB"), dtype=np.float32)
    a3 = shifted_alpha[..., None]
    new = bg * (1.0 - a3) + shifted_rgb * a3
    new = np.clip(np.round(new), 0, 255).astype(np.uint8)
    new_path = KEYFRAMES / "new_start.png"
    Image.fromarray(new).save(new_path)

    # Centroid / bbox (hard mask >0.5 to match G0 bbox semantics).
    ys, xs = np.where(mask > 0.5)
    ny, nx = np.where(shifted_alpha > 0.5)
    orig_cx, orig_cy = float(xs.mean()), float(ys.mean())
    new_cx, new_cy = float(nx.mean()), float(ny.mean())
    delta_x = new_cx - orig_cx

    # RGB integrity inside the character mask.
    rgb_diff = np.abs(frame.astype(np.int16) - char_rgb.astype(np.int16))
    char_rgb_mae = float(rgb_diff[mask > 0].mean())

    # Background outside mask should be untouched.
    bg_diff = np.abs(new.astype(np.int16) - bg.astype(np.int16))
    outside_mae = float(bg_diff[shifted_alpha == 0].mean())
    feather_mae = float(bg_diff[(shifted_alpha > 0) & (shifted_alpha < 1)].mean())

    # Prove the new keyframe is not in the original source.
    new_i16 = new.astype(np.int16)
    maes = [
        float(np.abs(new_i16 - f.astype(np.int16)).mean())
        for f in frames
    ]
    min_mae = min(maes)
    argmin_mae = int(np.argmin(maes))

    # Save prompt and official input line.
    (WORK / "prompt.txt").write_text(PROMPT, encoding="ascii")
    input_line = f"{PROMPT}@@{new_path}&&0"
    (WORK / "g1b_input.txt").write_text(input_line, encoding="ascii")

    meta = {
        "source_frame": {
            "path": str(SOURCE),
            "sha256": sha256(SOURCE),
            "frame_index": FRAME_IDX,
            "timestamp_seconds": round(FRAME_IDX / fps, 4),
            "resolution": [W, H],
        },
        "background": {
            "path": str(BACKGROUND),
            "sha256": sha256(BACKGROUND),
            "resolution": list(Image.open(BACKGROUND).size),
            "provenance": "G0-L1: per-pixel temporal median where character absent + Telea inpaint for never-visible pixels (see layered_video_g0/outputs/report.json)",
        },
        "mask": {
            "path": str(MASK),
            "semantics": "SAM2.1 character mask (255=character, 127=soft edge, 0=background)",
            "frame0_area_ratio": round(float((mask > 0.5).mean()), 4),
        },
        "composition": {
            "method": "original character RGB + existing background + horizontal translation",
            "original_character_center_x": round(orig_cx, 2),
            "new_character_center_x": round(new_cx, 2),
            "delta_x_pixels": round(delta_x, 2),
            "delta_x_ratio": round(delta_x / W, 4),
            "dx_pixels": dx,
            "dx_fraction": DX_FRACTION,
            "scale": 1.0,
            "rotation": 0.0,
            "mirror": False,
            "deterministic": True,
            "character_rgb_mae": char_rgb_mae,
            "background_outside_mae": outside_mae,
            "feather_band_mae": feather_mae,
            "original_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "new_bbox": [int(nx.min()), int(ny.min()), int(nx.max()), int(ny.max())],
            "clipped": bool(nx.max() >= W - 1 or ny.max() >= H - 1),
        },
        "new_keyframe": {
            "path": str(new_path),
            "sha256": sha256(new_path),
            "resolution": [W, H],
            "not_in_source": {
                "min_mae_vs_any_source_frame": round(min_mae, 2),
                "argmin_source_frame": argmin_mae,
            },
        },
        "prompt": PROMPT,
        "input_line": input_line,
    }
    (WORK / "prep_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    # ---- QA sheet 1: character asset ----
    mask_gray = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    checker_bg = checker(frame)
    over_checker = (checker_bg * (1 - a3) + shifted_rgb * a3).astype(np.uint8)
    rows = [(
        "Character asset",
        [
            frame,
            mask_gray,
            over_checker,
            new,
        ],
    )]
    make_qa_sheet(
        OUTPUTS / "g1b_character_asset_qa.png",
        "G1-B character asset: Source / Mask / RGBA over checker / New composition",
        rows,
        ["Source", "Mask", "RGBA/checker", "New composition"],
        cell_w=480,
        cell_h=270,
    )

    # ---- QA sheet 2: keyframe ----
    def with_markers(img, box, cx):
        im = Image.fromarray(img).convert("RGB")
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=3)
        d.line([cx, 0, cx, im.height], fill=(255, 0, 255), width=3)
        return np.asarray(im)

    orig_box = meta["composition"]["original_bbox"]
    new_box = meta["composition"]["new_bbox"]
    rows = [(
        "Keyframe",
        [
            with_markers(frame, orig_box, orig_cx),
            over_checker,
            bg.astype(np.uint8),
            with_markers(new, new_box, new_cx),
        ],
    )]
    make_qa_sheet(
        OUTPUTS / "g1b_new_keyframe_qa.png",
        "G1-B new keyframe: Source / Character RGBA / Background / New Constructed Keyframe",
        rows,
        ["Original Source", "RGBA/checker", "Background", "New Keyframe"],
        cell_w=480,
        cell_h=270,
    )


if __name__ == "__main__":
    main()
