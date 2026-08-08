#!/usr/bin/env python
"""G1-A2: AniSora V3/V3.1 arbitrary-frame baseline (start+middle+end).

Prepares guides from the G0 source video, writes the official input line
`prompt@@start.png,middle.png,end.png&&0,0.5,1`, waits for checkpoint
downloads, then runs the official generate-pi-i2v-any.py once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
GUIDES = WORK / "guides"
SOURCE = Path("/root/autodl-tmp/anime-remix-agent/experiments/layered_video_g0/work/g0_l2/source.mp4")
MODEL_DIR = Path("/root/autodl-tmp/anisora-v3-g1/models/V3.1")
V3_DIR = Path("/root/autodl-tmp/anisora-v3-g1/anisoraV3")
GUIDE_IDX = {"start": 0, "middle": 56, "end": 111}  # 0% / 50% / ~100% of 112 frames

PROMPT = (
    "A 2D anime girl with light golden-brown hair and a warm smile stands in a "
    "fixed scene. The camera is completely still. She only makes very subtle "
    "movements: gentle blinking, slight head turns, small facial expression "
    "changes and natural breathing. Her hair, face, clothing and the background "
    "remain stable. aesthetic score: 5.5. motion score: 2.5. "
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
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out) if out else None


def make_qa_sheet(path: Path, title: str, rows: list[tuple[str, list[np.ndarray]]], cols: list[str], cell_w: int = 340, cell_h: int = 191) -> None:
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


def prep() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    GUIDES.mkdir(parents=True, exist_ok=True)
    frames = decode_rgb(SOURCE)
    fps = 30.175394
    info = {}
    rows = []
    for name, idx in GUIDE_IDX.items():
        img = Image.fromarray(frames[idx])
        p = GUIDES / f"{name}.png"
        img.save(p)
        info[name] = {
            "path": str(p),
            "source_frame_index": idx,
            "timestamp_seconds": round(idx / fps, 4),
            "position": {"start": 0.0, "middle": 0.5, "end": 1.0}[name],
            "sha256": sha256(p),
            "resolution": list(img.size),
        }
        rows.append((name, [np.asarray(img)]))
    make_qa_sheet(
        OUTPUTS / "g1a2_guides_qa.png",
        "G1-A2 guide frames: Start (0%) / Middle (50%) / End (~100%) from source",
        rows,
        ["Guide"],
        cell_w=600,
        cell_h=338,
    )
    (WORK / "prompt.txt").write_text(PROMPT, encoding="ascii")
    line = (
        f"{PROMPT}@@{GUIDES / 'start.png'},{GUIDES / 'middle.png'},{GUIDES / 'end.png'}&&0,0.5,1"
    )
    (WORK / "g1a2_input.txt").write_text(line, encoding="ascii")
    info["prompt"] = PROMPT
    info["input_line"] = line
    info["source"] = {"path": str(SOURCE), "sha256": sha256(SOURCE)}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def wait_for_model() -> None:
    expected = {
        "model_part1.safetensors": 34342096376,
        "model_part2.safetensors": 31237553936,
        "models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": 4772359047,
        "Wan2.1_VAE.pth": 507609880,
        "diffusion_pytorch_model.safetensors.index.json": 81236,
        "config.json": 250,
    }
    while True:
        missing = []
        for name, size in expected.items():
            p = MODEL_DIR / name
            if not p.exists() or p.stat().st_size != size:
                missing.append(f"{name} {p.stat().st_size if p.exists() else 0}/{size}")
        if not missing:
            print("[model] all files complete")
            return
        print("[model] waiting:", "; ".join(missing), flush=True)
        time.sleep(60)


def mem_bytes(path: str) -> int:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return -1


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w") as f:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    f.write(f"{time.time()},{out.stdout.strip()}\n")
                    f.flush()
            except Exception:
                pass
            time.sleep(1)


def run_generation() -> dict:
    out_dir = OUTPUTS
    log_path = OUTPUTS / "g1a2_anymask.log"
    gpu_path = OUTPUTS / "g1a2_gpu.csv"
    env = os.environ.copy()
    env.update(
        {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HOME": "/root/autodl-tmp/anisora-v3-g1/.cache/hf",
            "TMPDIR": "/root/autodl-tmp/anisora-v3-g1/.tmp",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    # release checkpoint page cache (90GiB cgroup headroom)
    import os as _os
    for item in MODEL_DIR.rglob("*"):
        if item.is_file():
            fd = _os.open(str(item), _os.O_RDONLY)
            try:
                _os.posix_fadvise(fd, 0, 0, _os.POSIX_FADV_DONTNEED)
            finally:
                _os.close(fd)
    before = mem_bytes("/sys/fs/cgroup/memory.current")
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_path, stop), daemon=True)
    monitor.start()
    start = time.time()
    cmd = [
        "/root/autodl-tmp/anisora-g0/.venv/bin/python",
        "/root/autodl-tmp/anime-remix-agent/experiments/keyframe_i2v_g1a2/run_v3_bf16.py",
        "--task", "i2v-14B",
        "--size", "1280*720",
        "--ckpt_dir", str(MODEL_DIR),
        "--image", str(out_dir),
        "--prompt", str(WORK / "g1a2_input.txt"),
        "--base_seed", "4096",
        "--frame_num", "81",
    ]
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=V3_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = proc.returncode
    runtime = round(time.time() - start, 2)
    stop.set()
    monitor.join(timeout=5)
    after = mem_bytes("/sys/fs/cgroup/memory.current")
    peak = 0.0
    if gpu_path.exists():
        for line in gpu_path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip().replace(".", "", 1).isdigit():
                peak = max(peak, float(parts[1]))
    produced = out_dir / "0.mp4"
    ok = code == 0 and produced.exists()
    out_name = "g1a2_three_keyframe.mp4"
    if ok:
        shutil.copyfile(produced, out_dir / out_name)
    print(
        f"[run] exit={code} runtime_s={runtime} peak_vram_mib={peak} "
        f"output={out_dir / out_name if ok else 'MISSING'}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng1a2 exit_code={code} runtime_seconds={runtime} peak_vram_mib={peak} "
            f"memory_current_before_gib={before/1024**3:.2f} after_gib={after/1024**3:.2f}\n"
        )
    if code != 0 or not ok:
        raise SystemExit(f"g1a2 generation failed exit={code} ok={ok}")
    return {
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": peak,
        "memory_before_gib": round(before / 1024**3, 2),
        "memory_after_gib": round(after / 1024**3, 2),
        "output_path": str(out_dir / out_name),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()
    info = prep()
    if args.prep_only:
        print("[run] --prep-only: guides/prompt/input ready.")
        return
    wait_for_model()
    result = run_generation()
    (OUTPUTS / "g1a2_run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
