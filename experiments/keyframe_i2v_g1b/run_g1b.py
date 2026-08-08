#!/usr/bin/env python
"""G1-B: single V3 sampling from the constructed new keyframe (Tid=0)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUTPUTS = ROOT / "outputs"
SHIM = ROOT / "run_v3_bf16_g1b.py"
PROMPT_FILE = WORK / "g1b_input.txt"
MODEL_DIR = Path("/root/autodl-tmp/anisora-v3-g1/models/V3.1")
V3_DIR = Path("/root/autodl-tmp/anisora-v3-g1/anisoraV3")
PYTHON = "/root/autodl-tmp/anisora-g0/.venv/bin/python"


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
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
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


def mem_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w") as f:
        while not stop.is_set():
            v = mem_bytes("/sys/fs/cgroup/memory.current")
            if v >= 0:
                f.write(f"{time.time()},{v}\n")
                f.flush()
            time.sleep(1)


def release_page_cache() -> None:
    for item in MODEL_DIR.rglob("*"):
        if item.is_file():
            fd = os.open(str(item), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def main() -> None:
    assert PROMPT_FILE.exists(), f"missing {PROMPT_FILE}"
    OUTPUTS.mkdir(parents=True, exist_ok=True)

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

    release_page_cache()
    before = mem_bytes("/sys/fs/cgroup/memory.current")
    gpu_path = OUTPUTS / "g1b_gpu.csv"
    mem_path = OUTPUTS / "g1b_memory.csv"
    stop = threading.Event()
    threads = [
        threading.Thread(target=gpu_monitor, args=(gpu_path, stop), daemon=True),
        threading.Thread(target=mem_monitor, args=(mem_path, stop), daemon=True),
    ]
    for t in threads:
        t.start()

    log_path = OUTPUTS / "g1b_run.log"
    start = time.time()
    cmd = [
        PYTHON,
        str(SHIM),
        "--task", "i2v-14B",
        "--size", "1280*720",
        "--ckpt_dir", str(MODEL_DIR),
        "--image", str(OUTPUTS),
        "--prompt", str(PROMPT_FILE),
        "--base_seed", "4096",
        "--frame_num", "81",
        "--sample_steps", "40",
        "--sample_shift", "5",
        "--sample_guide_scale", "5",
        "--offload_model", "True",
    ]
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=V3_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = proc.returncode
    runtime = round(time.time() - start, 2)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    after = mem_bytes("/sys/fs/cgroup/memory.current")
    peak_mem = mem_bytes("/sys/fs/cgroup/memory.peak")

    peak_vram = 0.0
    if gpu_path.exists():
        for line in gpu_path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    peak_vram = max(peak_vram, float(parts[1]))
                except ValueError:
                    pass

    peak_ram = 0
    if mem_path.exists():
        for line in mem_path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    peak_ram = max(peak_ram, int(parts[1]))
                except ValueError:
                    pass
    peak_ram = max(peak_ram, peak_mem, before, after)

    produced = OUTPUTS / "0.mp4"
    ok = code == 0 and produced.exists()
    out_name = "g1b_new_shot.mp4"
    if ok:
        shutil.copyfile(produced, OUTPUTS / out_name)

    result = {
        "exit_code": code,
        "runtime_seconds": runtime,
        "peak_vram_mib": round(peak_vram, 1),
        "peak_ram_gib": round(peak_ram / 1024**3, 2),
        "memory_before_gib": round(before / 1024**3, 2),
        "memory_after_gib": round(after / 1024**3, 2),
        "output_path": str(OUTPUTS / out_name) if ok else None,
    }
    (OUTPUTS / "g1b_run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"\ng1b exit_code={code} runtime_seconds={runtime} "
            f"peak_vram_mib={peak_vram} peak_ram_gib={peak_ram/1024**3:.2f}\n"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if code != 0 or not ok:
        raise SystemExit(f"g1b generation failed exit={code} ok={ok}")


if __name__ == "__main__":
    main()
