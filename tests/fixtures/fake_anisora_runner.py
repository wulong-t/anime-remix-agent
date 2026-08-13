#!/usr/bin/env python
"""Fake AniSora BF16 runner for G1-MK1-R-PREP-L tests.

The real ``remote_sample`` tool invokes exactly one frozen AniSora command.
This fake parses the same arguments and synthesizes a deterministic MP4 with
the real ffmpeg (or simulates a failure), so tests never touch repository
media. Behavior is selected by the ``ANISORA_FAKE_MODE`` environment variable:

  success         - write a valid 0.mp4 (81 frames, 1280x704, 16fps H.264)
  invalid_frames  - write 0.mp4 with the wrong frame count
  invalid_size    - write 0.mp4 with the wrong canvas
  no_output       - exit 0 without writing 0.mp4
  fail            - exit non-zero without writing 0.mp4
  fail_with_output- write a valid 0.mp4 then exit non-zero
  extra_mp4       - write 0.mp4 and 1.mp4 then exit 0
  mutate_input    - rewrite the runtime input file then write 0.mp4

Every invocation writes an ``invoked.json`` sidecar recording the exact argv,
working directory, mode and runtime prompt content so tests can assert the
frozen command and runtime input. When ``ANISORA_FAKE_DELETE_FILE`` is set,
the fake runner deletes that file best-effort (used to test best-effort
failure evidence publication).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FAKE_MODES = {
    "success",
    "invalid_frames",
    "invalid_size",
    "no_output",
    "fail",
    "fail_with_output",
    "extra_mp4",
    "mutate_input",
}
RAW_SIZE = "1280x704"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fake_anisora_runner")
    parser.add_argument("--task", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base_seed", required=True)
    parser.add_argument("--frame_num", required=True)
    parser.add_argument("--sample_steps", required=True)
    parser.add_argument("--sample_shift", required=True)
    parser.add_argument("--sample_guide_scale", required=True)
    parser.add_argument("--offload_model", required=True)
    return parser


def _synthesize(out_dir: Path, *, frames: int, size: str) -> int:
    ffmpeg = os.environ.get("ANISORA_FAKE_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        print("fake runner: ffmpeg not found", file=sys.stderr)
        return 2
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={size}:r=16",
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_dir / "0.mp4"),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        print(
            f"fake runner: ffmpeg failed: {completed.stdout[-2000:]}",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.image)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("ANISORA_FAKE_MODE", "success")
    if mode not in FAKE_MODES:
        mode = "success"
    try:
        prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    except OSError:
        prompt_text = "<unreadable>"
    invoked = {
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "mode": mode,
        "prompt_file": args.prompt,
        "prompt_text": prompt_text,
        "image_dir": args.image,
    }
    (out_dir / "invoked.json").write_text(
        json.dumps(invoked, indent=2), encoding="utf-8"
    )
    delete_file = os.environ.get("ANISORA_FAKE_DELETE_FILE")
    if delete_file:
        try:
            os.remove(delete_file)
        except OSError:
            pass
    if mode == "fail":
        print("fake runner: simulated technical failure", file=sys.stderr)
        return 23
    if mode == "fail_with_output":
        code = _synthesize(out_dir, frames=81, size=RAW_SIZE)
        if code != 0:
            return code
        print("fake runner: simulated failure after output", file=sys.stderr)
        return 23
    if mode == "no_output":
        print("fake runner: simulated missing output")
        return 0
    if mode == "mutate_input":
        try:
            with Path(args.prompt).open("a", encoding="utf-8") as handle:
                handle.write("MUTATED\n")
        except OSError:
            pass
        return _synthesize(out_dir, frames=int(args.frame_num), size=RAW_SIZE)
    if mode == "invalid_frames":
        return _synthesize(out_dir, frames=20, size=RAW_SIZE)
    if mode == "invalid_size":
        return _synthesize(out_dir, frames=81, size="160x120")
    code = _synthesize(out_dir, frames=int(args.frame_num), size=RAW_SIZE)
    if mode == "extra_mp4" and code == 0:
        shutil.copyfile(out_dir / "0.mp4", out_dir / "1.mp4")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
