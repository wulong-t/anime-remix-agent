"""Wall-clock 30x1000 retrieval benchmark (Phase B, run via -m benchmark)."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from dataset import (
    build_clips,
    build_requirements,
    scan_statistics,
    selection_summary,
)

from anime_remix.services.clip_retriever import retrieve

pytestmark = pytest.mark.benchmark

REPORT_PATH = (
    Path(__file__).resolve().parents[2] / ".tmp" / "retrieval-benchmark.json"
)

MEDIAN_TARGET_SECONDS = 5.0
MAX_TARGET_SECONDS = 8.0


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_30x1000_wall_clock_benchmark() -> None:
    requirements = build_requirements()
    clips = build_clips()

    def run_once() -> tuple[float, Any, dict[str, Any]]:
        started = time.perf_counter()
        selections, audit = retrieve(requirements, clips)
        elapsed = time.perf_counter() - started
        return elapsed, selections, audit

    run_once()  # warm-up >= 1
    times: list[float] = []
    summaries: list[tuple[tuple[Any, ...], ...]] = []
    audits: list[dict[str, Any]] = []
    for _ in range(5):
        elapsed, selections, audit = run_once()
        times.append(elapsed)
        summaries.append(selection_summary(selections))
        audits.append(audit)

    deterministic = all(summary == summaries[0] for summary in summaries)
    average_scanned, max_scanned = scan_statistics(audits[-1])

    tracemalloc.start()
    retrieve(requirements, clips)
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    min_seconds = min(times)
    median_seconds = statistics.median(times)
    max_seconds = max(times)
    report = {
        "benchmark": {
            "shots": len(requirements),
            "clips": len(clips),
            "pairs": len(requirements) * len(clips),
            "runs": len(times),
            "min_seconds": min_seconds,
            "median_seconds": median_seconds,
            "max_seconds": max_seconds,
            "deterministic": deterministic,
            "average_scanned_candidates": average_scanned,
            "max_scanned_candidates": max_scanned,
            "peak_python_bytes": peak_bytes,
        }
    }
    _write_report(report)

    assert deterministic
    assert median_seconds <= MEDIAN_TARGET_SECONDS, (
        f"median={median_seconds:.3f}s > {MEDIAN_TARGET_SECONDS}s"
    )
    assert max_seconds <= MAX_TARGET_SECONDS, (
        f"max={max_seconds:.3f}s > {MAX_TARGET_SECONDS}s"
    )
