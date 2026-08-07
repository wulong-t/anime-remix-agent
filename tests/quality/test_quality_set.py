"""Phase B quality set hard gate (AGENTS.md v1.14 section 18.10 / B1)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from quality_cases import QualityCase, build_cases

from anime_remix.services.clip_retriever import retrieve

REPORT_PATH = (
    Path(__file__).resolve().parents[2] / ".tmp" / "quality-report.json"
)


def _run_case(case: QualityCase) -> dict[str, Any]:
    selections, audit = retrieve([case.requirement], list(case.candidates))
    selected = selections[case.requirement.id]
    shot = audit["shots"][0]
    trace = shot["selection_trace"]
    return {
        "selected_asset_id": (
            selected.asset.asset.id if selected.asset is not None else None
        ),
        "strategy": shot["selected"]["selected_strategy"],
        "reason_code": selected.reason_code,
        "source_in_frame": selected.source_in_frame,
        "source_frame_count": selected.source_frame_count,
        "global_rank": selected.rank,
        "freeze_fallback_asset_id": trace["freeze_fallback_asset_id"],
        "stop_reason": trace["stop_reason"],
    }


def _case_passed(case: QualityCase, actual: dict[str, Any]) -> bool:
    checks = [
        actual["selected_asset_id"] == case.expected_selected_asset_id,
        actual["strategy"] == case.expected_strategy,
        actual["reason_code"] == case.expected_reason_code,
    ]
    if case.expected_source_in_frame is not None:
        checks.append(
            actual["source_in_frame"] == case.expected_source_in_frame
        )
    if case.expected_source_frame_count is not None:
        checks.append(
            actual["source_frame_count"] == case.expected_source_frame_count
        )
    if case.expected_global_rank is not None:
        checks.append(actual["global_rank"] == case.expected_global_rank)
    if case.expected_freeze_fallback_asset_id is not None:
        checks.append(
            actual["freeze_fallback_asset_id"]
            == case.expected_freeze_fallback_asset_id
        )
    if case.expected_stop_reason is not None:
        checks.append(actual["stop_reason"] == case.expected_stop_reason)
    return all(checks)


def _failure_text(case: QualityCase, actual: dict[str, Any]) -> str:
    return (
        f"case_id={case.case_id}\n"
        f"case_tags={','.join(case.case_tags)}\n"
        f"expected_selected_asset_id={case.expected_selected_asset_id!r} "
        f"actual_selected_asset_id={actual['selected_asset_id']!r}\n"
        f"expected_strategy={case.expected_strategy!r} "
        f"actual_strategy={actual['strategy']!r}\n"
        f"expected_reason_code={case.expected_reason_code!r} "
        f"actual_reason_code={actual['reason_code']!r}\n"
        f"actual_global_rank={actual['global_rank']!r}\n"
        f"actual_stop_reason={actual['stop_reason']!r}\n"
        f"actual_freeze_fallback_asset_id="
        f"{actual['freeze_fallback_asset_id']!r}\n"
        f"rationale={case.expected_rationale}\n"
    )


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_quality_set_100_percent() -> None:
    cases = build_cases()
    assert 30 <= len(cases) <= 50, len(cases)

    failures: list[str] = []
    selection_hits = 0
    strategy_hits = 0
    reason_hits = 0
    by_tag: dict[str, list[bool]] = {}

    for case in cases:
        actual = _run_case(case)
        if case.case_id == "q24_asset_emotion_none_score_zero":
            selections, _ = retrieve([case.requirement], list(case.candidates))
            score = selections[case.requirement.id].score
            assert score is not None
            assert score.emotion == Decimal("0.000000")
        selection_hits += (
            actual["selected_asset_id"] == case.expected_selected_asset_id
        )
        strategy_hits += actual["strategy"] == case.expected_strategy
        reason_hits += actual["reason_code"] == case.expected_reason_code
        passed = _case_passed(case, actual)
        for tag in case.case_tags:
            by_tag.setdefault(tag, []).append(passed)
        if not passed:
            failures.append(_failure_text(case, actual))

    total = len(cases)
    report = {
        "quality": {
            "total_cases": total,
            "passed_cases": total - len(failures),
            "selection_accuracy": selection_hits / total,
            "strategy_accuracy": strategy_hits / total,
            "reason_code_accuracy": reason_hits / total,
            "by_tag": {
                tag: f"{sum(values)}/{len(values)}"
                for tag, values in sorted(by_tag.items())
            },
        }
    }
    _write_report(report)

    assert selection_hits == total, (
        f"selection_accuracy={selection_hits}/{total}\n" + "\n".join(failures)
    )
    assert strategy_hits == total, (
        f"strategy_accuracy={strategy_hits}/{total}\n" + "\n".join(failures)
    )
    assert reason_hits == total, (
        f"reason_code_accuracy={reason_hits}/{total}\n" + "\n".join(failures)
    )
    assert not failures, "\n".join(failures)
