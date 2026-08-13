"""I2 review loop: emit, preview and validate ``shot_plan.json``.

The Director result is written to a review directory together with a
human-readable Markdown preview.  The user edits the JSON (the preview is
derived and never parsed back), then ``validate_shot_plan_file`` re-checks
the edited document before it is consumed by later stages.
"""

from __future__ import annotations

from pathlib import Path

from anime_remix.json_io import dump_json_atomic
from anime_remix.services.script.shot_plan import (
    ShotPlanDocument,
    load_shot_plan,
)


def build_review_preview(document: ShotPlanDocument) -> str:
    """Render a Markdown preview of a shot plan for human review."""

    lines = [
        "# Shot Plan Review Preview",
        "",
        f"- schema: `{document.schema_version}`",
        f"- shots: {len(document.shots)}",
        f"- total duration: {sum(shot.duration_seconds for shot in document.shots):.1f}s",
        "",
    ]
    for shot in document.shots:
        lines.append(f"## {shot.shot_id} (order {shot.order}, {shot.duration_seconds:g}s)")
        lines.append("")
        lines.append(f"- scene: `{shot.scene_id}` | scale: `{shot.shot_scale}`")
        lines.append(f"- purpose: {shot.narrative_purpose}")
        lines.append(f"- composition: {shot.composition}")
        lines.append(
            f"- camera: {shot.camera_position} | motion: {shot.camera_motion}"
        )
        lines.append(f"- subjects: {', '.join(shot.subjects)}")
        lines.append(f"- setting: {shot.setting}")
        lines.append(f"- props: {', '.join(shot.props) or '—'}")
        lines.append(f"- start: {shot.start_state}")
        lines.append("- beats:")
        for beat in shot.action_beats:
            lines.append(
                f"  - @{beat.time_seconds:g}s {beat.description}"
            )
        lines.append(f"- end: {shot.end_state}")
        lines.append(f"- emotion arc: {shot.emotion_arc}")
        if shot.dialogue:
            lines.append(f"- dialogue: {shot.dialogue}")
        if shot.continuity_in:
            lines.append(f"- continuity in: {shot.continuity_in}")
        if shot.continuity_out:
            lines.append(f"- continuity out: {shot.continuity_out}")
        lines.append("")
    return "\n".join(lines)


def write_review_artifacts(
    review_dir: Path,
    document: ShotPlanDocument,
    *,
    run_manifest: dict[str, object] | None = None,
) -> Path:
    """Write ``shot_plan.json`` and ``shot_plan.review.md`` into a review dir."""

    review_dir.mkdir(parents=True, exist_ok=True)
    shot_plan_path = review_dir / "shot_plan.json"
    dump_json_atomic(
        shot_plan_path,
        document.model_dump(mode="json"),
        sort_keys=True,
    )
    (review_dir / "shot_plan.review.md").write_text(
        build_review_preview(document),
        encoding="utf-8",
    )
    if run_manifest is not None:
        dump_json_atomic(
            review_dir / "run_manifest.json",
            run_manifest,
            sort_keys=True,
        )
    return shot_plan_path


def validate_shot_plan_file(path: Path) -> ShotPlanDocument:
    """Strictly validate an edited shot_plan.json; raises on any violation."""

    return load_shot_plan(path)
