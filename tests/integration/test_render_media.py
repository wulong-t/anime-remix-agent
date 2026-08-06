from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import (
    RenderProfile,
    ShotRequirement,
    Timeline,
    TimelineItem,
)
from anime_remix.errors import (
    OutputValidationError,
    SourceDriftError,
)
from anime_remix.json_io import sha256_file
from anime_remix.workflows.render_workflow import render_timeline


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    assert binary, "ffmpeg must be installed for media tests"
    return binary


def _encode_clip(ffmpeg: str, output: Path, *, frames: int) -> None:
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=24:duration={frames / 24:.6f}",
        "-vf",
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-bf",
        "0",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-video_track_timescale",
        "48000",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-chroma_sample_location",
        "left",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "mp4",
        str(output),
    ]
    subprocess.run(args, check=True, shell=False)


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("media")
    clips = root / "clips"
    clips.mkdir()
    ffmpeg = _ffmpeg()
    source_a = clips / "source_a.mp4"
    source_b = clips / "source_b.mp4"
    _encode_clip(ffmpeg, source_a, frames=96)
    _encode_clip(ffmpeg, source_b, frames=120)
    return {"root": root, "a": source_a, "b": source_b}


def _requirement(
    shot_id: str,
    order: int,
    action: str,
    target_frames: int,
) -> ShotRequirement:
    return ShotRequirement(
        id=shot_id,
        order=order,
        source_text=f"{action}。",
        action=action,
        target_frames=target_frames,
    )


def _clip_item(
    *,
    shot_id: str,
    order: int,
    action: str,
    target_frames: int,
    source: Path,
    timeline_dir: Path,
) -> TimelineItem:
    return TimelineItem(
        shot_id=shot_id,
        order=order,
        requirement=_requirement(shot_id, order, action, target_frames),
        strategy=TimelineStrategy.CLIP,
        source_asset_id="source_a",
        source_path=str(source.relative_to(timeline_dir).as_posix()),
        source_size_bytes=source.stat().st_size,
        source_sha256=sha256_file(source),
        source_in_frame=(96 - target_frames) // 2,
        source_frame_count=target_frames,
        target_frames=target_frames,
        score=None,
        reason_code="center_trim",
        reason="center_trim",
    )


def _timeline(
    clips: dict[str, Path],
    timeline_dir: Path,
    *,
    extra_items: list[TimelineItem] | None = None,
):
    items = [
        _clip_item(
            shot_id="shot_001",
            order=1,
            action="独自站立",
            target_frames=72,
            source=clips["a"],
            timeline_dir=timeline_dir,
        ),
        TimelineItem(
            shot_id="shot_002",
            order=2,
            requirement=_requirement("shot_002", 2, "纯黑", 36),
            strategy=TimelineStrategy.PLACEHOLDER,
            source_in_frame=0,
            source_frame_count=0,
            target_frames=36,
            score=None,
            reason_code="no_candidate",
            reason="no candidate",
        ),
    ]
    if extra_items:
        items.extend(extra_items)
    return Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=items,
    )


def test_render_clip_and_placeholder(media: dict[str, Path], tmp_path: Path) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "out.mp4"
    result = render_timeline(
        timeline_path,
        output,
        toolkit=FFmpegToolkit(),
    )
    toolkit = FFmpegToolkit()
    toolkit.validate_final(result, total_frames=108, profile=RenderProfile())
    assert result.exists() and result.stat().st_size > 0


def test_rerender_after_reordering(media: dict[str, Path], tmp_path: Path) -> None:
    timeline = _timeline(media, media["root"])
    original_path = media["root"] / "original.json"
    original_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    original_output = tmp_path / "original.mp4"
    render_timeline(original_path, original_output, toolkit=FFmpegToolkit())

    items = list(reversed(timeline.items))
    for index, item in enumerate(items, start=1):
        item.order = index
        item.requirement.order = index
    timeline.items = items
    timeline_path = media["root"] / "edited.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "rerendered.mp4"
    result = render_timeline(timeline_path, output, toolkit=FFmpegToolkit())
    assert result.exists()
    # Reordering + renumbering must change the actual rendered output.
    assert sha256_file(original_output) != sha256_file(result)


def test_public_render_creates_missing_parent_dirs(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "a" / "b" / "deep.mp4"
    result = render_timeline(timeline_path, output, toolkit=FFmpegToolkit())
    assert result == output.resolve()
    assert output.parent.exists()


def test_source_drift_fails(media: dict[str, Path], tmp_path: Path) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with media["a"].open("ab") as handle:
        handle.write(b"drift")
    try:
        with pytest.raises(SourceDriftError):
            render_timeline(
                timeline_path,
                tmp_path / "drift.mp4",
                toolkit=FFmpegToolkit(),
            )
    finally:
        # restore size for other tests (content no longer matches sha, but this
        # module fixture is shared; rewrite the clip cleanly)
        _encode_clip(_ffmpeg(), media["a"], frames=96)


def test_output_must_not_equal_source(media: dict[str, Path], tmp_path: Path) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(OutputValidationError):
        render_timeline(
            timeline_path,
            media["a"],
            toolkit=FFmpegToolkit(),
        )


def test_public_render_requires_overwrite(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"old")
    with pytest.raises(OutputValidationError):
        render_timeline(timeline_path, output, toolkit=FFmpegToolkit())
    result = render_timeline(
        timeline_path,
        output,
        overwrite=True,
        toolkit=FFmpegToolkit(),
    )
    assert result.stat().st_size > 4


def test_output_inside_managed_run_rejected(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    timeline = _timeline(media, media["root"])
    timeline_path = media["root"] / "timeline.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".anime-remix-run").write_text("{}", encoding="utf-8")
    with pytest.raises(OutputValidationError):
        render_timeline(
            timeline_path,
            managed / "out.mp4",
            toolkit=FFmpegToolkit(),
        )
