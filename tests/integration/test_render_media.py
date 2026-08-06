from __future__ import annotations

import shutil
import subprocess
from itertools import pairwise
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
    TimelineValidationError,
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
    source_short = clips / "source_short.mp4"
    _encode_clip(ffmpeg, source_a, frames=96)
    _encode_clip(ffmpeg, source_b, frames=120)
    _encode_clip(ffmpeg, source_short, frames=36)
    return {
        "root": root,
        "a": source_a,
        "b": source_b,
        "short": source_short,
    }


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


def _freeze_item(
    *,
    shot_id: str,
    order: int,
    action: str,
    target_frames: int,
    source: Path,
    timeline_dir: Path,
    source_in_frame: int = 0,
    source_frame_count: int | None = None,
) -> TimelineItem:
    if source_frame_count is None:
        source_frame_count = 36
    return TimelineItem(
        shot_id=shot_id,
        order=order,
        requirement=_requirement(shot_id, order, action, target_frames),
        strategy=TimelineStrategy.FREEZE_FRAME,
        source_asset_id="source_short",
        source_path=str(source.relative_to(timeline_dir).as_posix()),
        source_size_bytes=source.stat().st_size,
        source_sha256=sha256_file(source),
        source_in_frame=source_in_frame,
        source_frame_count=source_frame_count,
        target_frames=target_frames,
        score=None,
        reason_code="short_source_freeze",
        reason="short_source_freeze",
    )


_FRAME_BYTES = 1280 * 720 * 3 // 2


def _decode_frames(path: Path) -> list[bytes]:
    completed = subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    data = completed.stdout
    assert len(data) % _FRAME_BYTES == 0
    return [
        data[i * _FRAME_BYTES : (i + 1) * _FRAME_BYTES]
        for i in range(len(data) // _FRAME_BYTES)
    ]


def _mean_abs_diff(a: bytes, b: bytes) -> float:
    assert len(a) == len(b)
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


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


def test_render_freeze_with_clip_and_placeholder(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    items = [
        _clip_item(
            shot_id="shot_001",
            order=1,
            action="独自站立",
            target_frames=72,
            source=media["a"],
            timeline_dir=media["root"],
        ),
        _freeze_item(
            shot_id="shot_002",
            order=2,
            action="短素材定格",
            target_frames=72,
            source=media["short"],
            timeline_dir=media["root"],
        ),
        TimelineItem(
            shot_id="shot_003",
            order=3,
            requirement=_requirement("shot_003", 3, "纯黑", 36),
            strategy=TimelineStrategy.PLACEHOLDER,
            source_in_frame=0,
            source_frame_count=0,
            target_frames=36,
            score=None,
            reason_code="no_candidate",
            reason="no candidate",
        ),
    ]
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=items,
    )
    timeline_path = media["root"] / "freeze-mix.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "freeze-mix.mp4"
    result = render_timeline(
        timeline_path,
        output,
        toolkit=FFmpegToolkit(),
    )
    toolkit = FFmpegToolkit()
    toolkit.validate_final(result, total_frames=180, profile=RenderProfile())
    assert result.exists() and result.stat().st_size > 0


def test_freeze_segment_exact_frames_and_last_frame_clone(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    item = _freeze_item(
        shot_id="shot_001",
        order=1,
        action="短素材定格",
        target_frames=72,
        source=media["short"],
        timeline_dir=media["root"],
    )
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )
    timeline_path = media["root"] / "freeze.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "freeze.mp4"
    result = render_timeline(
        timeline_path,
        output,
        toolkit=FFmpegToolkit(),
    )
    toolkit = FFmpegToolkit()
    toolkit.validate_final(result, total_frames=72, profile=RenderProfile())

    frames = _decode_frames(result)
    assert len(frames) == 72
    last_source_index = 36 - 1
    # H.264 is lossy: cloned frames are not bit-identical, so compare with a
    # small mean-absolute-difference budget instead of byte equality.
    assert _mean_abs_diff(frames[last_source_index], frames[-1]) <= 2.0
    tail = frames[last_source_index:]
    for left, right in pairwise(tail):
        assert _mean_abs_diff(left, right) <= 2.0
    # The source portion itself is still moving (testsrc2 changes per frame),
    # which distinguishes cloning from a static/black source.
    assert _mean_abs_diff(
        frames[last_source_index - 1],
        frames[last_source_index],
    ) > 0.5


def test_freeze_filter_graph_tpad_after_fps(
    media: dict[str, Path],
) -> None:
    toolkit = FFmpegToolkit()
    item = _freeze_item(
        shot_id="shot_001",
        order=1,
        action="短素材定格",
        target_frames=72,
        source=media["short"],
        timeline_dir=media["root"],
    )
    graph = toolkit.freeze_frame_filter_graph(item)
    assert graph.index("fps=fps=24") < graph.index("tpad=stop_mode=clone")
    assert graph.index("tpad=stop_mode=clone") < graph.index(
        "trim=end_frame=72"
    )
    assert graph.count("tpad") == 1


def test_freeze_hand_edited_interval_within_source(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    item = _freeze_item(
        shot_id="shot_001",
        order=1,
        action="短素材定格",
        target_frames=72,
        source=media["short"],
        timeline_dir=media["root"],
        source_in_frame=5,
        source_frame_count=20,
    )
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )
    timeline_path = media["root"] / "freeze-hand.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    output = tmp_path / "freeze-hand.mp4"
    result = render_timeline(
        timeline_path,
        output,
        toolkit=FFmpegToolkit(),
    )
    FFmpegToolkit().validate_final(
        result,
        total_frames=72,
        profile=RenderProfile(),
    )


def test_freeze_interval_out_of_bounds_fails(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    item = _freeze_item(
        shot_id="shot_001",
        order=1,
        action="短素材定格",
        target_frames=72,
        source=media["short"],
        timeline_dir=media["root"],
        source_in_frame=20,
        source_frame_count=20,
    )
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )
    timeline_path = media["root"] / "freeze-oob.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(TimelineValidationError):
        render_timeline(
            timeline_path,
            tmp_path / "freeze-oob.mp4",
            toolkit=FFmpegToolkit(),
        )


def test_freeze_source_drift_fails(
    media: dict[str, Path],
    tmp_path: Path,
) -> None:
    drifted = tmp_path / "drifted.mp4"
    shutil.copyfile(media["short"], drifted)
    item = _freeze_item(
        shot_id="shot_001",
        order=1,
        action="短素材定格",
        target_frames=72,
        source=drifted,
        timeline_dir=tmp_path,
    )
    timeline = Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=RenderProfile(),
        items=[item],
    )
    timeline_path = tmp_path / "freeze-drift.json"
    timeline_path.write_text(
        timeline.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with drifted.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(SourceDriftError):
        render_timeline(
            timeline_path,
            tmp_path / "freeze-drift.mp4",
            toolkit=FFmpegToolkit(),
        )
