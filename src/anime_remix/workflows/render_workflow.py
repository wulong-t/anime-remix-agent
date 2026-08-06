"""Independent render workflow: timeline -> verified MP4."""

from __future__ import annotations

import datetime as _datetime
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path

from pydantic import TypeAdapter

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import Timeline
from anime_remix.errors import (
    OutputValidationError,
    RenderError,
    SourceDriftError,
    TimelineValidationError,
)
from anime_remix.json_io import load_json_object, sha256_file
from anime_remix.services.input_loader import validate_timeline_source_path


def load_timeline(path: Path) -> Timeline:
    data = load_json_object(path)
    try:
        return TypeAdapter(Timeline).validate_python(data)
    except Exception as exc:
        raise TimelineValidationError(
            f"invalid timeline: {exc}",
            actual=path,
        ) from exc


def _log(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    timestamp = _datetime.datetime.now(_datetime.UTC).isoformat()
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{timestamp} {line}\n")
    except OSError:
        pass


def _ensure_output_safety(
    timeline_path: Path,
    output_path: Path,
    timeline: Timeline,
    sources: dict[str, Path],
    *,
    allow_managed_output: bool,
) -> None:
    output = output_path.resolve()
    timeline_resolved = timeline_path.resolve()
    if output == timeline_resolved:
        raise OutputValidationError("output must not equal the timeline file")
    if output in {source.resolve() for source in sources.values()}:
        raise OutputValidationError("output must not equal any source asset")
    for source in sources.values():
        try:
            if os.path.samefile(output, source):
                raise OutputValidationError(
                    "output must not be the same inode as a source asset",
                    actual=source,
                )
        except (FileNotFoundError, OSError):
            pass
    if output_path.is_symlink():
        raise OutputValidationError("output path must not be a symlink")
    if allow_managed_output:
        return
    ancestor = output.parent
    while True:
        if (ancestor / ".anime-remix-run").is_file():
            raise OutputValidationError(
                "output must not be inside a managed run directory",
                actual=ancestor,
            )
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent


def render_timeline(
    timeline_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    allow_managed_output: bool = False,
    log_path: Path | None = None,
    toolkit: FFmpegToolkit | None = None,
) -> Path:
    """Render a timeline into a verified MP4 using only timeline + sources."""

    timeline = load_timeline(timeline_path)
    toolkit = toolkit or FFmpegToolkit()
    toolkit.check_capabilities()

    sources: dict[str, Path] = {}
    for item in timeline.items:
        if item.strategy not in (
            TimelineStrategy.CLIP,
            TimelineStrategy.FREEZE_FRAME,
        ):
            continue
        source = validate_timeline_source_path(
            timeline_path.resolve().parent,
            item.source_path or "",
        )
        actual_size = source.stat().st_size
        if actual_size != item.source_size_bytes:
            raise SourceDriftError(
                "source size mismatch",
                asset_id=item.source_asset_id,
                shot_id=item.shot_id,
                field="source_size_bytes",
                actual=(item.source_size_bytes, actual_size),
            )
        actual_sha256 = sha256_file(source)
        if actual_sha256 != item.source_sha256:
            raise SourceDriftError(
                "source SHA256 mismatch",
                asset_id=item.source_asset_id,
                shot_id=item.shot_id,
                field="source_sha256",
            )
        if item.strategy is TimelineStrategy.FREEZE_FRAME:
            source_nb_frames = toolkit.source_nb_frames(source)
            if (
                item.source_in_frame + item.source_frame_count
                > source_nb_frames
            ):
                raise TimelineValidationError(
                    "freeze_frame source interval exceeds actual "
                    "source frame count",
                    shot_id=item.shot_id,
                    field="source_frame_count",
                    actual=(
                        item.source_in_frame,
                        item.source_frame_count,
                        source_nb_frames,
                    ),
                )
        sources[item.shot_id] = source

    _ensure_output_safety(
        timeline_path,
        output_path,
        timeline,
        sources,
        allow_managed_output=allow_managed_output,
    )
    if output_path.exists() and not overwrite and not allow_managed_output:
        raise OutputValidationError(
            "output already exists; pass --overwrite to replace it",
            actual=output_path,
        )

    total_frames = sum(item.target_frames for item in timeline.items)
    total_samples = total_frames * 48000 // 24

    if allow_managed_output:
        workdir = output_path.resolve().parent / "normalized"
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(
            tempfile.mkdtemp(
                prefix=".anime-remix-render-",
                dir=output_path.resolve().parent,
            )
        )

    _log(log_path, f"render start: {len(timeline.items)} items, {total_frames} frames")
    try:
        segments: list[Path] = []
        signatures: list[dict[str, object]] = []
        # Playback order is exclusively Timeline.items array order; the
        # renderer must never re-sort by item.order.
        for index, item in enumerate(timeline.items):
            segment = workdir / f"segment_{index:03d}.mp4"
            if item.strategy is TimelineStrategy.CLIP:
                _log(log_path, f"segment {index}: clip {item.source_asset_id}")
                toolkit.render_clip(
                    item,
                    sources[item.shot_id],
                    segment,
                    profile=timeline.render_profile,
                )
            elif item.strategy is TimelineStrategy.FREEZE_FRAME:
                _log(
                    log_path,
                    f"segment {index}: freeze_frame {item.source_asset_id}",
                )
                toolkit.render_freeze_frame(
                    item,
                    sources[item.shot_id],
                    segment,
                    profile=timeline.render_profile,
                )
            elif item.strategy is TimelineStrategy.PLACEHOLDER:
                _log(log_path, f"segment {index}: placeholder")
                toolkit.render_placeholder(
                    item,
                    segment,
                    profile=timeline.render_profile,
                )
            else:
                raise RenderError(
                    "unsupported strategy",
                    shot_id=item.shot_id,
                    actual=item.strategy,
                )
            signature = toolkit.validate_segment(
                segment,
                target_frames=item.target_frames,
                shot_id=item.shot_id,
            )
            segments.append(segment)
            signatures.append(signature)

        toolkit.concat_signatures_equal(signatures)
        joined = workdir / "joined_video.mp4"
        _log(log_path, "concat segments")
        toolkit.concat_video(
            segments,
            joined,
            durations=[Decimal(item.target_frames) / Decimal(24) for item in timeline.items],
        )
        final = workdir / "final.mp4"
        _log(log_path, "mux final AAC")
        toolkit.mux_final(
            joined,
            final,
            total_samples=total_samples,
            profile=timeline.render_profile,
        )
        _log(log_path, "validate final")
        toolkit.validate_final(
            final,
            total_frames=total_frames,
            profile=timeline.render_profile,
        )

        os.replace(final, output_path)
        _log(log_path, f"render complete: {output_path}")
        return output_path
    except BaseException as exc:
        _log(log_path, f"render failed: {type(exc).__name__}")
        raise
    finally:
        if not allow_managed_output:
            shutil.rmtree(workdir, ignore_errors=True)
