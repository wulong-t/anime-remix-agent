"""Compile ShotRequirements + selections into a Timeline 1.9 document."""

from __future__ import annotations

import os
from pathlib import Path

from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import (
    RenderProfile,
    ShotRequirement,
    Timeline,
    TimelineItem,
)
from anime_remix.errors import TimelineValidationError
from anime_remix.services.clip_retriever import Selection


def _relative_source_path(source: Path, target_dir: Path) -> str:
    source_abs = source.resolve()
    target_abs = target_dir.resolve()
    if source_abs.anchor != target_abs.anchor:
        raise TimelineValidationError(
            "source and build target must be on the same drive; "
            "cannot construct a relative path",
            actual={
                "source_path": os.fspath(source_abs),
                "source_anchor": source_abs.anchor,
                "target_path": os.fspath(target_abs),
                "target_anchor": target_abs.anchor,
            },
        )
    try:
        rel = os.path.relpath(source_abs, start=target_abs)
    except ValueError as exc:
        raise TimelineValidationError(
            "source and build target must be on the same drive",
            actual=(source_abs, target_abs),
        ) from exc
    return Path(rel).as_posix()


def compile_timeline(
    requirements: list[ShotRequirement],
    selections: dict[str, Selection],
    target_dir: Path,
    *,
    render_profile: RenderProfile | None = None,
    source_sha256: dict[str, str] | None = None,
) -> Timeline:
    source_sha256 = source_sha256 or {}
    items: list[TimelineItem] = []
    for requirement in requirements:
        selection = selections.get(requirement.id)
        if selection is None:
            raise TimelineValidationError(
                "missing selection",
                shot_id=requirement.id,
            )
        if selection.asset is None:
            items.append(
                TimelineItem(
                    shot_id=requirement.id,
                    order=requirement.order,
                    requirement=requirement,
                    strategy=TimelineStrategy.PLACEHOLDER,
                    source_in_frame=0,
                    source_frame_count=0,
                    target_frames=requirement.target_frames,
                    score=None,
                    reason_code=selection.reason_code,
                    reason="no candidate passed gates",
                )
            )
            continue
        source = selection.asset.resolved_path
        sha256 = source_sha256.get(selection.asset.asset.id)
        if not sha256:
            raise TimelineValidationError(
                "missing source sha256",
                asset_id=selection.asset.asset.id,
                shot_id=requirement.id,
            )
        items.append(
            TimelineItem(
                shot_id=requirement.id,
                order=requirement.order,
                requirement=requirement,
                strategy=TimelineStrategy.CLIP,
                source_asset_id=selection.asset.asset.id,
                source_path=_relative_source_path(source, target_dir),
                source_size_bytes=selection.asset.size_bytes,
                source_sha256=sha256,
                source_in_frame=selection.source_in_frame,
                source_frame_count=selection.source_frame_count,
                target_frames=requirement.target_frames,
                score=selection.score,
                reason_code=selection.reason_code,
                reason=selection.reason_code,
            )
        )
    return Timeline(
        schema_version="1.9",
        path_base="timeline_dir",
        render_profile=render_profile or RenderProfile(),
        items=items,
    )
