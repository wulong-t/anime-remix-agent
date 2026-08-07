"""Deterministic synthetic dataset for the 30x1000 retrieval benchmark.

Phase B only measures the already-locked retriever. The data is generated
without global random / time / UUID / hash() and never touches the filesystem.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from anime_remix.domain.models import (
    CharacterRef,
    ClipAsset,
    ProbedClip,
    ShotRequirement,
)

CHARACTERS = [
    ("char_lin_xia", "林夏"),
    ("char_lu_chen", "陆辰"),
    ("char_passerby", "路人"),
]
LOCATIONS = [
    ("loc_school_rooftop", "学校天台"),
    ("loc_classroom", "教室"),
    ("loc_garden", "花园"),
    ("loc_street", "街道"),
]
ACTIONS = [
    "独自站立",
    "沉默注视",
    "转身离开",
    "并肩站立",
    "骑车",
    "看书",
    "散步",
]
EMOTIONS = [None, "happy", "sad", "angry", "calm", "tense"]
SHOT_SCALES = [None, "close_up", "medium", "wide"]
FRAMES = [24, 30, 48, 72, 96, 120, 144, 192]

LONG_TEXT_CODEPOINTS = 300


def _long_text(base: str, length: int = LONG_TEXT_CODEPOINTS) -> str:
    return base + "啊" * max(0, length - len(base))


def build_clips(count: int = 1000) -> list[ProbedClip]:
    """Return ``count`` stable synthetic ProbedClip objects."""

    clips: list[ProbedClip] = []
    for index in range(1, count + 1):
        clip_id = f"bench_clip_{index:04d}"
        first = CHARACTERS[index % len(CHARACTERS)]
        characters = [first]
        if index % 7 == 0:
            characters.append(CHARACTERS[(index + 1) % len(CHARACTERS)])
        location = LOCATIONS[index % len(LOCATIONS)]
        action = ACTIONS[index % len(ACTIONS)]
        description = f"{first[1]}在{location[1]}里{action}。"
        if index % 50 == 0:
            description = _long_text(description)
        emotion = EMOTIONS[index % len(EMOTIONS)]
        shot_scale = SHOT_SCALES[index % len(SHOT_SCALES)]
        frames = FRAMES[index % len(FRAMES)]
        asset = ClipAsset(
            id=clip_id,
            path=f"clips/{clip_id}.mp4",
            characters=[
                CharacterRef(id=char_id, name=char_name)
                for char_id, char_name in characters
            ],
            location_id=location[0],
            location_name=location[1],
            action=action,
            description=description,
            emotion=emotion,
            shot_scale=shot_scale,
        )
        clips.append(
            ProbedClip(
                asset=asset,
                resolved_path=Path(f"clips/{clip_id}.mp4"),
                size_bytes=1000,
                width=1280,
                height=720,
                fps_num=24,
                fps_den=1,
                nb_frames=frames,
                duration_seconds=Decimal(frames) / Decimal(24),
            )
        )
    return clips


def build_requirements(count: int = 30) -> list[ShotRequirement]:
    """Return ``count`` stable synthetic ShotRequirement objects."""

    requirements: list[ShotRequirement] = []
    for index in range(1, count + 1):
        shot_id = f"bench_shot_{index:03d}"
        characters: list[tuple[str, str]] = []
        if index % 11 != 0:
            characters.append(CHARACTERS[index % len(CHARACTERS)])
        if index % 7 == 0:
            characters.append(CHARACTERS[(index + 1) % len(CHARACTERS)])
        has_location = index % 5 != 0
        location = LOCATIONS[(index * 3) % len(LOCATIONS)] if has_location else None
        action = ACTIONS[index % len(ACTIONS)]
        source_text = (
            f"第{index}镜：{action}发生在"
            f"{location[1] if location is not None else '某个地方'}。"
        )
        if index >= 26:
            source_text = _long_text(source_text)
        target_frames = FRAMES[(index * 2) % len(FRAMES)]
        emotion = EMOTIONS[index % len(EMOTIONS)]
        shot_scale = SHOT_SCALES[index % len(SHOT_SCALES)]
        requirements.append(
            ShotRequirement(
                id=shot_id,
                order=index,
                source_text=source_text,
                characters=[
                    CharacterRef(id=char_id, name=char_name)
                    for char_id, char_name in characters
                ],
                location_id=location[0] if location is not None else None,
                location_name=location[1] if location is not None else None,
                action=action,
                target_frames=target_frames,
                dialogue=None,
                emotion=emotion,
                shot_scale=shot_scale,
            )
        )
    return requirements


def selection_summary(selections: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Stable per-shot summary for determinism comparisons."""

    rows: list[tuple[Any, ...]] = []
    for shot_id in sorted(selections):
        selected = selections[shot_id]
        rows.append(
            (
                shot_id,
                selected.asset.asset.id if selected.asset is not None else None,
                selected.rank,
                selected.reason_code,
                selected.source_in_frame,
                selected.source_frame_count,
            )
        )
    return tuple(rows)


def scan_statistics(audit: dict[str, Any]) -> tuple[float, int]:
    """Average and max scanned candidate counts across all shots."""

    counts = [
        len(shot["selection_trace"]["scanned_candidates"])
        for shot in audit["shots"]
    ]
    average = sum(counts) / len(counts) if counts else 0.0
    return average, max(counts) if counts else 0
