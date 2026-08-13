"""Unit tests for episode frame timestamp planning (pure helpers)."""

from __future__ import annotations

from anime_remix.services.episode_assets.sampler import _spread_timestamps


def test_spread_timestamps_keeps_all_when_below_cap() -> None:
    timestamps = [1.0, 2.0, 3.0]
    assert _spread_timestamps(timestamps, max_frames=5) == timestamps


def test_spread_timestamps_covers_full_range_when_capped() -> None:
    timestamps = [float(index) for index in range(1, 201)]
    selected = _spread_timestamps(timestamps, max_frames=10)
    assert len(selected) == 10
    assert selected[0] == 1.0
    assert selected[-1] == 200.0
    # roughly even coverage across the full range
    spans = [
        selected[index + 1] - selected[index]
        for index in range(len(selected) - 1)
    ]
    assert min(spans) >= 10
    assert max(spans) <= 30


def test_spread_timestamps_handles_single_frame_cap() -> None:
    timestamps = [1.0, 2.0, 3.0]
    assert _spread_timestamps(timestamps, max_frames=1) == [1.0]
