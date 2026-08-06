"""Core enums."""

from __future__ import annotations

from enum import Enum


class TimelineStrategy(str, Enum):
    CLIP = "clip"
    PLACEHOLDER = "placeholder"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

