"""Core enums."""

from __future__ import annotations

from enum import Enum


class TimelineStrategy(str, Enum):
    CLIP = "clip"
    FREEZE_FRAME = "freeze_frame"
    PLACEHOLDER = "placeholder"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Emotion(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    TENSE = "tense"
    CALM = "calm"


class ShotScale(str, Enum):
    CLOSE_UP = "close_up"
    MEDIUM = "medium"
    WIDE = "wide"
