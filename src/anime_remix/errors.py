"""Error types and exit-code mapping for anime-remix."""

from __future__ import annotations


class AnimeRemixError(Exception):
    """Base class for all anime-remix errors."""

    stage = "unknown"
    exit_code = 1

    def __init__(
        self,
        message: str,
        *,
        asset_id: str | None = None,
        shot_id: str | None = None,
        field: str | None = None,
        actual: object | None = None,
    ) -> None:
        self.asset_id = asset_id
        self.shot_id = shot_id
        self.field = field
        self.actual = actual
        parts = [f"stage={self.stage}"]
        if asset_id:
            parts.append(f"asset_id={asset_id}")
        if shot_id:
            parts.append(f"shot_id={shot_id}")
        if field:
            parts.append(f"field={field}")
        if actual is not None:
            parts.append(f"actual={actual!r}")
        super().__init__(f"{message} ({', '.join(parts)})")


class InputValidationError(AnimeRemixError):
    stage = "input_validation"
    exit_code = 2


class UnsafePathError(AnimeRemixError):
    stage = "path_safety"
    exit_code = 2


class EnvironmentCapabilityError(AnimeRemixError):
    stage = "environment"
    exit_code = 3


class MediaProbeError(AnimeRemixError):
    stage = "media_probe"
    exit_code = 3


class UnsupportedMediaError(AnimeRemixError):
    stage = "media_contract"
    exit_code = 3


class RetrievalError(AnimeRemixError):
    stage = "retrieval"
    exit_code = 4


class TimelineValidationError(AnimeRemixError):
    stage = "timeline"
    exit_code = 4


class UnsupportedStrategyError(AnimeRemixError):
    stage = "timeline"
    exit_code = 4


class SourceDriftError(AnimeRemixError):
    stage = "source_drift"
    exit_code = 5


class RenderError(AnimeRemixError):
    stage = "render"
    exit_code = 5


class OutputValidationError(AnimeRemixError):
    stage = "output_validation"
    exit_code = 5


class PublicationError(AnimeRemixError):
    stage = "publication"
    exit_code = 6


ERROR_EXIT_CODES: dict[type[AnimeRemixError], int] = {
    InputValidationError: 2,
    UnsafePathError: 2,
    EnvironmentCapabilityError: 3,
    MediaProbeError: 3,
    UnsupportedMediaError: 3,
    RetrievalError: 4,
    TimelineValidationError: 4,
    UnsupportedStrategyError: 4,
    SourceDriftError: 5,
    RenderError: 5,
    OutputValidationError: 5,
    PublicationError: 6,
}

