"""Automatic reference-asset extraction from one authorized anime episode."""

from anime_remix.services.episode_assets.classifier import (
    DashScopeEpisodeClassifier,
    EpisodeClassifier,
    FrameClassification,
    StubEpisodeClassifier,
    parse_classification,
)
from anime_remix.services.episode_assets.cropper import (
    DashScopeCropper,
    StubCropper,
)
from anime_remix.services.episode_assets.episode_assets import (
    EpisodeAssetExtractResult,
    extract_episode_assets,
)

__all__ = [
    "DashScopeCropper",
    "DashScopeEpisodeClassifier",
    "EpisodeAssetExtractResult",
    "EpisodeClassifier",
    "FrameClassification",
    "StubCropper",
    "StubEpisodeClassifier",
    "extract_episode_assets",
    "parse_classification",
]
