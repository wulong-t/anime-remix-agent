"""Perceptual-hash frame deduplication for episode extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image


def dhash(image: Image.Image, *, size: int = 8) -> str:
    """Return a 64-bit difference hash as a binary string."""

    gray = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = gray.tobytes()
    bits: list[str] = []
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            left = pixels[offset + column]
            right = pixels[offset + column + 1]
            bits.append("1" if left > right else "0")
    return "".join(bits)


def color_signature(image: Image.Image, *, grid: int = 4) -> str:
    """Return a coarse quantized RGB signature so flat colors stay distinct."""

    small = image.convert("RGB").resize((grid, grid), Image.LANCZOS)
    bits: list[str] = []
    raw = small.tobytes()
    for offset in range(0, len(raw), 3):
        red, green, blue = raw[offset], raw[offset + 1], raw[offset + 2]
        bits.append(format(red >> 6, "02b"))
        bits.append(format(green >> 6, "02b"))
        bits.append(format(blue >> 6, "02b"))
    return "".join(bits)


def frame_hash(image: Image.Image, *, size: int = 8, grid: int = 4) -> str:
    """Combined structural + color hash for robust frame deduplication."""

    return dhash(image, size=size) + color_signature(image, grid=grid)


def split_hash(value: str) -> tuple[str, str]:
    """Split a combined frame hash into (dhash_part, color_part)."""

    if len(value) != 64 + 4 * 4 * 6:
        raise ValueError("unexpected combined frame hash length")
    return value[:64], value[64:]


def near_duplicates(
    left: str,
    right: str,
    *,
    dhash_threshold: int = 12,
    color_threshold: int = 2,
) -> bool:
    """True when two combined frame hashes look like the same shot."""

    left_dhash, left_color = split_hash(left)
    right_dhash, right_color = split_hash(right)
    return (
        hamming_distance(left_color, right_color) <= color_threshold
        and hamming_distance(left_dhash, right_dhash) <= dhash_threshold
    )


def hamming_distance(left: str, right: str) -> int:
    """Hamming distance between two equal-length binary hashes."""

    if len(left) != len(right):
        raise ValueError("dhash strings must have equal length")
    return sum(a != b for a, b in zip(left, right))


@dataclass(frozen=True)
class FrameCluster:
    representative_sha256: str
    member_sha256s: tuple[str, ...]


def cluster_frames(
    hashes: dict[str, str],
    *,
    dhash_threshold: int = 12,
    color_threshold: int = 2,
) -> list[FrameCluster]:
    """Greedily cluster frame sha256s by perceptual-hash distance.

    The first seen frame becomes a cluster representative; later frames
    join the first cluster whose representative is a near duplicate.
    """

    if dhash_threshold < 0 or color_threshold < 0:
        raise ValueError("thresholds must be non-negative")
    clusters: list[FrameCluster] = []
    for sha256, digest in hashes.items():
        for index, cluster in enumerate(clusters):
            if near_duplicates(
                digest,
                hashes[cluster.representative_sha256],
                dhash_threshold=dhash_threshold,
                color_threshold=color_threshold,
            ):
                clusters[index] = FrameCluster(
                    representative_sha256=cluster.representative_sha256,
                    member_sha256s=(*cluster.member_sha256s, sha256),
                )
                break
        else:
            clusters.append(FrameCluster(representative_sha256=sha256, member_sha256s=(sha256,)))
    return clusters


def unique_frame_sha256s(
    hashes: dict[str, str],
    *,
    dhash_threshold: int = 12,
    color_threshold: int = 2,
) -> list[str]:
    """Return representative sha256s in first-seen order after dedup."""

    return [
        cluster.representative_sha256
        for cluster in cluster_frames(
            hashes,
            dhash_threshold=dhash_threshold,
            color_threshold=color_threshold,
        )
    ]


def representative_for(
    hashes: dict[str, str],
    clusters: Sequence[FrameCluster],
    sha256: str,
    *,
    dhash_threshold: int = 12,
    color_threshold: int = 2,
) -> str:
    """Return the representative sha256 of the cluster containing ``sha256``."""

    digest = hashes[sha256]
    for cluster in clusters:
        if near_duplicates(
            digest,
            hashes[cluster.representative_sha256],
            dhash_threshold=dhash_threshold,
            color_threshold=color_threshold,
        ):
            return cluster.representative_sha256
    return sha256
