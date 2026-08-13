"""``character_extract`` stage: RGB CharacterCandidate -> RGBA CharacterLayer.

Round 7 freeze: extraction is an independent stage with its own producer
lineage; segmentation must never be hidden inside ``character_synthesis``.
The v1 implementation is deliberately dumb and deterministic: pixels close
to the image border colour become transparent.
"""

from __future__ import annotations

from PIL import Image

from anime_remix.services.execution.imaging import max_channel_distance


def extract_character_layer(
    candidate: Image.Image, background_tolerance: int = 24
) -> Image.Image:
    """Remove a near-uniform background and return an RGBA character layer."""

    rgba = candidate.convert("RGBA")
    border = rgba.convert("RGB").getpixel((0, 0))
    distance = max_channel_distance(rgba, border)
    alpha = distance.point(
        lambda value: 255 if value >= background_tolerance else 0
    )
    layer = rgba.copy()
    layer.putalpha(alpha)
    return layer
