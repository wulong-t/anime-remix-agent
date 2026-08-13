"""``geometry_extract`` stage: CharacterLayer -> CharacterGeometry."""

from __future__ import annotations

from PIL import Image

from anime_remix.services.execution.imaging import alpha_bbox


def extract_character_geometry(layer: Image.Image) -> dict:
    """Derive alpha bbox and basic anchors from a character layer."""

    rgba = layer.convert("RGBA")
    alpha = rgba.getchannel("A")
    box = alpha_bbox(alpha)
    if box == (-1, -1, -1, -1):
        raise ValueError("character layer has no opaque pixels")
    width, height = rgba.size
    x0, y0, x1, y1 = box
    nx0, ny0, nx1, ny1 = (
        x0 / width,
        y0 / height,
        x1 / width,
        y1 / height,
    )
    center_x = (nx0 + nx1) / 2
    center_y = (ny0 + ny1) / 2
    return {
        "source_size": [width, height],
        "alpha_bbox": [nx0, ny0, nx1, ny1],
        "anchors": {
            "center": [center_x, center_y],
            "top_center": [center_x, ny0],
            "bottom_center": [center_x, ny1],
            "left_center": [nx0, center_y],
            "right_center": [nx1, center_y],
        },
    }
