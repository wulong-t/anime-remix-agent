"""Technical validator for the CharacterLayer (Gate 1, deterministic)."""

from __future__ import annotations

from PIL import Image

from anime_remix.services.execution.imaging import alpha_bbox, count_nonzero

MIN_ALPHA_RATIO = 0.05
MAX_ALPHA_RATIO = 0.95


def validate_character_layer(
    layer: Image.Image, *, canvas_w: int, canvas_h: int
) -> tuple[bool, list[dict]]:
    """Return ``(valid, checks)`` for one character layer."""

    checks: list[dict] = []
    rgba = layer.convert("RGBA")
    alpha = rgba.getchannel("A")
    box = alpha_bbox(alpha)
    if box == (-1, -1, -1, -1):
        checks.append(
            {
                "check_id": "alpha_non_empty",
                "passed": False,
                "detail": "character layer has no opaque pixels",
            }
        )
        return False, checks
    checks.append({"check_id": "alpha_non_empty", "passed": True})

    ratio = count_nonzero(alpha) / (canvas_w * canvas_h)
    if not (MIN_ALPHA_RATIO <= ratio <= MAX_ALPHA_RATIO):
        checks.append(
            {
                "check_id": "alpha_area_ratio",
                "passed": False,
                "detail": f"alpha ratio {ratio:.3f} outside "
                f"[{MIN_ALPHA_RATIO}, {MAX_ALPHA_RATIO}]",
            }
        )
        return False, checks
    checks.append({"check_id": "alpha_area_ratio", "passed": True})

    x0, y0, x1, y1 = box
    if x0 <= 0 or y0 <= 0 or x1 >= layer.width - 1 or y1 >= layer.height - 1:
        checks.append(
            {
                "check_id": "subject_not_cropped",
                "passed": False,
                "detail": "subject touches the image border",
            }
        )
        return False, checks
    checks.append({"check_id": "subject_not_cropped", "passed": True})
    return True, checks
