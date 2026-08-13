"""Technical validator for the FinalKeyframe (deterministic)."""

from __future__ import annotations

from PIL import Image


def validate_final_keyframe(
    image: Image.Image, *, canvas_w: int, canvas_h: int
) -> tuple[bool, list[dict]]:
    checks: list[dict] = []
    if image.size != (canvas_w, canvas_h):
        checks.append(
            {
                "check_id": "canvas_size",
                "passed": False,
                "detail": f"expected {canvas_w}x{canvas_h}, got {image.size}",
            }
        )
        return False, checks
    checks.append({"check_id": "canvas_size", "passed": True})
    grey = image.convert("L")
    low, high = grey.getextrema()
    if low == high:
        checks.append(
            {
                "check_id": "not_blank",
                "passed": False,
                "detail": "final keyframe is a single flat colour",
            }
        )
        return False, checks
    checks.append({"check_id": "not_blank", "passed": True})
    return True, checks
