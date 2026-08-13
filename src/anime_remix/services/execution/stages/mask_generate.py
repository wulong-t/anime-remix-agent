"""``mask_generate`` stage: LayoutPlan -> CompositeMask + InpaintMask."""

from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw

from anime_remix.errors import InputValidationError
from anime_remix.services.execution.imaging import (
    count_nonzero,
    dilate_edges,
    polygon_l_mask,
    resize_uniform,
)

MAX_MASK_AREA_RATIO = 0.30


def generate_masks(
    *,
    character_layer: Image.Image,
    layout_plan: dict,
    canvas_w: int,
    canvas_h: int,
    max_mask_area_ratio: float = MAX_MASK_AREA_RATIO,
) -> tuple[Image.Image, Image.Image]:
    """Build composite mask (alpha minus occlusions) and inpaint mask."""

    character = next(
        layer
        for layer in layout_plan["layers"]
        if layer["layer_id"] == "character"
    )
    scale = character["transform"]["scale"]
    tx, ty = character["transform"]["translate"]
    alpha = character_layer.convert("RGBA").getchannel("A")
    scaled_alpha = resize_uniform(alpha, scale)
    composite_mask = Image.new("L", (canvas_w, canvas_h), 0)
    composite_mask.paste(
        scaled_alpha,
        (round(tx * canvas_w), round(ty * canvas_h)),
    )
    for occlusion in layout_plan["occlusions"]:
        polygon = polygon_l_mask(
            (canvas_w, canvas_h), occlusion["polygon"]
        )
        composite_mask = ImageChops.subtract(composite_mask, polygon)

    dilation_px = int(
        layout_plan["mask_generation"]["parameters"].get(
            "edge_dilation_px", 12
        )
    )
    dilated = dilate_edges(composite_mask, dilation_px)
    inpaint_mask = ImageChops.subtract(dilated, composite_mask)
    for contact in layout_plan["contacts"]:
        cx, cy = contact["target_anchor"]
        radius = max(dilation_px, 8)
        draw = ImageDraw.Draw(inpaint_mask)
        draw.ellipse(
            (
                round(cx * canvas_w) - radius,
                round(cy * canvas_h) - radius,
                round(cx * canvas_w) + radius,
                round(cy * canvas_h) + radius,
            ),
            fill=255,
        )

    area = count_nonzero(inpaint_mask) / (canvas_w * canvas_h)
    if area > max_mask_area_ratio:
        raise InputValidationError(
            f"inpaint mask area {area:.3f} exceeds policy maximum "
            f"{max_mask_area_ratio}"
        )
    return composite_mask, inpaint_mask
