"""``composite`` stage: deterministic layer compositing."""

from __future__ import annotations

from PIL import Image, ImageChops

from anime_remix.services.execution.imaging import (
    crop_source_rect,
    polygon_l_mask,
    resize_uniform,
)


def composite_keyframe(
    *,
    scene_image: Image.Image,
    character_layer: Image.Image,
    layout_plan: dict,
) -> Image.Image:
    """Deterministically place the character into the scene."""

    canvas_w = layout_plan["canvas"]["width"]
    canvas_h = layout_plan["canvas"]["height"]
    scene = crop_source_rect(
        scene_image, layout_plan["scene_crop"]["source_rect"]
    ).resize((canvas_w, canvas_h))
    base = scene.convert("RGBA")

    character = next(
        layer
        for layer in layout_plan["layers"]
        if layer["layer_id"] == "character"
    )
    scale = character["transform"]["scale"]
    tx, ty = character["transform"]["translate"]
    rgba = character_layer.convert("RGBA")
    scaled = resize_uniform(rgba, scale)
    scaled_w, scaled_h = scaled.size
    bx = round(tx * canvas_w)
    by = round(ty * canvas_h)
    left = max(0, bx)
    top = max(0, by)
    right = min(canvas_w, bx + scaled_w)
    bottom = min(canvas_h, by + scaled_h)
    if left >= right or top >= bottom:
        return base.convert("RGB")
    crop_box = (left - bx, top - by, right - bx, bottom - by)
    alpha = Image.new("L", (canvas_w, canvas_h), 0)
    alpha.paste(scaled.getchannel("A").crop(crop_box), (left, top))
    for occlusion in layout_plan["occlusions"]:
        polygon = polygon_l_mask((canvas_w, canvas_h), occlusion["polygon"])
        alpha = ImageChops.subtract(alpha, polygon)
    base.paste(
        scaled.convert("RGB").crop(crop_box),
        (left, top),
        alpha.crop((left, top, right, bottom)),
    )
    return base.convert("RGB")
