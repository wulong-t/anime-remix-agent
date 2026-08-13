"""Minimal Pillow helpers used by the Image-First execution stages."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def load_rgba(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_png(image: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(Path(path), "PNG")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop/resize an image to exactly ``target_w x target_h`` (cover)."""

    if image.size == (target_w, target_h):
        return image.copy()
    source_w, source_h = image.size
    source_ratio = source_w / source_h
    target_ratio = target_w / target_h
    if source_ratio > target_ratio:
        crop_w = round(source_h * target_ratio)
        x0 = (source_w - crop_w) // 2
        cropped = image.crop((x0, 0, x0 + crop_w, source_h))
    else:
        crop_h = round(source_w / target_ratio)
        y0 = (source_h - crop_h) // 2
        cropped = image.crop((0, y0, source_w, y0 + crop_h))
    return cropped.resize((target_w, target_h), Image.Resampling.BILINEAR)


def crop_source_rect(
    image: Image.Image, source_rect: list[float]
) -> Image.Image:
    """Crop a normalized source rect ``[x0, y0, x1, y1]`` from an image."""

    width, height = image.size
    x0, y0, x1, y1 = (float(v) for v in source_rect)
    box = (
        round(x0 * width),
        round(y0 * height),
        round(x1 * width),
        round(y1 * height),
    )
    return image.crop(box)


def resize_uniform(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )


def polygon_l_mask(size: tuple[int, int], polygon: list[list[float]]) -> Image.Image:
    """White-on-black polygon mask in normalized canvas coordinates."""

    mask = Image.new("L", size, 0)
    width, height = size
    points = [(round(x * width), round(y * height)) for x, y in polygon]
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return mask


def max_channel_distance(
    image: Image.Image, border: tuple[int, int, int]
) -> Image.Image:
    """Per-pixel maximum absolute channel difference from ``border``."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    r0, g0, b0 = border
    r, g, b = rgb.split()
    dr = ImageChops.difference(r, Image.new("L", (width, height), r0))
    dg = ImageChops.difference(g, Image.new("L", (width, height), g0))
    db = ImageChops.difference(b, Image.new("L", (width, height), b0))
    return ImageChops.lighter(ImageChops.lighter(dr, dg), db)


def dilate_edges(mask: Image.Image, radius_px: int) -> Image.Image:
    if radius_px <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.MaxFilter(radius_px * 2 + 1))


def alpha_bbox(alpha: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of non-zero alpha pixels; (-1,-1,-1,-1) when empty."""

    box = alpha.getbbox()
    if box is None:
        return (-1, -1, -1, -1)
    return tuple(int(v) for v in box)  # type: ignore[return-value]


def count_nonzero(mask: Image.Image) -> int:
    """Count non-zero pixels of an L-mode mask."""

    return mask.point(lambda v: 255 if v > 0 else 0).histogram()[255]


def paste_with_mask(
    canvas: Image.Image,
    layer: Image.Image,
    translate: tuple[int, int],
    mask: Image.Image,
) -> None:
    canvas.paste(layer, (round(translate[0]), round(translate[1])), mask)
