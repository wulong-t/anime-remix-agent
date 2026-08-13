"""Unit tests for perceptual-hash frame deduplication."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from anime_remix.services.episode_assets.deduper import (
    cluster_frames,
    color_signature,
    dhash,
    frame_hash,
    hamming_distance,
    representative_for,
    unique_frame_sha256s,
)


def _png(colour: tuple[int, int, int], *, size: tuple[int, int] = (64, 48)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, colour).save(output, format="PNG")
    return output.getvalue()


def _textured(
    *,
    background: tuple[int, int, int],
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
) -> bytes:
    output = BytesIO()
    image = Image.new("RGB", (64, 48), background)
    ImageDraw.Draw(image).rectangle(box, fill=fill)
    image.save(output, format="PNG")
    return output.getvalue()


def _hash_of(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        return dhash(image)


def test_dhash_identical_images_are_equal_and_different_images_differ() -> None:
    first = _hash_of(
        _textured(background=(10, 20, 30), box=(8, 8, 40, 32), fill=(200, 40, 90))
    )
    second = _hash_of(
        _textured(background=(10, 20, 30), box=(8, 8, 40, 32), fill=(200, 40, 90))
    )
    third = _hash_of(
        _textured(background=(90, 10, 10), box=(40, 20, 60, 44), fill=(30, 200, 40))
    )
    assert first == second
    assert first != third
    assert len(first) == 64
    assert hamming_distance(first, second) == 0
    assert hamming_distance(first, third) > 10


def test_cluster_frames_groups_near_duplicates() -> None:
    with Image.open(
        BytesIO(_textured(background=(30, 40, 50), box=(8, 8, 40, 32), fill=(200, 40, 90)))
    ) as image:
        frame_a = frame_hash(image)
    with Image.open(
        BytesIO(_textured(background=(32, 42, 52), box=(9, 9, 41, 33), fill=(200, 40, 90)))
    ) as image:
        frame_b = frame_hash(image)
    with Image.open(
        BytesIO(_textured(background=(220, 120, 60), box=(4, 30, 60, 44), fill=(10, 10, 200)))
    ) as image:
        frame_c = frame_hash(image)
    hashes = {
        "a": frame_a,
        "a-copy": frame_a,
        "b": frame_b,
        "c": frame_c,
    }
    clusters = cluster_frames(hashes)
    assert len(clusters) == 2
    assert unique_frame_sha256s(hashes) == ["a", "c"]
    assert representative_for(hashes, clusters, "b") == "a"
    assert representative_for(hashes, clusters, "c") == "c"


def test_frame_hash_distinguishes_flat_colors() -> None:
    red = _png((210, 30, 30))
    green = _png((30, 180, 40))
    blue = _png((30, 50, 200))
    with Image.open(BytesIO(red)) as image:
        red_hash = frame_hash(image)
    with Image.open(BytesIO(green)) as image:
        green_hash = frame_hash(image)
    with Image.open(BytesIO(blue)) as image:
        blue_hash = frame_hash(image)
    assert hamming_distance(red_hash, green_hash) > 5
    assert hamming_distance(red_hash, blue_hash) > 5
    assert hamming_distance(green_hash, blue_hash) > 5


def test_color_signature_is_deterministic() -> None:
    first = _png((120, 130, 140))
    second = _png((120, 130, 140))
    with Image.open(BytesIO(first)) as image:
        sig_a = color_signature(image)
    with Image.open(BytesIO(second)) as image:
        sig_b = color_signature(image)
    assert sig_a == sig_b
    assert len(sig_a) == 4 * 4 * 6
