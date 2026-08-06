"""Batch-0 FFmpeg feasibility prototype (AGENTS.md 18.2 / 19.2)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import TimelineStrategy
from anime_remix.domain.models import (
    RenderProfile,
    ShotRequirement,
    TimelineItem,
)


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    assert binary, "ffmpeg must be installed for media tests"
    return binary


def _ffprobe() -> str:
    binary = shutil.which("ffprobe")
    assert binary, "ffprobe must be installed for media tests"
    return binary


def _probe(path: Path, *, count_frames: bool = True) -> dict:
    args = [_ffprobe(), "-v", "error", "-print_format", "json"]
    if count_frames:
        args.append("-count_frames")
    args.extend(["-show_streams", "-show_format", str(path)])
    completed = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return json.loads(completed.stdout)


def _encode_with_setparams(
    ffmpeg: str,
    output: Path,
    *,
    frames: int,
    vf: str,
) -> None:
    args = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=24:duration={frames / 24:.6f}",
        "-vf",
        vf,
        "-profile:v",
        "high",
        "-level:v",
        "3.1",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-bf",
        "0",
        "-g",
        "48",
        "-keyint_min",
        "48",
        "-sc_threshold",
        "0",
        "-video_track_timescale",
        "48000",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-chroma_sample_location",
        "left",
        "-an",
        "-f",
        "mp4",
        str(output),
    ]
    subprocess.run(args, check=True, shell=False)


def _requirement(shot_id: str, order: int, action: str, frames: int) -> ShotRequirement:
    return ShotRequirement(
        id=shot_id,
        order=order,
        source_text=f"{action}。",
        action=action,
        target_frames=frames,
    )


@pytest.fixture(scope="module")
def segments(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Render the three batch-0 prototype segments: 24 clip / 36 freeze / 12 placeholder."""

    root = tmp_path_factory.mktemp("batch0")
    work = root / "work"
    work.mkdir()
    ffmpeg = _ffmpeg()
    toolkit = FFmpegToolkit()

    source = root / "source_48f.mp4"
    _encode_with_setparams(
        ffmpeg,
        source,
        frames=48,
        vf="setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
    )

    clip_item = TimelineItem(
        shot_id="shot_001",
        order=1,
        requirement=_requirement("shot_001", 1, "站立", 24),
        strategy=TimelineStrategy.CLIP,
        source_asset_id="source_48f",
        source_path=str(source.name),
        source_size_bytes=source.stat().st_size,
        source_sha256="a" * 64,  # fingerprint unused by direct segment rendering
        source_in_frame=12,
        source_frame_count=24,
        target_frames=24,
        reason_code="center_trim",
        reason="center_trim",
    )
    clip = work / "segment_000.mp4"
    toolkit.render_clip(
        clip_item,
        source,
        clip,
        profile=RenderProfile(),
    )

    freeze = work / "segment_001.mp4"
    freeze_vf = (
        "trim=end_frame=1,setpts=PTS-STARTPTS,"
        "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog,"
        "fps=fps=24:start_time=0:round=near,"
        "tpad=stop_mode=clone:stop=36,trim=end_frame=36,setpts=N/(24*TB)"
    )
    _encode_with_setparams(ffmpeg, freeze, frames=1, vf=freeze_vf)

    placeholder_item = TimelineItem(
        shot_id="shot_003",
        order=3,
        requirement=_requirement("shot_003", 3, "纯黑", 12),
        strategy=TimelineStrategy.PLACEHOLDER,
        source_in_frame=0,
        source_frame_count=0,
        target_frames=12,
        reason_code="no_candidate",
        reason="no candidate",
    )
    placeholder = work / "segment_002.mp4"
    toolkit.render_placeholder(
        placeholder_item,
        placeholder,
        profile=RenderProfile(),
    )
    return {
        "root": root,
        "work": work,
        "source": source,
        "clip": clip,
        "freeze": freeze,
        "placeholder": placeholder,
    }


def test_batch0_prototype_72_frames(segments: dict[str, Path]) -> None:
    toolkit = FFmpegToolkit()
    order = [segments["clip"], segments["freeze"], segments["placeholder"]]
    expected = [24, 36, 12]
    signatures: list[dict] = []
    for path, frames in zip(order, expected):
        info = _probe(path)
        streams = info["streams"]
        videos = [s for s in streams if s.get("codec_type") == "video"]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        assert len(videos) == 1 and not audios, "segment must be video-only"
        assert int(videos[0].get("nb_read_frames") or 0) == frames
        assert videos[0]["color_primaries"] == "bt709"
        assert videos[0]["color_transfer"] == "bt709"
        assert videos[0]["color_space"] == "bt709"
        assert videos[0]["color_range"] == "tv"
        assert videos[0]["field_order"] == "progressive"
        signatures.append(
            toolkit.validate_segment(
                path,
                target_frames=frames,
                shot_id=f"segment_{frames}",
            )
        )

    # Identical encoder settings => identical H.264 extradata.
    assert len({sig["extradata_sha256"] for sig in signatures}) == 1
    # The hash must be a real extradata digest, not a placeholder for missing data.
    assert signatures[0]["extradata_sha256"] != hashlib.sha256(b"").hexdigest()
    assert all(len(sig["extradata_sha256"]) == 64 for sig in signatures)

    # Concat signature must never include length/rate-dependent fields.
    for excluded in (
        "avg_frame_rate",
        "duration",
        "nb_frames",
        "nb_read_frames",
        "bit_rate",
    ):
        assert all(excluded not in sig for sig in signatures)

    # avg_frame_rate is not a compatibility field even when it differs;
    # on this FFmpeg it is 24/1 for every segment, which is recorded here.
    avg_rates = []
    for path in order:
        video = _probe(path)["streams"][0]
        avg_rates.append(video["avg_frame_rate"])
    assert len(avg_rates) == 3

    toolkit.concat_signatures_equal(signatures)
    joined = segments["work"] / "joined_video.mp4"
    toolkit.concat_video(
        order,
        joined,
        durations=[Decimal(f) / Decimal(24) for f in expected],
    )
    joined_info = _probe(joined)
    assert int(joined_info["streams"][0]["nb_read_frames"]) == 72

    total_frames = sum(expected)
    total_samples = total_frames * 48000 // 24
    final = segments["work"] / "final.mp4"
    toolkit.mux_final(
        joined,
        final,
        total_samples=total_samples,
        profile=RenderProfile(),
    )
    toolkit.validate_final(final, total_frames=72, profile=RenderProfile())

    info = _probe(final)
    video = info["streams"][0]
    audio = info["streams"][1]
    assert video["codec_name"] == "h264"
    assert int(video["nb_read_frames"]) == 72
    assert Decimal(video["start_time"]) == 0
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 48000
    assert int(audio["channels"]) == 2
    assert Decimal(audio["start_time"]) == 0
    assert abs(Decimal(audio["duration"]) - Decimal(3)) <= Decimal("0.042")
    assert abs(Decimal(info["format"]["duration"]) - Decimal(3)) <= Decimal("0.25")
    assert toolkit.max_volume(final) <= Decimal(-90)


def test_setparams_writes_complete_bt709_labels(
    segments: dict[str, Path],
) -> None:
    """Encoder tags alone omit primaries/transfer; setparams completes them."""

    ffmpeg = _ffmpeg()
    work = segments["work"]
    without = work / "without_setparams.mp4"
    with_ = work / "with_setparams.mp4"

    # Product-like encoder tags but NO setparams filter.
    _encode_with_setparams(
        ffmpeg,
        without,
        frames=24,
        vf="scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
        "fps=fps=24:start_time=0:round=near,trim=end_frame=24,setpts=N/(24*TB)",
    )
    _encode_with_setparams(
        ffmpeg,
        with_,
        frames=24,
        vf="scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog,"
        "fps=fps=24:start_time=0:round=near,trim=end_frame=24,setpts=N/(24*TB)",
    )

    without_video = _probe(without)["streams"][0]
    with_video = _probe(with_)["streams"][0]
    # Reproduced on FFmpeg 9.0 (gyan.dev): primaries/transfer are absent
    # without setparams even when encoder color tags are passed.
    assert without_video.get("color_primaries") is None
    assert without_video.get("color_transfer") is None
    assert with_video["color_primaries"] == "bt709"
    assert with_video["color_transfer"] == "bt709"
    assert with_video["color_space"] == "bt709"
    assert with_video["color_range"] == "tv"
    assert with_video["field_order"] == "progressive"


def test_avg_frame_rate_never_breaks_concat_equality(
    segments: dict[str, Path],
) -> None:
    """A signature copy with a different avg_frame_rate still compares equal."""

    toolkit = FFmpegToolkit()
    signature = toolkit.validate_segment(
        segments["clip"],
        target_frames=24,
        shot_id="clip",
    )
    modified = dict(signature)
    modified["avg_frame_rate"] = "999/1"
    toolkit.concat_signatures_equal([signature, modified])
