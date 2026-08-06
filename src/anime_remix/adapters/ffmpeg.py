"""FFmpeg/ffprobe adapter. All external commands live here."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from anime_remix.domain.models import (
    ClipAsset,
    ProbedClip,
    RenderProfile,
    TimelineItem,
)
from anime_remix.errors import (
    AnimeRemixError,
    EnvironmentCapabilityError,
    MediaProbeError,
    OutputValidationError,
    RenderError,
    UnsupportedMediaError,
)

PROBE_TIMEOUT = 120
ENCODE_TIMEOUT = 300
CONCAT_TIMEOUT = 300
MUX_TIMEOUT = 300
VOLUME_TIMEOUT = 120

REQUIRED_FILTERS = {
    "scale",
    "pad",
    "setsar",
    "format",
    "setparams",
    "fps",
    "trim",
    "setpts",
    "atrim",
    "asetpts",
    "color",
    "anullsrc",
}


def _fraction(value: str | None) -> Fraction | None:
    if not value or "/" not in value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError):
        return None


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except ArithmeticError:
        return None


def _summarize_stderr(stderr: str, *, limit: int = 4000) -> str:
    lines = [line[:300] for line in stderr.strip().splitlines()[-40:]]
    return "\n".join(lines)[-limit:]


class FFmpegToolkit:
    """Thin, strict wrapper around ffmpeg/ffprobe."""

    def __init__(
        self,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg")
        self.ffprobe = ffprobe or shutil.which("ffprobe")

    def _run(
        self,
        args: list[str],
        *,
        timeout: int,
        error_cls: type[AnimeRemixError],
        stage: str,
    ) -> subprocess.CompletedProcess[str]:
        if not args:
            raise error_cls(f"empty command for {stage}")
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise EnvironmentCapabilityError(
                f"{stage}: binary not found",
                actual=args[0],
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise error_cls(
                f"{stage} timed out after {timeout}s",
                actual=exc.cmd,
            ) from exc
        except OSError as exc:
            raise error_cls(f"{stage} failed to start", actual=str(exc)) from exc
        if completed.returncode != 0:
            raise error_cls(
                f"{stage} failed with exit code {completed.returncode}",
                actual=_summarize_stderr(completed.stderr),
            )
        return completed

    def ffmpeg_version(self) -> str:
        completed = self._run(
            [self.ffmpeg or "ffmpeg", "-version"],
            timeout=PROBE_TIMEOUT,
            error_cls=EnvironmentCapabilityError,
            stage="ffmpeg_version",
        )
        return completed.stdout.splitlines()[0].strip() if completed.stdout else ""

    def ffprobe_version(self) -> str:
        completed = self._run(
            [self.ffprobe or "ffprobe", "-version"],
            timeout=PROBE_TIMEOUT,
            error_cls=EnvironmentCapabilityError,
            stage="ffprobe_version",
        )
        return completed.stdout.splitlines()[0].strip() if completed.stdout else ""

    def check_capabilities(self) -> None:
        if not self.ffmpeg:
            raise EnvironmentCapabilityError("ffmpeg not found on PATH")
        if not self.ffprobe:
            raise EnvironmentCapabilityError("ffprobe not found on PATH")
        missing: list[str] = []
        encoders = self._run(
            [self.ffmpeg, "-hide_banner", "-encoders"],
            timeout=PROBE_TIMEOUT,
            error_cls=EnvironmentCapabilityError,
            stage="capabilities",
        ).stdout
        for encoder in ("libx264", "aac"):
            if not re.search(rf"(^|\s){re.escape(encoder)}(\s|$)", encoders):
                missing.append(f"encoder:{encoder}")
        filters = self._run(
            [self.ffmpeg, "-hide_banner", "-filters"],
            timeout=PROBE_TIMEOUT,
            error_cls=EnvironmentCapabilityError,
            stage="capabilities",
        ).stdout
        for filter_name in sorted(REQUIRED_FILTERS):
            if not re.search(
                rf"(^|\s){re.escape(filter_name)}\s",
                filters,
            ):
                missing.append(f"filter:{filter_name}")
        demuxers = self._run(
            [self.ffmpeg, "-hide_banner", "-demuxers"],
            timeout=PROBE_TIMEOUT,
            error_cls=EnvironmentCapabilityError,
            stage="capabilities",
        ).stdout
        if not re.search(r"(^|\s)concat(\s|$)", demuxers):
            missing.append("demuxer:concat")
        if missing:
            raise EnvironmentCapabilityError(
                "missing FFmpeg capabilities",
                actual=", ".join(missing),
            )

    def _probe(
        self,
        path: Path,
        *,
        count_frames: bool = False,
        select: str | None = None,
        data_hash: bool = False,
    ) -> dict[str, Any]:
        args = [
            self.ffprobe or "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
        ]
        if count_frames:
            args.append("-count_frames")
        if select:
            args.extend(["-select_streams", select])
        if data_hash:
            args.extend(["-show_data_hash", "sha256"])
        args.extend(["-show_streams", "-show_format", str(path)])
        completed = self._run(
            args,
            timeout=PROBE_TIMEOUT,
            error_cls=MediaProbeError,
            stage="probe",
        )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaProbeError(
                "ffprobe returned invalid JSON",
                actual=_summarize_stderr(completed.stdout),
            ) from exc

    def _has_rotation(self, stream: dict[str, Any]) -> bool:
        tags = stream.get("tags") or {}
        if tags.get("rotate") not in (None, "0"):
            return True
        for side in stream.get("side_data_list") or []:
            if side.get("rotation") not in (None, 0):
                return True
            if side.get("side_data_type") == "Display Matrix" and side.get(
                "rotation"
            ):
                return True
        return False

    def _validate_source_stream(
        self,
        stream: dict[str, Any],
        *,
        fallback_duration: Decimal | None = None,
    ) -> list[str]:
        errors: list[str] = []
        codec = stream.get("codec_name")
        if codec != "h264":
            errors.append(f"codec={codec or 'missing'}")
        if stream.get("pix_fmt") != "yuv420p":
            errors.append(f"pix_fmt={stream.get('pix_fmt') or 'missing'}")
        width = stream.get("width")
        height = stream.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            errors.append("width/height missing")
        else:
            if width <= 0 or height <= 0 or width % 2 or height % 2:
                errors.append(f"dimensions={width}x{height}")
            if width > 3840 or height > 2160:
                errors.append(f"dimensions too large={width}x{height}")
        r_fps = _fraction(stream.get("r_frame_rate"))
        avg_fps = _fraction(stream.get("avg_frame_rate"))
        if r_fps != Fraction(24, 1):
            errors.append(f"r_frame_rate={stream.get('r_frame_rate') or 'missing'}")
        if avg_fps != Fraction(24, 1):
            errors.append(f"avg_frame_rate={stream.get('avg_frame_rate') or 'missing'}")
        nb_frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
        try:
            nb_frames = int(nb_frames_raw)
        except (TypeError, ValueError):
            nb_frames = 0
        if nb_frames <= 0:
            errors.append(f"nb_frames={nb_frames_raw!r}")
        duration = _decimal(stream.get("duration")) or fallback_duration
        if duration is None or duration <= 0 or not duration.is_finite():
            errors.append(f"duration={stream.get('duration') or 'missing'}")
        else:
            if abs(duration * 24 - nb_frames) > 1:
                errors.append(f"duration/frames mismatch={duration}")
            if duration > 120:
                errors.append(f"duration_seconds={duration}")
        field_order = stream.get("field_order")
        if field_order not in ("progressive", "unknown", None):
            errors.append(f"field_order={field_order}")
        sar = stream.get("sample_aspect_ratio")
        if sar != "1:1":
            errors.append(f"SAR={sar or 'missing'}")
        if self._has_rotation(stream):
            errors.append("rotation")
        color_space = stream.get("color_space")
        color_primaries = stream.get("color_primaries")
        color_transfer = stream.get("color_transfer")
        color_range = stream.get("color_range")
        chroma = stream.get("chroma_location")
        if color_space not in (None, "bt709"):
            errors.append(f"color_space={color_space}")
        if color_primaries not in (None, "bt709"):
            errors.append(f"color_primaries={color_primaries}")
        if color_transfer not in (None, "bt709"):
            errors.append(f"color_transfer={color_transfer}")
        if color_range not in (None, "tv"):
            errors.append(f"color_range={color_range}")
        if chroma not in (None, "left", "unknown"):
            errors.append(f"chroma_location={chroma}")
        return errors

    def probe_asset(self, path: Path, asset: ClipAsset) -> ProbedClip:
        info = self._probe(path, count_frames=True)
        streams = info.get("streams") or []
        format_duration = _decimal((info.get("format") or {}).get("duration"))
        videos = [s for s in streams if s.get("codec_type") == "video"]
        if len(videos) != 1:
            raise UnsupportedMediaError(
                "source must have exactly one video stream",
                asset_id=asset.id,
                actual=len(videos),
            )
        stream = videos[0]
        errors = self._validate_source_stream(
            stream,
            fallback_duration=format_duration,
        )
        if errors:
            raise UnsupportedMediaError(
                "source violates MVR media contract",
                asset_id=asset.id,
                actual="; ".join(errors),
            )
        nb_frames = int(
            stream.get("nb_read_frames") or stream.get("nb_frames") or 0
        )
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise MediaProbeError(
                "cannot stat source",
                asset_id=asset.id,
                actual=str(exc),
            ) from exc
        if size_bytes > 1024 * 1024 * 1024:
            raise UnsupportedMediaError(
                "source exceeds 1 GiB",
                asset_id=asset.id,
                actual=size_bytes,
            )
        duration = _decimal(stream.get("duration")) or format_duration or Decimal(0)
        assumed = any(
            stream.get(key) is None
            for key in (
                "color_space",
                "color_primaries",
                "color_transfer",
                "color_range",
            )
        )
        return ProbedClip(
            asset=asset,
            resolved_path=path.resolve(),
            size_bytes=size_bytes,
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps_num=24,
            fps_den=1,
            nb_frames=nb_frames,
            duration_seconds=duration,
            assumed_color_metadata=assumed,
        )

    def _encode_args(
        self,
        profile: RenderProfile,
        output: Path,
    ) -> list[str]:
        return [
            "-an",
            "-c:v",
            profile.video_codec,
            "-profile:v",
            "high",
            "-level:v",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            profile.video_preset,
            "-crf",
            str(profile.video_crf),
            "-bf",
            str(profile.max_b_frames),
            "-g",
            str(profile.gop_frames),
            "-keyint_min",
            str(profile.gop_frames),
            "-sc_threshold",
            "0",
            "-video_track_timescale",
            str(profile.video_track_timescale),
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
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-f",
            "mp4",
            str(output),
        ]

    def render_clip(
        self,
        item: TimelineItem,
        source: Path,
        output: Path,
        *,
        profile: RenderProfile,
    ) -> None:
        start = item.source_in_frame
        end = start + item.source_frame_count
        filter_graph = ",".join(
            [
                f"trim=start_frame={start}:end_frame={end}",
                "setpts=PTS-STARTPTS",
                "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
                "setsar=1",
                "format=yuv420p",
                "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
                "fps=fps=24:start_time=0:round=near",
                f"trim=end_frame={item.target_frames}",
                "setpts=N/(24*TB)",
            ]
        )
        args = [
            self.ffmpeg or "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            filter_graph,
        ]
        args.extend(self._encode_args(profile, output))
        self._run(
            args,
            timeout=ENCODE_TIMEOUT,
            error_cls=RenderError,
            stage="render_clip",
        )

    def render_placeholder(
        self,
        item: TimelineItem,
        output: Path,
        *,
        profile: RenderProfile,
    ) -> None:
        filter_graph = ",".join(
            [
                f"trim=end_frame={item.target_frames}",
                "setpts=N/(24*TB)",
                "format=yuv420p",
                "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
            ]
        )
        args = [
            self.ffmpeg or "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:r=24",
            "-map",
            "0:v:0",
            "-vf",
            filter_graph,
        ]
        args.extend(self._encode_args(profile, output))
        self._run(
            args,
            timeout=ENCODE_TIMEOUT,
            error_cls=RenderError,
            stage="render_placeholder",
        )

    def _stream_signature(self, stream: dict[str, Any]) -> dict[str, Any]:
        extradata_hash = stream.get("extradata_hash") or ""
        digest = extradata_hash.split(":", 1)[1].strip().lower() if ":" in extradata_hash else ""
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RenderError(
                "segment has invalid/missing extradata_hash",
                actual=extradata_hash,
            )
        return {
            "codec_name": stream.get("codec_name"),
            "profile": stream.get("profile"),
            "level": stream.get("level"),
            "codec_tag_string": stream.get("codec_tag_string"),
            "extradata_sha256": digest,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "sample_aspect_ratio": stream.get("sample_aspect_ratio"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "time_base": stream.get("time_base"),
            "color_range": stream.get("color_range"),
            "color_space": stream.get("color_space"),
            "color_transfer": stream.get("color_transfer"),
            "color_primaries": stream.get("color_primaries"),
            "field_order": stream.get("field_order"),
            "chroma_location": stream.get("chroma_location"),
        }

    def validate_segment(
        self,
        path: Path,
        *,
        target_frames: int,
        shot_id: str,
    ) -> dict[str, Any]:
        info = self._probe(path, count_frames=True, data_hash=True)
        streams = info.get("streams") or []
        videos = [s for s in streams if s.get("codec_type") == "video"]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        if len(videos) != 1 or audios:
            raise RenderError(
                "segment must have exactly one video stream and no audio",
                shot_id=shot_id,
                actual=f"v={len(videos)} a={len(audios)}",
            )
        stream = videos[0]
        problems: list[str] = []
        if stream.get("codec_name") != "h264":
            problems.append(f"codec={stream.get('codec_name')}")
        if stream.get("profile") != "High":
            problems.append(f"profile={stream.get('profile')}")
        if stream.get("pix_fmt") != "yuv420p":
            problems.append(f"pix_fmt={stream.get('pix_fmt')}")
        if (stream.get("width"), stream.get("height")) != (1280, 720):
            problems.append(
                f"size={stream.get('width')}x{stream.get('height')}"
            )
        if stream.get("sample_aspect_ratio") != "1:1":
            problems.append(f"SAR={stream.get('sample_aspect_ratio')}")
        if _fraction(stream.get("r_frame_rate")) != Fraction(24, 1):
            problems.append(f"r_frame_rate={stream.get('r_frame_rate')}")
        if stream.get("time_base") != "1/48000":
            problems.append(f"time_base={stream.get('time_base')}")
        if stream.get("color_range") != "tv":
            problems.append(f"color_range={stream.get('color_range')}")
        if stream.get("color_space") != "bt709":
            problems.append(f"color_space={stream.get('color_space')}")
        if stream.get("color_transfer") != "bt709":
            problems.append(f"color_transfer={stream.get('color_transfer')}")
        if stream.get("color_primaries") != "bt709":
            problems.append(f"color_primaries={stream.get('color_primaries')}")
        if stream.get("field_order") != "progressive":
            problems.append(f"field_order={stream.get('field_order')}")
        if stream.get("chroma_location") != "left":
            problems.append(f"chroma_location={stream.get('chroma_location')}")
        start = _decimal(stream.get("start_time"))
        if start is None or start < 0 or start > Decimal(1) / Decimal(48000):
            problems.append(f"start_time={stream.get('start_time')}")
        nb_read_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
        try:
            nb_read = int(nb_read_raw)
        except (TypeError, ValueError):
            nb_read = -1
        if nb_read != target_frames:
            problems.append(
                f"nb_read_frames={nb_read_raw!r} expected={target_frames}"
            )
        if problems:
            raise RenderError(
                "segment validation failed",
                shot_id=shot_id,
                actual="; ".join(problems),
            )
        return self._stream_signature(stream)

    def concat_signatures_equal(
        self,
        signatures: list[dict[str, Any]],
    ) -> None:
        if not signatures:
            raise RenderError("no segment signatures to compare")
        first = signatures[0]
        for index, signature in enumerate(signatures[1:], start=2):
            differing = [
                key
                for key in first
                if first.get(key) != signature.get(key)
            ]
            if differing:
                raise RenderError(
                    "segment concat signature mismatch",
                    actual=f"segment {index}: {', '.join(differing)}",
                )

    def concat_video(
        self,
        segments: list[Path],
        output: Path,
        *,
        durations: list[Decimal] | None = None,
    ) -> None:
        if not segments:
            raise RenderError("concat requires at least one segment")
        if durations is not None and len(durations) != len(segments):
            raise RenderError(
                "concat durations must match segment count",
                actual=(len(segments), len(durations)),
            )
        list_path = output.parent / f".{output.stem}.concat.txt"
        try:
            with list_path.open("w", encoding="utf-8", newline="\n") as handle:
                for index, segment in enumerate(segments):
                    escaped = str(segment).replace("'", "'\\''")
                    handle.write(f"file '{escaped}'\n")
                    if durations is not None:
                        # Frame-derived duration (not probed stream duration) so
                        # the concat demuxer translates timestamps by exact
                        # target_frames / 24 for every segment.
                        handle.write(f"duration {float(durations[index]):.12f}\n")
            args = [
                self.ffmpeg or "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-f",
                "mp4",
                str(output),
            ]
            self._run(
                args,
                timeout=CONCAT_TIMEOUT,
                error_cls=RenderError,
                stage="concat",
            )
        finally:
            try:
                list_path.unlink()
            except OSError:
                pass

    def mux_final(
        self,
        joined_video: Path,
        output: Path,
        *,
        total_samples: int,
        profile: RenderProfile,
    ) -> None:
        args = [
            self.ffmpeg or "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(joined_video),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            f"atrim=end_sample={total_samples},asetpts=PTS-STARTPTS",
            "-c:a",
            profile.audio_codec,
            "-b:a",
            f"{profile.audio_bitrate_kbps}k",
            "-ar",
            str(profile.audio_sample_rate),
            "-ac",
            str(profile.audio_channels),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-f",
            "mp4",
            str(output),
        ]
        self._run(
            args,
            timeout=MUX_TIMEOUT,
            error_cls=RenderError,
            stage="mux_final",
        )

    def max_volume(self, path: Path) -> Decimal:
        completed = self._run(
            [
                self.ffmpeg or "ffmpeg",
                "-v",
                "info",
                "-i",
                str(path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            timeout=VOLUME_TIMEOUT,
            error_cls=OutputValidationError,
            stage="volumedetect",
        )
        match = re.search(
            r"max_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB",
            completed.stderr,
        )
        if not match:
            raise OutputValidationError(
                "volumedetect produced no max_volume",
                actual=_summarize_stderr(completed.stderr),
            )
        return Decimal(match.group(1))

    def validate_final(
        self,
        path: Path,
        *,
        total_frames: int,
        profile: RenderProfile,
    ) -> None:
        info = self._probe(path, count_frames=True)
        streams = info.get("streams") or []
        videos = [s for s in streams if s.get("codec_type") == "video"]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise OutputValidationError(
                "final MP4 must have exactly one video and one audio stream",
                actual=f"v={len(videos)} a={len(audios)}",
            )
        video = videos[0]
        audio = audios[0]
        problems: list[str] = []
        if video.get("codec_name") != "h264":
            problems.append(f"video codec={video.get('codec_name')}")
        if video.get("pix_fmt") != "yuv420p":
            problems.append(f"pix_fmt={video.get('pix_fmt')}")
        if (video.get("width"), video.get("height")) != (1280, 720):
            problems.append(f"size={video.get('width')}x{video.get('height')}")
        if video.get("sample_aspect_ratio") != "1:1":
            problems.append(f"video SAR={video.get('sample_aspect_ratio')}")
        if _fraction(video.get("r_frame_rate")) != Fraction(24, 1):
            problems.append(f"video fps={video.get('r_frame_rate')}")
        if video.get("color_range") != "tv":
            problems.append(f"video range={video.get('color_range')}")
        if video.get("color_space") != "bt709":
            problems.append(f"video space={video.get('color_space')}")
        if video.get("color_transfer") != "bt709":
            problems.append(f"video transfer={video.get('color_transfer')}")
        if video.get("color_primaries") != "bt709":
            problems.append(f"video primaries={video.get('color_primaries')}")
        if video.get("field_order") != "progressive":
            problems.append(f"field_order={video.get('field_order')}")
        if video.get("chroma_location") != "left":
            problems.append(f"chroma={video.get('chroma_location')}")
        nb_read_raw = video.get("nb_read_frames") or video.get("nb_frames")
        try:
            nb_read = int(nb_read_raw)
        except (TypeError, ValueError):
            nb_read = -1
        if nb_read != total_frames:
            problems.append(
                f"video nb_read_frames={nb_read_raw!r} expected={total_frames}"
            )
        video_start = _decimal(video.get("start_time"))
        if video_start is None or video_start < 0 or video_start > Decimal(1) / Decimal(48000):
            problems.append(f"video start={video.get('start_time')}")
        if audio.get("codec_name") != "aac":
            problems.append(f"audio codec={audio.get('codec_name')}")
        try:
            sample_rate = int(audio.get("sample_rate"))
        except (TypeError, ValueError):
            sample_rate = -1
        try:
            channels = int(audio.get("channels"))
        except (TypeError, ValueError):
            channels = -1
        if sample_rate != 48000:
            problems.append(f"audio sample_rate={audio.get('sample_rate')!r}")
        if channels != 2:
            problems.append(f"audio channels={audio.get('channels')!r}")
        audio_start = _decimal(audio.get("start_time"))
        if audio_start is None or audio_start < 0 or audio_start > Decimal(1) / Decimal(48000):
            problems.append(f"audio start={audio.get('start_time')}")
        expected_duration = Decimal(total_frames) / Decimal(24)
        tolerance = max(Decimal(1024) / Decimal(48000), Decimal(1) / Decimal(24))
        audio_duration = _decimal(audio.get("duration"))
        if audio_duration is None or abs(audio_duration - expected_duration) > tolerance:
            problems.append(
                f"audio duration={audio.get('duration')} expected={expected_duration}"
            )
        format_duration = _decimal((info.get("format") or {}).get("duration"))
        if format_duration is None or abs(format_duration - expected_duration) > Decimal("0.25"):
            problems.append(
                f"container duration={format_duration} expected={expected_duration}"
            )
        if problems:
            raise OutputValidationError(
                "final MP4 validation failed",
                actual="; ".join(problems),
            )
        volume = self.max_volume(path)
        if volume > Decimal(-90):
            raise OutputValidationError(
                "final audio is not silent",
                actual=f"max_volume={volume} dB",
            )
