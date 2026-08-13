"""Orchestrator: episode video -> sampled frames -> dedup -> catalog + manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.errors import InputValidationError
from anime_remix.json_io import dump_json_atomic, sha256_file
from anime_remix.services.episode_assets.catalog_builder import (
    asset_id_for,
    build_catalog_entry,
    build_crop_entry,
    write_episode_catalog,
)
from anime_remix.services.episode_assets.classifier import (
    EpisodeClassifier,
    FrameClassification,
)
from anime_remix.services.episode_assets.cropper import (
    EpisodeCropper,
    apply_crop,
)
from anime_remix.services.episode_assets.deduper import (
    cluster_frames,
    frame_hash,
    representative_for,
)
from anime_remix.services.episode_assets.sampler import (
    SampledFrame,
    sample_frames,
)


@dataclass(frozen=True)
class EpisodeAssetExtractResult:
    """Durable result of one episode asset extraction run."""

    run_id: str
    video_path: Path
    video_sha256: str
    video_duration_seconds: float
    extracted_frame_count: int
    unique_frame_count: int
    crop_count: int
    catalog_path: Path
    manifest_path: Path


def _default_run_id(video: Path) -> str:
    digest = sha256_file(video)[:12]
    return f"episode-assets-{video.stem[:24]}-{digest}"


def extract_episode_assets(
    *,
    video: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    title: str | None = None,
    max_frames: int = 40,
    scene_threshold: float = 0.30,
    min_gap_seconds: float = 1.0,
    classifier: EpisodeClassifier | None = None,
    cropper: EpisodeCropper | None = None,
    rights_status: str = (
        "user-owned episode frame; private experiment, no redistribution"
    ),
    toolkit: FFmpegToolkit | None = None,
) -> EpisodeAssetExtractResult:
    """Extract deduplicated reference frames and register them in a catalog."""

    video_path = Path(video)
    if not video_path.is_file():
        raise InputValidationError(f"video file does not exist: {video_path}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    asset_dir = out_dir / "assets"
    frame_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    if not run_id or not run_id.strip():
        run_id = _default_run_id(video_path)
    toolkit = toolkit or FFmpegToolkit()

    video_sha = sha256_file(video_path)
    duration = toolkit.probe_duration(video_path)
    sampled = sample_frames(
        video_path,
        frame_dir,
        max_frames=max_frames,
        scene_threshold=scene_threshold,
        min_gap_seconds=min_gap_seconds,
        toolkit=toolkit,
    )
    hashes = {frame.sha256: _frame_hash_of(frame) for frame in sampled}
    clusters = cluster_frames(hashes)
    by_sha = {frame.sha256: frame for frame in sampled}
    unique = [
        by_sha[cluster.representative_sha256]
        for cluster in clusters
        if cluster.representative_sha256 in by_sha
    ]

    classifications: dict[str, FrameClassification] = {}
    classification_errors: dict[str, str] = {}
    classification_request_ids: dict[str, str] = {}
    if classifier is not None:
        for frame in unique:
            try:
                classifications[frame.sha256] = classifier.classify(frame.path)
                request_id = getattr(classifier, "last_request_id", None)
                if request_id:
                    classification_request_ids[frame.sha256] = str(request_id)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                classification_errors[frame.sha256] = (
                    f"{type(exc).__name__}: {exc}"
                )

    entries: list[dict] = []
    manifest_frames: list[dict] = []
    frame_asset_ids: dict[str, str] = {}
    for index, frame in enumerate(unique, start=1):
        asset_id = asset_id_for(index, title=title)
        frame_asset_ids[frame.sha256] = asset_id
        target = asset_dir / f"{asset_id}.png"
        target.write_bytes(frame.path.read_bytes())
        classification = classifications.get(frame.sha256)
        rel_path = f"assets/{target.name}"
        entries.append(
            build_catalog_entry(
                asset_id=asset_id,
                rel_path=rel_path,
                frame_sha256=frame.sha256,
                timestamp_seconds=frame.timestamp_seconds,
                video_name=video_path.name,
                classification=classification,
                rights_status=rights_status,
            )
        )
    crop_records: list[dict] = []
    crop_errors: dict[str, list[str]] = {}
    if cropper is not None:
        for frame in unique:
            frame_asset_id = frame_asset_ids[frame.sha256]
            classification = classifications.get(frame.sha256)
            frame_subject = (
                classification.subject_or_scene_id
                if classification is not None
                else f"frame at t={frame.timestamp_seconds:.3f}s"
            )
            try:
                regions = cropper.crop_regions(frame.path)
                crop_request_id = getattr(cropper, "last_request_id", None)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                crop_errors.setdefault(frame.sha256, []).append(
                    f"crop_regions: {type(exc).__name__}: {exc}"
                )
                continue
            for crop_index, box in enumerate(regions.boxes, start=1):
                label = box.label
                if label == "region":
                    if (
                        classification is not None
                        and classification.asset_type == "character"
                    ):
                        label = "character"
                    else:
                        continue
                try:
                    crop_asset_id = f"{frame_asset_id}_crop{crop_index}"[:64]
                    crop_path = asset_dir / f"{crop_asset_id}.png"
                    width, height = apply_crop(frame.path, box, crop_path)
                    crop_sha = sha256_file(crop_path)
                    entries.append(
                        build_crop_entry(
                            asset_id=crop_asset_id,
                            rel_path=f"assets/{crop_path.name}",
                            crop_sha256=crop_sha,
                            parent_asset_id=frame_asset_id,
                            parent_sha256=frame.sha256,
                            timestamp_seconds=frame.timestamp_seconds,
                            video_name=video_path.name,
                            label=label,
                            frame_subject=frame_subject,
                            rights_status=rights_status,
                        )
                    )
                    crop_records.append(
                        {
                            "asset_id": crop_asset_id,
                            "path": f"assets/{crop_path.name}",
                            "sha256": crop_sha,
                            "width": width,
                            "height": height,
                            "parent_asset_id": frame_asset_id,
                            "request_id": (
                                str(crop_request_id) if crop_request_id else None
                            ),
                            "box": box.model_dump(mode="json"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - keep the batch alive
                    crop_errors.setdefault(frame.sha256, []).append(
                        f"apply_crop {crop_index}: {type(exc).__name__}: {exc}"
                    )
                    continue
    for index, frame in enumerate(sampled, start=1):
        rep = representative_for(hashes, clusters, frame.sha256)
        manifest_frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": frame.timestamp_seconds,
                "sha256": frame.sha256,
                "width": frame.width,
                "height": frame.height,
                "dedup_representative_sha256": rep,
                "asset_id": (
                    asset_id_for(unique.index(by_sha[rep]) + 1, title=title)
                    if rep in by_sha
                    else None
                ),
                "classification": (
                    classifications[rep].model_dump(mode="json")
                    if rep in classifications
                    else None
                ),
                "classification_error": classification_errors.get(frame.sha256),
                "crop_errors": crop_errors.get(frame.sha256, []),
                "classification_request_id": (
                    classification_request_ids.get(rep)
                    if rep in classification_request_ids
                    else None
                ),
            }
        )

    catalog_path = write_episode_catalog(
        output_dir=out_dir,
        video_name=video_path.name,
        rights_status=rights_status,
        entries=entries,
    )
    manifest_path = out_dir / "episode_assets_manifest.json"
    dump_json_atomic(
        manifest_path,
        {
            "schema_version": "episode-assets-extract-v1",
            "run_id": run_id,
            "video": {
                "path": video_path.name,
                "sha256": video_sha,
                "duration_seconds": duration,
            },
            "extracted_frame_count": len(sampled),
            "unique_frame_count": len(unique),
            "crop_count": len(crop_records),
            "frames": manifest_frames,
            "crops": crop_records,
            "catalog": catalog_path.name,
        },
        sort_keys=True,
    )
    return EpisodeAssetExtractResult(
        run_id=run_id,
        video_path=video_path,
        video_sha256=video_sha,
        video_duration_seconds=duration,
        extracted_frame_count=len(sampled),
        unique_frame_count=len(unique),
        crop_count=len(crop_records),
        catalog_path=catalog_path,
        manifest_path=manifest_path,
    )


def _frame_hash_of(frame: SampledFrame) -> str:
    with Image.open(frame.path) as image:
        return frame_hash(image)
