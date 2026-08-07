"""Build workflow: script + clips -> staging -> verified target run dir."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

from anime_remix import __version__
from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.domain.enums import RunStatus
from anime_remix.domain.models import ProbedClip
from anime_remix.errors import (
    AnimeRemixError,
    InputValidationError,
    PublicationError,
    UnsupportedMediaError,
)
from anime_remix.json_io import dump_json_atomic, sha256_file
from anime_remix.services.aliases import load_aliases_document
from anime_remix.services.clip_retriever import retrieve
from anime_remix.services.input_loader import (
    load_clips_document,
    load_script_text,
    validate_clip_path,
)
from anime_remix.services.script_parser import parse_script
from anime_remix.services.timeline_compiler import compile_timeline
from anime_remix.workflows.render_workflow import render_timeline

MARKER = {
    "schema_version": "1.9",
    "application": "anime-remix",
    "managed_entries": [
        ".anime-remix-run",
        "normalized",
        "output.mp4",
        "parsed_script.json",
        "render.log",
        "retrieval_results.json",
        "run_manifest.json",
        "timeline.json",
    ],
}


def _now() -> str:
    return _datetime.datetime.now(_datetime.UTC).isoformat()


def _application_version() -> str:
    try:
        return importlib.metadata.version("anime-remix-agent")
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _sha256_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _build_manifest(
    *,
    status: RunStatus,
    toolkit: FFmpegToolkit,
    script_path: Path,
    clips_path: Path,
    aliases_path: Path | None = None,
    started_at: str,
    finished_at: str | None,
    selected_source_sha256: dict[str, str] | None = None,
    core_artifact_sha256: str | None = None,
    selected_source_frame_audit: dict[str, dict[str, int]] | None = None,
    core_artifact_member_sha256: dict[str, str] | None = None,
    output_sha256: str | None = None,
    assumed_color_metadata_asset_ids: list[str] | None = None,
    failed_stage: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.9",
        "status": status.value,
        "application_version": _application_version(),
        "python_version": platform.python_version(),
        "ffmpeg_version": toolkit.ffmpeg_version(),
        "ffprobe_version": toolkit.ffprobe_version(),
        "requested_parser": "rule",
        "actual_parser": "rule",
        "script_sha256": sha256_file(script_path),
        "clips_json_sha256": sha256_file(clips_path),
        "aliases_sha256": (
            sha256_file(aliases_path) if aliases_path is not None else None
        ),
        "selected_source_sha256": selected_source_sha256 or {},
        "core_artifact_sha256": core_artifact_sha256,
        "selected_source_frame_audit": selected_source_frame_audit or {},
        "core_artifact_member_sha256": core_artifact_member_sha256,
        "output_sha256": output_sha256,
        "assumed_color_metadata_asset_ids": assumed_color_metadata_asset_ids or [],
        "started_at": started_at,
        "finished_at": finished_at,
        "failed_stage": failed_stage,
        "error_type": error_type,
    }


def _validate_target(
    target: Path,
    *,
    script_path: Path,
    clips_path: Path,
    clip_files: list[Path],
) -> None:
    resolved = target.resolve()
    if resolved.exists():
        raise PublicationError("build target must not exist", actual=target)
    protected = [script_path.resolve(), clips_path.resolve(), *clip_files]
    for file in protected:
        if resolved == file or _is_within(file, resolved):
            raise PublicationError(
                "build target must not equal or contain input files",
                actual=target,
            )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _planning_documents(
    requirements: list[Any],
    retrieval_doc: dict[str, Any],
    timeline_doc: dict[str, Any],
) -> dict[str, Any]:
    return {
        "parsed_script.json": {
            "schema_version": "1.9",
            "shots": [req.model_dump(mode="json") for req in requirements],
        },
        "retrieval_results.json": retrieval_doc,
        "timeline.json": timeline_doc,
    }


def build(
    script_path: Path,
    clips_path: Path,
    output_dir: Path,
    *,
    parser: str = "rule",
    aliases_path: Path | None = None,
) -> Path:
    """Run the fixed build workflow and publish a verified run directory."""

    if parser != "rule":
        raise InputValidationError(
            "only --parser rule is supported in MVR",
            actual=parser,
        )
    started_at = _now()
    toolkit = FFmpegToolkit()
    staging: Path | None = None
    aliases_doc = None
    selected_source_frame_audit: dict[str, dict[str, int]] = {}
    core_artifact_sha256: str | None = None
    core_artifact_member_sha256: dict[str, str] | None = None
    try:
        script_text = load_script_text(script_path)
        clips_doc = load_clips_document(clips_path)
        clip_files = [
            validate_clip_path(
                clips_path.resolve().parent,
                str(clip.path),
                clip_id=clip.id,
            )
            for clip in clips_doc.clips
        ]
        _validate_target(
            output_dir,
            script_path=script_path,
            clips_path=clips_path,
            clip_files=clip_files,
        )
        if aliases_path is not None:
            aliases_doc = load_aliases_document(aliases_path, clips_doc)
        toolkit.check_capabilities()

        probes: list[ProbedClip] = []
        # Serial probe in fixed asset-id order; never parallel/non-deterministic.
        ordered_pairs = sorted(
            zip(clips_doc.clips, clip_files),
            key=lambda pair: pair[0].id,
        )
        for clip, resolved in ordered_pairs:
            probes.append(toolkit.probe_asset(resolved, clip))

        requirements = parse_script(script_text, clips_doc, aliases=aliases_doc)
        selections, retrieval_doc = retrieve(requirements, probes)
        source_sha256: dict[str, str] = {}
        for selection in selections.values():
            if selection.asset is not None:
                asset_id = selection.asset.asset.id
                source_sha256.setdefault(
                    asset_id,
                    sha256_file(selection.asset.resolved_path),
                )
        # Stable manifest ordering: one entry per selected asset, sorted by id.
        source_sha256 = dict(sorted(source_sha256.items()))

        # Selected-source count_frames audit (AGENTS v1.13 section 12.8):
        # runs before staging creation, in UTF-8 byte order by asset_id, only
        # for unique selected assets. Mismatch fails immediately without
        # staging / render / reretrieval / fallback.
        selected_asset_ids = sorted(
            {
                selection.asset.asset.id
                for selection in selections.values()
                if selection.asset is not None
            },
            key=lambda asset_id: asset_id.encode("utf-8"),
        )
        probe_by_id = {probe.asset.id: probe for probe in probes}
        for asset_id in selected_asset_ids:
            probe = probe_by_id[asset_id]
            counted = toolkit.count_source_frames(probe.resolved_path)
            if counted != probe.nb_frames:
                raise UnsupportedMediaError(
                    "selected source count_frames audit mismatch",
                    asset_id=asset_id,
                    field="nb_frames",
                    actual={
                        "metadata_nb_frames": probe.nb_frames,
                        "counted_nb_frames": counted,
                    },
                )
            selected_source_frame_audit[asset_id] = {
                "metadata_nb_frames": probe.nb_frames,
                "counted_nb_frames": counted,
            }

        timeline = compile_timeline(
            requirements,
            selections,
            target_dir=output_dir,
            source_sha256=source_sha256,
        )
        documents = _planning_documents(
            requirements,
            retrieval_doc,
            timeline.model_dump(mode="json"),
        )

        staging = output_dir.resolve().parent / f".{output_dir.name}.staging"
        if staging.exists():
            raise PublicationError(
                "staging directory already exists",
                actual=staging,
            )
        staging.mkdir(parents=True, exist_ok=False)

        dump_json_atomic(staging / ".anime-remix-run", MARKER)
        assumed_ids = [
            probe.asset.id for probe in probes if probe.assumed_color_metadata
        ]
        running_manifest = _build_manifest(
            status=RunStatus.RUNNING,
            toolkit=toolkit,
            script_path=script_path,
            clips_path=clips_path,
            aliases_path=aliases_path,
            started_at=started_at,
            finished_at=None,
            selected_source_sha256=source_sha256,
            selected_source_frame_audit=selected_source_frame_audit,
            core_artifact_member_sha256=None,
            assumed_color_metadata_asset_ids=assumed_ids,
        )
        dump_json_atomic(staging / "run_manifest.json", running_manifest)
        for name, doc in documents.items():
            dump_json_atomic(staging / name, doc)
        core_artifact_names = (
            "parsed_script.json",
            "retrieval_results.json",
            "timeline.json",
        )
        core_artifact_sha256 = _sha256_bytes(
            [(staging / name).read_bytes() for name in core_artifact_names]
        )
        core_artifact_member_sha256 = {
            name: sha256_file(staging / name) for name in core_artifact_names
        }
        running_manifest["core_artifact_sha256"] = core_artifact_sha256
        running_manifest[
            "core_artifact_member_sha256"
        ] = core_artifact_member_sha256
        dump_json_atomic(staging / "run_manifest.json", running_manifest)

        render_timeline(
            timeline_path=staging / "timeline.json",
            output_path=staging / "output.mp4",
            allow_managed_output=True,
            log_path=staging / "render.log",
            toolkit=toolkit,
        )
        output_path = staging / "output.mp4"
        toolkit.validate_final(
            output_path,
            total_frames=sum(item.target_frames for item in timeline.items),
            profile=timeline.render_profile,
        )
        succeeded_manifest = _build_manifest(
            status=RunStatus.SUCCEEDED,
            toolkit=toolkit,
            script_path=script_path,
            clips_path=clips_path,
            aliases_path=aliases_path,
            started_at=started_at,
            finished_at=_now(),
            selected_source_sha256=source_sha256,
            core_artifact_sha256=core_artifact_sha256,
            selected_source_frame_audit=selected_source_frame_audit,
            core_artifact_member_sha256=core_artifact_member_sha256,
            output_sha256=sha256_file(output_path),
            assumed_color_metadata_asset_ids=assumed_ids,
        )
        dump_json_atomic(staging / "run_manifest.json", succeeded_manifest)
        output_dir.resolve().parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
        return output_dir
    except AnimeRemixError as exc:
        if staging is not None and staging.exists():
            failed_manifest = _build_manifest(
                status=RunStatus.FAILED,
                toolkit=toolkit,
                script_path=script_path,
                clips_path=clips_path,
                aliases_path=aliases_path,
                started_at=started_at,
                finished_at=_now(),
                selected_source_sha256=source_sha256,
                core_artifact_sha256=core_artifact_sha256,
                selected_source_frame_audit=selected_source_frame_audit,
                core_artifact_member_sha256=core_artifact_member_sha256,
                failed_stage=exc.stage,
                error_type=type(exc).__name__,
            )
            try:
                dump_json_atomic(
                    staging / "run_manifest.json",
                    failed_manifest,
                )
            except AnimeRemixError:
                pass
        raise
