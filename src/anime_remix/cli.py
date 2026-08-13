"""Typer CLI entry point for anime-remix."""

from __future__ import annotations

import copy
import re
import unicodedata
from pathlib import Path
from typing import Any

import typer
from pydantic import TypeAdapter, ValidationError

from anime_remix import __version__
from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.errors import (
    ERROR_EXIT_CODES,
    AnimeRemixError,
    InputValidationError,
)
from anime_remix.json_io import (
    dump_json_atomic,
    load_json_object,
    sha256_file,
)
from anime_remix.services.aliases import load_aliases_document
from anime_remix.services.episode_assets import (
    DashScopeCropper,
    DashScopeEpisodeClassifier,
    StubCropper,
    StubEpisodeClassifier,
    extract_episode_assets,
)
from anime_remix.services.episode_assets.review import (
    apply_review,
    write_review_sheet,
)
from anime_remix.services.execution.adapter import (
    QwenImage30Adapter,
    StubImageExecutor,
)
from anime_remix.services.execution.dashscope_executor import (
    DashScopeQwenExecutor,
)
from anime_remix.services.execution.first_frame_composer import (
    run_first_frame_composition,
)
from anime_remix.services.execution.generated_shot_pipeline import (
    parse_generated_shot_inputs,
    run_generated_shot_pipeline,
)
from anime_remix.services.execution.handoff_frame_composer import (
    run_handoff_frame_composition,
)
from anime_remix.services.execution.prepared_component_composer import (
    run_prepared_component_composition,
)
from anime_remix.services.image_assets import (
    ImageAssetsDocument,
    load_image_assets,
    probe_image_file,
)
from anime_remix.services.input_loader import (
    load_clips_document,
    load_script_text,
)
from anime_remix.services.script.binding import (
    auto_bind,
    generate_binding_template,
    load_binding,
    validate_binding_against_plan,
    write_binding_template,
    write_reference_bundles,
)
from anime_remix.services.script.director import run_director_for_script
from anime_remix.services.script.first_frame_content_plan import (
    approve_first_frame_content_plan,
    build_first_frame_content_plan,
)
from anime_remix.services.script.first_frame_plan import (
    approve_first_frame_plan,
    build_first_frame_plan,
    parse_first_frame_plan,
)
from anime_remix.services.script.generation_segment_plan import (
    approve_generation_segment_plan,
    build_generation_segment_plan,
    parse_generation_segment_plan,
)
from anime_remix.services.script.prepared_component_plan import (
    approve_prepared_component_plan,
    build_prepared_component_plan,
    complete_prepared_component_plan,
    parse_prepared_component_plan,
)
from anime_remix.services.script.review import (
    validate_shot_plan_file,
    write_review_artifacts,
)
from anime_remix.services.script.shot_plan import load_shot_plan
from anime_remix.workflows.build_workflow import build
from anime_remix.workflows.render_workflow import render_timeline

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Anime Remix Agent: script + clips -> editable timeline -> MP4",
)
director_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Director LLM shot planning and review loop.",
)
app.add_typer(director_app, name="director")
assets_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Register and promote image assets in an image_assets.json catalog.",
)
app.add_typer(assets_app, name="assets")
episode_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Extract reference assets from one authorized anime episode.",
)
app.add_typer(episode_app, name="episode-assets")

_ASSET_TYPES = frozenset(
    {"character", "background", "foreground", "prop", "style"}
)
_REFERENCE_ROLES = frozenset(
    {
        "identity_reference",
        "pose_reference",
        "expression_reference",
        "outfit_reference",
        "scene_reference",
        "prop_reference",
        "style_reference",
    }
)
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"anime-remix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show Python tracebacks on errors.",
    ),
) -> None:
    ctx.obj = {"verbose": verbose}


def _handle(ctx: typer.Context, operation: Any) -> None:
    try:
        operation()
    except AnimeRemixError as exc:
        if ctx.obj.get("verbose"):
            raise
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(ERROR_EXIT_CODES.get(type(exc), exc.exit_code)) from exc
    except Exception as exc:  # unexpected
        if ctx.obj.get("verbose"):
            raise
        typer.echo(f"error: internal failure: {exc}", err=True)
        raise typer.Exit(1) from exc


class ReviewOnlyExecutor:
    """Resume guard: any unexpected model execution is fatal in review-only mode."""

    provider = "review-only"

    def execute(self, **_: object) -> bytes:
        raise InputValidationError(
            "review-only resume attempted an unexpected model call; approve "
            "completed outputs without executing new model stages"
        )


def _resolve_generation_executor(
    *,
    executor_name: str,
    confirm_paid: bool,
    seed: int = 0,
    size: str = "1280*720",
) -> tuple[QwenImage30Adapter, Any]:
    """Resolve the image-generation adapter and executor from CLI flags."""

    normalized = executor_name.strip().casefold()
    if normalized not in {"stub", "dashscope", "review-only"}:
        raise InputValidationError(
            "--executor must be 'stub', 'dashscope' or 'review-only'",
            actual=executor_name,
        )
    if normalized == "dashscope" and not confirm_paid:
        raise InputValidationError(
            "DashScope generation is paid-capable; rerun with --confirm-paid "
            "after confirming the requested new model stages/frames"
        )
    if normalized != "dashscope" and confirm_paid:
        raise InputValidationError(
            "--confirm-paid is only valid with --executor dashscope"
        )
    adapter = QwenImage30Adapter(seed=seed, size=size)
    if normalized == "dashscope":
        executor: Any = DashScopeQwenExecutor()
    elif normalized == "review-only":
        executor = ReviewOnlyExecutor()
    else:
        executor = StubImageExecutor()
    return adapter, executor


def _validate_roles(roles: list[str] | None) -> list[str]:
    """Validate and deduplicate --roles values."""

    if not roles:
        return []
    cleaned: list[str] = []
    for role in roles:
        stripped = role.strip()
        if stripped not in _REFERENCE_ROLES:
            raise InputValidationError(
                f"invalid --roles value {role!r}; expected one of: "
                + ", ".join(sorted(_REFERENCE_ROLES)),
                field="reference_roles",
                actual=role,
            )
        if stripped not in cleaned:
            cleaned.append(stripped)
    return cleaned


def _slugify_asset_id(stem: str) -> str:
    """Deterministic ASCII asset_id derived from a file stem."""

    norm = unicodedata.normalize("NFKC", stem).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", norm).strip("_-")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "asset"
    if not re.match(r"^[A-Za-z0-9]", cleaned):
        cleaned = f"asset_{cleaned}"
    return cleaned[:64]


def _unique_asset_id(base: str, used: set[str]) -> str:
    """First free id: base, base-2, base-3, ... (deterministic)."""

    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _load_asset_document(
    catalog: Path,
) -> tuple[dict[str, Any], object | None]:
    """Load catalog JSON plus a validated catalog (None when empty/missing)."""

    if not catalog.exists():
        return {"schema_version": "image-assets-v1", "assets": []}, None
    data = load_json_object(catalog)
    if data.get("schema_version") != "image-assets-v1":
        raise InputValidationError(
            "catalog schema_version must be image-assets-v1",
            actual=data.get("schema_version"),
        )
    assets = data.get("assets")
    if not isinstance(assets, list):
        raise InputValidationError(
            "catalog must contain an assets list",
            field="assets",
            actual=catalog,
        )
    if not assets:
        return data, None
    return data, load_image_assets(catalog)


def _sha_index(
    catalog: object,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (asset_id -> sha256, sha256 -> first asset_id) indexes.

    The reverse index drives exact-file SHA-256 dedup: a byte-identical
    file already registered under any id is never registered twice.
    """

    from anime_remix.services.image_assets import ImageAssetCatalog

    if not isinstance(catalog, ImageAssetCatalog):
        return {}, {}
    sha_by_id: dict[str, str] = {}
    id_by_sha: dict[str, str] = {}
    for record in catalog.records:
        prov = record.provenance or {}
        digest = prov.get("sha256")
        if not digest:
            digest = sha256_file(record.resolved_path)
        sha_by_id.setdefault(record.asset_id, digest)
        id_by_sha.setdefault(digest, record.asset_id)
    return sha_by_id, id_by_sha


def _collect_image_files(
    paths: list[Path] | None,
    dir_path: Path | None,
) -> list[Path]:
    """Explicit --paths plus top-level --dir image files (no recursion)."""

    if not paths and dir_path is None:
        raise InputValidationError(
            "provide at least one --paths file or a --dir directory"
        )
    resolved_files: list[Path] = []
    seen: set[str] = set()
    if dir_path is not None:
        base = dir_path.resolve()
        if not base.is_dir():
            raise InputValidationError(
                f"--dir is not a directory: {dir_path}",
                actual=str(base),
            )
        try:
            children = sorted(base.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise InputValidationError(
                f"cannot list --dir {dir_path}",
                actual=str(exc),
            ) from exc
        for child in children:
            if not child.is_file():
                continue
            if child.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            key = str(child.resolve())
            if key not in seen:
                seen.add(key)
                resolved_files.append(child)
    base_for_paths = dir_path.resolve() if dir_path is not None else Path.cwd()
    for raw in paths or []:
        candidate = raw if raw.is_absolute() else base_for_paths / raw
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise InputValidationError(
                f"image file does not exist: {raw}",
                actual=str(candidate),
            ) from exc
        if not resolved.is_file():
            raise InputValidationError(
                f"image path is not a regular file: {raw}",
                actual=str(resolved),
            )
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            resolved_files.append(resolved)
    return resolved_files


def _relative_catalog_path(
    catalog_dir: Path,
    resolved: Path,
    *,
    label: str,
) -> str:
    """Store images as plain relative paths inside the catalog directory."""

    try:
        rel = resolved.relative_to(catalog_dir)
    except ValueError:
        raise InputValidationError(
            f"{label} must live inside the catalog directory: {catalog_dir}",
            actual=str(resolved),
        ) from None
    return rel.as_posix()


def _new_asset_entry(
    *,
    asset_id: str,
    rel_path: str,
    asset_type: str,
    rights: str,
    digest: str,
    source_path: Path,
    tier: str,
    roles: list[str],
    note: str | None,
    parent_asset_id: str | None,
    parent_sha256: str | None,
) -> dict[str, Any]:
    """One strict image_assets.json entry for a freshly registered file."""

    return {
        "asset_id": asset_id,
        "path": rel_path,
        "asset_type": asset_type,
        "rights_status": rights,
        "subject_or_scene_id": None,
        "view_angle": None,
        "pose": None,
        "expression": None,
        "outfit": None,
        "time_of_day": None,
        "quality_notes": note,
        "source_tier": tier,
        "reference_roles": roles,
        "provenance": {
            "source_path": str(source_path),
            "sha256": digest,
            "parent_asset_id": parent_asset_id,
            "parent_sha256": parent_sha256,
            "note": note,
        },
        "analysis_status": "pending",
    }


def _write_and_verify_catalog(
    catalog: Path,
    data: dict[str, Any],
    original: dict[str, Any] | None,
) -> None:
    """Atomic write, then full loader verification (restores on failure)."""

    try:
        TypeAdapter(ImageAssetsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid catalog after registration: {exc}",
            actual=catalog,
        ) from exc
    dump_json_atomic(catalog, data, sort_keys=True)
    try:
        load_image_assets(catalog)
    except AnimeRemixError:
        if original is not None:
            dump_json_atomic(catalog, original, sort_keys=True)
        else:
            catalog.unlink(missing_ok=True)
        raise


def _register_assets(
    *,
    catalog: Path,
    paths: list[Path] | None,
    dir_path: Path | None,
    asset_type: str | None,
    roles: list[str] | None,
    note: str | None,
    rights: str,
    tier: str,
    generated_from: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Shared register/register-candidate logic; returns (data, entries, duplicates)."""

    data, loaded = _load_asset_document(catalog)
    sha_by_id, id_by_sha = (
        _sha_index(loaded) if loaded is not None else ({}, {})
    )
    used_ids = {entry["asset_id"] for entry in data["assets"]}

    parent_asset_id: str | None = None
    parent_sha256: str | None = None
    if generated_from:
        if loaded is None or loaded.get(generated_from) is None:
            raise InputValidationError(
                f"parent asset not found in catalog: {generated_from}",
                asset_id=generated_from,
            )
        parent = loaded.get(generated_from)
        parent_asset_id = generated_from
        parent_sha256 = sha_by_id.get(generated_from)
        if asset_type is None:
            asset_type = parent.asset_type
    if asset_type is None:
        raise InputValidationError(
            "an asset_type is required (--type, or --generated-from parent)",
            field="asset_type",
        )
    if asset_type not in _ASSET_TYPES:
        raise InputValidationError(
            f"invalid asset_type {asset_type!r}; expected one of: "
            + ", ".join(sorted(_ASSET_TYPES)),
            field="asset_type",
            actual=asset_type,
        )
    if not rights.strip():
        raise InputValidationError(
            "rights_status must be non-empty",
            field="rights_status",
            actual=rights,
        )
    roles_clean = _validate_roles(roles)
    files = _collect_image_files(paths, dir_path)
    if not files:
        raise InputValidationError(
            "no image files to register (check --paths / --dir)"
        )

    catalog_dir = catalog.resolve().parent
    new_entries: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    for file_path in files:
        probe_image_file(file_path)
        digest = sha256_file(file_path)
        existing_id = id_by_sha.get(digest)
        if existing_id is not None:
            duplicates.append(
                {"path": str(file_path), "asset_id": existing_id}
            )
            continue
        base = _slugify_asset_id(file_path.stem)
        asset_id = _unique_asset_id(base, used_ids)
        rel_path = _relative_catalog_path(
            catalog_dir,
            file_path,
            label=file_path.name,
        )
        new_entries.append(
            _new_asset_entry(
                asset_id=asset_id,
                rel_path=rel_path,
                asset_type=asset_type,
                rights=rights.strip(),
                digest=digest,
                source_path=file_path,
                tier=tier,
                roles=roles_clean,
                note=note.strip() if note else None,
                parent_asset_id=parent_asset_id,
                parent_sha256=parent_sha256,
            )
        )
        id_by_sha.setdefault(digest, asset_id)
        sha_by_id[asset_id] = digest
        used_ids.add(asset_id)
    return data, new_entries, duplicates


@app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    script: Path = typer.Option(..., "--script", help="Path to script.md"),
    clips: Path = typer.Option(..., "--clips", help="Path to clips.json"),
    aliases: Path | None = typer.Option(
        None,
        "--aliases",
        help="Path to aliases.json (optional; static target/conflict validation).",
    ),
    probe_media: bool = typer.Option(
        False,
        "--probe-media",
        help="Also run FFprobe and enforce the MVR media contract.",
    ),
) -> None:
    """Validate script, clips and optional aliases (and media with --probe-media)."""

    def operation() -> None:
        script_text = load_script_text(script)
        clips_doc = load_clips_document(clips)
        if aliases is not None:
            load_aliases_document(aliases, clips_doc)
        if probe_media:
            toolkit = FFmpegToolkit()
            toolkit.check_capabilities()
            for clip in clips_doc.clips:
                toolkit.probe_asset(
                    (clips.resolve().parent / clip.path).resolve(),
                    clip,
                )
            typer.echo(
                f"ok: static + media validation passed "
                f"({len(clips_doc.clips)} clips, {len(script_text.splitlines())} lines)"
            )
        else:
            typer.echo(
                f"ok: static validation passed "
                f"({len(clips_doc.clips)} clips)"
            )

    _handle(ctx, operation)


@app.command("render")
def render_cmd(
    ctx: typer.Context,
    timeline: Path = typer.Option(..., "--timeline", help="Path to timeline.json"),
    output: Path = typer.Option(..., "--output", help="Output MP4 path"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing output MP4.",
    ),
) -> None:
    """Render a timeline into a verified MP4 using only timeline + sources."""

    def operation() -> None:
        result = render_timeline(
            timeline_path=timeline,
            output_path=output,
            overwrite=overwrite,
        )
        typer.echo(f"ok: rendered {result}")

    _handle(ctx, operation)


@director_app.command("plan")
def director_plan_cmd(
    ctx: typer.Context,
    script: Path = typer.Option(..., "--script", help="Path to script.md"),
    output: Path = typer.Option(..., "--output", help="Review directory"),
) -> None:
    """Run the Director, then write shot_plan.json + review preview."""

    def operation() -> None:
        result = run_director_for_script(script, out_dir=output)
        write_review_artifacts(
            output,
            result.document,
            run_manifest={
                "schema_version": "director-review-v1",
                "prompt_version": result.prompt_version,
                "script": result.script_name,
                "script_sha256": result.script_sha256,
                "model_identity": result.model_identity,
                "session_id": result.session_id,
                "tokens_used": result.tokens_used,
                "tries": result.tries,
                "status": "needs_review",
            },
        )
        total = sum(s.duration_seconds for s in result.document.shots)
        typer.echo(
            f"ok: wrote {output / 'shot_plan.json'} and "
            f"{output / 'shot_plan.review.md'} "
            f"({len(result.document.shots)} shots, {total:.1f}s total) - "
            f"edit the JSON, then run: anime-remix director validate "
            f"--shot-plan {output / 'shot_plan.json'}"
        )

    _handle(ctx, operation)


@director_app.command("validate")
def director_validate_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(..., "--shot-plan", help="Path to shot_plan.json"),
) -> None:
    """Strictly validate an (edited) shot plan JSON."""

    def operation() -> None:
        document = validate_shot_plan_file(shot_plan)
        total = sum(s.duration_seconds for s in document.shots)
        typer.echo(
            f"ok: shot plan valid ({len(document.shots)} shots, "
            f"{total:.1f}s total)"
        )

    _handle(ctx, operation)


@director_app.command("bind-template")
def director_bind_template_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(..., "--shot-plan", help="Path to shot_plan.json"),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Path to image_assets.json catalog",
    ),
    output: Path = typer.Option(..., "--output", help="Output directory"),
) -> None:
    """Generate an editable per-shot reference-image binding template."""

    def operation() -> None:
        plan = load_shot_plan(shot_plan)
        catalog = load_image_assets(image_assets)
        template = generate_binding_template(plan, catalog)
        path = write_binding_template(template, output)
        typer.echo(
            f"ok: wrote {path} ({len(template.shots)} shots with candidates) - "
            f"edit it to add bindings, then run director bind-validate"
        )

    _handle(ctx, operation)


@director_app.command("bind-auto")
def director_bind_auto_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(..., "--shot-plan", help="Path to shot_plan.json"),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Path to image_assets.json catalog",
    ),
    output: Path = typer.Option(..., "--output", help="Output directory"),
) -> None:
    """Auto-bind reference assets per shot; write a draft for confirmation."""

    def operation() -> None:
        plan = load_shot_plan(shot_plan)
        catalog = load_image_assets(image_assets)
        draft, report = auto_bind(plan, catalog)
        output.mkdir(parents=True, exist_ok=True)
        draft_path = output / "shot_asset_binding.auto.json"
        dump_json_atomic(
            draft_path,
            draft.model_dump(mode="json"),
            sort_keys=True,
        )
        summary: list[str] = []
        for shot_id, shot_report in sorted(report.items()):
            decision = shot_report["decision"]
            summary.append(f"  {shot_id} [{decision}]")
            for entry in shot_report["entries"]:
                summary.append(
                    f"    {entry['asset_id']} ({entry['asset_type']}, "
                    f"tier={entry['tier']}, score={entry['score']}, "
                    f"confidence={entry['confidence']}, "
                    f"reason={entry['reason']})"
                )
        typer.echo(f"ok: wrote auto-binding draft {draft_path}")
        typer.echo("auto-binding report:")
        typer.echo("\n".join(summary))
        typer.echo(
            "review/edit the draft, then run: anime-remix director "
            "bind-validate --binding <draft> ..."
        )

    _handle(ctx, operation)


@director_app.command("bind-validate")
def director_bind_validate_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(..., "--shot-plan", help="Path to shot_plan.json"),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Path to image_assets.json catalog",
    ),
    binding: Path = typer.Option(..., "--binding", help="Path to edited binding JSON"),
    output: Path = typer.Option(..., "--output", help="Reference bundle directory"),
    shots: str | None = typer.Option(
        None,
        "--shots",
        help="Comma-separated shot_ids to bind; default all shots",
    ),
) -> None:
    """Validate bindings and write per-shot reference bundles."""

    def operation() -> None:
        plan = load_shot_plan(shot_plan)
        catalog = load_image_assets(image_assets)
        binding_doc = load_binding(binding)
        bundles = validate_binding_against_plan(binding_doc, plan, catalog)
        selected = (
            {item.strip() for item in shots.split(",") if item.strip()}
            if shots
            else set(bundles)
        )
        missing_selected = sorted(selected - set(bundles))
        if missing_selected:
            raise InputValidationError(
                f"unknown shot_ids: {', '.join(missing_selected)}",
                actual=shots,
            )
        selected_bundles = {
            shot_id: bundles[shot_id] for shot_id in bundles if shot_id in selected
        }
        write_reference_bundles(selected_bundles, output)
        typer.echo(
            f"ok: wrote reference bundles for {len(selected_bundles)} shots "
            f"into {output}"
        )

    _handle(ctx, operation)


@director_app.command("first-frame-content")
def director_first_frame_content_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(
        ..., "--shot-plan", help="Path to reviewed shot_plan.json"
    ),
    shot_id: str = typer.Option(..., "--shot-id", help="Shot to plan"),
    output: Path = typer.Option(
        ..., "--output", help="Output first_frame_content_plan.json path"
    ),
    assembly_policy: Path | None = typer.Option(
        None,
        "--assembly-policy",
        help="Optional policy whose interaction requirements seed the contact graph",
    ),
) -> None:
    """Scaffold the editable, model-independent visual truth for K0."""

    def operation() -> None:
        document = load_shot_plan(shot_plan)
        shot = next((item for item in document.shots if item.shot_id == shot_id), None)
        if shot is None:
            raise InputValidationError(
                f"shot_id not found in shot plan: {shot_id}", actual=shot_id
            )
        plan = build_first_frame_content_plan(
            shot,
            assembly_policy=(
                load_json_object(assembly_policy)
                if assembly_policy is not None
                else None
            ),
        )
        dump_json_atomic(output, plan.model_dump(mode="json"), sort_keys=True)
        typer.echo(
            f"ok: wrote draft {output} (decision={plan.decision}, "
            f"layers={len(plan.layers)}, contacts={len(plan.contact_graph)})"
        )
        for warning in plan.warnings:
            typer.echo(f"warning: {warning}")
        typer.echo(
            "edit layer order, state, contact and information truth; then run "
            "director first-frame-content-approve"
        )

    _handle(ctx, operation)


@director_app.command("first-frame-content-approve")
def director_first_frame_content_approve_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Reviewed first-frame-content-plan-v1 draft"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output approved content plan JSON"
    ),
) -> None:
    """Approve the reviewed K0 content truth before selecting assets."""

    def operation() -> None:
        approved = approve_first_frame_content_plan(load_json_object(plan))
        dump_json_atomic(output, approved.model_dump(mode="json"), sort_keys=True)
        typer.echo(f"ok: approved first-frame content {approved.plan_id}; wrote {output}")

    _handle(ctx, operation)


@director_app.command("component-plan")
def director_component_plan_cmd(
    ctx: typer.Context,
    content_plan: Path = typer.Option(
        ..., "--content-plan", help="Approved first-frame-content-plan-v1 JSON"
    ),
    reference_bundle: Path = typer.Option(
        ..., "--reference-bundle", help="Preparation reference-bundle-v1 JSON"
    ),
    image_assets: Path = typer.Option(
        ..., "--image-assets", help="Path to image_assets.json catalog"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output prepared_component_plan.json path"
    ),
    assembly_policy: Path | None = typer.Option(
        None,
        "--assembly-policy",
        help="Optional authority policy for WHO/HOW/control-only references",
    ),
) -> None:
    """Plan WHO/HOW component plates without invoking an image model."""

    def operation() -> None:
        catalog = load_image_assets(image_assets)
        plan = build_prepared_component_plan(
            load_json_object(content_plan),
            reference_bundle=load_json_object(reference_bundle),
            catalog=catalog,
            assembly_policy=(
                load_json_object(assembly_policy)
                if assembly_policy is not None
                else None
            ),
        )
        dump_json_atomic(output, plan.model_dump(mode="json"), sort_keys=True)
        typer.echo(
            f"ok: wrote draft {output} (decision={plan.decision}, "
            f"tasks={len(plan.tasks)})"
        )
        for warning in plan.warnings:
            typer.echo(f"warning: {warning}")
        typer.echo("review tasks; then run director component-approve")

    _handle(ctx, operation)


@director_app.command("component-approve")
def director_component_approve_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Reviewed prepared-component-plan-v1 draft"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output generation-approved component plan JSON"
    ),
) -> None:
    """Approve component tasks for a separately authorized generation run."""

    def operation() -> None:
        approved = approve_prepared_component_plan(load_json_object(plan))
        dump_json_atomic(output, approved.model_dump(mode="json"), sort_keys=True)
        typer.echo(f"ok: approved component plan {approved.plan_id}; wrote {output}")

    _handle(ctx, operation)


@director_app.command("component-complete")
def director_component_complete_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Plan with manually approved task results"
    ),
    image_assets: Path = typer.Option(
        ..., "--image-assets", help="Catalog containing approved generated outputs"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output completed component plan JSON"
    ),
) -> None:
    """Complete preparation after approved outputs are registered and analyzed."""

    def operation() -> None:
        completed = complete_prepared_component_plan(
            load_json_object(plan), catalog=load_image_assets(image_assets)
        )
        dump_json_atomic(output, completed.model_dump(mode="json"), sort_keys=True)
        typer.echo(
            f"ok: completed component plan {completed.plan_id}; wrote {output}"
        )

    _handle(ctx, operation)


@director_app.command("component-compose")
def director_component_compose_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Generation-approved prepared-component-plan-v1 JSON"
    ),
    image_assets: Path = typer.Option(
        ..., "--image-assets", help="Catalog containing exact task input assets"
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="New or resumable component-generation run directory"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Stable id for resume safety"),
    executor_name: str = typer.Option(
        "stub",
        "--executor",
        help=(
            "stub (offline), dashscope (paid qwen-image-3.0) or "
            "review-only (check a run without new model calls)"
        ),
    ),
    size: str = typer.Option(
        "1280*720", "--size", help="Frozen qwen-image-3.0 output size"
    ),
    seed: int = typer.Option(0, "--seed", help="Frozen image-generation seed"),
    max_new_tasks: int = typer.Option(
        1,
        "--max-new-tasks",
        help="Maximum new paid-capable tasks in this invocation",
    ),
    confirm_paid: bool = typer.Option(
        False,
        "--confirm-paid",
        help="Required with --executor dashscope; acknowledges the requested calls",
    ),
) -> None:
    """Generate recoverable component candidates; never auto-approve them."""

    def operation() -> None:
        adapter, executor = _resolve_generation_executor(
            executor_name=executor_name,
            confirm_paid=confirm_paid,
            seed=seed,
            size=size,
        )
        plan_doc = parse_prepared_component_plan(load_json_object(plan))
        catalog = load_image_assets(image_assets)
        input_ids = {
            item.asset_id for task in plan_doc.tasks for item in task.model_inputs
        }
        missing = sorted(input_ids - set(catalog.ids))
        if missing:
            raise InputValidationError(
                f"component task inputs are missing from the catalog: {missing}"
            )
        result = run_prepared_component_composition(
            run_dir=run_dir,
            run_id=run_id,
            plan=plan_doc,
            asset_map={
                asset_id: catalog.get(asset_id).resolved_path for asset_id in input_ids
            },
            adapter=adapter,
            executor=executor,
            max_new_model_tasks=max_new_tasks,
        )
        typer.echo(
            f"ok: component run {result.run_id} status={result.status}; "
            f"completed={len(result.completed_task_ids)}; "
            f"next={result.next_task_id or '-'}"
        )
        for asset_id, path in result.output_paths:
            typer.echo(f"candidate {asset_id}: {path}")
        typer.echo(f"manifest: {result.manifest_path}")
        if result.output_paths:
            typer.echo(
                "manual review is required; register candidates, promote only "
                "accepted outputs, add review notes and pass results for every "
                "structured gate, then run component-complete"
            )

    _handle(ctx, operation)


@director_app.command("component-review")
def director_component_review_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Approved prepared-component-plan-v1 JSON"
    ),
    task_id: str | None = typer.Option(
        None,
        "--task-id",
        help="Task to record the review for; default requires a single task",
    ),
    result: str = typer.Option(
        ..., "--result", help="Manual verdict: approved or rejected"
    ),
    gate_result: list[str] | None = typer.Option(
        None,
        "--gate-result",
        help="gate_id=pass|fail for one structured gate; repeatable",
    ),
    note: str = typer.Option(
        ..., "--note", help="Review note recorded on the task result"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output reviewed component plan JSON"
    ),
) -> None:
    """Record the manual component review verdict on a prepared task."""

    def operation() -> None:
        normalized_result = result.strip().casefold()
        if normalized_result not in {"approved", "rejected"}:
            raise InputValidationError(
                "--result must be 'approved' or 'rejected'",
                actual=result,
            )
        if not note.strip():
            raise InputValidationError("--note must be a non-empty review note")
        plan_doc = parse_prepared_component_plan(load_json_object(plan))
        if plan_doc.review_status != "approved":
            raise InputValidationError("component review requires an approved plan")
        if task_id is None:
            if len(plan_doc.tasks) != 1:
                raise InputValidationError(
                    "--task-id is required when the plan has multiple tasks"
                )
            selected = plan_doc.tasks[0]
        else:
            selected = next(
                (item for item in plan_doc.tasks if item.task_id == task_id),
                None,
            )
            if selected is None:
                raise InputValidationError(
                    f"task_id not found in component plan: {task_id}",
                    actual=task_id,
                )
        gate_by_id = {item.gate_id: item for item in selected.review_gates}
        results: list[dict] = []
        for raw in gate_result or ():
            if "=" not in raw:
                raise InputValidationError(
                    "--gate-result must be gate_id=pass|fail",
                    actual=raw,
                )
            gate_id, verdict = raw.split("=", 1)
            verdict = verdict.strip().casefold()
            if verdict not in {"pass", "fail"}:
                raise InputValidationError(
                    "--gate-result verdict must be pass or fail",
                    actual=verdict,
                )
            if gate_id not in gate_by_id:
                raise InputValidationError(
                    f"unknown review gate: {gate_id}",
                    actual=gate_id,
                )
            results.append({"gate_id": gate_id, "result": verdict, "note": note.strip()})
        if len(results) != len({item["gate_id"] for item in results}):
            raise InputValidationError("--gate-result ids must be unique")
        if normalized_result == "approved":
            missing = sorted(set(gate_by_id) - {item["gate_id"] for item in results})
            if missing:
                raise InputValidationError(
                    "approved result requires a verdict for every review gate; "
                    f"missing: {missing}"
                )
            if any(item["result"] != "pass" for item in results):
                raise InputValidationError(
                    "approved result requires every review gate to pass"
                )
        payload = plan_doc.model_dump(mode="json")
        task_payload = next(
            item for item in payload["tasks"] if item["task_id"] == selected.task_id
        )
        task_payload["result"] = normalized_result
        task_payload["result_review_notes"] = note.strip()
        task_payload["gate_results"] = results
        reviewed = parse_prepared_component_plan(payload)
        dump_json_atomic(
            output,
            reviewed.model_dump(mode="json"),
            sort_keys=True,
        )
        typer.echo(
            f"ok: recorded {normalized_result} for {selected.task_id}; "
            f"wrote {output}"
        )

    _handle(ctx, operation)


@director_app.command("first-frame-plan")
def director_first_frame_plan_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(
        ..., "--shot-plan", help="Path to reviewed shot_plan.json"
    ),
    shot_id: str = typer.Option(..., "--shot-id", help="Shot to plan"),
    reference_bundle: Path = typer.Option(
        ...,
        "--reference-bundle",
        help="Validated reference-bundle-v1 JSON for this shot",
    ),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Path to image_assets.json catalog",
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output first_frame_plan.json path"
    ),
    full_frame_anchor: str | None = typer.Option(
        None,
        "--full-frame-anchor",
        help="Bound background/style asset_id that is already the target frame",
    ),
    assembly_policy: Path | None = typer.Option(
        None,
        "--assembly-policy",
        help=(
            "Optional first-frame-assembly-policy-v1 JSON with reference "
            "authority and interaction gates"
        ),
    ),
    content_plan: Path | None = typer.Option(
        None,
        "--content-plan",
        help="Optional approved first-frame-content-plan-v1 JSON",
    ),
    prepared_component_plan: Path | None = typer.Option(
        None,
        "--prepared-component-plan",
        help="Optional completed prepared-component-plan-v1 JSON",
    ),
) -> None:
    """Build a reviewable staged reference-first first-frame plan."""

    def operation() -> None:
        document = load_shot_plan(shot_plan)
        shot = next((item for item in document.shots if item.shot_id == shot_id), None)
        if shot is None:
            raise InputValidationError(
                f"shot_id not found in shot plan: {shot_id}", actual=shot_id
            )
        bundle = load_json_object(reference_bundle)
        catalog = load_image_assets(image_assets)
        plan = build_first_frame_plan(
            shot,
            reference_bundle=bundle,
            catalog=catalog,
            full_frame_anchor_asset_id=full_frame_anchor,
            assembly_policy=(
                load_json_object(assembly_policy)
                if assembly_policy is not None
                else None
            ),
            content_plan=(
                load_json_object(content_plan) if content_plan is not None else None
            ),
            prepared_component_plan=(
                load_json_object(prepared_component_plan)
                if prepared_component_plan is not None
                else None
            ),
        )
        dump_json_atomic(output, plan.model_dump(mode="json"), sort_keys=True)
        model_stages = sum(
            stage.operation not in {"adopt_anchor", "composite_overlay"}
            for stage in plan.stages
        )
        typer.echo(
            f"ok: wrote draft {output} "
            f"(decision={plan.decision}, stages={len(plan.stages)}, "
            f"model_stages={model_stages})"
        )
        for warning in plan.warnings:
            typer.echo(f"warning: {warning}")
        typer.echo(
            "review the components, coverage and stages; then run director "
            "first-frame-approve"
        )

    _handle(ctx, operation)


@director_app.command("first-frame-approve")
def director_first_frame_approve_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Reviewed first_frame_plan.json draft"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output approved first_frame_plan.json path"
    ),
) -> None:
    """Explicitly approve a reviewed first-frame plan for execution."""

    def operation() -> None:
        approved = approve_first_frame_plan(load_json_object(plan))
        dump_json_atomic(
            output,
            approved.model_dump(mode="json"),
            sort_keys=True,
        )
        typer.echo(
            f"ok: approved first-frame plan {approved.plan_id}; wrote {output}"
        )

    _handle(ctx, operation)


@director_app.command("segment-plan")
def director_segment_plan_cmd(
    ctx: typer.Context,
    shot_plan: Path = typer.Option(
        ..., "--shot-plan", help="Path to reviewed shot_plan.json"
    ),
    shot_id: str = typer.Option(..., "--shot-id", help="Editorial shot to split"),
    first_frame_plan: Path = typer.Option(
        ...,
        "--first-frame-plan",
        help="Path to approved first-frame-plan-v1 JSON",
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output generation_segment_plan.json path"
    ),
    boundaries: Path | None = typer.Option(
        None,
        "--boundaries",
        help=(
            "Optional segment-boundary-intents-v1 JSON. Without it the "
            "planner emits one minimal start-to-end generation segment."
        ),
    ),
) -> None:
    """Build a reviewable small-shot plan with shared boundary anchors."""

    def operation() -> None:
        document = load_shot_plan(shot_plan)
        shot = next((item for item in document.shots if item.shot_id == shot_id), None)
        if shot is None:
            raise InputValidationError(
                f"shot_id not found in shot plan: {shot_id}", actual=shot_id
            )
        intent_data = load_json_object(boundaries) if boundaries is not None else None
        plan = build_generation_segment_plan(
            shot,
            first_frame_plan=load_json_object(first_frame_plan),
            boundary_intents=intent_data,
        )
        dump_json_atomic(output, plan.model_dump(mode="json"), sort_keys=True)
        typer.echo(
            f"ok: wrote draft {output} "
            f"(decision={plan.decision}, anchors={len(plan.anchors)}, "
            f"segments={len(plan.segments)})"
        )
        for warning in plan.warnings:
            typer.echo(f"warning: {warning}")
        typer.echo(
            "review shared anchors and split reasons; then run director segment-approve"
        )

    _handle(ctx, operation)


@director_app.command("segment-approve")
def director_segment_approve_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Reviewed generation_segment_plan.json draft"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output approved generation segment plan"
    ),
) -> None:
    """Explicitly approve shared anchors before any frame generation."""

    def operation() -> None:
        approved = approve_generation_segment_plan(load_json_object(plan))
        dump_json_atomic(output, approved.model_dump(mode="json"), sort_keys=True)
        typer.echo(
            f"ok: approved generation segment plan {approved.plan_id}; wrote {output}"
        )

    _handle(ctx, operation)


@director_app.command("first-frame-compose")
def director_first_frame_compose_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Approved first-frame-plan-v1 JSON"
    ),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Catalog containing the exact selected first-frame assets",
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="New or resumable first-frame run directory"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Stable id for resume safety"),
    executor_name: str = typer.Option(
        "stub",
        "--executor",
        help=(
            "stub (offline), dashscope (paid qwen-image-3.0) or "
            "review-only (approve completed outputs without new model calls)"
        ),
    ),
    size: str = typer.Option(
        "1280*720", "--size", help="Frozen qwen-image-3.0 output size"
    ),
    seed: int = typer.Option(0, "--seed", help="Frozen image-generation seed"),
    approve_stage: list[str] | None = typer.Option(
        None,
        "--approve-stage",
        help="Completed stage id accepted in this resume; repeatable",
    ),
    max_new_stages: int = typer.Option(
        1,
        "--max-new-stages",
        help="Maximum new paid-capable model stages in this invocation",
    ),
    confirm_paid: bool = typer.Option(
        False,
        "--confirm-paid",
        help="Required with --executor dashscope; acknowledges the requested calls",
    ),
) -> None:
    """Execute or resume staged canonical first-frame fusion."""

    def operation() -> None:
        adapter, executor = _resolve_generation_executor(
            executor_name=executor_name,
            confirm_paid=confirm_paid,
            seed=seed,
            size=size,
        )
        plan_doc = parse_first_frame_plan(load_json_object(plan))
        catalog = load_image_assets(image_assets)
        missing = sorted(set(plan_doc.selected_asset_ids) - set(catalog.ids))
        if missing:
            raise InputValidationError(
                "selected first-frame assets are missing from the catalog: "
                + ", ".join(missing)
            )
        result = run_first_frame_composition(
            run_dir=run_dir,
            run_id=run_id,
            plan=plan_doc,
            asset_map={
                asset_id: catalog.get(asset_id).resolved_path
                for asset_id in plan_doc.selected_asset_ids
            },
            adapter=adapter,
            executor=executor,
            approved_stage_ids=set(approve_stage or ()),
            auto_approve=False,
            max_new_model_stages=max_new_stages,
        )
        typer.echo(
            f"ok: first-frame run {result.run_id} status={result.status}; "
            f"completed={len(result.completed_stage_ids)}; "
            f"next={result.next_stage_id or '-'}"
        )
        if result.final_frame_path is not None:
            typer.echo(f"final frame: {result.final_frame_path}")
        typer.echo(f"manifest: {result.manifest_path}")
        if result.status in {"awaiting_review", "paused_limit"}:
            typer.echo(
                "review the stage output, then resume with "
                "--approve-stage <id> to accept it (or use --executor "
                "review-only to approve without new model calls)"
            )

    _handle(ctx, operation)


@director_app.command("handoff-compose")
def director_handoff_compose_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Approved generation-segment-plan-v1 JSON"
    ),
    first_frame: Path = typer.Option(
        ...,
        "--first-frame",
        help="Approved canonical first-frame image path (anchor 1)",
    ),
    image_assets: Path = typer.Option(
        ...,
        "--image-assets",
        help="Catalog containing the exact handoff reference assets",
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="New or resumable handoff-frame run directory"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Stable id for resume safety"),
    executor_name: str = typer.Option(
        "stub",
        "--executor",
        help=(
            "stub (offline), dashscope (paid qwen-image-3.0) or "
            "review-only (approve completed outputs without new model calls)"
        ),
    ),
    size: str = typer.Option(
        "1280*720", "--size", help="Frozen qwen-image-3.0 output size"
    ),
    seed: int = typer.Option(0, "--seed", help="Frozen image-generation seed"),
    approve_anchor: list[str] | None = typer.Option(
        None,
        "--approve-anchor",
        help="Completed anchor id accepted in this resume; repeatable",
    ),
    max_new_frames: int = typer.Option(
        1,
        "--max-new-frames",
        help=(
            "Maximum new paid-capable model frames in this invocation; "
            "0 prepares reusable anchors and pauses before a model call"
        ),
    ),
    confirm_paid: bool = typer.Option(
        False,
        "--confirm-paid",
        help="Required with --executor dashscope; acknowledges the requested calls",
    ),
) -> None:
    """Execute or resume shared handoff anchors from the approved first frame."""

    def operation() -> None:
        adapter, executor = _resolve_generation_executor(
            executor_name=executor_name,
            confirm_paid=confirm_paid,
            seed=seed,
            size=size,
        )
        plan_doc = parse_generation_segment_plan(load_json_object(plan))
        catalog = load_image_assets(image_assets)
        required = {
            anchor.reference_asset_id
            for anchor in plan_doc.anchors[1:]
            if anchor.reference_asset_id is not None
        }
        missing = sorted(required - set(catalog.ids))
        if missing:
            raise InputValidationError(
                "handoff reference assets are missing from the catalog: "
                + ", ".join(missing)
            )
        result = run_handoff_frame_composition(
            run_dir=run_dir,
            run_id=run_id,
            plan=plan_doc,
            first_frame_path=first_frame,
            asset_map={
                asset_id: catalog.get(asset_id).resolved_path for asset_id in required
            },
            adapter=adapter,
            executor=executor,
            approved_anchor_ids=set(approve_anchor or ()),
            auto_approve=False,
            max_new_model_frames=max_new_frames,
        )
        typer.echo(
            f"ok: handoff run {result.run_id} status={result.status}; "
            f"completed={len(result.completed_anchor_ids)}; "
            f"next={result.next_anchor_id or '-'}"
        )
        if result.final_frame_path is not None:
            typer.echo(f"final frame: {result.final_frame_path}")
        typer.echo(f"manifest: {result.manifest_path}")
        if result.status in {"awaiting_review", "paused_limit"}:
            typer.echo(
                "review the anchor output, then resume with "
                "--approve-anchor <id> to accept it (or use --executor "
                "review-only to approve without new model calls)"
            )

    _handle(ctx, operation)


@app.command("shot-generate")
def shot_generate_cmd(
    ctx: typer.Context,
    plan: Path = typer.Option(
        ..., "--plan", help="Approved generation-segment-plan-v1 JSON"
    ),
    anchors: Path = typer.Option(
        ...,
        "--anchors",
        help="Recoverable handoff-frame-compose-run-v1 manifest",
    ),
    inputs: Path = typer.Option(
        ...,
        "--inputs",
        help="generated-shot-inputs-v1 with approved raw segment videos",
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", help="New or resumable GeneratedShot run directory"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Stable id for resume safety"),
    max_new_normalizations: int = typer.Option(
        1,
        "--max-new-normalizations",
        help="Maximum new local segment normalizations in this invocation",
    ),
    retry_segment: list[str] | None = typer.Option(
        None,
        "--retry-segment",
        help="Failed local normalization segment explicitly allowed to retry",
    ),
) -> None:
    """Import approved Vidu outputs and build one recoverable GeneratedShot."""

    def operation() -> None:
        result = run_generated_shot_pipeline(
            run_dir=run_dir,
            run_id=run_id,
            plan=parse_generation_segment_plan(load_json_object(plan)),
            anchor_manifest_path=anchors,
            inputs=parse_generated_shot_inputs(load_json_object(inputs)),
            inputs_base_dir=inputs.resolve().parent,
            max_new_normalizations=max_new_normalizations,
            retry_failed_segment_ids=set(retry_segment or ()),
        )
        typer.echo(
            f"ok: GeneratedShot run {result.run_id} status={result.status}; "
            f"completed={len(result.completed_segment_ids)}; "
            f"next={result.next_segment_id or '-'}"
        )
        if result.shot_video_path is not None:
            typer.echo(f"generated shot: {result.shot_video_path}")
        typer.echo(f"manifest: {result.manifest_path}")
        if result.status == "awaiting_anchors":
            typer.echo("resume after the next shared anchor is approved")
        elif result.status == "awaiting_video":
            typer.echo("add the next approved raw video to --inputs and resume")
        elif result.status == "paused_limit":
            typer.echo("resume to normalize the next registered segment")
        elif result.status == "failed":
            typer.echo("inspect the manifest, then explicitly use --retry-segment")

    _handle(ctx, operation)


@assets_app.command("register")
def assets_register_cmd(
    ctx: typer.Context,
    catalog: Path = typer.Option(
        ...,
        "--catalog",
        help="Path to image_assets.json (created when missing)",
    ),
    paths: list[Path] | None = typer.Option(
        None,
        "--paths",
        help="Explicit image file(s); repeatable",
    ),
    asset_type: str = typer.Option(
        ...,
        "--type",
        help="character|background|foreground|prop|style",
    ),
    roles: list[str] | None = typer.Option(
        None,
        "--roles",
        help="Reference role(s); repeatable",
    ),
    note: str | None = typer.Option(
        None,
        "--note",
        help="quality_notes text for the registered assets",
    ),
    dir_path: Path | None = typer.Option(
        None,
        "--dir",
        help="Base dir: its top-level PNG/JPEG files are registered and "
        "relative --paths resolve here (no recursion)",
    ),
    rights: str = typer.Option(
        "user-owned",
        "--rights",
        help="rights_status claim recorded for the new assets",
    ),
) -> None:
    """Register explicit PNG/JPEG files as canonical assets (SHA-256 dedup)."""

    def operation() -> None:
        data, new_entries, duplicates = _register_assets(
            catalog=catalog,
            paths=paths,
            dir_path=dir_path,
            asset_type=asset_type,
            roles=roles,
            note=note,
            rights=rights,
            tier="canonical",
            generated_from=None,
        )
        if not new_entries:
            typer.echo(
                f"ok: no new assets; {len(duplicates)} duplicate(s) skipped"
            )
            for dup in duplicates:
                typer.echo(
                    f"duplicate skipped: {dup['path']} already registered "
                    f"as {dup['asset_id']}"
                )
            return
        original = copy.deepcopy(data) if catalog.exists() else None
        data["assets"] = data["assets"] + new_entries
        _write_and_verify_catalog(catalog, data, original)
        ids = ", ".join(entry["asset_id"] for entry in new_entries)
        typer.echo(
            f"ok: registered {len(new_entries)} canonical asset(s) into "
            f"{catalog}"
        )
        for dup in duplicates:
            typer.echo(
                f"duplicate skipped: {dup['path']} already registered "
                f"as {dup['asset_id']}"
            )
        typer.echo(f"asset ids: {ids}")

    _handle(ctx, operation)


@assets_app.command("register-candidate")
def assets_register_candidate_cmd(
    ctx: typer.Context,
    catalog: Path = typer.Option(
        ...,
        "--catalog",
        help="Path to image_assets.json",
    ),
    paths: list[Path] | None = typer.Option(
        None,
        "--paths",
        help="Explicit image file(s); repeatable",
    ),
    asset_type: str | None = typer.Option(
        None,
        "--type",
        help="Overrides the parent's asset_type",
    ),
    dir_path: Path | None = typer.Option(
        None,
        "--dir",
        help="Base dir: its top-level PNG/JPEG files are registered and "
        "relative --paths resolve here (no recursion)",
    ),
    generated_from: str | None = typer.Option(
        None,
        "--generated-from",
        help="Parent asset_id this candidate was generated from",
    ),
    rights: str = typer.Option(
        "user-owned",
        "--rights",
        help="rights_status claim recorded for the new assets",
    ),
    note: str | None = typer.Option(
        None,
        "--note",
        help="provenance.note text for the new candidate (lineage summary)",
    ),
) -> None:
    """Register model-generated images as generated_candidate assets."""

    def operation() -> None:
        data, new_entries, duplicates = _register_assets(
            catalog=catalog,
            paths=paths,
            dir_path=dir_path,
            asset_type=asset_type,
            roles=None,
            note=note,
            rights=rights,
            tier="generated_candidate",
            generated_from=generated_from,
        )
        if not new_entries:
            typer.echo(
                f"ok: no new assets; {len(duplicates)} duplicate(s) skipped"
            )
            for dup in duplicates:
                typer.echo(
                    f"duplicate skipped: {dup['path']} already registered "
                    f"as {dup['asset_id']}"
                )
            return
        original = copy.deepcopy(data) if catalog.exists() else None
        data["assets"] = data["assets"] + new_entries
        _write_and_verify_catalog(catalog, data, original)
        ids = ", ".join(entry["asset_id"] for entry in new_entries)
        parent_note = (
            f" (generated from {generated_from})" if generated_from else ""
        )
        typer.echo(
            f"ok: registered {len(new_entries)} generated_candidate "
            f"asset(s) into {catalog}{parent_note}"
        )
        for dup in duplicates:
            typer.echo(
                f"duplicate skipped: {dup['path']} already registered "
                f"as {dup['asset_id']}"
            )
        typer.echo(f"asset ids: {ids}")

    _handle(ctx, operation)


@assets_app.command("promote")
def assets_promote_cmd(
    ctx: typer.Context,
    catalog: Path = typer.Option(
        ...,
        "--catalog",
        help="Path to image_assets.json",
    ),
    asset_id: str = typer.Option(
        ...,
        "--asset-id",
        help="generated_candidate asset_id to promote",
    ),
) -> None:
    """Promote a generated_candidate asset to approved_generated."""

    def operation() -> None:
        if not catalog.exists():
            raise InputValidationError(
                f"catalog does not exist: {catalog}",
                actual=catalog,
            )
        data = load_json_object(catalog)
        try:
            TypeAdapter(ImageAssetsDocument).validate_python(data)
        except (ValidationError, TypeError) as exc:
            raise InputValidationError(
                f"invalid image_assets.json catalog: {exc}",
                actual=catalog,
            ) from exc
        entries = data["assets"]
        target = next(
            (entry for entry in entries if entry.get("asset_id") == asset_id),
            None,
        )
        if target is None:
            raise InputValidationError(
                f"asset not found in catalog: {asset_id}",
                asset_id=asset_id,
            )
        tier = target.get("source_tier", "canonical")
        if tier != "generated_candidate":
            raise InputValidationError(
                f"only generated_candidate assets can be promoted; "
                f"{asset_id} is {tier!r}",
                asset_id=asset_id,
            )
        provenance = target.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            target["provenance"] = provenance
        provenance["note"] = "promoted"
        target["source_tier"] = "approved_generated"
        try:
            TypeAdapter(ImageAssetsDocument).validate_python(data)
        except (ValidationError, TypeError) as exc:
            raise InputValidationError(
                f"invalid catalog after promotion: {exc}",
                asset_id=asset_id,
            ) from exc
        dump_json_atomic(catalog, data, sort_keys=True)
        typer.echo(
            f"ok: promoted {asset_id} "
            f"(generated_candidate -> approved_generated)"
        )

    _handle(ctx, operation)


@episode_app.command("extract")
def episode_assets_extract_cmd(
    ctx: typer.Context,
    video: Path = typer.Option(
        ..., "--video", help="Path to one authorized anime episode video"
    ),
    output: Path = typer.Option(
        ..., "--output", help="New output directory for frames, catalog and manifest"
    ),
    title: str | None = typer.Option(
        None, "--title", help="Episode title used for asset ids"
    ),
    max_frames: int = typer.Option(
        40,
        "--max-frames",
        min=1,
        help="Maximum sampled frames",
    ),
    scene_threshold: float = typer.Option(
        0.30,
        "--scene-threshold",
        min=0.0,
        max=1.0,
        help="Scene-change threshold (0..1)",
    ),
    min_gap: float = typer.Option(
        1.0,
        "--min-gap",
        min=0.0,
        help="Minimum seconds between sampled frames",
    ),
    executor_name: str = typer.Option(
        "stub",
        "--executor",
        help=(
            "stub (offline placeholder labels) or dashscope "
            "(paid qwen-vl classification)"
        ),
    ),
    confirm_paid: bool = typer.Option(
        False,
        "--confirm-paid",
        help=(
            "Required when --executor dashscope or --crop dashscope; "
            "acknowledges paid classification/cropping"
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Only probe the video and print the planned frame timestamps; "
            "no extraction, no paid calls"
        ),
    ),
    crop_mode: str = typer.Option(
        "none",
        "--crop",
        help=(
            "none, stub (offline center crop) or dashscope "
            "(paid face/character boxes)"
        ),
    ),
    rights: str = typer.Option(
        "user-owned episode frame; private experiment, no redistribution",
        "--rights",
        help="rights_status recorded for the extracted assets",
    ),
) -> None:
    """Extract deduplicated reference frames and register them as assets."""

    def operation() -> None:
        if dry_run:
            from anime_remix.services.episode_assets.sampler import plan_timestamps

            duration, timestamps = plan_timestamps(
                video,
                max_frames=max_frames,
                scene_threshold=scene_threshold,
                min_gap_seconds=min_gap,
            )
            typer.echo(
                f"ok: dry-run duration={duration:.3f}s "
                f"planned_frames={len(timestamps)}"
            )
            for index, timestamp in enumerate(timestamps, start=1):
                typer.echo(f"  frame {index:03d}: t={timestamp:.3f}s")
            return
        normalized = executor_name.strip().casefold()
        if normalized not in {"stub", "dashscope"}:
            raise InputValidationError(
                "--executor must be 'stub' or 'dashscope'",
                actual=executor_name,
            )
        normalized_crop = crop_mode.strip().casefold()
        if normalized_crop not in {"none", "stub", "dashscope"}:
            raise InputValidationError(
                "--crop must be 'none', 'stub' or 'dashscope'",
                actual=crop_mode,
            )
        paid_used = normalized == "dashscope" or normalized_crop == "dashscope"
        if paid_used and not confirm_paid:
            raise InputValidationError(
                "DashScope classification/cropping is paid-capable; rerun with "
                "--confirm-paid after confirming the sampled frame count"
            )
        if not paid_used and confirm_paid:
            raise InputValidationError(
                "--confirm-paid is only valid with --executor dashscope"
            )
        classifier = (
            DashScopeEpisodeClassifier()
            if normalized == "dashscope"
            else StubEpisodeClassifier()
        )
        cropper = (
            DashScopeCropper()
            if normalized_crop == "dashscope"
            else (StubCropper() if normalized_crop == "stub" else None)
        )
        result = extract_episode_assets(
            video=video,
            output_dir=output,
            title=title,
            max_frames=max_frames,
            scene_threshold=scene_threshold,
            min_gap_seconds=min_gap,
            classifier=classifier,
            cropper=cropper,
            rights_status=rights,
        )
        typer.echo(
            f"ok: extracted {result.extracted_frame_count} frames, "
            f"{result.unique_frame_count} unique, {result.crop_count} crops"
        )
        typer.echo(f"catalog: {result.catalog_path}")
        typer.echo(f"manifest: {result.manifest_path}")
        if normalized == "dashscope":
            typer.echo(
                "vision classification recorded; review labels and asset types "
                "before keyframe use"
            )

    _handle(ctx, operation)


@episode_app.command("review-sheet")
def episode_assets_review_sheet_cmd(
    ctx: typer.Context,
    catalog: Path = typer.Option(
        ..., "--catalog", help="Path to an image_assets.json catalog"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output CSV review worksheet path"
    ),
) -> None:
    """Generate an editable CSV review worksheet from a catalog."""

    def operation() -> None:
        write_review_sheet(catalog=catalog, output=output)
        typer.echo(f"ok: wrote review worksheet {output}")
        typer.echo(
            "edit decision/keep|revise|reject, corrected_asset_type, "
            "corrected_subject, corrected_roles (semicolon-separated) and "
            "review_notes; then run episode-assets apply-review"
        )

    _handle(ctx, operation)


@episode_app.command("apply-review")
def episode_assets_apply_review_cmd(
    ctx: typer.Context,
    catalog: Path = typer.Option(
        ..., "--catalog", help="Path to the source image_assets.json catalog"
    ),
    worksheet: Path = typer.Option(
        ..., "--worksheet", help="Path to the edited CSV review worksheet"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Output corrected image_assets.json catalog"
    ),
    review_record: Path = typer.Option(
        ...,
        "--review-record",
        help="Output review record JSON (decisions and notes)",
    ),
) -> None:
    """Apply reviewed decisions and write the corrected catalog."""

    def operation() -> None:
        result = apply_review(
            catalog=catalog,
            worksheet=worksheet,
            output_catalog=output,
            review_record=review_record,
        )
        typer.echo(
            f"ok: kept={result.kept_count} revised={result.revised_count} "
            f"rejected={result.rejected_count}"
        )
        typer.echo(f"corrected catalog: {result.catalog_path}")
        typer.echo(f"review record: {result.review_record_path}")

    _handle(ctx, operation)


@app.command("build")
def build_cmd(
    ctx: typer.Context,
    script: Path = typer.Option(..., "--script", help="Path to script.md"),
    clips: Path = typer.Option(..., "--clips", help="Path to clips.json"),
    aliases: Path | None = typer.Option(
        None,
        "--aliases",
        help="Path to aliases.json (optional; extends rule parser dictionaries).",
    ),
    output: Path = typer.Option(..., "--output", help="New run directory"),
    parser: str = typer.Option(
        "rule",
        "--parser",
        help="Parser strategy (MVR supports only 'rule').",
    ),
) -> None:
    """Build a complete run: parse, retrieve, render, and publish."""

    def operation() -> None:
        result = build(
            script_path=script,
            clips_path=clips,
            aliases_path=aliases,
            output_dir=output,
            parser=parser,
        )
        typer.echo(f"ok: published {result}")

    _handle(ctx, operation)


if __name__ == "__main__":
    app()
