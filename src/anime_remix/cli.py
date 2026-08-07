"""Typer CLI entry point for anime-remix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from anime_remix import __version__
from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.errors import (
    ERROR_EXIT_CODES,
    AnimeRemixError,
)
from anime_remix.services.aliases import load_aliases_document
from anime_remix.services.input_loader import (
    load_clips_document,
    load_script_text,
)
from anime_remix.workflows.build_workflow import build
from anime_remix.workflows.render_workflow import render_timeline

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Anime Remix Agent: script + clips -> editable timeline -> MP4",
)


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
