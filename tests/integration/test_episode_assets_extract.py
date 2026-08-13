"""End-to-end tests for episode asset extraction (offline stub path)."""

from __future__ import annotations

import csv
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.cli import app
from anime_remix.services.episode_assets import (
    StubCropper,
    StubEpisodeClassifier,
    extract_episode_assets,
)
from anime_remix.services.image_assets import load_image_assets

runner = CliRunner()


def _png(colour: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, 180), colour).save(output, format="PNG")
    return output.getvalue()


def _make_video(tmp_path: Path) -> Path:
    """Build a 1s 12fps video with three solid-color scenes."""

    frame_dir = tmp_path / "source-frames"
    frame_dir.mkdir()
    scenes = [(210, 30, 30), (30, 180, 40), (30, 50, 200)]
    for index in range(12):
        (frame_dir / f"frame_{index + 1:02d}.png").write_bytes(
            _png(scenes[index // 4])
        )
    video = tmp_path / "episode.mp4"
    toolkit = FFmpegToolkit()
    toolkit.run(
        [
            toolkit.ffmpeg or "ffmpeg",
            "-y",
            "-framerate",
            "12",
            "-i",
            str(frame_dir / "frame_%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        timeout=120,
        stage="test_video_encode",
    )
    return video


def test_extract_episode_assets_dedups_and_writes_valid_catalog(
    tmp_path: Path,
) -> None:
    video = _make_video(tmp_path)
    out = tmp_path / "out"
    result = extract_episode_assets(
        video=video,
        output_dir=out,
        title="ep01",
        max_frames=12,
        scene_threshold=0.2,
        min_gap_seconds=0.05,
        classifier=StubEpisodeClassifier(),
    )

    assert result.extracted_frame_count >= 3
    assert result.unique_frame_count == 3
    assert result.catalog_path.is_file()
    assert result.manifest_path.is_file()

    catalog = load_image_assets(result.catalog_path)
    assert len(catalog) == 3
    for record in catalog:
        assert record.source_tier == "derived"
        assert record.rights_status.startswith("user-owned episode frame")
        assert record.analysis_status == "analyzed"
        assert record.resolved_path.is_file()
        assert record.reference_roles == ("scene_reference",)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "episode-assets-extract-v1"
    assert manifest["unique_frame_count"] == 3
    assert manifest["video"]["sha256"] == result.video_sha256
    reps = {
        frame["dedup_representative_sha256"]
        for frame in manifest["frames"]
    }
    assert len(reps) == 3
    for frame in manifest["frames"]:
        assert frame["timestamp_seconds"] >= 0
        if frame["dedup_representative_sha256"] == frame["sha256"]:
            assert frame["asset_id"] is not None
            assert frame["classification_request_id"] == "stub-classify"


def test_cli_extract_offline_and_paid_flag_guards(tmp_path: Path) -> None:
    video = _make_video(tmp_path)
    out = tmp_path / "cli-out"
    ok = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(out),
            "--title",
            "ep01",
            "--executor",
            "stub",
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "3 unique" in ok.output
    assert (out / "image_assets.json").is_file()
    assert (out / "episode_assets_manifest.json").is_file()

    paid_without_confirm = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(tmp_path / "paid-out"),
            "--executor",
            "dashscope",
        ],
    )
    assert paid_without_confirm.exit_code != 0
    assert "paid-capable" in paid_without_confirm.output

    confirm_with_stub = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(tmp_path / "stub-paid"),
            "--executor",
            "stub",
            "--confirm-paid",
        ],
    )
    assert confirm_with_stub.exit_code != 0
    assert "only valid with --executor dashscope" in confirm_with_stub.output


def test_extract_with_stub_cropper_registers_child_assets(
    tmp_path: Path,
) -> None:
    video = _make_video(tmp_path)
    out = tmp_path / "crop-out"
    result = extract_episode_assets(
        video=video,
        output_dir=out,
        title="ep01",
        max_frames=12,
        scene_threshold=0.2,
        min_gap_seconds=0.05,
        classifier=StubEpisodeClassifier(asset_type="character"),
        cropper=StubCropper(),
    )

    assert result.unique_frame_count == 3
    assert result.crop_count == 3
    catalog = load_image_assets(result.catalog_path)
    assert len(catalog) == 6
    frames = [record for record in catalog if "_crop" not in record.asset_id]
    crops = [record for record in catalog if "_crop" in record.asset_id]
    assert len(frames) == 3
    assert len(crops) == 3
    for crop in crops:
        assert crop.asset_type == "character"
        assert crop.source_tier == "derived"
        assert "identity_reference" in crop.reference_roles
        assert crop.provenance is not None
        assert crop.provenance.get("parent_asset_id") is not None
        assert crop.resolved_path.is_file()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["crop_count"] == 3
    assert len(manifest["crops"]) == 3
    assert all(crop["request_id"] == "stub-crop" for crop in manifest["crops"])

    cli = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(tmp_path / "crop-cli"),
            "--crop",
            "stub",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert "3 crops" in cli.output

    crop_paid = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(tmp_path / "crop-paid"),
            "--crop",
            "dashscope",
        ],
    )
    assert crop_paid.exit_code != 0
    assert "paid-capable" in crop_paid.output


def test_cli_dry_run_prints_plan_without_writing(
    tmp_path: Path,
) -> None:
    video = _make_video(tmp_path)
    out = tmp_path / "dry-out"
    result = runner.invoke(
        app,
        [
            "episode-assets",
            "extract",
            "--video",
            str(video),
            "--output",
            str(out),
            "--max-frames",
            "12",
            "--dry-run",
            "--executor",
            "dashscope",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "planned_frames=" in result.output
    assert "t=" in result.output
    assert not out.exists()


def test_review_sheet_and_apply_review_roundtrip(
    tmp_path: Path,
) -> None:
    video = _make_video(tmp_path)
    out = tmp_path / "rev-out"
    extract_episode_assets(
        video=video,
        output_dir=out,
        title="ep01",
        max_frames=12,
        scene_threshold=0.2,
        min_gap_seconds=0.05,
        classifier=StubEpisodeClassifier(asset_type="character"),
    )
    catalog = out / "image_assets.json"
    sheet = tmp_path / "review.csv"
    made = runner.invoke(
        app,
        [
            "episode-assets",
            "review-sheet",
            "--catalog",
            str(catalog),
            "--output",
            str(sheet),
        ],
    )
    assert made.exit_code == 0, made.output
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig", newline="")))
    assert len(rows) == 3
    rows[0]["decision"] = "reject"
    rows[0]["review_notes"] = "motion blur, unusable"
    rows[1]["decision"] = "revise"
    rows[1]["corrected_asset_type"] = "prop"
    rows[1]["corrected_subject"] = "red key"
    rows[1]["corrected_roles"] = "prop_reference"
    rows[1]["review_notes"] = "this is the prop plate"
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    applied = runner.invoke(
        app,
        [
            "episode-assets",
            "apply-review",
            "--catalog",
            str(catalog),
            "--worksheet",
            str(sheet),
            "--output",
            str(out / "image_assets.reviewed.json"),
            "--review-record",
            str(tmp_path / "asset_review.json"),
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert "kept=1 revised=1 rejected=1" in applied.output
    reviewed = load_image_assets(out / "image_assets.reviewed.json")
    assert len(reviewed) == 2
    record = json.loads(
        (tmp_path / "asset_review.json").read_text(encoding="utf-8")
    )
    decisions = {
        asset_id: item["decision"] for asset_id, item in record["reviews"].items()
    }
    assert set(decisions.values()) == {"reject", "revise", "keep"}

    with sheet.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writerow(
            {
                **{key: "" for key in rows[0]},
                "asset_id": "nope",
                "decision": "keep",
            }
        )
    bad = runner.invoke(
        app,
        [
            "episode-assets",
            "apply-review",
            "--catalog",
            str(catalog),
            "--worksheet",
            str(sheet),
            "--output",
            str(tmp_path / "bad.json"),
            "--review-record",
            str(tmp_path / "bad-review.json"),
        ],
    )
    assert bad.exit_code != 0
    assert "unknown asset_id" in bad.output
