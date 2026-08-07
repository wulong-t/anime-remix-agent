"""End-to-end aliases workflow and CLI tests (AGENTS.md 18.7)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from anime_remix.adapters.ffmpeg import FFmpegToolkit
from anime_remix.cli import app
from anime_remix.domain.models import RenderProfile
from anime_remix.errors import InputValidationError
from anime_remix.json_io import sha256_file
from anime_remix.workflows.build_workflow import build
from anime_remix.workflows.render_workflow import render_timeline

runner = CliRunner()


def _encode_clip(ffmpeg: str, output: Path, *, frames: int) -> None:
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
        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709:field_mode=prog",
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
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-f",
        "mp4",
        str(output),
    ]
    subprocess.run(args, check=True, shell=False)


@pytest.fixture(scope="module")
def aliases_demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("aliases-demo")
    clips_dir = root / "clips"
    clips_dir.mkdir()
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg must be installed for media tests"
    specs = [
        ("clip_001", 96, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "独自站立", "林夏独自站在学校天台。"),
        ("clip_002", 120, "char_lu_chen", "陆辰", "loc_classroom", "教室", "沉默注视", "陆辰在教室沉默注视窗外。"),
        ("clip_003", 72, "char_lin_xia", "林夏", "loc_school_rooftop", "学校天台", "转身离开", "林夏在天台转身离开。"),
    ]
    clips = []
    for clip_id, frames, char_id, name, loc_id, location, action, description in specs:
        path = clips_dir / f"{clip_id}.mp4"
        _encode_clip(ffmpeg, path, frames=frames)
        clips.append(
            {
                "id": clip_id,
                "path": f"clips/{clip_id}.mp4",
                "characters": [{"id": char_id, "name": name}],
                "location_id": loc_id,
                "location_name": location,
                "action": action,
                "description": description,
            }
        )
    script = root / "script.md"
    script.write_text(
        "“天台的风很大。”小夏独自站在学校楼顶，望着远方的城市。\n\n"
        "阿辰在课堂上沉默注视窗外，没有说一句话。\n\n"
        "“走吧，要上课了。”小夏转身离开学校楼顶，阿辰跟在身后。",
        encoding="utf-8",
    )
    clips_path = root / "clips.json"
    clips_path.write_text(
        json.dumps(
            {"schema_version": "1.9", "clips": clips},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    aliases_path = root / "aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                "schema_version": "1.9",
                "character_aliases": [
                    {"target_id": "char_lin_xia", "aliases": ["小夏"]},
                    {"target_id": "char_lu_chen", "aliases": ["阿辰"]},
                ],
                "location_aliases": [
                    {"target_id": "loc_school_rooftop", "aliases": ["楼顶"]},
                    {"target_id": "loc_classroom", "aliases": ["课堂"]},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "script": script,
        "clips": clips_path,
        "aliases": aliases_path,
    }


class TestCli:
    def test_validate_aliases_static_success(
        self,
        aliases_demo: dict[str, Path],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "validate",
                "--script",
                str(aliases_demo["script"]),
                "--clips",
                str(aliases_demo["clips"]),
                "--aliases",
                str(aliases_demo["aliases"]),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ok: static validation passed" in result.output

    def test_validate_aliases_target_error_exits_2(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad-aliases.json"
        bad.write_text(
            json.dumps(
                {
                    "schema_version": "1.9",
                    "character_aliases": [
                        {"target_id": "char_nobody", "aliases": ["某人"]}
                    ],
                    "location_aliases": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "validate",
                "--script",
                str(aliases_demo["script"]),
                "--clips",
                str(aliases_demo["clips"]),
                "--aliases",
                str(bad),
            ],
        )
        assert result.exit_code == 2

    def test_render_help_has_no_aliases(self) -> None:
        result = runner.invoke(app, ["render", "--help"])
        assert result.exit_code == 0
        assert "--aliases" not in result.output


class TestBuildWorkflow:
    def test_build_with_aliases_success_and_canonical_output(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "runs" / "aliases-001"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_demo["aliases"],
            output_dir=target,
        )
        assert target.exists()
        parsed = json.loads(
            (target / "parsed_script.json").read_text(encoding="utf-8")
        )
        shot = parsed["shots"][0]
        assert shot["characters"][0] == {
            "id": "char_lin_xia",
            "name": "林夏",
        }
        assert shot["location_id"] == "loc_school_rooftop"
        assert shot["location_name"] == "学校天台"
        timeline = json.loads(
            (target / "timeline.json").read_text(encoding="utf-8")
        )
        requirement = timeline["items"][0]["requirement"]
        assert requirement["characters"][0] == {
            "id": "char_lin_xia",
            "name": "林夏",
        }
        assert requirement["location_id"] == "loc_school_rooftop"
        assert requirement["location_name"] == "学校天台"
        # Alias strings must not leak into identity fields.
        assert "小夏" not in {
            requirement["characters"][0]["id"],
            requirement["characters"][0]["name"],
            requirement["location_id"],
            requirement["location_name"],
        }

    def test_aliases_target_error_creates_no_staging(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad-aliases.json"
        bad.write_text(
            json.dumps(
                {
                    "schema_version": "1.9",
                    "character_aliases": [
                        {"target_id": "char_nobody", "aliases": ["某人"]}
                    ],
                    "location_aliases": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        target = tmp_path / "runs" / "aliases-fail"
        with pytest.raises(InputValidationError):
            build(
                script_path=aliases_demo["script"],
                clips_path=aliases_demo["clips"],
                aliases_path=bad,
                output_dir=target,
            )
        assert not target.exists()
        assert not (tmp_path / "runs" / ".aliases-fail.staging").exists()

    def test_manifest_aliases_sha256_is_real_file_hash(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "runs" / "aliases-hash"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_demo["aliases"],
            output_dir=target,
        )
        manifest = json.loads(
            (target / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["aliases_sha256"] == sha256_file(
            aliases_demo["aliases"]
        )

    def test_manifest_aliases_sha256_null_without_aliases(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "runs" / "no-aliases"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            output_dir=target,
        )
        manifest = json.loads(
            (target / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["aliases_sha256"] is None

    def test_aliases_input_order_byte_identical_core_artifacts(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        aliases_reversed = tmp_path / "aliases-reversed.json"
        payload = json.loads(
            aliases_demo["aliases"].read_text(encoding="utf-8")
        )
        payload["character_aliases"] = list(
            reversed(payload["character_aliases"])
        )
        payload["location_aliases"] = list(reversed(payload["location_aliases"]))
        aliases_reversed.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        first = tmp_path / "runs" / "order-a"
        second = tmp_path / "runs" / "order-b"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_demo["aliases"],
            output_dir=first,
        )
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_reversed,
            output_dir=second,
        )
        for name in (
            "parsed_script.json",
            "retrieval_results.json",
            "timeline.json",
        ):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name
        first_manifest = json.loads(
            (first / "run_manifest.json").read_text(encoding="utf-8")
        )
        second_manifest = json.loads(
            (second / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert first_manifest["core_artifact_sha256"] == second_manifest[
            "core_artifact_sha256"
        ]

    def test_independent_rerender_without_script_clips_aliases_retrieval(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "runs" / "aliases-rerender"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_demo["aliases"],
            output_dir=target,
        )
        timeline = json.loads(
            (target / "timeline.json").read_text(encoding="utf-8")
        )
        total_frames = sum(item["target_frames"] for item in timeline["items"])
        backups = [
            aliases_demo["script"].with_suffix(".bak"),
            aliases_demo["clips"].with_suffix(".bak"),
            aliases_demo["aliases"].with_suffix(".bak"),
            target / "retrieval_results.json.bak",
        ]
        try:
            aliases_demo["script"].rename(backups[0])
            aliases_demo["clips"].rename(backups[1])
            aliases_demo["aliases"].rename(backups[2])
            (target / "retrieval_results.json").rename(backups[3])
            rerendered = tmp_path / "aliases-rerendered.mp4"
            render_timeline(
                target / "timeline.json",
                rerendered,
                toolkit=FFmpegToolkit(),
            )
            assert rerendered.exists()
            FFmpegToolkit().validate_final(
                rerendered,
                total_frames=total_frames,
                profile=RenderProfile(),
            )
        finally:
            backups[0].rename(aliases_demo["script"])
            backups[1].rename(aliases_demo["clips"])
            backups[2].rename(aliases_demo["aliases"])
            backups[3].rename(target / "retrieval_results.json")

    def test_build_aliases_output_media_regression(
        self,
        aliases_demo: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "runs" / "aliases-media"
        build(
            script_path=aliases_demo["script"],
            clips_path=aliases_demo["clips"],
            aliases_path=aliases_demo["aliases"],
            output_dir=target,
        )
        timeline = json.loads(
            (target / "timeline.json").read_text(encoding="utf-8")
        )
        total_frames = sum(item["target_frames"] for item in timeline["items"])
        FFmpegToolkit().validate_final(
            target / "output.mp4",
            total_frames=total_frames,
            profile=RenderProfile(),
        )
