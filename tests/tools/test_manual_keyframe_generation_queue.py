"""G1-MK5-L tests: deterministic placeholder -> manual-generation queue.

The queue tool is an opt-in experimental read-only planner handoff: it
captures the explicitly passed timeline bytes exactly once, validates that
captured object with the existing ``Timeline`` Pydantic model, and atomically
publishes exactly one ``generation_queue.json``.  Tests use synthetic
timelines only and assert no source media reads, no render, no FFmpeg, no
subprocess, and no directory discovery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anime_remix.domain.models import RenderProfile
from experiments.manual_keyframe_mvp import generation_queue


def _requirement(
    shot_id: str,
    order: int,
    *,
    emotion: str | None = None,
    shot_scale: str | None = None,
    characters: list[dict[str, str]] | None = None,
    location_id: str | None = None,
    location_name: str | None = None,
) -> dict[str, object]:
    return {
        "id": shot_id,
        "order": order,
        "source_text": f"source text for {shot_id}",
        "action": f"action for {shot_id}",
        "target_frames": 72,
        "dialogue": None,
        "emotion": emotion,
        "shot_scale": shot_scale,
        "characters": characters if characters is not None else [],
        "location_id": location_id,
        "location_name": location_name,
    }


def _clip_item(shot_id: str = "shot_001", order: int = 1) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": _requirement(shot_id, order),
        "strategy": "clip",
        "source_asset_id": f"asset_{shot_id}",
        "source_path": f"clips/{shot_id}.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "a" * 64,
        "source_in_frame": 0,
        "source_frame_count": 72,
        "target_frames": 72,
        "score": None,
        "reason_code": "exact_length",
        "reason": "exact_length",
    }


def _freeze_item(shot_id: str = "shot_002", order: int = 2) -> dict[str, object]:
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": _requirement(shot_id, order),
        "strategy": "freeze_frame",
        "source_asset_id": f"asset_{shot_id}",
        "source_path": f"clips/{shot_id}.mp4",
        "source_size_bytes": 1000,
        "source_sha256": "b" * 64,
        "source_in_frame": 0,
        "source_frame_count": 24,
        "target_frames": 72,
        "score": None,
        "reason_code": "short_source_freeze",
        "reason": "short_source_freeze",
    }


def _placeholder_item(
    shot_id: str = "shot_003", order: int = 3, **overrides: object
) -> dict[str, object]:
    requirement = _requirement(shot_id, order)
    requirement.update(overrides)
    return {
        "shot_id": shot_id,
        "order": order,
        "requirement": requirement,
        "strategy": "placeholder",
        "source_asset_id": None,
        "source_path": None,
        "source_size_bytes": None,
        "source_sha256": None,
        "source_in_frame": 0,
        "source_frame_count": 0,
        "target_frames": 72,
        "score": None,
        "reason_code": "no_candidate",
        "reason": "no candidate",
    }


def _timeline_doc(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.9",
        "path_base": "timeline_dir",
        "render_profile": RenderProfile().model_dump(mode="json"),
        "items": items,
    }


def _write_timeline(
    root: Path, items: list[dict[str, object]], *, name: str = "timeline.json"
) -> Path:
    path = root / name
    path.write_text(
        json.dumps(_timeline_doc(items), indent=2), encoding="utf-8"
    )
    return path


def _staging_leftovers(parent: Path) -> list[Path]:
    return [
        path for path in parent.iterdir() if ".staging-" in path.name
    ]


def _mixed_items() -> list[dict[str, object]]:
    return [
        _clip_item("shot_001", 1),
        _freeze_item("shot_002", 2),
        _placeholder_item(
            "shot_003",
            3,
            emotion="calm",
            shot_scale="medium",
            characters=[{"id": "char_a", "name": "Mira"}],
            location_id="loc_roof",
            location_name="Rooftop",
        ),
        _clip_item("shot_004", 4),
        _placeholder_item("shot_005", 5),
    ]


def test_mixed_timeline_emits_only_placeholders_in_order(
    tmp_path: Path,
) -> None:
    items = _mixed_items()
    timeline = _write_timeline(tmp_path, items)
    before = timeline.read_bytes()
    output = tmp_path / "queue-out"

    queue = generation_queue.cmd_plan(timeline, output)

    assert set(queue) == generation_queue.QUEUE_FIELDS
    assert queue["schema_version"] == generation_queue.QUEUE_SCHEMA
    assert queue["timeline_sha256"] == generation_queue._sha256_bytes(before)
    assert queue["timeline_schema_version"] == "1.9"
    assert queue["total_timeline_items"] == 5
    assert queue["pending_count"] == 2
    assert timeline.read_bytes() == before
    assert [path.name for path in sorted(output.iterdir())] == [
        "generation_queue.json"
    ]

    first, second = queue["items"]
    for item in (first, second):
        assert set(item) == generation_queue.ITEM_FIELDS
        assert item["status"] == "needs_manual_keyframes"
        assert item["required_human_inputs"] == [
            "subject_description",
            "scene_description",
            "start_state",
            "end_state",
            "k0_png",
            "k_end_png",
            "k0_provenance",
            "k_end_provenance",
            "rights_and_visual_approval",
        ]
    assert first["shot_id"] == "shot_003"
    assert first["order"] == 3
    assert first["target_frames"] == 72
    assert first["source_text"] == "source text for shot_003"
    assert first["action"] == "action for shot_003"
    assert first["emotion"] == "calm"
    assert first["shot_scale"] == "medium"
    assert first["characters"] == [{"id": "char_a", "name": "Mira"}]
    assert first["location_id"] == "loc_roof"
    assert first["location_name"] == "Rooftop"
    assert second["shot_id"] == "shot_005"
    assert second["order"] == 5
    assert second["emotion"] is None
    assert second["shot_scale"] is None
    assert second["characters"] == []
    assert second["location_id"] is None
    assert second["location_name"] is None
    assert queue["next_action"] == {
        "action": "provide_manual_keyframes",
        "target_shot_id": "shot_003",
    }


def test_no_placeholders_emits_empty_queue_and_none(
    tmp_path: Path,
) -> None:
    timeline = _write_timeline(
        tmp_path, [_clip_item("shot_001", 1), _freeze_item("shot_002", 2)]
    )
    output = tmp_path / "queue-out"

    queue = generation_queue.cmd_plan(timeline, output)

    assert queue["pending_count"] == 0
    assert queue["items"] == []
    assert queue["next_action"] == {"action": "none", "target_shot_id": None}
    assert queue["total_timeline_items"] == 2
    assert (output / "generation_queue.json").is_file()


def test_deterministic_bytes_across_equivalent_runs(tmp_path: Path) -> None:
    timeline = _write_timeline(tmp_path, _mixed_items())
    first = tmp_path / "queue-1"
    second = tmp_path / "queue-2"

    generation_queue.cmd_plan(timeline, first)
    generation_queue.cmd_plan(timeline, second)

    assert (first / "generation_queue.json").read_bytes() == (
        second / "generation_queue.json"
    ).read_bytes()


def test_queue_binds_to_captured_timeline_bytes(tmp_path: Path) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    first_queue = generation_queue.cmd_plan(timeline, tmp_path / "q-1")

    timeline.write_text(
        json.dumps(
            _timeline_doc(
                [_placeholder_item("shot_001", 1, emotion="calm")]
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    second_queue = generation_queue.cmd_plan(timeline, tmp_path / "q-2")

    assert first_queue["timeline_sha256"] != second_queue["timeline_sha256"]
    assert second_queue["timeline_sha256"] == generation_queue._sha256_bytes(
        timeline.read_bytes()
    )


@pytest.mark.parametrize(
    ("name", "mutator"),
    [
        ("unknown top-level key", lambda doc: doc.update(sneaky=True)),
        ("future schema", lambda doc: doc.update(schema_version="2.0")),
        ("bad strategy", lambda doc: doc["items"][0].update(strategy="generate")),
        ("empty items", lambda doc: doc.update(items=[])),
        (
            "order mismatch",
            lambda doc: doc["items"][0].update(order=2),
        ),
    ],
)
def test_invalid_timeline_rejected_atomically(
    tmp_path: Path, name: str, mutator
) -> None:
    doc = _timeline_doc([_placeholder_item("shot_001", 1)])
    mutator(doc)
    timeline = tmp_path / "bad-timeline.json"
    timeline.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    output = tmp_path / "queue-out"

    with pytest.raises(generation_queue.QueueError) as exc:
        generation_queue.cmd_plan(timeline, output)

    assert exc.value.layer == "input_contract"
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []


def test_existing_output_rejected_and_left_untouched(tmp_path: Path) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    before = timeline.read_bytes()
    output = tmp_path / "queue-out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(generation_queue.QueueError, match="already exists"):
        generation_queue.cmd_plan(timeline, output)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert timeline.read_bytes() == before
    assert _staging_leftovers(tmp_path) == []


def test_symlink_timeline_rejected_where_supported(tmp_path: Path) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    link = tmp_path / "timeline-link.json"
    try:
        os.symlink(timeline, link)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks: {exc}")
    except NotImplementedError:
        pytest.skip("symlinks are not supported on this platform")

    with pytest.raises(generation_queue.QueueError, match="symlink"):
        generation_queue.cmd_plan(link, tmp_path / "queue-out")


def test_symlink_output_rejected_where_supported(tmp_path: Path) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "queue-link"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks: {exc}")
    except NotImplementedError:
        pytest.skip("symlinks are not supported on this platform")

    with pytest.raises(generation_queue.QueueError, match="symlink"):
        generation_queue.cmd_plan(timeline, link)


def test_link_detector_rejects_input_and_output_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    output = tmp_path / "queue-out"
    real_detector = generation_queue._is_link_or_reparse

    def _fake_output_link(path: Path) -> bool:
        return path == output or real_detector(path)

    monkeypatch.setattr(
        generation_queue, "_is_link_or_reparse", _fake_output_link
    )
    with pytest.raises(generation_queue.QueueError, match="symlink"):
        generation_queue.cmd_plan(timeline, output)
    assert not output.exists()
    assert _staging_leftovers(tmp_path) == []

    other = tmp_path / "other-out"

    def _fake_input_link(path: Path) -> bool:
        return path == timeline or real_detector(path)

    monkeypatch.setattr(
        generation_queue, "_is_link_or_reparse", _fake_input_link
    )
    with pytest.raises(generation_queue.QueueError, match="symlink"):
        generation_queue.cmd_plan(timeline, other)
    assert not other.exists()
    assert _staging_leftovers(tmp_path) == []


def test_injected_publish_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _write_timeline(tmp_path, [_placeholder_item("shot_001", 1)])
    before = timeline.read_bytes()
    output = tmp_path / "queue-out"

    def _boom(*args, **kwargs):
        raise generation_queue.QueueError(
            "evidence_incomplete", "injected publish failure"
        )

    monkeypatch.setattr(generation_queue, "_publish_dir", _boom)
    with pytest.raises(generation_queue.QueueError, match="injected"):
        generation_queue.cmd_plan(timeline, output)

    assert not output.exists()
    assert timeline.read_bytes() == before
    assert _staging_leftovers(tmp_path) == []


def test_single_capture_and_no_source_media_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [_clip_item("shot_001", 1), _placeholder_item("shot_002", 2)]
    timeline = _write_timeline(tmp_path, items)
    media = tmp_path / "clips" / "shot_001.mp4"
    media.parent.mkdir()
    media.write_bytes(b"fake media bytes")
    before_media = media.read_bytes()
    output = tmp_path / "queue-out"

    reads: list[Path] = []
    real_read_bytes = Path.read_bytes

    def _counting_read_bytes(self: Path) -> bytes:
        reads.append(self)
        return real_read_bytes(self)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", _counting_read_bytes)
        generation_queue.cmd_plan(timeline, output)

    assert [str(path) for path in reads] == [str(timeline)]
    assert media.read_bytes() == before_media
    assert (output / "generation_queue.json").is_file()


def test_source_has_no_media_render_subprocess_or_discovery() -> None:
    text = Path(generation_queue.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "iterdir",
        "listdir",
        "rglob",
        "glob(",
        "subprocess",
        "ffmpeg",
        "ffprobe",
        "os.environ",
        "API_KEY",
        "SECRET",
        "TOKEN",
    ):
        assert forbidden not in text


def test_cli_plan_json_stdout_and_error_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    timeline = _write_timeline(tmp_path, _mixed_items())
    output = tmp_path / "queue-out"

    rc = generation_queue.main(
        ["plan", "--timeline", str(timeline), "--output", str(output)]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["pending_count"] == 2
    assert payload["next_action"]["action"] == "provide_manual_keyframes"

    rc = generation_queue.main(
        ["plan", "--timeline", str(timeline), "--output", str(output)]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert captured.out == ""

    bad = tmp_path / "bad-timeline.json"
    bad.write_text('{"schema_version": "2.0"}', encoding="utf-8")
    bad_output = tmp_path / "bad-out"
    rc = generation_queue.main(
        ["plan", "--timeline", str(bad), "--output", str(bad_output)]
    )
    assert rc == 2
    assert not bad_output.exists()
