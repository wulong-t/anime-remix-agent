from __future__ import annotations

import pytest

from anime_remix.domain.models import ClipsDocument
from anime_remix.errors import InputValidationError
from anime_remix.services.script_parser import (
    action_from_paragraph,
    compute_target_frames,
    extract_dialogues,
    find_longest_non_overlapping,
    parse_script,
    split_paragraphs,
)


def test_split_paragraphs() -> None:
    text = "第一段。\n\n第二段。\n\n\n\n第三段。"
    assert split_paragraphs(text) == ["第一段。", "第二段。", "第三段。"]


def test_extract_dialogues_and_action() -> None:
    text = "“天台的风很大。”林夏独自站在学校天台。"
    dialogue, spans = extract_dialogues(text)
    assert dialogue == "天台的风很大。"
    assert spans == [(0, 9)]
    action = action_from_paragraph(text, spans)
    assert "天台的风很大" not in action
    assert "林夏独自站在学校天台" in action


def test_unclosed_dialogue_not_extracted() -> None:
    text = "“没有闭合的引号 林夏站在原地。"
    dialogue, spans = extract_dialogues(text)
    assert dialogue == ""
    assert spans == []


def test_target_frames_formula() -> None:
    assert compute_target_frames(None) == 72
    assert compute_target_frames("") == 72
    # 24 chars -> 24/4.5 + 0.6 = 5.9333s -> ceil(142.4) = 143 frames
    assert compute_target_frames("对" * 24) == 143
    # very long dialogue is capped at 192
    assert compute_target_frames("对" * 100) == 192


def test_paragraph_count_limits() -> None:
    with pytest.raises(InputValidationError):
        parse_script("只有一段。", _clips())
    text = "\n\n".join(f"段落{i}" for i in range(11))
    with pytest.raises(InputValidationError):
        parse_script(text, _clips())


def test_longest_non_overlapping_match() -> None:
    entries = [
        {"key": "a", "display": "林夏"},
        {"key": "b", "display": "林夏和陆辰"},
    ]
    matched = find_longest_non_overlapping("林夏和陆辰站在天台。", entries)
    assert [entry["key"] for entry in matched] == ["b"]


def test_parse_script_stable_ids_and_characters() -> None:
    clips = _clips()
    text = (
        "“天台的风很大。”林夏独自站在学校天台，望着远方。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "“走吧。”林夏转身离开学校天台。"
    )
    requirements = parse_script(text, clips)
    assert [req.id for req in requirements] == ["shot_001", "shot_002", "shot_003"]
    assert [req.order for req in requirements] == [1, 2, 3]
    assert requirements[0].characters[0].id == "char_lin_xia"
    assert requirements[0].location_id == "loc_school_rooftop"
    assert requirements[1].characters[0].id == "char_lu_chen"
    assert requirements[1].location_id == "loc_classroom"
    assert requirements[0].target_frames == 72
    assert "天台的风很大" not in requirements[0].action


def test_dialogue_drives_longer_target_frames() -> None:
    clips = _clips()
    text = (
        f"“{('对' * 30)}”林夏独自站在学校天台。\n\n"
        "陆辰在教室沉默注视窗外。\n\n"
        "林夏转身离开学校天台。"
    )
    requirements = parse_script(text, clips)
    assert requirements[0].target_frames > 72


def _clips() -> ClipsDocument:
    return ClipsDocument(
        clips=[
            {
                "id": "clip_001",
                "path": "clips/clip_001.mp4",
                "characters": [
                    {"id": "char_lin_xia", "name": "林夏"},
                    {"id": "char_lu_chen", "name": "陆辰"},
                ],
                "location_id": "loc_school_rooftop",
                "location_name": "学校天台",
                "action": "独自站立",
                "description": "林夏站在学校天台。",
            },
            {
                "id": "clip_002",
                "path": "clips/clip_002.mp4",
                "characters": [{"id": "char_lu_chen", "name": "陆辰"}],
                "location_id": "loc_classroom",
                "location_name": "教室",
                "action": "沉默注视",
                "description": "陆辰在教室沉默注视窗外。",
            },
        ]
    )

