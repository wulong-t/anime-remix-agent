"""Unit tests for aliases.json model, loading and rule-parser integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from anime_remix.domain.models import AliasesDocument, ClipsDocument
from anime_remix.errors import InputValidationError
from anime_remix.services.aliases import alias_key, load_aliases_document
from anime_remix.services.input_loader import canonicalize_character_refs
from anime_remix.services.script_parser import parse_script


def _clips_doc() -> ClipsDocument:
    doc = ClipsDocument(
        clips=[
            {
                "id": "clip_001",
                "path": "clips/clip_001.mp4",
                "characters": [
                    {"id": "char_lin_xia", "name": "林夏"},
                ],
                "location_id": "loc_school_rooftop",
                "location_name": "学校天台",
                "action": "独自站立",
                "description": "林夏独自站在学校天台。",
            },
            {
                "id": "clip_002",
                "path": "clips/clip_002.mp4",
                "characters": [
                    {"id": "char_lu_chen", "name": "陆辰"},
                ],
                "location_id": "loc_classroom",
                "location_name": "教室",
                "action": "沉默注视",
                "description": "陆辰在教室沉默注视窗外。",
            },
        ]
    )
    return canonicalize_character_refs(doc)


def _write_aliases(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _valid_payload() -> dict:
    return {
        "schema_version": "1.9",
        "character_aliases": [
            {"target_id": "char_lin_xia", "aliases": ["小夏"]},
            {"target_id": "char_lu_chen", "aliases": ["阿辰"]},
        ],
        "location_aliases": [
            {"target_id": "loc_school_rooftop", "aliases": ["楼顶"]},
            {"target_id": "loc_classroom", "aliases": ["课堂"]},
        ],
    }


class TestModelAndInput:
    def test_valid_aliases_document_passes(
        self,
        tmp_path: Path,
    ) -> None:
        path = _write_aliases(tmp_path, _valid_payload())
        doc = load_aliases_document(path, _clips_doc())
        assert doc.schema_version == "1.9"
        assert len(doc.character_aliases) == 2
        assert len(doc.location_aliases) == 2

    def test_top_level_bare_array_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "aliases.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["extra"] = 1
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_empty_alias_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": ["  "]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_overlong_alias_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": ["x" * 129]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_empty_aliases_list_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": []}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_missing_target_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_nobody", "aliases": ["某人"]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_character_name_as_target_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "林夏", "aliases": ["小夏"]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_location_name_as_target_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["location_aliases"] = [
            {"target_id": "学校天台", "aliases": ["楼顶"]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_same_alias_key_different_characters_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": ["小夏"]},
            {"target_id": "char_lu_chen", "aliases": ["小夏"]},
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_same_alias_key_different_locations_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _valid_payload()
        payload["location_aliases"] = [
            {"target_id": "loc_school_rooftop", "aliases": ["天台"]},
            {"target_id": "loc_classroom", "aliases": ["天台"]},
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_duplicate_alias_key_same_target_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": ["小夏", " 小夏 "]}
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_alias_conflicting_with_canonical_term_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        for bad_alias in ("林夏", "char_lin_xia", "陆辰"):
            payload = _valid_payload()
            payload["character_aliases"] = [
                {"target_id": "char_lin_xia", "aliases": [bad_alias]}
            ]
            path = _write_aliases(tmp_path, payload)
            with pytest.raises(InputValidationError):
                load_aliases_document(path, _clips_doc())

    def test_nfkc_casefold_conflict_rejected(self, tmp_path: Path) -> None:
        payload = _valid_payload()
        payload["character_aliases"] = [
            {"target_id": "char_lin_xia", "aliases": ["ｓｈａｎｇｘｉａ"]},
            {"target_id": "char_lu_chen", "aliases": ["shangxia"]},
        ]
        path = _write_aliases(tmp_path, payload)
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_file_over_1_mib_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "aliases.json"
        path.write_bytes(b" " * (1024 * 1024 + 1))
        with pytest.raises(InputValidationError):
            load_aliases_document(path, _clips_doc())

    def test_model_rejects_non_string_target_and_bool_aliases(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(AliasesDocument).validate_python(
                {
                    "schema_version": "1.9",
                    "character_aliases": [
                        {"target_id": 123, "aliases": ["x"]}
                    ],
                }
            )
        with pytest.raises(ValidationError):
            TypeAdapter(AliasesDocument).validate_python(
                {
                    "schema_version": "1.9",
                    "character_aliases": [
                        {"target_id": "char_lin_xia", "aliases": [True]}
                    ],
                }
            )


class TestParserIntegration:
    def _aliases(self) -> AliasesDocument:
        return TypeAdapter(AliasesDocument).validate_python(_valid_payload())

    def test_character_alias_outputs_canonical_ref(self) -> None:
        requirements = parse_script(
            "小夏独自站在学校楼顶，望着远方。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            self._aliases(),
        )
        assert requirements[0].characters[0].id == "char_lin_xia"
        assert requirements[0].characters[0].name == "林夏"

    def test_location_alias_outputs_canonical_location(self) -> None:
        requirements = parse_script(
            "小夏独自站在学校楼顶，望着远方。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            self._aliases(),
        )
        assert requirements[0].location_id == "loc_school_rooftop"
        assert requirements[0].location_name == "学校天台"

    def test_alias_not_saved_as_identity(self) -> None:
        requirements = parse_script(
            "小夏独自站在学校楼顶，望着远方。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            self._aliases(),
        )
        dumped = requirements[0].model_dump(mode="json")
        assert dumped["characters"][0] == {
            "id": "char_lin_xia",
            "name": "林夏",
        }
        assert dumped["location_id"] == "loc_school_rooftop"
        assert dumped["location_name"] == "学校天台"
        assert "小夏" not in dumped["characters"][0].values()
        assert "楼顶" not in {
            dumped["location_id"],
            dumped["location_name"],
        }

    def test_longest_non_overlapping_across_alias_and_canonical(self) -> None:
        aliases = TypeAdapter(AliasesDocument).validate_python(
            {
                "schema_version": "1.9",
                "character_aliases": [
                    {
                        "target_id": "char_lin_xia",
                        "aliases": ["小夏", "小夏和陆辰"],
                    }
                ],
                "location_aliases": [],
            }
        )
        requirements = parse_script(
            "小夏和陆辰在学校楼顶。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            aliases,
        )
        assert [ref.id for ref in requirements[0].characters] == [
            "char_lin_xia"
        ]
        assert len(requirements[0].characters) == 1

    def test_ascii_alias_uses_word_boundary(self) -> None:
        aliases = TypeAdapter(AliasesDocument).validate_python(
            {
                "schema_version": "1.9",
                "character_aliases": [
                    {"target_id": "char_lin_xia", "aliases": ["linxia"]}
                ],
                "location_aliases": [],
            }
        )
        requirements = parse_script(
            "linxia is standing here.\n\n"
            "xlinxia_extra must not match.\n\n"
            "third paragraph.",
            _clips_doc(),
            aliases,
        )
        assert [ref.id for ref in requirements[0].characters] == [
            "char_lin_xia"
        ]
        assert requirements[1].characters == []

    def test_regex_special_characters_are_escaped(self) -> None:
        aliases = TypeAdapter(AliasesDocument).validate_python(
            {
                "schema_version": "1.9",
                "character_aliases": [
                    {"target_id": "char_lin_xia", "aliases": ["a+b[1]"]}
                ],
                "location_aliases": [],
            }
        )
        requirements = parse_script(
            "call a+b[1] now.\n\n"
            "call aab1 now must not match.\n\n"
            "third paragraph.",
            _clips_doc(),
            aliases,
        )
        assert [ref.id for ref in requirements[0].characters] == [
            "char_lin_xia"
        ]
        assert requirements[1].characters == []

    def test_same_target_multiple_hits_output_once(self) -> None:
        requirements = parse_script(
            "小夏和林夏站在学校楼顶。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。",
            _clips_doc(),
            self._aliases(),
        )
        assert len(requirements[0].characters) == 1
        assert requirements[0].characters[0].id == "char_lin_xia"

    def test_aliases_input_order_does_not_change_output(self) -> None:
        forward = TypeAdapter(AliasesDocument).validate_python(_valid_payload())
        reversed_payload = _valid_payload()
        reversed_payload["character_aliases"] = list(
            reversed(reversed_payload["character_aliases"])
        )
        reversed_payload["location_aliases"] = list(
            reversed(reversed_payload["location_aliases"])
        )
        backward = TypeAdapter(AliasesDocument).validate_python(
            reversed_payload
        )
        text = (
            "小夏独自站在学校楼顶，望着远方。\n\n"
            "阿辰在课堂上沉默注视窗外。\n\n"
            "小夏转身离开学校楼顶。"
        )
        clips = _clips_doc()
        forward_out = [
            req.model_dump(mode="json")
            for req in parse_script(text, clips, forward)
        ]
        backward_out = [
            req.model_dump(mode="json")
            for req in parse_script(text, clips, backward)
        ]
        assert forward_out == backward_out

    def test_no_aliases_keeps_existing_behavior(self) -> None:
        text = (
            "林夏独自站在学校天台，望着远方。\n\n"
            "陆辰在教室沉默注视窗外。\n\n"
            "林夏转身离开学校天台。"
        )
        clips = _clips_doc()
        without = parse_script(text, clips)
        explicit_none = parse_script(text, clips, None)
        assert [req.model_dump(mode="json") for req in without] == [
            req.model_dump(mode="json") for req in explicit_none
        ]
        assert without[0].characters[0].id == "char_lin_xia"
        assert without[0].location_id == "loc_school_rooftop"


def test_alias_key_normalization() -> None:
    assert alias_key("  ＳｈａｎｇＸｉａ  ") == "shangxia"
    assert alias_key("小夏") == alias_key(" 小夏 ")
