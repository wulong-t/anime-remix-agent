"""Image-First I1: deterministic image asset manifest contract.

Loads one exact ``image_assets.json`` document (schema
``image-assets-v1``) and returns an ordered, immutable catalog.  Each
record carries the declared metadata plus the resolved absolute path and
minimal header-probed format, width and height.  Only explicitly listed
relative image paths are opened, and only their headers are inspected:
nothing is discovered, decoded or hashed, and no external process,
network or environment access happens.

Since the ASSET-BOOTSTRAP-MVP contract the entry also carries four
optional, defaulted provenance fields (``source_tier``,
``reference_roles``, ``provenance``, ``analysis_status``); old manifests
without them keep loading unchanged.
"""

from __future__ import annotations

import os
import re
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from anime_remix.errors import InputValidationError, UnsafePathError
from anime_remix.json_io import load_json_object

_SCHEMA_VERSION = "image-assets-v1"
_MAX_DIMENSION = 16384
_JPEG_HEADER_SCAN_LIMIT = 1024 * 1024
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_SOI = b"\xff\xd8"
_REPARSE_POINT = 0x400  # Windows FILE_ATTRIBUTE_REPARSE_POINT

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_URL_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WINDOWS_DRIVE = re.compile(r"^[a-zA-Z]:")
_WINDOWS_UNC = re.compile(r"^\\\\")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_STRICT_CONFIG = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

SourceTier = Literal[
    "canonical",
    "derived",
    "generated_candidate",
    "approved_generated",
]
ReferenceRole = Literal[
    "identity_reference",
    "pose_reference",
    "expression_reference",
    "outfit_reference",
    "scene_reference",
    "prop_reference",
    "style_reference",
]
AnalysisStatus = Literal["pending", "analyzed"]
_REFERENCE_ROLES = frozenset(ReferenceRole.__args__)


class Provenance(BaseModel):
    """Optional import/generation provenance for one image asset."""

    model_config = _STRICT_CONFIG

    source_path: str | None = None
    sha256: str | None = None
    parent_asset_id: str | None = None
    parent_sha256: str | None = None
    note: str | None = None

    @field_validator("source_path", "note", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("provenance text must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("provenance text must be non-empty when provided")
        return stripped

    @field_validator("sha256", "parent_sha256", mode="before")
    @classmethod
    def _sha256(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(
                "sha256 must be 64 lowercase hex characters or null"
            )
        return value

    @field_validator("parent_asset_id", mode="before")
    @classmethod
    def _parent_asset_id(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("parent_asset_id must be a string or null")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid parent_asset_id {value!r}")
        return stripped


class ImageAssetEntry(BaseModel):
    """One declared image asset with strict, nullable optional metadata."""

    model_config = _STRICT_CONFIG

    asset_id: str
    path: str
    asset_type: Literal["character", "background", "foreground", "prop", "style"]
    rights_status: str
    subject_or_scene_id: str | None = None
    view_angle: str | None = None
    pose: str | None = None
    expression: str | None = None
    outfit: str | None = None
    time_of_day: str | None = None
    quality_notes: str | None = None
    source_tier: SourceTier = "canonical"
    reference_roles: list[ReferenceRole] = []
    provenance: Provenance | None = None
    analysis_status: AnalysisStatus = "pending"

    @field_validator("asset_id", mode="before")
    @classmethod
    def _validate_asset_id(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("asset_id must be a string")
        stripped = value.strip()
        if not _ID_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid asset_id {value!r}")
        return stripped

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("path must be a string")
        if not value.strip():
            raise ValueError("path must be non-empty")
        return value

    @field_validator("rights_status", mode="before")
    @classmethod
    def _validate_rights(cls, value: object) -> object:
        if not isinstance(value, str):
            raise TypeError("rights_status must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("rights_status must be non-empty")
        return stripped

    @field_validator(
        "subject_or_scene_id",
        "view_angle",
        "pose",
        "expression",
        "outfit",
        "time_of_day",
        "quality_notes",
        mode="before",
    )
    @classmethod
    def _validate_optional_metadata(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("optional metadata must be a string or null")
        stripped = value.strip()
        if not stripped:
            raise ValueError("optional metadata must be non-empty when provided")
        return stripped

    @field_validator("reference_roles", mode="before")
    @classmethod
    def _validate_reference_roles(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("reference_roles must be a list")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or item not in _REFERENCE_ROLES:
                raise ValueError(f"invalid reference_role {item!r}")
            if item in cleaned:
                raise ValueError(f"duplicate reference_role {item!r}")
            cleaned.append(item)
        return cleaned


class ImageAssetsDocument(BaseModel):
    """Strict image_assets.json document (experimental schema)."""

    model_config = _STRICT_CONFIG

    schema_version: Literal["image-assets-v1"] = _SCHEMA_VERSION
    assets: list[ImageAssetEntry]

    @model_validator(mode="after")
    def _validate_document(self) -> ImageAssetsDocument:
        if not self.assets:
            raise ValueError("assets must not be empty")
        ids = [entry.asset_id for entry in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset ids must be globally unique")
        return self


@dataclass(frozen=True)
class ImageAssetRecord:
    """One validated, header-probed image asset."""

    asset_id: str
    asset_type: str
    path: str
    rights_status: str
    resolved_path: Path
    format: str
    width: int
    height: int
    subject_or_scene_id: str | None = None
    view_angle: str | None = None
    pose: str | None = None
    expression: str | None = None
    outfit: str | None = None
    time_of_day: str | None = None
    quality_notes: str | None = None
    source_tier: str = "canonical"
    reference_roles: tuple[str, ...] = ()
    provenance: dict[str, str | None] | None = None
    analysis_status: str = "pending"


@dataclass(frozen=True)
class ImageAssetCatalog:
    """Ordered immutable catalog with exact asset_id lookup."""

    records: tuple[ImageAssetRecord, ...]
    _by_id: Mapping[str, ImageAssetRecord]

    @classmethod
    def build(cls, records: list[ImageAssetRecord]) -> ImageAssetCatalog:
        return cls(records=tuple(records), _by_id={r.asset_id: r for r in records})

    def get(self, asset_id: str) -> ImageAssetRecord | None:
        """Return the record for an exact asset_id, or None."""

        return self._by_id.get(asset_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.asset_id for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ImageAssetRecord]:
        return iter(self.records)


def _reject_unsafe_path(raw: str, *, asset_id: str) -> None:
    if not raw.strip():
        raise UnsafePathError("path must not be empty", asset_id=asset_id)
    if _URL_PREFIX.match(raw) or "://" in raw:
        raise UnsafePathError("URL/URI paths are rejected", asset_id=asset_id, actual=raw)
    if (
        Path(raw).is_absolute()
        or _WINDOWS_DRIVE.match(raw)
        or _WINDOWS_UNC.match(raw)
    ):
        raise UnsafePathError(
            "absolute/device paths are rejected",
            asset_id=asset_id,
            actual=raw,
        )
    if any(part == ".." for part in Path(raw).parts):
        raise UnsafePathError(
            "parent traversal is rejected",
            asset_id=asset_id,
            actual=raw,
        )
    name = Path(raw).name.upper()
    if name in _DEVICE_NAMES or re.fullmatch(r"CON[0-9]|COM[0-9]|LPT[0-9]", name):
        raise UnsafePathError("device paths are rejected", asset_id=asset_id, actual=raw)


def _is_link_or_reparse(path: Path) -> bool:
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None:
        try:
            if isjunction(path):
                return True
        except OSError:
            pass
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & _REPARSE_POINT)


def _reject_link_components(base: Path, raw: str, *, asset_id: str) -> None:
    current = base
    for part in Path(raw).parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise UnsafePathError(
                "symlink/reparse point in image path is rejected",
                asset_id=asset_id,
                actual=raw,
            )


def _resolve_image_path(base: Path, raw: str, *, asset_id: str) -> Path:
    _reject_unsafe_path(raw, asset_id=asset_id)
    _reject_link_components(base, raw, asset_id=asset_id)
    try:
        resolved = (base / raw).resolve(strict=True)
    except OSError as exc:
        raise UnsafePathError(
            "image file does not exist or is unreadable",
            asset_id=asset_id,
            actual=raw,
        ) from exc
    if not resolved.is_file():
        raise UnsafePathError(
            "image path is not a regular file",
            asset_id=asset_id,
            actual=raw,
        )
    try:
        resolved.relative_to(base)
    except ValueError:
        raise UnsafePathError(
            "resolved image path escapes manifest directory",
            asset_id=asset_id,
            actual=raw,
        ) from None
    return resolved


def _format_for_path(raw: str) -> str:
    suffix = Path(raw).suffix.lower()
    if suffix == ".png":
        return "png"
    if suffix in (".jpg", ".jpeg"):
        return "jpeg"
    raise InputValidationError(
        f"unsupported image format {suffix!r}",
        field="path",
        actual=raw,
    )


def _check_dimensions(width: int, height: int, *, asset_id: str) -> None:
    if width <= 0 or height <= 0:
        raise InputValidationError(
            "image dimensions must be positive",
            asset_id=asset_id,
            actual=[width, height],
        )
    if width > _MAX_DIMENSION or height > _MAX_DIMENSION:
        raise InputValidationError(
            f"image dimensions exceed {_MAX_DIMENSION}",
            asset_id=asset_id,
            actual=[width, height],
        )


def _parse_png(header: bytes, *, asset_id: str) -> tuple[int, int]:
    if len(header) < 24 or not header.startswith(_PNG_MAGIC):
        raise InputValidationError("not a valid PNG header", asset_id=asset_id)
    (length,) = struct.unpack(">I", header[8:12])
    if length != 13 or header[12:16] != b"IHDR":
        raise InputValidationError(
            "PNG header has no valid IHDR chunk",
            asset_id=asset_id,
        )
    width, height = struct.unpack(">II", header[16:24])
    _check_dimensions(width, height, asset_id=asset_id)
    return width, height


def _is_sof(marker: int) -> bool:
    return 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)


def _parse_jpeg(handle: BinaryIO, *, asset_id: str) -> tuple[int, int]:
    if handle.read(2) != _JPEG_SOI:
        raise InputValidationError("not a valid JPEG header", asset_id=asset_id)
    scanned = 2
    while True:
        byte = handle.read(1)
        if not byte:
            raise InputValidationError(
                "malformed JPEG: no SOF marker",
                asset_id=asset_id,
            )
        scanned += 1
        if byte[0] != 0xFF:
            raise InputValidationError("malformed JPEG marker", asset_id=asset_id)
        while byte[0] == 0xFF:
            byte = handle.read(1)
            if not byte:
                raise InputValidationError(
                    "malformed JPEG: truncated marker",
                    asset_id=asset_id,
                )
            scanned += 1
        marker = byte[0]
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            raise InputValidationError(
                "JPEG has no SOF marker before EOI",
                asset_id=asset_id,
            )
        if marker == 0x00:
            raise InputValidationError(
                "malformed JPEG marker byte",
                asset_id=asset_id,
            )
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise InputValidationError(
                "malformed JPEG segment length",
                asset_id=asset_id,
            )
        scanned += 2
        (length,) = struct.unpack(">H", length_bytes)
        if length < 2:
            raise InputValidationError(
                "malformed JPEG segment length",
                asset_id=asset_id,
            )
        payload_size = length - 2
        if _is_sof(marker):
            if payload_size < 5:
                raise InputValidationError(
                    "malformed JPEG SOF segment",
                    asset_id=asset_id,
                )
            payload = handle.read(payload_size)
            if len(payload) != payload_size:
                raise InputValidationError(
                    "malformed JPEG SOF segment",
                    asset_id=asset_id,
                )
            height, width = struct.unpack(">HH", payload[1:5])
            _check_dimensions(width, height, asset_id=asset_id)
            return width, height
        remaining = payload_size
        while remaining:
            chunk = handle.read(min(remaining, 65536))
            if not chunk:
                raise InputValidationError(
                    "malformed JPEG: truncated segment",
                    asset_id=asset_id,
                )
            scanned += len(chunk)
            remaining -= len(chunk)
        if scanned > _JPEG_HEADER_SCAN_LIMIT:
            raise InputValidationError(
                "JPEG SOF not found within header scan limit",
                asset_id=asset_id,
            )


def _probe_image(
    resolved: Path, expected_format: str, *, asset_id: str
) -> tuple[int, int]:
    with resolved.open("rb") as handle:
        if expected_format == "png":
            return _parse_png(handle.read(24), asset_id=asset_id)
        return _parse_jpeg(handle, asset_id=asset_id)


def probe_image_file(path: Path | str) -> tuple[str, int, int]:
    """Header-probe one PNG/JPEG file outside a manifest.

    Returns ``(format, width, height)`` using exactly the same probe rules
    as the manifest loader.  This is the public entry used by the
    asset-bootstrap CLI; the loader itself never enumerates directories
    and never hashes file contents.
    """

    raw = Path(path)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise UnsafePathError(
            "image file does not exist or is unreadable",
            actual=str(raw),
        ) from exc
    if not resolved.is_file():
        raise UnsafePathError(
            "image path is not a regular file",
            actual=str(raw),
        )
    expected_format = _format_for_path(resolved.name)
    width, height = _probe_image(
        resolved, expected_format, asset_id=resolved.name
    )
    return expected_format, width, height


def load_image_assets(manifest_path: Path | str) -> ImageAssetCatalog:
    """Load one exact ``image_assets.json`` manifest deterministically.

    Only the explicitly listed relative image paths are opened (headers
    only); no directory is enumerated and no other file is read.
    """

    path = Path(manifest_path)
    data = load_json_object(path)
    try:
        document = TypeAdapter(ImageAssetsDocument).validate_python(data)
    except (ValidationError, TypeError) as exc:
        raise InputValidationError(
            f"invalid image_assets.json schema: {exc}",
            actual=path,
        ) from exc
    base = path.resolve().parent
    records: list[ImageAssetRecord] = []
    for entry in document.assets:
        resolved = _resolve_image_path(base, entry.path, asset_id=entry.asset_id)
        expected_format = _format_for_path(entry.path)
        width, height = _probe_image(
            resolved, expected_format, asset_id=entry.asset_id
        )
        records.append(
            ImageAssetRecord(
                asset_id=entry.asset_id,
                asset_type=entry.asset_type,
                path=entry.path,
                rights_status=entry.rights_status,
                resolved_path=resolved,
                format=expected_format,
                width=width,
                height=height,
                subject_or_scene_id=entry.subject_or_scene_id,
                view_angle=entry.view_angle,
                pose=entry.pose,
                expression=entry.expression,
                outfit=entry.outfit,
                time_of_day=entry.time_of_day,
                quality_notes=entry.quality_notes,
                source_tier=entry.source_tier,
                reference_roles=tuple(entry.reference_roles),
                provenance=(
                    entry.provenance.model_dump(mode="json")
                    if entry.provenance is not None
                    else None
                ),
                analysis_status=entry.analysis_status,
            )
        )
    return ImageAssetCatalog.build(records)
