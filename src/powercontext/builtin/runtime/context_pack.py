# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned Context Packs and prefix-reusing incremental deltas.

A Context Pack is the complete prepared context as a deterministic sequence
of UTF-8 chunks. A delta keeps a prefix of the previous pack and appends only
the target suffix, so a receiver holding the previous wire Pack can rebuild
the new one without any sender-side state.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

# Chunks stay well inside the caller's ceiling so a change to an early citation
# or metadata byte keeps a longer reusable prefix across consecutive rounds.
_PREFERRED_CHUNK_BYTES = 128

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when a Context Pack or delta cannot satisfy its wire contract."""


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError("chunk arrays must be JSON lists")  # noqa: TRY003
    items: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError("chunks must be standard Base64 strings")  # noqa: TRY003
        try:
            decoded = base64.b64decode(entry, validate=True)
        except (binascii.Error, ValueError):
            raise ContextPackError("chunks must be standard Base64 strings") from None  # noqa: TRY003
        if base64.b64encode(decoded).decode("ascii") != entry:
            raise ContextPackError("chunks must be standard Base64 strings")  # noqa: TRY003
        items.append(decoded)
    return tuple(items)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(items)}
    return "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _validate_items(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, tuple | list):
        raise ContextPackError("pack items must be a bytes tuple")  # noqa: TRY003
    normalized = tuple(items)
    for item in normalized:
        if not isinstance(item, bytes):
            raise ContextPackError("pack items must be bytes")  # noqa: TRY003
    return normalized


def _validate_digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ContextPackError("digests must use sha256:<hex>")  # noqa: TRY003
    return value


def _decode_document(data: object) -> tuple[dict[str, object], int]:
    if not isinstance(data, bytes | bytearray):
        raise ContextPackError("wire documents must be bytes")  # noqa: TRY003
    raw = bytes(data)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ContextPackError("wire documents must be valid UTF-8") from None  # noqa: TRY003
    try:
        payload = json.loads(text)
    except ValueError:
        raise ContextPackError("wire documents must be valid JSON") from None  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError("wire documents must be JSON objects")  # noqa: TRY003
    return payload, len(raw)


def _require_fields(payload: dict[str, object], expected: frozenset[str]) -> None:
    if set(payload) != expected:
        raise ContextPackError("wire document fields do not match the protocol")  # noqa: TRY003


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete context as deterministic chunks with a content digest."""

    items: tuple[bytes, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _validate_items(self.items))
        if not isinstance(self.digest, str) or not self.digest:
            raise ContextPackError("pack digest must be a non-empty string")  # noqa: TRY003

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a complete pack from an explicit chunk sequence."""

        normalized = _validate_items(items)
        return cls(items=normalized, digest=_pack_digest(normalized))

    @classmethod
    def from_content(cls, content: str | bytes | None, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Chunk UTF-8 content deterministically into valid, bounded chunks."""

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        elif isinstance(content, bytes | bytearray):
            try:
                text = bytes(content).decode("utf-8")
            except UnicodeDecodeError:
                raise ContextPackError("content must be valid UTF-8") from None  # noqa: TRY003
        else:
            raise ContextPackError("content must be text or bytes")  # noqa: TRY003
        try:
            return cls.build(_chunk_text(text, min(chunk_bytes, _PREFERRED_CHUNK_BYTES)))
        except UnicodeEncodeError:
            raise ContextPackError("content must be encodable as UTF-8") from None  # noqa: TRY003

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify one complete wire pack."""

        payload, size = _decode_document(data)
        _require_fields(payload, _PACK_FIELDS)
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("pack schema is not supported")  # noqa: TRY003
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        if not isinstance(digest, str) or digest != _pack_digest(items):
            raise ContextPackError("pack digest does not match its items")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count != size:
            raise ContextPackError("pack byte_count does not match its wire bytes")  # noqa: TRY003
        return cls(items=items, digest=digest)

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        payload: dict[str, object] = {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_items(self.items),
            "digest": self.digest,
        }
        count = 0
        while True:
            payload["byte_count"] = count
            data = _canonical_json(payload)
            if len(data) == count:
                return data
            count = len(data)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A reusable base prefix plus the target suffix of one transition."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_digest, str) or not isinstance(self.target_digest, str):
            raise ContextPackError("delta digests must be strings")  # noqa: TRY003
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        object.__setattr__(self, "appended", _validate_items(self.appended))

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Reuse the longest common chunk prefix and carry the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be Context Packs")  # noqa: TRY003
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify one wire delta."""

        payload, _size = _decode_document(data)
        _require_fields(payload, _DELTA_FIELDS)
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("delta schema is not supported")  # noqa: TRY003
        base_digest = _validate_digest(payload["base_digest"])
        target_digest = _validate_digest(payload["target_digest"])
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        appended = _decode_items(payload["appended"])
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical_json(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Validate one delta against its base and rebuild the target pack."""

    if isinstance(base, bytes | bytearray):
        base = ContextPack.from_bytes(base)
    if isinstance(delta, bytes | bytearray):
        delta = ContextPackDelta.from_bytes(delta)
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a Pack and a Delta")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the held pack")  # noqa: TRY003
    if type(delta.prefix_items) is not int or delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the held pack")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")  # noqa: TRY003
    return target


def _chunk_text(text: str, chunk_bytes: int) -> tuple[bytes, ...]:
    chunks: list[bytes] = []
    current = bytearray()
    for character in text:
        encoded = character.encode("utf-8")
        if len(encoded) > chunk_bytes:
            raise ContextPackError("chunk_bytes cannot hold one UTF-8 character")  # noqa: TRY003
        if len(current) + len(encoded) > chunk_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current.extend(encoded)
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
