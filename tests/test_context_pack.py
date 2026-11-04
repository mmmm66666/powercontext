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

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import pytest

from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, PrepareContextRequest, open_builtin_runtime
from powercontext.builtin.runtime.context_pack import (
    CONTEXT_PACK_DELTA_SCHEMA,
    CONTEXT_PACK_SCHEMA,
    ContextPack,
    ContextPackDelta,
    ContextPackError,
    apply_context_pack_delta,
)
from powercontext.builtin.scope import ScopeDraft


def _pack_payload(items: tuple[bytes, ...]) -> dict:
    unsigned = {
        "schema": CONTEXT_PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    return unsigned


def test_pack_wire_matches_the_independent_oracle() -> None:
    items = (b"alpha\n", "\u6d77\u6d0b".encode())
    canonical = json.dumps(_pack_payload(items), sort_keys=True, separators=(",", ":")).encode()
    pack = ContextPack.build(items)
    payload = json.loads(pack.to_bytes())
    assert payload == {
        **_pack_payload(items),
        "digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "byte_count": len(pack.to_bytes()),
    }
    assert pack.to_bytes() == json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert pack.digest.startswith("sha256:")
    assert pack.byte_count == len(pack.to_bytes())


def test_delta_wire_matches_the_protocol() -> None:
    base = ContextPack.build((b"alpha", b"beta"))
    target = ContextPack.build((b"alpha", b"beta", b"gamma"))
    delta = ContextPackDelta.between(base, target)
    payload = json.loads(delta.to_bytes())
    assert set(payload) == {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
    assert payload["schema"] == CONTEXT_PACK_DELTA_SCHEMA
    assert payload["base_digest"] == base.digest
    assert payload["target_digest"] == target.digest
    assert payload["prefix_items"] == 2
    assert payload["appended"] == [base64.b64encode(b"gamma").decode("ascii")]
    assert delta.byte_count == len(delta.to_bytes())

    rebuilt = apply_context_pack_delta(
        ContextPack.from_bytes(base.to_bytes()),
        ContextPackDelta.from_bytes(delta.to_bytes()),
    )
    assert rebuilt.to_bytes() == target.to_bytes()


def test_from_content_chunks_are_bounded_valid_and_deterministic() -> None:
    content = "alpha \u6d77\u6d0b beta \U0001f600 gamma\n" * 20
    first = ContextPack.from_content(content, chunk_bytes=7)
    second = ContextPack.from_content(content, chunk_bytes=7)
    assert first == second
    assert len(first.items) > 1
    assert b"".join(first.items).decode("utf-8") == content
    for chunk in first.items:
        assert chunk
        assert len(chunk) <= 7
        chunk.decode("utf-8")

    default = ContextPack.from_content(content)
    assert all(len(chunk) <= 512 for chunk in default.items)
    assert b"".join(default.items).decode("utf-8") == content


@pytest.mark.parametrize("chunk_bytes", [0, -1, 1.0, True])
def test_invalid_chunk_bounds_fail(chunk_bytes) -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("alpha", chunk_bytes=chunk_bytes)


def test_content_boundary_that_cannot_fit_one_character_fails() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.from_content("\u6d77", chunk_bytes=2)


def test_empty_packs_and_deltas_round_trip() -> None:
    empty = ContextPack.from_content("")
    assert empty.items == ()
    assert ContextPack.from_content(None) == empty
    assert ContextPack.from_bytes(empty.to_bytes()) == empty

    same = ContextPackDelta.between(empty, ContextPack.from_content(""))
    assert same.prefix_items == 0 and same.appended == ()
    assert apply_context_pack_delta(empty, ContextPackDelta.from_bytes(same.to_bytes())) == empty

    target = ContextPack.from_content("later")
    delta = ContextPackDelta.between(empty, target)
    assert apply_context_pack_delta(empty, delta) == target


@pytest.mark.parametrize(
    "target_items",
    [(b"a", b"b", b"new", b"tail"), (b"new", b"b", b"old"), (b"a", b"new", b"old"), ()],
)
def test_delta_reconstructs_changes_at_any_position(target_items: tuple[bytes, ...]) -> None:
    base = ContextPack.build((b"a", b"b", b"old"))
    target = ContextPack.build(target_items)
    delta = ContextPackDelta.between(base, target)
    rebuilt = apply_context_pack_delta(
        ContextPack.from_bytes(base.to_bytes()),
        ContextPackDelta.from_bytes(delta.to_bytes()),
    )
    assert rebuilt.to_bytes() == target.to_bytes()

    full_suffix = {
        **json.loads(delta.to_bytes()),
        "prefix_items": 0,
        "appended": [base64.b64encode(item).decode("ascii") for item in target_items],
    }
    decoded = ContextPackDelta.from_bytes(json.dumps(full_suffix).encode())
    assert apply_context_pack_delta(base, decoded).to_bytes() == target.to_bytes()


def test_pack_decoder_rejects_tampering_and_missing_fields() -> None:
    pack = ContextPack.build((b"alpha",))
    payload = json.loads(pack.to_bytes())

    for field in payload:
        incomplete = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ContextPackError):
            ContextPack.from_bytes(json.dumps(incomplete).encode())

    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "extra": True}).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(pack.to_bytes().replace(pack.digest.encode(), b"sha256:" + b"0" * 64))
    tampered_count = {**payload, "byte_count": payload["byte_count"] + 1}
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps(tampered_count, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b"{not json")
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(b"\xff\xfe")
    with pytest.raises(ContextPackError):
        ContextPack.from_bytes(json.dumps({**payload, "schema": "other"}).encode())


def test_delta_decoder_rejects_tampering_and_bad_prefixes() -> None:
    pack = ContextPack.build((b"alpha",))
    delta = ContextPackDelta.between(pack, ContextPack.build((b"alpha", b"beta")))
    payload = json.loads(delta.to_bytes())

    for field in payload:
        incomplete = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ContextPackError):
            ContextPackDelta.from_bytes(json.dumps(incomplete).encode())

    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "byte_count": 1}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "prefix_items": -1}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "prefix_items": True}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "prefix_items": 1.5}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "base_digest": "sha256:xyz"}).encode())
    with pytest.raises(ContextPackError):
        ContextPackDelta.from_bytes(json.dumps({**payload, "appended": ["***"]}).encode())


def test_apply_rejects_wrong_base_bounds_and_target_digest() -> None:
    base = ContextPack.build((b"one",))
    delta = ContextPackDelta.between(base, ContextPack.build((b"one", b"two")))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build((b"other",)), delta)

    with pytest.raises(ContextPackError):
        apply_context_pack_delta(ContextPack.build(()), ContextPackDelta(base.digest, delta.target_digest, 2, ()))
    with pytest.raises(ContextPackError):
        apply_context_pack_delta(
            base, ContextPackDelta(base.digest, "sha256:" + "0" * 64, delta.prefix_items, delta.appended)
        )


def test_apply_accepts_serialized_packs_and_deltas() -> None:
    base = ContextPack.build((b"one", b"two"))
    target = ContextPack.build((b"one", b"three"))
    delta = ContextPackDelta.between(base, target)
    rebuilt = apply_context_pack_delta(base.to_bytes(), delta.to_bytes())
    assert rebuilt.to_bytes() == target.to_bytes()


def test_build_and_between_validate_types() -> None:
    with pytest.raises(ContextPackError):
        ContextPack.build((b"ok", "not-bytes"))  # type: ignore[arg-type]
    with pytest.raises(ContextPackError):
        ContextPackDelta.between(ContextPack.build((b"a",)), "not-a-pack")  # type: ignore[arg-type]


def test_repeated_rounds_reuse_the_previous_prefix_and_save_transport() -> None:
    content = "\n\n".join(
        f"Speaker {index % 3}: turn {index} discusses remote controls, meeting logistics and product status in detail."
        for index in range(80)
    )
    full_bytes = len(content.encode("utf-8"))
    first = ContextPack.from_content(content)
    repeated = ContextPack.from_content(content)
    delta = ContextPackDelta.between(first, repeated)
    assert delta.prefix_items == len(first.items)
    assert delta.appended == ()
    assert delta.byte_count < full_bytes

    incremental = len(first.to_bytes()) + 39 * len(delta.to_bytes())
    raw_full = 40 * full_bytes
    assert 1 - incremental / raw_full >= 0.45

    changed = ContextPack.from_content(content + "\n\nSpeaker 1: one more appended turn.")
    moving = ContextPackDelta.between(first, changed)
    assert moving.prefix_items >= len(first.items) - 1


def test_prepare_incremental_supports_a_legal_empty_context(tmp_path) -> None:
    async def scenario() -> None:
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"))
        async with open_builtin_runtime(config) as runtime:
            scope = await runtime.scopes.create(
                ScopeDraft(title="Empty", summary="Empty", idempotency_key="context-pack-empty")
            )
            context = runtime.context.for_scope(scope.scope_id)
            request = PrepareContextRequest(query="empty", max_bytes=512)
            first = await context.prepare_incremental(request)
            assert isinstance(first, ContextPack)
            assert first.items == ()

            second = await context.prepare_incremental(request, base_pack=first)
            assert isinstance(second, ContextPackDelta)
            assert second.prefix_items == 0 and second.appended == ()
            assert apply_context_pack_delta(first, second) == first

    asyncio.run(scenario())
