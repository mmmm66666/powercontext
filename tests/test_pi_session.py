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
import json
from pathlib import Path

import pytest

from powercontext.sources import (
    ConnectorBinding,
    SourceDefinitionRegistry,
    SourceMaterialization,
    SourceRef,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.connectors import ConnectorLifecycle
from powercontext.sources.pi_session import (
    PI_EVENT_SOURCE_DEFINITION,
    PiEventInput,
    PiEventSource,
    PiSessionConnector,
    PiSessionInput,
    parse_pi_session,
    pi_event_source_id,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY

SESSION_ID = "session-1"


def _session(records: list[dict], *, header: dict | None = None) -> bytes:
    first = header if header is not None else {"type": "session", "version": 3, "id": SESSION_ID}
    return ("\n".join(json.dumps(record, ensure_ascii=False) for record in [first, *records]) + "\n").encode()


def _parse(records: list[dict], leaf: str | None = None) -> tuple[PiEventInput, ...]:
    return parse_pi_session(PiSessionInput(content=_session(records), leaf_id=leaf))


def test_visible_bodies_follow_the_pi_event_rules() -> None:
    records = [
        {
            "type": "message",
            "id": "user",
            "parentId": None,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}],
            },
        },
        {
            "type": "message",
            "id": "assistant",
            "parentId": "user",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": "answer"},
                    {"type": "toolCall", "name": "read", "id": "call-1", "arguments": {"z": 1, "a": "caf\u00e9"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "result",
            "parentId": "assistant",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            },
        },
        {
            "type": "message",
            "id": "bash",
            "parentId": "result",
            "message": {"role": "bashExecution", "command": "printf x", "output": "x", "content": []},
        },
    ]
    events = _parse(records)
    assert [event.content for event in events] == [
        "user: line one\nline two",
        'assistant: answer\n[tool_call read call-1] {"a":"caf\u00e9","z":1}',
        "toolResult: [tool_result read call-1] first\nsecond",
        "bashExecution: $ printf x\nx",
    ]
    assert [event.role for event in events] == ["user", "assistant", "toolResult", "bashExecution"]
    assert [event.event_type for event in events] == ["message"] * 4


def test_summaries_custom_messages_and_control_nodes() -> None:
    records = [
        {"type": "message", "id": "a", "parentId": None, "message": {"role": "user", "content": "before"}},
        {"type": "thinking_level_change", "id": "b", "parentId": "a"},
        {"type": "compaction", "id": "c", "parentId": "b", "summary": "compacted"},
        {"type": "branch_summary", "id": "d", "parentId": "c", "summary": "branched"},
        {
            "type": "custom_message",
            "id": "e",
            "parentId": "d",
            "content": [{"type": "text", "text": "custom evidence"}],
        },
        {
            "type": "message",
            "id": "f",
            "parentId": "e",
            "message": {
                "role": "assistant",
                "content": [{"type": "image", "data": "ignored"}, {"type": "thinking", "thinking": "ignored"}],
            },
        },
    ]
    events = _parse(records, leaf="f")
    assert [event.event_id for event in events] == ["a", "c", "d", "e"]
    assert [event.content for event in events] == [
        "user: before",
        "compaction: compacted",
        "branch_summary: branched",
        "custom: custom evidence",
    ]
    assert events[1].parent_id == "b"
    assert events[2].parent_id == "c"
    assert events[3].event_type == "custom_message"
    assert events[3].role == "custom"
    assert events[1].record["summary"] == "compacted"


def test_default_leaf_is_the_last_record_and_empty_records_are_ignored() -> None:
    content = (
        b'{"type":"session","version":3,"id":"session-1"}\n'
        b'\n'
        b'{"type":"message","id":"a","parentId":null,"message":{"role":"user","content":"hello"}}\n'
        b"\n"
    )
    events = parse_pi_session(PiSessionInput(content=content))
    assert [event.event_id for event in events] == ["a"]
    assert parse_pi_session(PiSessionInput(content=b"\xef\xbb\xbf" + content)) == events


@pytest.mark.parametrize(
    "records,leaf",
    [
        ([], "a"),
        ([{"type": "message", "id": "a", "parentId": "missing"}], "a"),
        ([{"type": "message", "id": "a", "parentId": "a"}], "a"),
        ([{"type": "message", "id": "a", "parentId": None}] * 2, "a"),
        ([{"type": "message", "id": "a", "parentId": None}], "absent"),
        ([{"type": "message", "id": "a", "parentId": None}], ""),
    ],
)
def test_invalid_graphs_fail_atomically(records: list[dict], leaf: str) -> None:
    with pytest.raises(ValueError):
        _parse(records, leaf)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\n\n",
        b"{}\n",
        b"[]\n",
        b"null\n",
        b"\xff",
        b'{"type":"session","version":2,"id":"x"}\n',
        b'{"type":"session","version":3,"id":""}\n',
        b'{"type":"session","version":3,"id":"x"}\n{bad}\n',
        b'{"type":"session","version":3,"id":"x"}\n[1]\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":1,"parentId":null}\n',
        b'{"type":"session","version":3,"id":"x"}\n{"type":"message","id":"a","parentId":7}\n',
    ],
)
def test_invalid_archives_fail(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_pi_session(PiSessionInput(content=raw))


def test_identity_uses_only_session_and_event_and_stays_bounded() -> None:
    assert pi_event_source_id("session:1", "event/1") == "pi:session%3A1:event%2F1"
    assert pi_event_source_id("session:1", "event/1") != pi_event_source_id("session:2", "event/1")
    assert pi_event_source_id("session:1", "event/1") != pi_event_source_id("session:1", "event:1")

    long_identity = pi_event_source_id("s" * 200, "e" * 200)
    assert len(long_identity) <= 256
    assert long_identity != pi_event_source_id("s" * 200, "e" * 201)
    assert long_identity != pi_event_source_id("s" * 201, "e" * 200)

    for session_id, event_id in [("", "e"), ("s", ""), (" ", "e"), ("s", "   ")]:
        with pytest.raises(ValueError):
            pi_event_source_id(session_id, event_id)


def test_definition_projection_and_manifest_preserve_the_complete_record() -> None:
    records = [
        {
            "type": "message",
            "id": "a",
            "parentId": None,
            "message": {"role": "user", "content": "hello"},
        }
    ]
    event = _parse(records)[0]

    assert PI_EVENT_SOURCE_DEFINITION.name == "pi-event"
    assert PI_EVENT_SOURCE_DEFINITION.version == "1"
    assert PI_EVENT_SOURCE_DEFINITION.input_class is PiEventInput
    assert PI_EVENT_SOURCE_DEFINITION.source_class is PiEventSource

    registry = SourceDefinitionRegistry((PI_EVENT_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(event))
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.name == source.source_id == "pi:session-1:a"
    assert asyncio.run(registry.read(source)) == event
    assert source.record == records[0]

    evidence = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    assert evidence == {
        "source_type": "pi-event",
        "source_id": "pi:session-1:a",
        "content": "user: hello",
        "metadata": {
            "session_id": "session-1",
            "event_id": "a",
            "parent_id": None,
            "event_type": "message",
            "role": "user",
        },
    }

    manifest = manifest_for_definition(PI_EVENT_SOURCE_DEFINITION)
    assert manifest.name == "pi-event"
    assert manifest.version == "1"
    assert [projection.key for projection in manifest.projections] == [TEXT_EVIDENCE_PROJECTION_KEY]

    observation = project_source_for_transport(registry, source)
    assert observation.projection(TEXT_EVIDENCE_PROJECTION_KEY) == evidence
    assert observation.payload["record"] == records[0]


class _RecordingSink:
    def __init__(self) -> None:
        self.items: list[PiEventInput] = []

    async def submit(self, binding, item_id, definition_name, value, /) -> SourceRef:
        assert definition_name == "pi-event"
        self.items.append(value)
        return SourceRef(source_type=definition_name, source_id=item_id)


class _CheckpointStore:
    def __init__(self) -> None:
        self.value = None

    async def load(self, binding):
        return self.value

    async def save(self, binding, checkpoint, /, *, expected):
        assert expected == self.value
        self.value = checkpoint


def _binding() -> ConnectorBinding:
    return ConnectorBinding(
        scope_id="scope-1",
        binding_id="pi",
        connector_name="pi-session-history",
        connector_version="1",
    )


def _run(path: Path, leaf: str | None, store: _CheckpointStore):
    sink = _RecordingSink()
    lifecycle = ConnectorLifecycle(sink=sink, checkpoints=store)
    result = asyncio.run(lifecycle.run(PiSessionConnector(path, leaf), _binding()))
    return result, sink.items


def test_connector_switch_append_restart_and_mutation_rejection(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "Shared"}},
        {"type": "message", "id": "left", "parentId": "root", "message": {"role": "assistant", "content": "Left"}},
        {"type": "model_change", "id": "control", "parentId": "root"},
        {"type": "message", "id": "right", "parentId": "control", "message": {"role": "assistant", "content": "Right"}},
    ]

    def write(current: list[dict]) -> None:
        path.write_bytes(_session(current))

    write(records)
    store = _CheckpointStore()

    first, items = _run(path, "left", store)
    assert [item.event_id for item in items] == ["root", "left"]
    assert first.committed_checkpoint == store.value

    second, items = _run(path, "right", store)
    assert [item.event_id for item in items] == ["right"]
    assert second.previous_checkpoint == first.committed_checkpoint
    assert second.items[0].status.value == "accepted"

    replay, items = _run(path, "right", store)
    assert replay.items == () and items == []

    branch_left, items = _run(path, "left", store)
    assert branch_left.items == () and items == []

    records.append(
        {"type": "message", "id": "tail", "parentId": "right", "message": {"role": "assistant", "content": "Tail"}}
    )
    write(records)
    added, items = _run(path, "tail", store)
    assert [item.event_id for item in items] == ["tail"]

    # A new Connector instance with the persisted checkpoint must not resubmit.
    restarted, items = _run(path, "tail", store)
    assert restarted.previous_checkpoint == added.committed_checkpoint
    assert restarted.items == () and items == []

    checkpoint_before_failure = store.value
    path.write_bytes(path.read_bytes() + b"{bad json}\n")
    with pytest.raises(ValueError):
        _run(path, "tail", store)
    assert store.value == checkpoint_before_failure

    write(records)
    repaired, items = _run(path, "tail", store)
    assert repaired.previous_checkpoint == added.committed_checkpoint
    assert repaired.items == () and items == []

    records[0]["message"]["content"] = "Rewritten"
    write(records)
    with pytest.raises(ValueError):
        _run(path, "tail", store)
    assert store.value == checkpoint_before_failure

    records[0]["message"]["content"] = "Shared"
    write(records)
    stable, items = _run(path, "tail", store)
    assert stable.previous_checkpoint == added.committed_checkpoint
    assert stable.items == () and items == []


def test_connector_detects_rewrites_on_an_unselected_branch(tmp_path: Path) -> None:
    path = tmp_path / "tree.jsonl"
    records = [
        {"type": "message", "id": "root", "parentId": None, "message": {"role": "user", "content": "Shared"}},
        {"type": "message", "id": "left", "parentId": "root", "message": {"role": "assistant", "content": "Left"}},
        {"type": "message", "id": "right", "parentId": "root", "message": {"role": "assistant", "content": "Right"}},
    ]
    path.write_bytes(_session(records))
    store = _CheckpointStore()

    _run(path, "left", store)
    _, items = _run(path, "right", store)
    assert [item.event_id for item in items] == ["right"]
    checkpoint = store.value

    records[1]["message"]["content"] = "Left rewritten"
    path.write_bytes(_session(records))
    with pytest.raises(ValueError):
        _run(path, "right", store)
    assert store.value == checkpoint
