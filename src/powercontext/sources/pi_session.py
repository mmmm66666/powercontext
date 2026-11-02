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

"""Captured Pi session-branch events with durable, resumable acquisition.

A Pi session file is a JSONL node graph: one session header plus records that
point at their parent. The selected leaf and its ancestors define the visible
history. This module turns every visible event on that path into immutable
evidence, and the Connector imports only events that its persisted checkpoint
does not already cover.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
PI_SESSION_HEADER_VERSION = 3
_CHECKPOINT_VERSION = 1


def _compact_json(value: object) -> str:
    """Serialize tool-call arguments with sorted keys, Unicode, and no spaces."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_digest(record: dict[str, Any]) -> str:
    """Fingerprint one raw input record regardless of JSON formatting details."""

    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build a stable identity from the session ID and the event ID only.

    Both components are percent-encoded separately so ``:`` inside either
    component cannot shift the boundary between them. Identities that would
    exceed the Source ID limit fall back to a deterministic digest so every
    event stays addressable.
    """

    _validate_identity_part("session_id", session_id)
    _validate_identity_part("event_id", event_id)
    identity = f"pi:{_quote(session_id)}:{_quote(event_id)}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    return f"pi:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _quote(component: str) -> str:
    return quote(component, safe="")


def _validate_identity_part(field: str, value: object) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"Pi {field} must be a non-empty string")  # noqa: TRY003
    return value


class PiSessionInput(BaseModel):
    """One complete Pi session archive and the branch leaf to read."""

    content: bytes
    leaf_id: str | None = None

    @field_validator("leaf_id")
    @classmethod
    def validate_leaf_id(cls, value: str | None) -> str | None:
        if value is not None and (not value or not value.strip()):
            raise ValueError("leaf_id must be non-empty when provided")  # noqa: TRY003
        return value


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its complete input record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


class PiEventSource(Source):
    """Captured Pi event evidence addressable by session and event identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one Pi session and return visible events on the selected branch.

    The whole archive is validated before any event is returned. Control nodes
    keep their graph position but never become text evidence, and compaction or
    branch summaries only add their own summary text.
    """

    session_id, records = _read_session(value.content)
    events = _branch_events(session_id, records, value.leaf_id)
    return events


def _read_session(content: bytes) -> tuple[str, dict[str, Any]]:
    records = _read_jsonl(content)
    if not records:
        raise ValueError("Pi session archive must contain a session header")  # noqa: TRY003

    header = records[0]
    if header.get("type") != "session":
        raise ValueError("Pi session archive must start with a session header")  # noqa: TRY003
    if header.get("version") != PI_SESSION_HEADER_VERSION:
        raise ValueError(f"Pi session header version must be {PI_SESSION_HEADER_VERSION}")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must contain a non-empty id")  # noqa: TRY003

    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Every Pi session record must contain a non-empty id")  # noqa: TRY003
        if node_id in nodes:
            raise ValueError(f"Pi session record id {node_id!r} is duplicated")  # noqa: TRY003
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str):
                raise ValueError(f"Pi session record {node_id!r} has a non-string parentId")  # noqa: TRY003
            if parent_id != "" and parent_id not in nodes:
                raise ValueError(  # noqa: TRY003
                    f"Pi session record {node_id!r} references unknown parent {parent_id!r}"
                )
        nodes[node_id] = record
        order.append(node_id)
    return session_id, {"nodes": nodes, "order": order}


def _read_jsonl(content: bytes) -> list[dict[str, Any]]:
    try:
        # ``utf-8-sig`` accepts one leading BOM and is otherwise strict.
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Pi session archive must be valid UTF-8") from None  # noqa: TRY003

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError("Pi session archive contains invalid JSON") from None  # noqa: TRY003
        if not isinstance(parsed, dict):
            raise ValueError("Pi session archive records must be JSON objects")  # noqa: TRY003
        records.append(parsed)
    return records


def _branch_events(
    session_id: str,
    graph: dict[str, Any],
    leaf_id: str | None,
) -> tuple[PiEventInput, ...]:
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    order: list[str] = graph["order"]

    leaf = leaf_id if leaf_id is not None else (order[-1] if order else None)
    if leaf is None or leaf not in nodes:
        raise ValueError(f"Pi session branch leaf {leaf!r} does not exist")  # noqa: TRY003

    chain: list[str] = []
    cursor: str | None = leaf
    while cursor is not None:
        chain.append(cursor)
        parent = nodes[cursor].get("parentId")
        cursor = parent if isinstance(parent, str) and parent else None
    chain.reverse()

    events: list[PiEventInput] = []
    for node_id in chain:
        event = _event_from_record(session_id, nodes[node_id])
        if event is not None:
            events.append(event)
    return tuple(events)


def _event_from_record(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    parent_id = record.get("parentId")
    parent = parent_id if isinstance(parent_id, str) and parent_id else None
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None

    if event_type == "message":
        rendered = _message_event(record.get("message"))
        if rendered is None:
            return None
        role, body = rendered
        return _event(session_id, record, parent, event_type, role, body)
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return _event(session_id, record, parent, event_type, event_type, summary)
    if event_type == "custom_message":
        body = _visible_text(record.get("content"), tool_calls=False)
        if body is None or body == "":
            return None
        return _event(session_id, record, parent, event_type, "custom", body)
    return None


def _event(
    session_id: str,
    record: dict[str, Any],
    parent_id: str | None,
    event_type: str,
    role: str,
    body: str,
) -> PiEventInput:
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_event(message: object) -> tuple[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "bashExecution":
        command = _string(message.get("command"))
        output = _string(message.get("output"))
        if not command and not output:
            return None
        return role, f"$ {command}\n{output}"
    if role == "toolResult":
        tool_name = _string(message.get("toolName"))
        tool_call_id = _string(message.get("toolCallId"))
        text = _tool_result_text(message.get("content"))
        if not tool_name and not tool_call_id and not text:
            return None
        return role, f"[tool_result {tool_name} {tool_call_id}] {text}"
    body = _visible_text(message.get("content"))
    if body is None or body == "":
        return None
    return role, body


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _visible_text(content: object, *, tool_calls: bool = True) -> str | None:
    """Join visible text (and message tool calls); return None when nothing is visible."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif tool_calls and block_type == "toolCall":
            parts.append(
                f"[tool_call {_string(block.get('name'))} {_string(block.get('id'))}] "
                f"{_compact_json(block.get('arguments'))}"
            )
    return "\n".join(parts)


class PiEventSourceAdapter:
    """Materialize Pi session events as captured Source evidence."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one Pi event through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one Pi branch, resuming from a checkpoint."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self._path.read_bytes()
        session_id, graph = _read_session(content)
        events = _branch_events(session_id, graph, self._leaf_id)

        digests = {node_id: _record_digest(record) for node_id, record in graph["nodes"].items()}
        imported = _imported_events(session.checkpoint, session_id)
        for event_id, known_digest in imported.items():
            current_digest = digests.get(event_id)
            if current_digest is not None and current_digest != known_digest:
                raise ValueError(  # noqa: TRY003
                    f"Pi event {event_id!r} was rewritten after it was imported"
                )

        for event in events:
            if event.event_id not in imported:
                await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)

        updated = dict(imported)
        updated.update({event.event_id: digests[event.event_id] for event in events})
        checkpoint = {
            "version": _CHECKPOINT_VERSION,
            "session_id": session_id,
            "events": updated,
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _imported_events(checkpoint: object, session_id: str) -> dict[str, str]:
    """Read the imported event digests that belong to the current session."""

    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003
    if checkpoint.get("version") != _CHECKPOINT_VERSION:
        return {}
    if checkpoint.get("session_id") != session_id:
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    imported: dict[str, str] = {}
    for event_id, digest in events.items():
        if isinstance(event_id, str) and isinstance(digest, str):
            imported[event_id] = digest
    return imported


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
