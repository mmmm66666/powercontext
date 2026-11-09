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

"""Source-dependency lifecycle for Memory and Topic Memory availability."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, RetireMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """A lifecycle graph, event, reference, or resolution contract violation."""


class SourceLifecycleEvent(BaseModel):
    """One explicit source availability transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(strict=True)
    revision: int = Field(strict=True, gt=0)
    sequence: int = Field(strict=True, gt=0)
    available: bool = Field(strict=True)
    reason: str = Field(strict=True)

    @field_validator("node_id", "reason")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")  # noqa: TRY003
        return value


_LIFECYCLE_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS pc_source_lifecycle_nodes (
        scope_id VARCHAR(255) NOT NULL,
        node_id VARCHAR(255) NOT NULL,
        family VARCHAR(16) NOT NULL,
        reference TEXT NOT NULL,
        parents TEXT NOT NULL,
        policy VARCHAR(8) NOT NULL,
        node_position INTEGER NOT NULL,
        available INTEGER NOT NULL,
        revision INTEGER,
        sequence INTEGER,
        last_reason TEXT,
        applied INTEGER NOT NULL,
        PRIMARY KEY (scope_id, node_id)
    )
    """,
)


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    family: Literal["source", "memory", "topic"]
    reference: str
    parents: tuple[str, ...]
    policy: Literal["all", "any"]
    position: int
    available: bool
    revision: int | None
    sequence: int | None
    last_reason: str | None
    applied: bool

    def with_available(self, available: bool) -> _Node:
        return replace(self, available=available)


class SourceLifecycleApplication:
    """Expose lifecycle state per Scope and filter unsupported Topic hits."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime
        self._locks: dict[str, asyncio.Lock] = {}

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self, validate_scope_id(scope_id))

    def _lock(self, scope_id: str) -> asyncio.Lock:
        return self._locks.setdefault(scope_id, asyncio.Lock())

    async def inactive_topic_refs(self, scope_id: str, /) -> frozenset[tuple[str, int]]:
        try:
            return await self.for_scope(scope_id)._inactive_topic_refs()
        except Exception:
            return frozenset()


class ScopedSourceLifecycle:
    """Mutable lifecycle graph and state for one Scope."""

    def __init__(self, application: SourceLifecycleApplication, scope_id: str) -> None:
        self._application = application
        self._runtime = application._runtime
        self.scope_id = scope_id

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int) -> None:
        _require_node_id(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("source revision must be a positive integer")
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source registration requires a SourceRef")
        await self._validate_source(source_ref)
        reference = _canonical(source_ref)
        async with self._application._lock(self.scope_id):
            rows = await self._load_nodes()
            existing = rows.get(node_id)
            if existing is not None:
                if (
                    existing.family == "source"
                    and existing.reference == reference
                    and existing.revision == revision
                ):
                    return
                raise SourceLifecycleError("node registration conflicts with the existing binding")
            position = _next_position(rows.values())
            await self._insert_node(
                _Node(
                    node_id=node_id,
                    family="source",
                    reference=reference,
                    parents=(),
                    policy="all",
                    position=position,
                    available=True,
                    revision=revision,
                    sequence=0,
                    last_reason=None,
                    applied=True,
                )
            )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory registration requires a MemoryCitation")
        await self._validate_memory(citation)
        await self._register_derived(node_id, "memory", citation, parents=parents, policy=policy)
        rows = await self._load_nodes()
        node = rows.get(node_id)
        if node is not None:
            await self._ensure_memory_state(node, reason="source lifecycle registration")

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef):
            raise SourceLifecycleError("topic registration requires an ArtifactRef")
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic registration requires a topic-memory ArtifactRef")
        await self._validate_topic(artifact_ref)
        await self._register_derived(node_id, "topic", artifact_ref, parents=parents, policy=policy)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("apply requires a SourceLifecycleEvent")
        async with self._application._lock(self.scope_id):
            rows = await self._load_nodes()
            source = rows.get(event.node_id)
            if source is None or source.family != "source":
                raise SourceLifecycleError("event node is not a registered Source")
            if source.revision != event.revision:
                raise SourceLifecycleError("event revision does not match the Source revision")
            current_sequence = source.sequence or 0
            if event.sequence < current_sequence:
                raise SourceLifecycleError("event sequence is older than the stored sequence")
            if event.sequence == current_sequence:
                if source.available != event.available or (source.last_reason or "") != event.reason:
                    raise SourceLifecycleError("event sequence was reused with different content")
                if source.applied:
                    return _available_ids(rows)
                updated = rows
            else:
                source_states = {node_id: node.available for node_id, node in rows.items()}
                source_states[event.node_id] = event.available
                states = _compute_states(rows, source_states)
                updated = {
                    node_id: node.with_available(states[node_id]) for node_id, node in rows.items()
                }
                await self._persist_event(event, updated)
            await self._reconcile_memory(updated, reason=event.reason)
            await self._mark_applied(event.node_id)
            return _available_ids(updated)

    async def resolve(self, node_id: str, /, *, purpose: Literal["generation", "audit"]) -> Any:
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown lifecycle purpose")
        rows = await self._load_nodes()
        node = rows.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("lifecycle node is not available for generation")
        return _reference_value(node)

    async def _inactive_topic_refs(self) -> frozenset[tuple[str, int]]:
        rows = await self._load_nodes(create=False)
        states: dict[tuple[str, int], list[bool]] = {}
        for node in rows.values():
            if node.family != "topic":
                continue
            reference = _reference_value(node)
            if isinstance(reference, ArtifactRef):
                states.setdefault((reference.artifact_id, reference.revision), []).append(node.available)
        return frozenset(key for key, values in states.items() if not any(values))

    async def _register_derived(
        self,
        node_id: str,
        family: Literal["memory", "topic"],
        reference: Any,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        _require_node_id(node_id)
        normalized_parents = tuple(parents)
        if not normalized_parents:
            raise SourceLifecycleError("derived node requires at least one parent")
        if any(not isinstance(parent, str) or not parent.strip() for parent in normalized_parents):
            raise SourceLifecycleError("parent node ids must not be blank")
        if len(set(normalized_parents)) != len(normalized_parents):
            raise SourceLifecycleError("parent node ids must be unique")
        if node_id in normalized_parents:
            raise SourceLifecycleError("a lifecycle node cannot depend on itself")
        if policy not in ("all", "any"):
            raise SourceLifecycleError("policy must be 'all' or 'any'")
        canonical = _canonical(reference)
        async with self._application._lock(self.scope_id):
            rows = await self._load_nodes()
            existing = rows.get(node_id)
            if existing is not None:
                if (
                    existing.family == family
                    and existing.reference == canonical
                    and existing.parents == normalized_parents
                    and existing.policy == policy
                ):
                    return
                raise SourceLifecycleError("node registration conflicts with the existing binding")
            parent_nodes = []
            for parent in normalized_parents:
                parent_node = rows.get(parent)
                if parent_node is None:
                    raise SourceLifecycleError("parent node is not registered")
                parent_nodes.append(parent_node)
            available = (
                all(parent.available for parent in parent_nodes)
                if policy == "all"
                else any(parent.available for parent in parent_nodes)
            )
            await self._insert_node(
                _Node(
                    node_id=node_id,
                    family=family,
                    reference=canonical,
                    parents=normalized_parents,
                    policy=policy,
                    position=_next_position(rows.values()),
                    available=available,
                    revision=None,
                    sequence=None,
                    last_reason=None,
                    applied=True,
                )
            )

    async def _persist_event(self, event: SourceLifecycleEvent, rows: dict[str, _Node]) -> None:
        async with self._runtime._provider.database.transaction() as connection:
                await _ensure_tables(connection)
                for node in rows.values():
                    await connection.execute(
                        text(
                            "UPDATE pc_source_lifecycle_nodes SET available = :available "
                            "WHERE scope_id = :scope_id AND node_id = :node_id"
                        ),
                        {
                            "available": 1 if node.available else 0,
                            "scope_id": self.scope_id,
                            "node_id": node.node_id,
                        },
                    )
                await connection.execute(
                    text(
                        "UPDATE pc_source_lifecycle_nodes SET sequence = :sequence, "
                        "last_reason = :reason, applied = 0 "
                        "WHERE scope_id = :scope_id AND node_id = :node_id"
                    ),
                    {
                        "sequence": event.sequence,
                        "reason": event.reason,
                        "scope_id": self.scope_id,
                        "node_id": event.node_id,
                    },
                )

    async def _mark_applied(self, node_id: str) -> None:
        async with self._runtime._provider.database.transaction() as connection:
                await _ensure_tables(connection)
                await connection.execute(
                    text(
                        "UPDATE pc_source_lifecycle_nodes SET applied = 1 "
                        "WHERE scope_id = :scope_id AND node_id = :node_id"
                    ),
                    {"scope_id": self.scope_id, "node_id": node_id},
                )

    async def _insert_node(self, node: _Node) -> None:
        async with self._runtime._provider.database.transaction() as connection:
                await _ensure_tables(connection)
                await connection.execute(
                    text(
                        "INSERT INTO pc_source_lifecycle_nodes "
                        "(scope_id, node_id, family, reference, parents, policy, node_position, "
                        "available, revision, sequence, last_reason, applied) "
                        "VALUES (:scope_id, :node_id, :family, :reference, :parents, :policy, "
                        ":node_position, :available, :revision, :sequence, :last_reason, :applied)"
                    ),
                    {
                        "scope_id": self.scope_id,
                        "node_id": node.node_id,
                        "family": node.family,
                        "reference": node.reference,
                        "parents": json.dumps(node.parents, separators=(",", ":")),
                        "policy": node.policy,
                        "node_position": node.position,
                        "available": 1 if node.available else 0,
                        "revision": node.revision,
                        "sequence": node.sequence,
                        "last_reason": node.last_reason,
                        "applied": 1 if node.applied else 0,
                    },
                )

    async def _load_nodes(self, *, create: bool = True) -> dict[str, _Node]:
        async with self._runtime._provider.database.transaction() as connection:
                if create:
                    await _ensure_tables(connection)
                else:
                    try:
                        await _ensure_tables(connection)
                    except Exception:
                        return {}
                result = await connection.execute(
                    text(
                        "SELECT node_id, family, reference, parents, policy, node_position, available, "
                        "revision, sequence, last_reason, applied "
                        "FROM pc_source_lifecycle_nodes WHERE scope_id = :scope_id "
                        "ORDER BY node_position"
                    ),
                    {"scope_id": self.scope_id},
                )
                rows = result.mappings().all()
        nodes: dict[str, _Node] = {}
        for row in rows:
            nodes[str(row["node_id"])] = _Node(
                node_id=str(row["node_id"]),
                family=str(row["family"]),  # type: ignore[arg-type]
                reference=str(row["reference"]),
                parents=tuple(json.loads(str(row["parents"]))),
                policy=str(row["policy"]),  # type: ignore[arg-type]
                position=int(row["node_position"]),
                available=bool(row["available"]),
                revision=None if row["revision"] is None else int(row["revision"]),
                sequence=None if row["sequence"] is None else int(row["sequence"]),
                last_reason=None if row["last_reason"] is None else str(row["last_reason"]),
                applied=bool(row["applied"]),
            )
        return nodes

    async def _ensure_memory_state(self, node: _Node, *, reason: str) -> None:
        if node.family != "memory":
            return
        citation = _reference_value(node)
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory node reference is invalid")
        memory = self._runtime.memory.for_scope(self.scope_id)
        entries = await memory.list(include_inactive=True)
        record = next((item for item in entries.entries if item.entry.entry_id == citation.entry_id), None)
        if record is None:
            raise SourceLifecycleError("memory citation is not readable")
        target = "active" if node.available else "inactive"
        if record.state == target:
            return
        request = RetireMemoryEntryRequest(citation=record.citation, reason=reason)
        if node.available:
            await memory.reactivate(request)
        else:
            await memory.retire(request)

    async def _reconcile_memory(self, rows: dict[str, _Node], *, reason: str) -> None:
        for node in sorted(rows.values(), key=lambda item: (item.position, item.node_id)):
            if node.family == "memory":
                await self._ensure_memory_state(node, reason=reason)

    async def _validate_source(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type, source_ref.source_id
            )
        except Exception as error:
            raise SourceLifecycleError("source reference is not readable") from error

    async def _validate_topic(self, artifact_ref: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision
            )
        except Exception as error:
            raise SourceLifecycleError("topic reference is not readable") from error

    async def _validate_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(
                GetMemoryEntryRequest(citation=citation)
            )
        except Exception as error:
            raise SourceLifecycleError("memory citation is not readable") from error


async def _ensure_tables(connection: Any) -> None:
    for statement in _LIFECYCLE_TABLES:
        await connection.execute(text(statement))


def _require_node_id(node_id: str) -> None:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node id must not be blank")


def _next_position(nodes: Any) -> int:
    return max((node.position for node in nodes), default=0) + 1


def _canonical(value: Any) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _reference_value(node: _Node) -> Any:
    payload = json.loads(node.reference)
    if node.family == "source":
        return SourceRef.model_validate(payload)
    if node.family == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


def _compute_states(rows: dict[str, _Node], source_states: dict[str, bool]) -> dict[str, bool]:
    memo: dict[str, bool] = {}

    def state(node_id: str) -> bool:
        if node_id in memo:
            return memo[node_id]
        node = rows[node_id]
        if node.family == "source":
            value = source_states.get(node_id, node.available)
        else:
            values = [state(parent) for parent in node.parents]
            value = all(values) if node.policy == "all" else any(values)
        memo[node_id] = value
        return value

    return {node_id: state(node_id) for node_id in rows}


def _available_ids(rows: dict[str, _Node]) -> list[str]:
    return sorted(node_id for node_id, node in rows.items() if node.available)


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]
