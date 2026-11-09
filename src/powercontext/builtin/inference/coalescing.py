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

"""Cross-call embedding coalescing with a bounded successful-vector cache."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True, eq=False)
class _Call:
    """One caller's immutable input and the state needed to assemble its result."""

    texts: tuple[str, ...]
    unique: tuple[str, ...]
    future: asyncio.Future[EmbeddingResult]
    vectors: dict[str, EmbeddingVector] = field(default_factory=dict)
    remaining: set[str] = field(default_factory=set)
    batches: dict[int, InferenceUsage] = field(default_factory=dict)
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.remaining = set(self.unique)


@dataclass(slots=True)
class _Batch:
    """One underlying provider request and the callers waiting on it."""

    batch_id: int
    texts: tuple[str, ...]
    calls: set[_Call] = field(default_factory=set)


class CoalescingEmbeddingModel:
    """Share in-flight embedding work across calls and cache successful vectors.

    The wrapper is intentionally independent of any provider: it preserves the
    delegate's ``profile`` and only coordinates batches, callers, usage totals and a
    bounded LRU of successful vectors.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be a non-negative finite number")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._lock = asyncio.Lock()
        self._pending: dict[str, list[_Call]] = {}
        self._inflight: dict[str, _Batch] = {}
        self._flush_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_batch_id = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return vectors for ``texts`` in input order, sharing pending work."""

        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(requested))
        future: asyncio.Future[EmbeddingResult] = asyncio.get_running_loop().create_future()
        call = _Call(texts=requested, unique=unique, future=future)

        async with self._lock:
            for text in unique:
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    self._resolve_text(call, text, cached)
                    continue
                batch = self._inflight.get(text)
                if batch is not None:
                    batch.calls.add(call)
                    call.batches[batch.batch_id] = _EMPTY_USAGE
                    continue
                self._pending.setdefault(text, []).append(call)
            self._finalize(call)
            if not call.future.done() and any(text in self._pending for text in unique):
                self._ensure_flush_locked()

        if call.future.done():
            return call.future.result()
        try:
            return await asyncio.shield(call.future)
        except asyncio.CancelledError:
            # The per-call future is deliberately not cancelled: other callers may
            # still depend on the same provider batch.
            call.cancelled = True
            raise

    def _ensure_flush_locked(self) -> None:
        task = self._flush_task
        if task is None or task.done():
            self._flush_task = asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            else:
                # A zero-millisecond window still yields to let calls already ready in
                # this event-loop turn register before the first snapshot.
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            while True:
                async with self._lock:
                    if not self._pending:
                        self._flush_task = None
                        return
                    pending = self._pending
                    self._pending = {}
                    ready: list[str] = []
                    for text, calls in pending.items():
                        cached = self._cache.get(text)
                        if cached is not None:
                            self._cache.move_to_end(text)
                            for call in calls:
                                self._resolve_text(call, text, cached)
                                self._finalize(call)
                            continue
                        batch = self._inflight.get(text)
                        if batch is not None:
                            for call in calls:
                                batch.calls.add(call)
                                call.batches[batch.batch_id] = _EMPTY_USAGE
                            continue
                        ready.append(text)
                    for index in range(0, len(ready), self._max_batch_size):
                        chunk = tuple(ready[index : index + self._max_batch_size])
                        batch = _Batch(batch_id=self._next_batch_id, texts=chunk)
                        self._next_batch_id += 1
                        for text in chunk:
                            self._inflight[text] = batch
                        for text in chunk:
                            for call in pending[text]:
                                batch.calls.add(call)
                                call.batches[batch.batch_id] = _EMPTY_USAGE
                        task = asyncio.create_task(self._execute(batch))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
        except asyncio.CancelledError:
            raise

    async def _execute(self, batch: _Batch) -> None:
        error: BaseException | None = None
        result: EmbeddingResult | None = None
        try:
            result = await self._model.embed(batch.texts)
            if len(result.vectors) != len(batch.texts):
                raise ValueError("embedding vector count does not match batch size")  # noqa: TRY003
        except BaseException as caught:  # provider failures belong to this batch only
            error = caught

        completed: list[_Call] = []
        async with self._lock:
            for text in batch.texts:
                self._inflight.pop(text, None)
            if error is not None:
                for call in batch.calls:
                    if not call.future.done() and not call.cancelled:
                        call.future.set_exception(error)
            else:
                assert result is not None
                usage = result.usage
                vectors_by_text = dict(zip(batch.texts, result.vectors, strict=True))
                for text, vector in vectors_by_text.items():
                    if self._cache_size:
                        self._cache[text] = vector
                        self._cache.move_to_end(text)
                self._trim_cache()
                for call in batch.calls:
                    if call.future.done():
                        continue
                    for text in batch.texts:
                        if text in call.remaining:
                            call.remaining.remove(text)
                            call.vectors[text] = vectors_by_text[text]
                    call.batches[batch.batch_id] = usage
                    completed.append(call)
        for call in completed:
            self._finalize(call)

    def _trim_cache(self) -> None:
        while self._cache_size and len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _resolve_text(call: _Call, text: str, vector: EmbeddingVector) -> None:
        if text in call.remaining:
            call.remaining.remove(text)
            call.vectors[text] = vector

    def _finalize(self, call: _Call) -> None:
        if call.future.done() or call.remaining:
            return
        call.future.set_result(
            EmbeddingResult(
                vectors=tuple(call.vectors[text] for text in call.texts),
                usage=_sum_usage(call.batches.values()),
            )
        )


_EMPTY_USAGE = InferenceUsage(requests=0)


def _sum_usage(usages: Any) -> InferenceUsage:
    """Sum distinct provider batches while preserving missing token fields."""

    values = tuple(usages)
    if not values:
        return InferenceUsage(requests=0)
    input_tokens = [value.input_tokens for value in values if value.input_tokens is not None]
    output_tokens = [value.output_tokens for value in values if value.output_tokens is not None]
    return InferenceUsage(
        requests=sum(value.requests for value in values),
        input_tokens=sum(input_tokens) if input_tokens else None,
        output_tokens=sum(output_tokens) if output_tokens else None,
    )


__all__ = ["CoalescingEmbeddingModel"]
