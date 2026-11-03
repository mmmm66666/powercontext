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

"""HTTP cross-encoder reranking for Memory search candidates.

The service scores one query against a batch of candidates and returns a
score per original candidate rank. The client must split oversized pools
without renumbering candidates, validate that every batch response covers
exactly the ranks it sent, and then choose the final hits from one global
ordering over the complete pool.
"""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score Memory candidates with one HTTP service, preserving their identities."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score the complete candidate pool in batches and select the global best."""

        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = list(range(start + 1, start + 1 + len(batch)))
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text}
                        for rank, candidate in zip(ranks, batch, strict=True)
                    ],
                },
            )
            requests += 1
            if not 200 <= response.status_code < 300:
                raise httpx.HTTPStatusError(
                    f"HTTP rerank service returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
            scores.update(_parse_scores(response, ranks))

        selected = tuple(
            rank for rank, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        )
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))


def _parse_scores(response: httpx.Response, expected_ranks: list[int]) -> dict[int, float]:
    """Validate one response against exactly the ranks that were requested."""

    try:
        body = response.json()
    except ValueError as error:
        raise ValueError("HTTP rerank response must be valid JSON") from error  # noqa: TRY003
    if not isinstance(body, dict) or set(body) != {"scores"}:
        raise ValueError("HTTP rerank response must contain exactly scores")  # noqa: TRY003
    records = body["scores"]
    if not isinstance(records, list):
        raise ValueError("HTTP rerank scores must be a list")  # noqa: TRY003

    scores: dict[int, float] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"rank", "score"}:
            raise ValueError("HTTP rerank score records must contain exactly rank and score")  # noqa: TRY003
        rank = record["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("HTTP rerank ranks must be integers")  # noqa: TRY003
        if rank in scores:
            raise ValueError("HTTP rerank response must not repeat a rank")  # noqa: TRY003
        scores[rank] = _finite_score(record["score"])

    if sorted(scores) != expected_ranks:
        raise ValueError("HTTP rerank response must cover exactly the requested ranks")  # noqa: TRY003
    return scores


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("HTTP rerank scores must be finite numbers")  # noqa: TRY003
    try:
        score = float(value)
    except OverflowError:
        raise ValueError("HTTP rerank scores must be finite numbers") from None  # noqa: TRY003
    if not math.isfinite(score):
        raise ValueError("HTTP rerank scores must be finite numbers")  # noqa: TRY003
    return score


__all__ = [
    "DEFAULT_POLICY_ID",
    "HTTPMemoryReranker",
]
