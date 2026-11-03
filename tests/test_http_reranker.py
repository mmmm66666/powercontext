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

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory.http_reranker import DEFAULT_POLICY_ID, HTTPMemoryReranker
from powercontext.builtin.artifacts.memory.models import MemoryHit


def _hit(rank: int, *, text: str | None = None) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=f"candidate {rank}" if text is None else text,
        score=1.0 / rank,
        matched_by=("fts",),
    )


def _candidates(count: int) -> tuple[MemoryHit, ...]:
    return tuple(_hit(rank) for rank in range(1, count + 1))


def _service(scores: dict[int, float], seen: list[dict] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        return httpx.Response(
            200,
            json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]},
        )

    return httpx.MockTransport(handler)


def test_batches_without_renumbering_and_orders_globally() -> None:
    seen: list[dict] = []

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=_service({1: 0.2, 2: 0.9, 3: 0.9}, seen)) as client:
            reranker = HTTPMemoryReranker("https://reranker.invalid/score", client=client, batch_size=2)
            decision = await reranker.rerank("query", _candidates(3), 2)

        assert decision.selected_ranks == (2, 3)
        assert decision.usage.requests == 2
        assert len(seen) == 2

    asyncio.run(scenario())

    assert seen[0]["query"] == "query"
    assert [len(request["candidates"]) for request in seen] == [2, 1]
    assert [item["rank"] for request in seen for item in request["candidates"]] == [1, 2, 3]
    for request in seen:
        for item in request["candidates"]:
            assert type(item["rank"]) is int
            assert item["text"] == f"candidate {item['rank']}"


def test_scores_can_arrive_out_of_order_and_custom_policy_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        records = [{"rank": item["rank"], "score": 1.0 / item["rank"]} for item in body["candidates"]]
        return httpx.Response(200, json={"scores": list(reversed(records))})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(
                "https://invalid", client=client, batch_size=2, policy_id="custom-policy"
            )
            decision = await reranker.rerank("q", _candidates(3), 3)
            assert decision.selected_ranks == (1, 2, 3)
            assert decision.usage.requests == 2

    asyncio.run(scenario())


def test_ties_break_by_original_global_rank_across_batches() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(transport=_service({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.9})) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client, batch_size=2)
            decision = await reranker.rerank("q", _candidates(4), 3)
            assert decision.selected_ranks == (4, 1, 2)

    asyncio.run(scenario())


def test_default_policy_id() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(transport=_service({1: 1.0})) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            assert reranker.policy_id == DEFAULT_POLICY_ID == "http-cross-encoder-v1"

    asyncio.run(scenario())


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def _no_request(_request: httpx.Request) -> httpx.Response:
    pytest.fail("no request expected")


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_fails_before_any_request(batch_size) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_no_request)) as client:
            with pytest.raises(ValueError):
                HTTPMemoryReranker("https://invalid", client=client, batch_size=batch_size)

    asyncio.run(scenario())


@pytest.mark.parametrize(("candidate_count", "limit"), [(0, 0), (1, 0), (1, 2)])
def test_invalid_bounds_fail_before_any_request(candidate_count: int, limit: int) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_no_request)) as client:
            reranker = HTTPMemoryReranker("https://invalid", client=client)
            with pytest.raises(ValueError):
                await reranker.rerank("q", _candidates(candidate_count), limit)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {"scores": []},
        {"scores": [], "extra": True},
        {},
        {"scores": {"rank": 1, "score": 0.5}},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 2, "score": 0.4}]},
        {"scores": [{"rank": 1, "score": 0.5}, {"rank": 1, "score": 0.4}]},
        {"scores": [{"rank": 1.0, "score": 0.5}]},
        {"scores": [{"rank": "1", "score": 0.5}]},
        {"scores": [{"rank": True, "score": 0.5}]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"rank": 1, "score": 0.5, "extra": True}]},
        {"scores": [{"rank": 1, "score": "0.5"}]},
        {"scores": [{"rank": 1, "score": True}]},
        {"scores": [{"rank": 1, "score": None}]},
        {"scores": [{"rank": 1, "score": float("nan")}]},
        {"scores": [{"rank": 1, "score": float("inf")}]},
        {"scores": [{"rank": 1, "score": float("-inf")}]},
    ],
)
def test_invalid_response_schema_and_scores_raise_value_error(payload) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_invalid_json_response_raises_value_error() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"{not json"))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_http_status_and_transport_errors_are_preserved() -> None:
    async def status_scenario(status: int) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(status))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    async def transport_scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.ConnectError):
                await HTTPMemoryReranker("https://invalid", client=client).rerank("q", (_hit(1),), 1)

    for status in (302, 404, 503):
        asyncio.run(status_scenario(status))
    asyncio.run(transport_scenario())


def test_scores_map_back_to_candidate_identity_by_original_rank() -> None:
    candidates = (_hit(1, text="first"), _hit(2, text="second"), _hit(3, text="third"))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=_service({1: 0.1, 2: 0.7, 3: 0.9})) as client:
            decision = await HTTPMemoryReranker("https://invalid", client=client, batch_size=1).rerank(
                "q", candidates, 2
            )

        assert decision.selected_ranks == (3, 2)
        assert [candidates[rank - 1].entry_version_id for rank in decision.selected_ranks] == ["v3", "v2"]

    asyncio.run(scenario())
