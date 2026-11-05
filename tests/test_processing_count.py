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

import pytest
from pydantic import ValidationError

from powercontext.builtin.runtime.artifact_processing import ArtifactProcessingBinding
from powercontext.builtin.runtime.config import RuntimeConfig

FIELDS = ("memory_source_count_threshold", "topic_memory_source_count_threshold")


def test_source_count_thresholds_are_strict_optional_positive_integers() -> None:
    for field in FIELDS:
        assert getattr(RuntimeConfig(), field) is None
        for value in (None, 1, 3, 100):
            assert getattr(RuntimeConfig(**{field: value}), field) == value
        for value in (True, False, 0, -1, 1.0, "3"):
            with pytest.raises(ValidationError):
                RuntimeConfig(**{field: value})


@pytest.mark.parametrize("threshold", [None, 1, 7])
def test_binding_accepts_positive_source_count_thresholds(threshold: int | None) -> None:
    binding = ArtifactProcessingBinding(
        binding_name="memory-source-window",
        artifact_family="memory",
        launcher=object(),  # type: ignore[arg-type]
        source_count_threshold=threshold,
    )
    assert binding.source_count_threshold == threshold


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5, "3"])
def test_binding_rejects_invalid_source_count_thresholds(threshold: object) -> None:
    with pytest.raises(ValueError):
        ArtifactProcessingBinding(
            binding_name="memory-source-window",
            artifact_family="memory",
            launcher=object(),  # type: ignore[arg-type]
            source_count_threshold=threshold,  # type: ignore[arg-type]
        )
