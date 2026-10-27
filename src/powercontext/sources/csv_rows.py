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

"""Captured per-row CSV Sources with stable, row-addressable identities."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"


def _is_blank(value: str) -> bool:
    """Return True when the value is empty or contains only whitespace.

    Uses ``str.strip`` only on a throwaway copy for classification; user values
    are never mutated — this does not perform a trim on stored data.
    """

    return not value.strip()


@dataclass(frozen=True, slots=True)
class CSVRowInput:
    """One validated data row ready to be materialized as a Source."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured exact CSV row addressable by primary key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(
    raw_bytes: bytes,
    document_id: str,
    primary_key: str,
) -> tuple[CSVRowInput, ...]:
    """Parse and validate a complete CSV document before returning any row."""

    rows = _read_rows(raw_bytes)
    if not rows or not rows[0]:
        raise ValueError("CSV document must contain a header row")  # noqa: TRY003

    columns = tuple(rows[0])
    primary_key_index = _validate_header(columns, primary_key)

    seen_primary_key_values: set[str] = set()
    result: list[CSVRowInput] = []
    for row in rows[1:]:
        if not any(row):
            continue
        primary_key_value = _validate_row(row, columns, primary_key_index, seen_primary_key_values)
        result.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(row),
            )
        )
    return tuple(result)


def csv_row_source_id(primary_key: str, primary_key_value: str) -> str:
    """Build the stable identity of one CSV row Source."""

    source_id = f"{quote(primary_key, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def _read_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("CSV document must be valid UTF-8") from None  # noqa: TRY003
    try:
        return list(
            csv.reader(
                io.StringIO(text, newline=""),
                strict=True,
                quotechar='"',
                delimiter=",",
            )
        )
    except csv.Error:
        raise ValueError("CSV document is malformed") from None  # noqa: TRY003


def _validate_header(columns: tuple[str, ...], primary_key: str) -> int:
    for column in columns:
        if _is_blank(column):
            raise ValueError("CSV header must not contain empty column names")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header must not contain duplicate column names")  # noqa: TRY003
    if primary_key not in columns:
        raise ValueError("CSV header must contain the primary key column")  # noqa: TRY003
    return columns.index(primary_key)


def _validate_row(
    row: list[str],
    columns: tuple[str, ...],
    primary_key_index: int,
    seen_primary_key_values: set[str],
) -> str:
    if len(row) != len(columns):
        raise ValueError("CSV row cell count must match the header column count")  # noqa: TRY003
    primary_key_value = row[primary_key_index]
    if _is_blank(primary_key_value):
        raise ValueError("CSV row primary key value must not be blank")  # noqa: TRY003
    if primary_key_value in seen_primary_key_values:
        raise ValueError("CSV row primary key values must be unique within a document")  # noqa: TRY003
    seen_primary_key_values.add(primary_key_value)
    return primary_key_value


class CSVRowSourceAdapter:
    """Materialize CSV row inputs as captured Source evidence."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput) -> CSVRowSource:
        source_id = csv_row_source_id(value.primary_key, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row through the shared text-evidence capability."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
