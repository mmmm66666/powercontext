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

"""Captured per-row CSV Sources with stable, row-addressable identities.

A table row is evidence only when its original fields, column order, and
values survive capture. Each row therefore becomes its own ``CAPTURED``
Source identified by the document plus the primary key value, so a citation
can be resolved back to the complete record after reordering or a restart.
"""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"


def _is_blank(value: str) -> bool:
    """Classify empty or whitespace-only values without altering them.

    ``str.strip`` only feeds the boolean result; stored fields are never
    trimmed, so surrounding spaces stay part of the original evidence.
    """

    return not value.strip()


def _validate_document_id(document_id: str) -> str:
    """Reject document identities that would make row identities ambiguous."""

    if not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and must not have surrounding whitespace")  # noqa: TRY003
    return document_id


def _validate_primary_key(primary_key: str) -> str:
    """Reject a missing primary key while keeping surrounding spaces meaningful."""

    if _is_blank(primary_key):
        raise ValueError("primary_key must be non-empty")  # noqa: TRY003
    return primary_key


def _validate_primary_key_value(primary_key_value: str) -> str:
    """Reject a row primary key value that cannot address a record."""

    if _is_blank(primary_key_value):
        raise ValueError("primary_key_value must be non-empty")  # noqa: TRY003
    return primary_key_value


class CSVDocumentInput(BaseModel):
    """One complete CSV document selected by a caller-stable identity."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_document_id(value)

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        return _validate_primary_key(value)


class CSVRowInput(BaseModel):
    """One validated data row ready to be materialized as a Source."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured exact CSV row addressable by document and primary key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a complete CSV document before returning any row.

    The first record is the header. Every data record must have the header
    width, a non-blank primary key value, and a primary key value unique in
    the document. Completely empty physical records are ignored; a document
    with only a header yields no rows.
    """

    rows = _read_rows(value.content)
    if not rows or not rows[0]:
        raise ValueError("CSV document must contain a non-empty header row")  # noqa: TRY003

    columns = tuple(rows[0])
    primary_key_index = _validate_header(columns, value.primary_key)

    seen_primary_key_values: set[str] = set()
    result: list[CSVRowInput] = []
    for row in rows[1:]:
        if not row:
            # csv.reader yields an empty list only for a completely empty
            # physical record. A record of empty fields still has cells and
            # must be validated like any other row.
            continue
        primary_key_value = _validate_row(row, columns, primary_key_index, seen_primary_key_values)
        result.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(row),
            )
        )
    return tuple(result)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable identity of one CSV row Source.

    Both components are percent-encoded separately before joining, so ``:``
    inside either component cannot shift the boundary between them.
    """

    quoted_document = quote(_validate_document_id(document_id), safe="")
    quoted_value = quote(_validate_primary_key_value(primary_key_value), safe="")
    source_id = f"csv:{quoted_document}:{quoted_value}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def _read_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        # ``utf-8-sig`` accepts one leading UTF-8 BOM and is otherwise strict.
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
        raise ValueError("CSV document has malformed quoting") from None  # noqa: TRY003


def _validate_header(columns: tuple[str, ...], primary_key: str) -> int:
    for column in columns:
        if _is_blank(column):
            raise ValueError("CSV header must not contain empty column names")  # noqa: TRY003
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header must not contain duplicate column names")  # noqa: TRY003
    try:
        return columns.index(primary_key)
    except ValueError:
        raise ValueError("CSV header must contain the primary key column exactly") from None  # noqa: TRY003


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
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
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
        row = _row_object(source.columns, source.values)
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


def _row_object(columns: tuple[str, ...], values: tuple[str, ...]) -> dict[str, str]:
    """Pair stored columns and values, preserving the original column order."""

    try:
        return dict(zip(columns, values, strict=True))
    except ValueError:
        raise ValueError("CSV row columns and values must have the same length") from None  # noqa: TRY003


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSVRowSourceAdapter(),
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
