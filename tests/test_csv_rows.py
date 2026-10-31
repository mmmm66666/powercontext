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
import csv
import io
import json

import pytest

from powercontext.sources import (
    SourceDefinitionRegistry,
    SourceMaterialization,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.csv_rows import (
    CSV_ROW_SOURCE_DEFINITION,
    CSV_ROW_SOURCE_NAME,
    CSVDocumentInput,
    CSVRowInput,
    CSVRowSource,
    CSVRowSourceAdapter,
    CSVRowTextEvidenceProjection,
    csv_row_source_id,
    parse_csv_document,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


def _document(content: bytes, *, document_id: str = "airports", primary_key: str = "id") -> CSVDocumentInput:
    return CSVDocumentInput(document_id=document_id, primary_key=primary_key, content=content)


def _write(records: list[list[str]], *, header: list[str] | None = None, lineterminator: str = "\n") -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator=lineterminator)
    if header is not None:
        writer.writerow(header)
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def test_parse_preserves_original_strings_column_order_and_embedded_newlines() -> None:
    content = (
        b' id ,name,note,elevation_ft\r\n'
        b'001,"Airport, East","first line\r\nsecond ""quoted"" line",0\r\n'
        b'002," \xc3\xa9\xe4\xb8\xad\xe6\x96\x87 ",plain,0010\r\n'
    )
    rows = parse_csv_document(_document(content, document_id="airports:control", primary_key=" id "))

    assert [row.columns for row in rows] == [(" id ", "name", "note", "elevation_ft")] * 2
    assert rows[0].values == ("001", "Airport, East", 'first line\r\nsecond "quoted" line', "0")
    assert rows[1].values == ("002", " é中文 ", "plain", "0010")
    assert rows[0].primary_key_value == "001"
    assert rows[1].primary_key_value == "002"


def test_bom_is_accepted_only_at_the_start_and_header_only_documents_are_valid() -> None:
    rows = parse_csv_document(_document(b"\xef\xbb\xbfid,name\n\n"))
    assert rows == ()

    # Only the leading mark is a byte-order mark; a second one is header content.
    doubled = parse_csv_document(_document(b"\xef\xbb\xbf\xef\xbb\xbfid\n1\n", primary_key="\ufeffid"))
    assert doubled[0].columns == ("\ufeffid",)


def test_empty_physical_records_are_ignored_but_empty_field_records_are_not() -> None:
    content = b"id,name\n\n1,one\n\n\n2,two\n\n"
    rows = parse_csv_document(_document(content))
    assert [row.values for row in rows] == [("1", "one"), ("2", "two")]

    # A record made only of separators has cells and a blank primary key; it
    # must fail validation instead of being silently skipped as an empty line.
    with pytest.raises(ValueError):
        parse_csv_document(_document(b"id,name\n,\n"))


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xef\xbb\xbf",
        b"\nid,name\n1,one\n",
        b"id,id\n1,1\n",
        b"id,\n1,one\n",
        b",name\n1,one\n",
        b" \t,name\n1,one\n",
        b"id,name\n1,one\n1,duplicate\n",
        b"id,name\n,one\n",
        b"id,name\n   ,one\n",
        b"id,name\n1\n",
        b"id,name\n1,one,extra\n",
        b'id,name\n1,"unclosed\n',
        b"id,name\n\xff,invalid\n",
    ],
)
def test_invalid_documents_fail_atomically(content: bytes) -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document(content))


def test_primary_key_must_match_a_header_column_exactly() -> None:
    with pytest.raises(ValueError):
        parse_csv_document(_document(b"id,name\n1,one\n", primary_key=" id"))
    with pytest.raises(ValueError):
        parse_csv_document(_document(b" id,name\n1,one\n", primary_key="id"))

    row = parse_csv_document(_document(b" id ,name\n x ,one\n", primary_key=" id "))[0]
    assert row.primary_key == " id "


def test_primary_key_values_keep_surrounding_spaces_and_stay_distinct() -> None:
    first = parse_csv_document(_document(b"id,name\n key ,first\n"))[0]
    second = parse_csv_document(_document(b"id,name\nkey ,second\n"))[0]

    assert first.primary_key_value == " key "
    assert second.primary_key_value == "key "
    assert csv_row_source_id(first.document_id, first.primary_key_value) != csv_row_source_id(
        second.document_id, second.primary_key_value
    )

    with pytest.raises(ValueError):
        parse_csv_document(_document(b"id,name\n key ,first\n key ,duplicate\n"))


def test_row_identity_is_delimiter_safe_bounded_and_validated() -> None:
    assert csv_row_source_id("doc:one", "key/one") == "csv:doc%3Aone:key%2Fone"
    assert csv_row_source_id("doc:one", "key/one") != csv_row_source_id("doc:two", "key/one")
    assert csv_row_source_id("doc:one", "key/one") != csv_row_source_id("doc:one", "key:one")
    assert csv_row_source_id("space-keys", " key ") == "csv:space-keys:%20key%20"

    # ``csv:`` + document + ``:`` + key of length one fills the 256 limit.
    assert len(csv_row_source_id("x" * 250, "k")) == 256
    with pytest.raises(ValueError):
        csv_row_source_id("x" * 251, "k")

    for document_id, primary_key_value in [
        ("", "key"),
        (" doc ", "key"),
        ("doc", ""),
        ("doc", "   "),
        ("doc", "x" * 260),
    ]:
        with pytest.raises(ValueError):
            csv_row_source_id(document_id, primary_key_value)


def test_definition_manifest_and_projection_preserve_the_complete_record() -> None:
    row = parse_csv_document(
        _document(b"id,name,note\n01,\xc3\xa9,first \"quoted\" note\n", document_id="airports:control")
    )[0]

    assert CSV_ROW_SOURCE_DEFINITION.name == CSV_ROW_SOURCE_NAME == "csv-row"
    assert CSV_ROW_SOURCE_DEFINITION.version == "1"
    assert CSV_ROW_SOURCE_DEFINITION.input_class is CSVRowInput
    assert CSV_ROW_SOURCE_DEFINITION.source_class is CSVRowSource

    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    source = asyncio.run(registry.resolve(row))

    assert isinstance(source, CSVRowSource)
    assert source.materialization is SourceMaterialization.CAPTURED
    assert source.source_id == source.name == "csv:airports%3Acontrol:01"
    assert asyncio.run(registry.read(source)) == row

    projection = registry.project(source, TEXT_EVIDENCE_PROJECTION_KEY)
    expected_row = {"id": "01", "name": "é", "note": 'first "quoted" note'}
    assert projection == {
        "source_type": "csv-row",
        "source_id": "csv:airports%3Acontrol:01",
        "content": json.dumps(expected_row, ensure_ascii=False, separators=(",", ":")),
        "metadata": {
            "document_id": "airports:control",
            "primary_key": "id",
            "primary_key_value": "01",
            "columns": ["id", "name", "note"],
            "row": expected_row,
        },
    }

    manifest = manifest_for_definition(CSV_ROW_SOURCE_DEFINITION)
    assert manifest.name == "csv-row"
    assert manifest.version == "1"
    assert [projection.key for projection in manifest.projections] == [TEXT_EVIDENCE_PROJECTION_KEY]

    observation = project_source_for_transport(registry, source)
    assert observation.source_type == "csv-row"
    assert observation.payload["document_id"] == "airports:control"
    assert observation.payload["values"] == ["01", "é", 'first "quoted" note']
    assert observation.projection(TEXT_EVIDENCE_PROJECTION_KEY) == projection


def test_adapter_and_projection_are_usable_directly() -> None:
    adapter = CSVRowSourceAdapter()
    row = CSVRowInput(
        document_id="control",
        primary_key="id",
        primary_key_value="01",
        columns=("id", "name"),
        values=("01", "Alpha"),
    )
    source = asyncio.run(adapter.resolve(row))
    assert asyncio.run(adapter.read(source)) == row
    evidence = CSVRowTextEvidenceProjection().project(source)
    assert evidence.content == '{"id":"01","name":"Alpha"}'


def test_reordering_and_non_key_edits_keep_row_identity() -> None:
    columns = ("id", "name", "note")

    def parse(records: list[list[str]]) -> dict[str, str]:
        rows = parse_csv_document(_document(_write(records, header=list(columns)), document_id="control-table"))
        return {row.primary_key_value: csv_row_source_id(row.document_id, row.primary_key_value) for row in rows}

    original = parse([["01", "Alpha", "first"], ["02", "Beta", "second"]])
    updated = parse([["02", "Beta", "reordered"], ["01", "Alpha edited", "first"]])

    assert original == updated == {"01": "csv:control-table:01", "02": "csv:control-table:02"}
