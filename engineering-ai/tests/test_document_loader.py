"""Tests for Stage 2: app/extraction/document_loader.py.

Run from the repo root (engineering-ai/):

    python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.document_loader import (
    ExtractedDocument,
    Table,
    UnsupportedFileTypeError,
    load_directory,
    load_document,
    load_many,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def find_table(doc: ExtractedDocument, location_substr: str) -> Table:
    for table in doc.tables:
        if location_substr in table.location:
            return table
    raise AssertionError(f"no table with location containing {location_substr!r}")


def test_txt_loads_text_with_provenance():
    doc = load_document(FIXTURES / "project_info.txt")
    assert doc.file_type == "txt"
    assert len(doc.text_chunks) == 1
    chunk = doc.text_chunks[0]
    assert chunk.location.startswith("line 1..")
    assert "324 mm OD" in chunk.text
    assert "Fy = 450 MPa" in chunk.text
    assert doc.warnings == []


def test_txt_empty_file_warns(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text(" ", encoding="utf-8")
    doc = load_document(empty)
    assert doc.total_text_chars() == 0
    assert any("empty" in w for w in doc.warnings)


def test_csv_comma_delimited():
    doc = load_document(FIXTURES / "pile_schedule.csv")
    assert doc.file_type == "csv"
    table = find_table(doc, "whole file")
    assert "Pile" in table.headers
    assert "HP1" in [r[0] for r in table.rows]
    assert str(324.0) not in {c for r in table.rows for c in r}  # floats normalised


def test_csv_semicolon_delimited_sniffed():
    doc = load_document(FIXTURES / "loads_semicolon.csv")
    table = find_table(doc, "whole file")
    assert "Load Case" in table.headers
    assert "ULS-1" in [r[0] for r in table.rows]
    assert "980" in table.rows[0]


def test_xlsx_two_sheets_with_provenance():
    doc = load_document(FIXTURES / "design_report.xlsx")
    assert doc.file_type == "xlsx"
    pile = find_table(doc, "sheet: Pile Schedule")
    assert "Diameter (mm)" in pile.headers
    assert "324" in [r[1] for r in pile.rows]
    loads = find_table(doc, "sheet: Loads")
    assert "Compression (kN)" in loads.headers
    assert "ULS-1" in [r[0] for r in loads.rows]
    assert any(c.location == "sheet: Pile Schedule, row 1" for c in doc.text_chunks)


def test_xlsx_formula_without_cached_value_warns(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "S1"
    ws.append(["A", "B"])
    ws.append([1, "=A2*2"])  # never saved by Excel -> no cached value
    path = tmp_path / "formula.xlsx"
    wb.save(path)

    doc = load_document(path)
    assert any("has no cached value" in w for w in doc.warnings)


def test_docx_paragraphs_and_table():
    doc = load_document(FIXTURES / "design_report.docx")
    assert doc.file_type == "docx"
    texts = " ".join(c.text for c in doc.text_chunks)
    assert "Helical Pile Design Report" in texts
    table = find_table(doc, "table 1")
    assert "Diameter (mm)" in table.headers
    assert len(table.rows) == len(["HP1", "HP2", "HP3"])


def test_pdf_text_and_detected_table():
    doc = load_document(FIXTURES / "design_report.pdf")
    assert doc.file_type == "pdf"
    texts = " ".join(c.text for c in doc.text_chunks)
    assert "Helical Pile Design Report" in texts
    assert "324 mm OD" in texts
    table = find_table(doc, "page 1, table")
    assert "Pile" in table.headers
    assert "HP1" in [r[0] for r in table.rows]


def test_scanned_pdf_flagged_not_silently_empty():
    doc = load_document(FIXTURES / "scanned_page.pdf")
    assert doc.total_text_chars() == 0
    assert any("OCR" in w for w in doc.warnings)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"not a pdf")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document(FIXTURES / "does_not_exist.pdf")


def test_directory_loads_only_supported():
    docs = load_directory(FIXTURES)
    types = sorted({d.file_type for d in docs})
    assert types == ["csv", "docx", "pdf", "txt", "xlsx"]
    by_name = {d.path.name: d for d in docs}
    assert set(by_name) == {
        "project_info.txt",
        "full_report.txt",
        "pile_schedule.csv",
        "loads_semicolon.csv",
        "design_report.xlsx",
        "design_report.docx",
        "design_report.pdf",
        "scanned_page.pdf",
    }


def test_load_many_mixes_files_and_directory(tmp_path):
    docs = load_many([FIXTURES / "project_info.txt", tmp_path])
    assert all(isinstance(d, ExtractedDocument) for d in docs)


def test_full_text_contains_locations():
    doc = load_document(FIXTURES / "design_report.xlsx")
    full = doc.full_text()
    assert "[sheet: Pile Schedule, row 1]" in full
    assert "324" in full