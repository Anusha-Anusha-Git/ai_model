"""Tests for Stage 1: app/report/pdf_report.py.

Run from the repo root (engineering-ai/):

    python -m pytest tests/test_pdf_report.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.report.pdf_report import (
    CONFLICT_FLAG,
    NOT_FOUND,
    SECTION_ORDER,
    ExtractionPayload,
    build_calculation_rows,
    build_evaluation_rows,
    build_project_rows,
    build_reference_rows,
    build_summary_paragraphs,
    generate_pdf,
    main,
)

# A realistic slice of ExtractionResult.to_dict(): a project title page, a pile
# section table, a helix table, load cases and a ratio summary.
FIXTURE: dict = {
    "fields": [
        {
            "parameter": "Project Name", "value": "Riverside Viaduct Pier 4",
            "unit": None, "source": "design_report.pdf",
            "location": "page 1, table 1, row 2", "context": "", "section": "",
            "raw": "Riverside Viaduct Pier 4",
        },
        {
            "parameter": "Client", "value": "Karnataka Roads Dept",
            "unit": None, "source": "design_report.pdf",
            "location": "page 1, table 1, row 4", "context": "", "section": "",
            "raw": "Karnataka Roads Dept",
        },
        {
            "parameter": "Dp", "value": 324.0, "unit": "mm",
            "source": "design_report.pdf", "location": "page 3, table 2, row 2",
            "context": "P1", "section": "PILE SECTION", "raw": "324",
        },
        {
            "parameter": "tP", "value": 12.7, "unit": "mm",
            "source": "design_report.pdf", "location": "page 3, table 2, row 2",
            "context": "P1", "section": "PILE SECTION", "raw": "12.7",
        },
        {
            "parameter": "Fy", "value": 450.0, "unit": "MPa",
            "source": "design_report.pdf", "location": "page 3, table 2, row 2",
            "context": "P1", "section": "PILE SECTION", "raw": "450",
        },
        {
            "parameter": "Numbe", "value": 3.0, "unit": "plates",
            "source": "design_report.pdf", "location": "page 4, table 3, row 1",
            "context": "Plate 1", "section": "HELIX", "raw": "3",
        },
        {
            "parameter": "compression", "value": 980.0, "unit": "kN",
            "source": "design_report.pdf", "location": "page 7, table 5, row 2",
            "context": "ULS-1", "section": "LOADS", "raw": "980",
        },
        {
            "parameter": "tension", "value": 410.0, "unit": "kN",
            "source": "design_report.pdf", "location": "page 7, table 5, row 2",
            "context": "ULS-2", "section": "LOADS", "raw": "410",
        },
        {
            "parameter": "R/CAP", "value": 0.87, "unit": None,
            "source": "design_report.pdf", "location": "page 9, table 7, row 3",
            "context": "ULS-1", "section": "RESULTS", "raw": "0.87",
        },
        {
            "parameter": "R/CAP", "value": 1.04, "unit": None,
            "source": "design_report.pdf", "location": "page 9, table 7, row 5",
            "context": "ULS-2", "section": "RESULTS", "raw": "1.04",
        },
        # Same parameter, two different values -> must be flagged, never merged.
        {
            "parameter": "Fy", "value": 345.0, "unit": "MPa",
            "source": "appendix.pdf", "location": "page 2, table 1, row 1",
            "context": "P9", "section": "APPENDIX", "raw": "345",
        },
        {
            "parameter": "settlement", "value": 14.2, "unit": "mm",
            "source": "design_report.pdf", "location": "page 10, table 8, row 2",
            "context": "ULS-1", "section": "RESULTS", "raw": "14.2",
        },
    ],
    "conflicts": [
        {
            "parameter": "Fy", "kind": "value",
            "fields": [],
            "detail": "value conflict for 'Fy': '450' (MPa) and '345' (MPa)",
        }
    ],
    "warnings": [
        "design_report.pdf, page 11, table 9, row 2: cell 'TBD' for "
        "'Corrosion' is not a number; kept as raw text.",
    ],
}


@pytest.fixture
def payload() -> ExtractionPayload:
    return ExtractionPayload.from_dict(FIXTURE)


# --------------------------------------------------------------------------
# Fixed layout
# --------------------------------------------------------------------------


def test_section_order_is_the_required_fixed_contract() -> None:
    assert SECTION_ORDER == (
        "Project Information",
        "Executive Summary",
        "Load Summary",
        "Engineering Calculations",
        "Results",
        "Engineering Evaluation",
        "References",
    )


def test_layout_is_identical_for_two_different_documents(payload: ExtractionPayload) -> None:
    """Only content may vary between documents -- never the section set."""
    other = ExtractionPayload.from_dict({"fields": [], "conflicts": [], "warnings": []})
    empty_project = {row.label for row in build_project_rows(other)}
    real_project = {row.label for row in build_project_rows(payload)}
    assert empty_project == real_project

    empty_calc = {row.label for row in build_calculation_rows(other)}
    real_calc = {row.label for row in build_calculation_rows(payload)}
    assert empty_calc == real_calc


# --------------------------------------------------------------------------
# Accuracy: value, unit, source page
# --------------------------------------------------------------------------


def test_value_keeps_its_unit_and_source_page(payload: ExtractionPayload) -> None:
    rows = {row.label: row for row in build_calculation_rows(payload)}
    diameter = rows["Pile Diameter (O.D.)"]
    assert diameter.value == "324 mm"
    assert "page 3" in diameter.citation
    assert "design_report.pdf" in diameter.citation


def test_missing_field_prints_not_found_in_document() -> None:
    empty = ExtractionPayload.from_dict({"fields": []})
    rows = {row.label: row for row in build_calculation_rows(empty)}
    assert rows["Pile Diameter (O.D.)"].display_value() == NOT_FOUND
    assert rows["Pile Diameter (O.D.)"].is_missing


def test_conflicting_parameter_is_flagged_not_silently_resolved(payload: ExtractionPayload) -> None:
    rows = {row.label: row for row in build_calculation_rows(payload)}
    yield_strength = rows["Yield Strength (Fy)"]
    assert yield_strength.is_conflict
    assert CONFLICT_FLAG in yield_strength.display_value()


def test_empty_payload_still_renders_all_rows() -> None:
    empty = ExtractionPayload.from_dict({"fields": []})
    assert len(build_project_rows(empty)) == 8
    assert all(row.is_missing for row in build_project_rows(empty))


# --------------------------------------------------------------------------
# Executive summary: numbers come from the JSON, never from a model
# --------------------------------------------------------------------------


def test_summary_numbers_are_interpolated_from_the_json(payload: ExtractionPayload) -> None:
    clauses, _missing = build_summary_paragraphs(payload)
    joined = " ".join(clauses)
    assert "324 mm" in joined
    assert "12.7 mm" in joined
    assert "450 MPa" in joined
    assert "3 plates" in joined
    assert "1.04" in joined  # largest ratio, taken from the JSON


def test_summary_reports_absent_values_instead_of_guessing() -> None:
    empty = ExtractionPayload.from_dict({"fields": []})
    clauses, missing = build_summary_paragraphs(empty)
    assert not [c for c in clauses if "324" in c]
    assert any("Pile Diameter (O.D.)" in item for item in missing)
    assert all(NOT_FOUND in item for item in missing)


def test_summary_flags_conflicts_without_resolving_them(payload: ExtractionPayload) -> None:
    clauses, _missing = build_summary_paragraphs(payload)
    assert any(CONFLICT_FLAG in clause for clause in clauses)


# --------------------------------------------------------------------------
# Deterministic evaluation (not certification)
# --------------------------------------------------------------------------


def test_evaluation_compares_each_reported_ratio_against_one(payload: ExtractionPayload) -> None:
    rows = {row.label: row for row in build_evaluation_rows(payload)}
    assert "within 1.00" in rows["ULS-1"].value
    assert "exceeds 1.00" in rows["ULS-2"].value
    assert rows["Governing ratio"].value == "1.04"


def test_evaluation_reports_missing_rather_than_defaulting() -> None:
    empty = ExtractionPayload.from_dict({"fields": []})
    rows = build_evaluation_rows(empty)
    assert len(rows) == 1 and rows[0].is_missing


# --------------------------------------------------------------------------
# References / traceability
# --------------------------------------------------------------------------


def test_references_list_every_source_document(payload: ExtractionPayload) -> None:
    labels = [row.label for row in build_reference_rows(payload)]
    assert "Source document: design_report.pdf" in labels
    assert "Source document: appendix.pdf" in labels


def test_references_carry_extractor_warnings(payload: ExtractionPayload) -> None:
    rows = build_reference_rows(payload)
    assert any(row.label == "Extractor warning" and "not a number" in row.value for row in rows)


# --------------------------------------------------------------------------
# PDF output
# --------------------------------------------------------------------------


def test_generate_pdf_writes_a_real_pdf(payload: ExtractionPayload, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "review.pdf"
    written = generate_pdf(payload, out)

    assert written.exists()
    assert written.stat().st_size > 1000
    assert written.read_bytes().startswith(b"%PDF-")


def test_generate_pdf_accepts_a_plain_dict(tmp_path: Path) -> None:
    out = generate_pdf(FIXTURE, tmp_path / "from_dict.pdf")
    assert out.exists() and out.read_bytes().startswith(b"%PDF-")


def test_generate_pdf_succeeds_on_an_empty_payload(tmp_path: Path) -> None:
    """A document with nothing extractable must still produce all 7 sections."""
    out = generate_pdf({"fields": [], "conflicts": [], "warnings": []}, tmp_path / "empty.pdf")
    assert out.exists() and out.stat().st_size > 1000


def test_main_cli_writes_the_pdf(payload: ExtractionPayload, tmp_path: Path) -> None:
    source = tmp_path / "extraction.json"
    source.write_text(json.dumps(FIXTURE), encoding="utf-8")
    out = tmp_path / "cli.pdf"

    assert main([str(source), str(out)]) == 0
    assert out.exists() and out.read_bytes().startswith(b"%PDF-")


def test_main_cli_reports_a_missing_input_file(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope.json"), str(tmp_path / "out.pdf")]) == 1


def test_main_cli_requires_two_arguments() -> None:
    assert main([]) == 2
