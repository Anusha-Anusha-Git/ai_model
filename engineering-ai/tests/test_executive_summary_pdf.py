"""Tests for the Executive Summary PDF export.

Run from the repo root (engineering-ai/):

    ./.venv/bin/python -m pytest tests/test_executive_summary_pdf.py -q

Text is asserted by extracting it back out of the rendered PDF, so these tests
prove what a reader actually sees on the page rather than what the builder
returned.
"""

from __future__ import annotations

import pymupdf
import pytest

from app.pdf_builder import (
    CONFLICT_FLAG,
    FOOTER_TEXT,
    MULTI_FLAG,
    NOT_FOUND,
    build_executive_summary_pdf,
    executive_summary_filename,
)


def _keypoint(key, value, unit=None, context="", source="design_report.pdf", location="page 3, table 2"):
    return {
        "category": "",
        "key": key,
        "label": key,
        "value": value,
        "unit": unit,
        "context": context,
        "source": source,
        "location": location,
        "raw": f"{key}: {value}",
    }


def _payload(**sections):
    base = {"project": [], "pile": [], "loads": [], "results": [], "missing": []}
    base.update(sections)
    return {
        "key_points": base,
        "conflicts": [],
        "warnings": [],
    }


def _full_payload():
    return _payload(
        project=[
            _keypoint("project_name", "Baytown Station", location="page 1"),
            _keypoint("structure", "Substation foundations", location="page 1"),
            _keypoint("location", "Comal County, Texas", location="page 1"),
            _keypoint("client", "Meridian Utilities Co.", location="page 1"),
            _keypoint("date", "2026-08-15", location="page 1"),
        ],
        pile=[
            _keypoint("pile_diameter", 324.0, "mm"),
            _keypoint("pile_thickness", 12.7, "mm"),
            _keypoint("yield_strength", 345.0, "MPa"),
            _keypoint("helix_count", 3.0),
            _keypoint("helix_dia_1", 12.0, "in"),
            _keypoint("helix_dia_2", 10.0, "in"),
        ],
        loads=[
            _keypoint("load_compression", 980.0, "kN", "ULS-1"),
            _keypoint("load_tension", 610.0, "kN", "ULS-1"),
            _keypoint("load_shear", 45.0, "kN", "ULS-1"),
            _keypoint("load_moment", 62.0, "kN*m", "ULS-1"),
        ],
        results=[_keypoint("result_ratio", 0.78, None, "ULS-1")],
    )


def _text(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def _pages(pdf: bytes) -> int:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return document.page_count


def test_populated_payload_renders_a_valid_pdf():
    pdf = build_executive_summary_pdf(_full_payload())
    assert pdf.startswith(b"%PDF-")


def test_empty_payload_renders_rather_than_raising():
    """A report where nothing could be extracted must still produce a document."""
    pdf = build_executive_summary_pdf({})
    assert pdf.startswith(b"%PDF-")
    assert NOT_FOUND in _text(pdf)


def test_missing_value_prints_not_found_and_is_listed_in_notes():
    """The headline case: pile_thickness is absent and must not be invented."""
    data = _full_payload()
    data["key_points"]["pile"] = [
        record
        for record in data["key_points"]["pile"]
        if record["key"] != "pile_thickness"
    ]

    text = _text(build_executive_summary_pdf(data))

    assert "Pile wall thickness" in text
    assert "Notes and Flags" in text
    notes = text.split("Notes and Flags", 1)[1]
    assert "Pile wall thickness" in notes
    assert NOT_FOUND in notes
    # The neighbours are still read normally, so "Not found" is item-specific.
    assert "324 mm" in text


def test_conflicting_values_are_both_shown_and_never_resolved():
    data = _full_payload()
    data["key_points"]["pile"].append(
        _keypoint("pile_diameter", 406.0, "mm", context="HP2", location="page 4, table 3")
    )

    text = _text(build_executive_summary_pdf(data))

    assert "324 mm" in text and "406 mm" in text
    assert MULTI_FLAG in text
    notes = text.split("Notes and Flags", 1)[1]
    assert "more than one value" in notes
    assert "none was selected" in notes


def test_extractor_conflict_entry_is_surfaced_with_both_sides():
    data = _full_payload()
    data["conflicts"] = [
        {
            "parameter": "Yield strength",
            "kind": "value",
            "fields": [
                {
                    "parameter": "Yield strength",
                    "value": 345.0,
                    "unit": "MPa",
                    "source": "design_report.pdf",
                    "location": "page 3",
                    "context": "",
                    "section": "",
                    "raw": "Fy: 345 MPa",
                },
                {
                    "parameter": "Yield strength",
                    "value": 450.0,
                    "unit": "MPa",
                    "source": "design_report.pdf",
                    "location": "page 9",
                    "context": "",
                    "section": "",
                    "raw": "Fy: 450 MPa",
                },
            ],
            "detail": "ignored, rebuilt from fields",
        }
    ]

    notes = _text(build_executive_summary_pdf(data)).split("Notes and Flags", 1)[1]

    assert "Yield strength" in notes
    assert "345 MPa" in notes and "450 MPa" in notes
    assert "page 3" in notes and "page 9" in notes


def test_unreadable_value_is_shown_verbatim_and_flagged():
    data = _full_payload()
    data["key_points"]["pile"].append(
        _keypoint("yield_strength", "mm wall", "MPa", context="Note")
    )

    text = _text(build_executive_summary_pdf(data))
    notes = text.split("Notes and Flags", 1)[1]

    assert "mm wall" in text
    assert "not a plain number" in notes


def test_value_without_a_unit_is_flagged_but_not_given_one():
    data = _full_payload()
    data["key_points"]["pile"] = [
        record
        for record in data["key_points"]["pile"]
        if record["key"] != "pile_diameter"
    ]
    data["key_points"]["pile"].append(_keypoint("pile_diameter", 324.0, None))

    notes = _text(build_executive_summary_pdf(data)).split("Notes and Flags", 1)[1]

    assert "no unit" in notes
    assert "was not assumed" in notes


def test_every_page_carries_the_footer_and_a_page_number():
    pdf = build_executive_summary_pdf(_payload())
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text()
            assert FOOTER_TEXT in text
            assert f"Page {index} of {document.page_count}" in text


def test_source_reference_is_printed_for_each_value():
    text = _text(build_executive_summary_pdf(_full_payload()))
    assert "design_report.pdf (page 3, table 2)" in text
    assert "design_report.pdf (page 1)" in text


def test_units_appear_next_to_their_numbers():
    text = _text(build_executive_summary_pdf(_full_payload()))
    for rendered in ("324 mm", "12.7 mm", "345 MPa", "12 in", "980 kN", "610 kN"):
        assert rendered in text


def test_populated_summary_fits_the_page_budget():
    assert _pages(build_executive_summary_pdf(_full_payload())) <= 2


def test_long_load_case_list_is_capped_and_the_cap_is_disclosed():
    data = _full_payload()
    data["key_points"]["loads"] = [
        _keypoint("load_compression", 100.0 + index, "kN", f"ULS-{index}")
        for index in range(1, 31)
    ]

    text = _text(build_executive_summary_pdf(data))
    notes = text.split("Notes and Flags", 1)[1]

    assert _pages(build_executive_summary_pdf(data)) <= 2
    assert "only the first" in notes
    assert "source report" in notes


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {"key_points": None},
        {"key_points": {"pile": None}},
        {"key_points": {"pile": ["not-a-dict", 7]}},
        {"conflicts": "not-a-list", "warnings": None},
    ],
)
def test_malformed_input_degrades_instead_of_raising(malformed):
    pdf = build_executive_summary_pdf(malformed)
    assert pdf.startswith(b"%PDF-")


def test_non_mapping_input_is_rejected():
    with pytest.raises(ValueError):
        build_executive_summary_pdf([])  # type: ignore[arg-type]


def test_filename_uses_the_project_name():
    assert (
        executive_summary_filename(_full_payload())
        == "Baytown_Station_Executive_Summary.pdf"
    )


def test_filename_falls_back_when_no_project_name():
    assert executive_summary_filename({}) == "Executive_Summary.pdf"
    assert executive_summary_filename(_payload()) == "Executive_Summary.pdf"


@pytest.mark.parametrize(
    "project_name",
    ['Acme "Quoted" / Site', "Ünicode Prøject", "   ", "../../etc/passwd", "x" * 200],
)
def test_filename_is_always_safe_for_a_content_disposition_header(project_name):
    name = executive_summary_filename(
        _payload(project=[_keypoint("project_name", project_name)])
    )
    assert not set(name) & set('/"\\\n\r\t')
    assert name.isascii()
    assert name == "Executive_Summary.pdf" or name.endswith("_Executive_Summary.pdf")
    assert len(name) <= 82