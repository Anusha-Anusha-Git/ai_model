"""Cover-page tests: layout limits, content, revision values, and prepending."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf
import pytest

from app.report.cover_page import (
    MARGIN,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    build_cover_page_pdf,
    prepend_cover_page_pdf,
    resolve_cover_fields,
)
from app.report.pdf_report import ExtractionPayload, generate_pdf


FIELDS = {
    "structure_name": "PEMB (PERIMETER A)",
    "project_name": "ROYSE DATA CENTER PROJECT",
    "project_address": "4853 COUNTY ROAD 2656, ROYSE CITY, TX, 75189",
    "company_address": "202 S Dixon St, Callisburg, TX 76240, United States",
    "date": "AUG 27, 2026",
    "rev": "0",
    "description": "Issue for Record",
    "originated_by": "PP",
    "checked_by": "SP",
    "approved_by": "SP",
}


def _open(pdf_bytes: bytes) -> pymupdf.Document:
    return pymupdf.open(stream=pdf_bytes, filetype="pdf")


def test_cover_is_one_a4_page() -> None:
    document = _open(build_cover_page_pdf(FIELDS))
    try:
        assert document.page_count == 1
        page = document[0]
        assert page.rect.width == pytest.approx(PAGE_WIDTH, abs=0.5)
        assert page.rect.height == pytest.approx(PAGE_HEIGHT, abs=0.5)
    finally:
        document.close()


def test_cover_contains_all_required_sections() -> None:
    document = _open(build_cover_page_pdf(FIELDS))
    try:
        text = document[0].get_text()
    finally:
        document.close()

    for expected in (
        "STRUCTURAL CALCULATION",
        "HELICAL PILE DESIGN",
        "PEMB (PERIMETER A)",
        "ROYSE DATA CENTER PROJECT",
        "4853 COUNTY ROAD 2656, ROYSE CITY, TX, 75189",
        "202 S Dixon St, Callisburg, TX 76240, United States",
        "PNA ASSOCIATES Inc.",
        "(dba PNA TECHNICAL SERVICES)",
        "(Texas Firm# 26780)",
        "HOUSTON, TEXAS",
        "REVISION RECORD",
        "Rev",
        "Date",
        "Description",
        "Originated by",
        "Checked by",
        "Approved by",
        "Issue for Record",
        "This document is the property of PNA Technical Services.",
        "NOT",
    ):
        assert expected in text


def test_revision_values_are_read_from_the_payload() -> None:
    changed = dict(FIELDS, rev="2", date="SEP 01, 2026",
                   description="Issued for Construction",
                   originated_by="AB", checked_by="CD", approved_by="EF")
    document = _open(build_cover_page_pdf(changed))
    try:
        text = document[0].get_text()
    finally:
        document.close()
    for expected in ("2", "SEP 01, 2026", "Issued for Construction", "AB", "CD", "EF"):
        assert expected in text


def test_none_and_null_are_never_printed() -> None:
    document = _open(build_cover_page_pdf({
        "structure_name": None,
        "project_name": "None",
        "project_address": "null",
        "company_address": None,
    }))
    try:
        text = document[0].get_text().casefold()
    finally:
        document.close()
    assert "none" not in text
    assert "null" not in text
    # Fixed company address is still present as a safe fallback.
    assert "202 s dixon st" in text


def test_title_and_company_text_stay_inside_the_content_margins() -> None:
    document = _open(build_cover_page_pdf(FIELDS))
    try:
        words = document[0].get_text("words")
    finally:
        document.close()

    assert words
    for x0, _y0, x1, _y1, word, *_rest in words:
        assert x0 >= MARGIN - 1.0, f"{word!r} starts left of the margin"
        assert x1 <= PAGE_WIDTH - MARGIN + 1.0, f"{word!r} runs past the margin"


def test_banner_is_drawn_as_layered_vector_shapes() -> None:
    document = _open(build_cover_page_pdf(FIELDS))
    try:
        drawings = document[0].get_drawings()
    finally:
        document.close()

    # The banner occupies the top 1.2 in; at least five differently coloured
    # polygons are expected there (the white page fill is the first drawing).
    banner_shapes = [
        drawing
        for drawing in drawings
        if drawing.get("fill") is not None
        and drawing["rect"].y0 >= 0
        and drawing["rect"].y1 <= 1.2 * 72 + 1
    ]
    assert len(banner_shapes) >= 5
    assert len({tuple(round(c, 3) for c in drawing["fill"]) for drawing in banner_shapes}) >= 4


def test_prepending_cover_keeps_every_original_page() -> None:
    payload = ExtractionPayload.from_dict({
        "fields": [
            {
                "parameter": "Dp", "value": 324.0, "unit": "mm",
                "source": "design_report.pdf", "location": "page 3, table 2, row 2",
            }
        ],
        "conflicts": [],
        "warnings": [],
    })

    with tempfile.TemporaryDirectory() as tmp:
        rest_path = generate_pdf(payload, Path(tmp) / "report.pdf")
        rest_bytes = rest_path.read_bytes()

    full = prepend_cover_page_pdf(rest_bytes, build_cover_page_pdf(FIELDS))
    document = _open(full)
    try:
        assert document.page_count >= 2
        assert "STRUCTURAL CALCULATION" in document[0].get_text()
        assert "Helical Pile Design Review Report" in document[1].get_text()
    finally:
        document.close()


def test_resolve_cover_fields_fills_from_report_metadata() -> None:
    data = {
        "key_points": {
            "project": [
                {"key": "structure", "value": "PEMB (PERIMETER A)"},
                {"key": "project_name", "value": "Royse Data Center Project"},
                {"key": "location", "value": "Royse City, TX"},
            ]
        },
        "fields": [{"parameter": "Project", "value": "Fallback Project"}],
    }
    resolved = resolve_cover_fields({}, data)
    assert resolved["structure_name"] == "PEMB (PERIMETER A)"
    assert resolved["project_name"] == "Royse Data Center Project"
    assert resolved["project_address"] == "Royse City, TX"
    assert resolved["rev"] == "0"
    assert resolved["description"] == "Issue for Record"
