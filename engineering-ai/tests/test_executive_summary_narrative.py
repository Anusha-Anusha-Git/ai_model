"""Tests for the five-paragraph Executive Summary narrative and its PDF export.

Run from the repo root (engineering-ai/):

    ./.venv/bin/python -m pytest tests/test_executive_summary_narrative.py -q
"""

from __future__ import annotations

import pymupdf
import pytest

from app.report.executive_summary import (
    TITLE,
    build_executive_summary_narrative_pdf,
    build_executive_summary_text,
)

NOT_STATED = "not stated in the report"


def _field(parameter, value, unit=None, context="", raw=None):
    return {
        "parameter": parameter,
        "value": value,
        "unit": unit,
        "context": context,
        "section": "",
        "source": "report.pdf",
        "location": "page 3",
        "raw": raw if raw is not None else f"{parameter}: {value}",
    }


def _payload(fields=None, key_points=None, references=None, warnings=None):
    key_points = key_points or {}
    if references is not None:
        key_points = {**key_points, "references": references}
    return {
        "fields": fields or [],
        "key_points": key_points,
        "conflicts": [],
        "warnings": warnings or [],
        "clean_engineering_summary": "",
    }


def _paragraphs(text: str) -> list[str]:
    return text.split("\n\n")


def _flat(text: str) -> str:
    """Whitespace-insensitive view, so PDF line wraps never break an assertion."""
    return " ".join(text.split())


STAGE_FIELDS = [
    _field("Pile diameter O.D.", 7.63, "in"),
    _field("Wall thickness", 0.375, "in"),
    _field("Yield strength (Fy)", 80, "ksi"),
    _field("Numbe", 3.0, None, "HP-1"),
    _field("Dhel", 24.0, "in", "Plate 1"),
    _field("Axial load", 150.0, "k", "Stage 0"),
    _field("Axial load", 19.975, "k", "Stage 1"),
    _field("Shear", 10.0, "k", "Stage 0"),
    _field("Shear", 10.0, "k", "Stage 1"),
]

KEY_POINTS = {
    "project": [
        {"key": "project_name", "value": "PEMB (PERIMETER A)", "unit": None, "context": ""},
        {"key": "software", "value": "HelixPile (DeepFND)", "unit": None, "context": ""},
    ],
    "pile": [
        {"key": "helix_count", "value": 3.0, "unit": None, "context": "HP-1"},
        {"key": "helix_dia_1", "value": 24.0, "unit": "in", "context": "Plate 1"},
    ],
    "loads": [
        {"key": "load_compression", "value": 150.0, "unit": "k", "context": "Stage 0"},
        {"key": "load_shear", "value": 10.0, "unit": "k", "context": "Stage 0"},
        {"key": "load_tension", "value": 19.975, "unit": "k", "context": "Stage 1"},
        {"key": "load_shear", "value": 10.0, "unit": "k", "context": "Stage 1"},
    ],
    "results": [],
}

REFERENCES = [
    {"text": "API RP 2A p-y and Vesic individual plate bearing method", "source": "report.pdf"},
    {"text": "ANSI/AISC 360-22 Specification for Structural Steel Buildings", "source": "report.pdf"},
    {"text": "ICC-AC358 Acceptance Criteria for Helical Pile Systems", "source": "report.pdf"},
]


def test_title_then_exactly_five_paragraphs():
    text = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    parts = _paragraphs(text)
    assert parts[0] == TITLE
    assert len(parts) == 6


def test_project_and_helix_details_are_copied_verbatim():
    text = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    assert "PEMB (PERIMETER A)" in _flat(text)
    assert "7.63 in" in text
    assert "0.375 in" in text
    assert "Fy = 80 ksi" in text
    assert "24 in diameter helical plate" in text


def test_load_cases_are_described_per_stage():
    text = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    loads = _paragraphs(text)[2]
    assert "2 governing design load cases" in loads
    assert "Stage 0 (compression) consisted of a 150 k axial compressive load" in loads
    assert "Stage 1 (uplift) consisted of 19.975 k uplift load" in loads
    assert "10 k lateral load" in loads


def test_software_methods_and_code_are_detected_from_text():
    data = _payload(STAGE_FIELDS, KEY_POINTS, references=REFERENCES)
    text = build_executive_summary_text(data)
    assert "HelixPile (DeepFND)" in text
    assert "Vesic" in text
    assert "API p-y" in text
    assert "ANSI/AISC 360-22" in text
    assert "ICC-AC358" in text


def test_missing_values_are_reported_as_not_stated():
    text = build_executive_summary_text({})
    assert NOT_STATED in _flat(text)
    assert "fully adequate for service" not in text


def test_all_ratios_within_1_but_no_settlement_criterion_requires_review():
    fields = [*STAGE_FIELDS, _field("R/GEO", 0.907, None, "Stage 0")]
    text = build_executive_summary_text(_payload(fields, KEY_POINTS, references=REFERENCES))
    assert "requires engineer review" in text
    assert "settlement criteria" in text


def test_over_capacity_ratio_requires_review_and_names_the_check():
    fields = [*STAGE_FIELDS, _field("R/GEO", 1.34, None, "Stage 0")]
    text = build_executive_summary_text(_payload(fields, KEY_POINTS, references=REFERENCES))
    assert "exceeds 1.0" in text
    assert "requires engineer review" in text


def test_fully_adequate_only_when_every_check_is_confirmed():
    fields = [
        *STAGE_FIELDS,
        _field("R/GEO", 0.907, None, "Stage 0"),
        _field("R/STR", 0.542, None, "Stage 0"),
        _field("R/CAP", 0.907, None, "Stage 0"),
        _field("Settlement @des. cap.", 0.4, "in", "Stage 0"),
        _field("Allowable settlement", 1.0, "in"),
    ]
    references = [
        *REFERENCES,
        {"text": "Installation verification criteria satisfied.", "source": "report.pdf"},
    ]
    text = build_executive_summary_text(_payload(fields, KEY_POINTS, references=references))
    assert "fully adequate for service" in text
    assert "requires engineer review" not in text


def test_bucket_pile_phrase_only_when_a_bucket_pile_is_described():
    plain = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    assert "combined bucket pile-helical pile foundation system" not in plain

    bucket = _payload(
        STAGE_FIELDS,
        KEY_POINTS,
        references=[{"text": "Combined bucket pile and helical pile system.", "source": "r.pdf"}],
    )
    text = build_executive_summary_text(bucket)
    assert "combined bucket pile-helical pile foundation system" in text


def test_pdf_renders_the_narrative_text():
    data = _payload(STAGE_FIELDS, KEY_POINTS, references=REFERENCES)
    pdf = build_executive_summary_narrative_pdf(data)
    assert pdf.startswith(b"%PDF-")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        assert document.page_count <= 2
        text = "\n".join(page.get_text() for page in document)
    assert "Executive Summary" in _flat(text)
    assert "PEMB (PERIMETER A)" in _flat(text)
    assert "fully adequate" in text or "engineer review" in text


def test_empty_payload_still_renders():
    pdf = build_executive_summary_narrative_pdf({})
    assert pdf.startswith(b"%PDF-")
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    assert NOT_STATED in _flat(text)


def test_non_mapping_input_is_rejected():
    with pytest.raises(ValueError):
        build_executive_summary_text([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_executive_summary_narrative_pdf([])  # type: ignore[arg-type]


def test_web_route_serves_the_narrative_summary():
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from app.web.server import _Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        data = _payload(STAGE_FIELDS, KEY_POINTS)
        request = urllib.request.Request(
            f"http://{host}:{port}/api/executive-summary.pdf",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.headers["Content-Type"] == "application/pdf"
            body = response.read()
    finally:
        server.shutdown()
        server.server_close()

    with pymupdf.open(stream=body, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    assert "Executive Summary: Helical Pile Design Verification" in _flat(text)
    assert "PEMB (PERIMETER A)" in _flat(text)

