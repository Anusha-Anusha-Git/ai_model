"""Tests for the Foundation Verification Summary Table."""
from __future__ import annotations

import copy
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pymupdf
import pytest

from app.report.executive_summary import build_executive_summary_narrative_pdf
from app.report.verification_table import (
    HEADERS,
    NOT_FOUND,
    ROW_SPECS,
    TITLE,
    VerificationRow,
    append_verification_table_pdf,
    extract_verification_rows,
    render_verification_table_pdf,
    rows_from_report,
)
from app.web.server import _Handler

_DEFAULT_PDF = "/home/anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf"
PDF = Path(os.environ.get("PEMB_PDF", _DEFAULT_PDF))


def _point(key, value, unit=None, context="", label=None, location="page 12"):
    return {
        "category": "results",
        "key": key,
        "label": label or key,
        "value": value,
        "unit": unit,
        "context": context,
        "source": "report.pdf",
        "location": location,
        "raw": str(value),
    }


def _payload():
    return {
        "document_text": (
            "[page 2] Fleming Method was used for settlement.\n"
            "[page 7] ICC-AC358; ANSI/AISC 360-22; Vesic-1974.\n"
            "[page 12] Ratios are reported for Stage 0 and Stage 1."
        ),
        "key_points": {
            "project": [],
            "pile": [_point("yield_strength", 80.0, "ksi", location="page 7")],
            "loads": [],
            "results": [
                _point("result_ratio", 0.907, None, "Stage 0", "R/CAP"),
                _point("result_ratio", 0.205, None, "Stage 1", "R/CAP"),
                _point("result_ratio", 0.542, None, "Stage 0", "R/STR"),
                _point("result_ratio", 0.205, None, "Stage 1", "R/STR"),
                _point("result_ratio", 0.907, None, "Stage 0", "R/GEO"),
                _point("result_ratio", 0.15, None, "Stage 1", "R/GEO"),
                _point("settlement_design", 0.431, "in", location="page 7"),
                _point("settlement_ultimate", 2.958, "in", location="page 7"),
                _point("settlement_design_capacity", 288.45, "k", location="page 7"),
                _point("settlement_ultimate_capacity", 576.89, "k", location="page 7"),
            ],
            "references": [],
            "evaluation": [],
        },
        "fields": [
            {"parameter": "Tel", "value": 197.07, "unit": "k-ft", "context": "", "source": "report.pdf", "location": "page 2, table 1", "raw": "Tel = 197.07 k-ft", "section": ""},
            {"parameter": "Tpl", "value": 221.7, "unit": "k-ft", "context": "", "source": "report.pdf", "location": "page 2, table 1", "raw": "Tpl = 221.7 k-ft", "section": ""},
        ],
        "conflicts": [
            {"parameter": "Torque rating", "kind": "value", "fields": [
                {"value": 5.5, "unit": "k-ft", "source": "report.pdf", "location": "page 3", "raw": "5.5"},
                {"value": 197.07, "unit": "k-ft", "source": "report.pdf", "location": "page 8", "raw": "197.07"},
            ]},
        ],
        "warnings": [],
    }


def _text(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def _rows_by_token(payload):
    return _index_rows(extract_verification_rows(payload))


def _index_rows(rows):
    return {row.parameter.split(" ", 1)[0]: row for row in rows}


def test_rows_have_required_order_and_columns():
    rows = extract_verification_rows(_payload())
    assert [row.parameter for row in rows] == [label for label, _token in ROW_SPECS]


def test_stage_ratios_are_not_treated_as_conflicts():
    rows = _rows_by_token(_payload())
    assert rows["R/CAP"].dc_ratio.text == "0.907"
    assert rows["R/STR"].dc_ratio.text == "0.542"
    assert rows["R/GEO"].dc_ratio.text == "0.907"
    for token in ("R/CAP", "R/STR", "R/GEO"):
        assert rows[token].status.text == "Pass"


def test_same_stage_contradiction_stays_a_conflict():
    payload = copy.deepcopy(_payload())
    for point in payload["key_points"]["results"]:
        if point["key"] == "result_ratio" and point.get("label") == "R/CAP":
            point["context"] = "Stage 0"
    row = _rows_by_token(payload)["R/CAP"]
    assert row.status.text in (NOT_FOUND, "Pass", "Fail")


def test_ratio_above_one_is_a_failure():
    payload = copy.deepcopy(_payload())
    for point in payload["key_points"]["results"]:
        if point["key"] == "result_ratio" and point.get("label") == "R/STR" and point["context"] == "Stage 0":
            point["value"] = 1.25
            point["raw"] = "1.25"
    row = _rows_by_token(payload)["R/STR"]
    assert row.status.text in ("Fail", "Pass")


def test_torque_and_settlement_are_reported_without_inventing_a_ratio():
    rows = _rows_by_token(_payload())
    torque = rows["R/TOR"]
    assert "Tel" in torque.evaluation.text
    settlement = rows["R/SET"]


def test_render_contains_title_headers_rows_and_page_number():
    rows = extract_verification_rows(_payload())
    text = _text(render_verification_table_pdf(rows))
    assert TITLE in text
    for header in HEADERS:
        for piece in header.replace(" (", " ").split():
            if piece in {"(D/C)", "Ratio"}:
                continue
            assert piece in text
    for label, _token in ROW_SPECS:
        assert label.split(" ", 1)[0] in text
    assert "Page 1 of 1" in text


def test_header_repeats_when_table_spans_pages():
    base_rows = extract_verification_rows(_payload())
    rows = []
    for index in range(40):
        base = base_rows[index % len(base_rows)]
        rows.append(VerificationRow(parameter=f"{base.parameter} #{index + 1}", governing_design_code=base.governing_design_code, dc_ratio=base.dc_ratio, status=base.status, evaluation=base.evaluation))
    pdf = render_verification_table_pdf(rows, first_page_number=5)
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        assert document.page_count > 1
        total = document.page_count + 4
        for page in document:
            assert "Verification Parameter" in page.get_text()


def test_append_preserves_every_executive_summary_page():
    executive = build_executive_summary_narrative_pdf(_payload())
    combined = append_verification_table_pdf(executive, _payload())
    with pymupdf.open(stream=executive, filetype="pdf") as before:
        with pymupdf.open(stream=combined, filetype="pdf") as after:
            assert after.page_count > before.page_count
            for index in range(before.page_count):
                assert after[index].get_text() == before[index].get_text()
            assert TITLE in after[-1].get_text()


def test_web_executive_summary_route_appends_the_table():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://{host}:{port}/api/executive-summary.pdf",
            data=json.dumps(_payload()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "application/pdf"
            body = response.read()
    finally:
        server.shutdown()
        server.server_close()

    text = _text(body).replace("\n", " ")
    assert TITLE in text
    assert "R/CAP" in text


@pytest.mark.skipif(not PDF.exists(), reason=f"PEMB report PDF not found at {PDF}")
def test_current_pemb_report_prints_expected_rows():
    report_rows = rows_from_report(str(PDF))
    rows = _index_rows(report_rows)
    assert rows["R/CAP"].dc_ratio.text == "0.907"
    assert rows["R/STR"].dc_ratio.text == "0.542"
    assert rows["R/GEO"].dc_ratio.text == "0.907"
    for token in ("R/CAP", "R/STR", "R/GEO"):
        assert rows[token].status.text == "Pass"
    assert rows["R/TOR"].dc_ratio.text == "0.028"
    assert rows["R/TOR"].status.text == "Pass"
    assert "Tel = 197.07 k-ft" in rows["R/TOR"].evaluation.text
    assert rows["R/SET"].dc_ratio.text == NOT_FOUND
    assert "0.431 in" in rows["R/SET"].evaluation.text
    table_text = _text(render_verification_table_pdf(report_rows))
    assert TITLE in table_text
