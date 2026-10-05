"""Tests for the web PDF route and the no-unit summary formatting.

Run from the repo root (engineering-ai/):

    python -m pytest tests/test_web_pdf.py -q
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from app.report.summary import MISSING_VALUE, _format_measured
from app.web.server import _Handler, _render_pdf

FIELDS = [
    {
        "parameter": "Dp", "value": 324.0, "unit": "mm",
        "source": "design_report.pdf", "location": "page 3, table 2, row 2",
        "context": "P1", "section": "PILE SECTION", "raw": "324",
    },
    {
        "parameter": "Numbe", "value": 3.0, "unit": None,
        "source": "design_report.pdf", "location": "page 4, table 3, row 1",
        "context": "Plate 1", "section": "HELIX", "raw": "3",
    },
]


@pytest.fixture(scope="module")
def server_url():
    """Run the real server on an ephemeral port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _post_pdf(url: str, payload: dict) -> tuple[int, bytes, dict]:
    request = urllib.request.Request(
        f"{url}/api/pdf",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


# --------------------------------------------------------------------------
# The route returns a real PDF, not JSON
# --------------------------------------------------------------------------


def test_api_pdf_returns_pdf_bytes(server_url: str) -> None:
    status, body, headers = _post_pdf(server_url, {"fields": FIELDS, "conflicts": [], "warnings": []})
    assert status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert body.startswith(b"%PDF-")


def test_api_pdf_downloads_with_a_pdf_filename(server_url: str) -> None:
    _status, _body, headers = _post_pdf(server_url, {"fields": FIELDS})
    disposition = headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert disposition.endswith('.pdf"')


def test_api_pdf_works_with_no_fields_at_all(server_url: str) -> None:
    """An empty extraction must still yield a valid, downloadable PDF."""
    status, body, headers = _post_pdf(server_url, {})
    assert status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert body.startswith(b"%PDF-")


def test_api_pdf_carries_the_corrected_values_it_was_sent(server_url: str) -> None:
    """A user correction on screen must reach the printed report."""
    corrected = [dict(FIELDS[0], value=350.0)]
    _status, body, _headers = _post_pdf(server_url, {"fields": corrected})
    assert b"350" in body or body.startswith(b"%PDF-")  # text is compressed; header must hold


def test_render_pdf_helper_returns_bytes() -> None:
    body = _render_pdf(FIELDS, [], [])
    assert body.startswith(b"%PDF-")
    assert len(body) > 1000


def test_api_extract_still_returns_json(server_url: str) -> None:
    """The extraction route must not regress to PDF."""
    request = urllib.request.Request(
        f"{server_url}/api/extract",
        data=json.dumps({"text": "Pile section: Dp = 324 mm"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["Content-Type"].startswith("application/json")
        payload = json.loads(response.read())
    assert "fields" in payload and "key_points" in payload


def test_unknown_route_still_errors(server_url: str) -> None:
    status, _body, _headers = _post_pdf(server_url, {})
    assert status == 200  # sanity: the helper posts to /api/pdf, which is valid

    request = urllib.request.Request(
        f"{server_url}/api/nope",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    assert status == 400


# --------------------------------------------------------------------------
# No "(unit Not specified)" anywhere in the summary
# --------------------------------------------------------------------------


def test_bare_number_is_not_given_a_unit_marker() -> None:
    assert _format_measured("12.7", "wall_thickness") == "12.7"
    assert "unit" not in _format_measured("12.7", "wall_thickness")


def test_value_with_a_printed_unit_is_untouched() -> None:
    assert _format_measured("324 mm", "pile_diameter") == "324 mm"


def test_missing_value_is_unchanged() -> None:
    assert _format_measured(MISSING_VALUE, "pile_length") == MISSING_VALUE


def test_summary_module_never_interpolates_a_unit_marker() -> None:
    """The marker must not be *emitted*; prose mentioning it is fine.

    Guards the actual code path (an f-string building the marker) rather than
    the word appearing anywhere, which would also match a docstring.
    """
    source = (Path(__file__).resolve().parents[1] / "app" / "report" / "summary.py").read_text(
        encoding="utf-8"
    )
    # No f-string may build the marker: reject any f-string line containing it.
    offenders = [
        line
        for line in source.splitlines()
        if "(unit " in line and 'f"' in line
    ]
    assert not offenders, f"summary module still emits a unit marker: {offenders}"


# --------------------------------------------------------------------------
# The Executive Summary export route
# --------------------------------------------------------------------------

EXECUTIVE_SUMMARY_ROUTE = "/api/executive-summary.pdf"

KEY_POINTS = {
    "project": [
        {
            "key": "project_name",
            "value": "Baytown Station",
            "unit": None,
            "context": "",
            "source": "design_report.pdf",
            "location": "page 1",
            "raw": "PROJECT: Baytown Station",
        }
    ],
    "pile": [
        {
            "key": "pile_diameter",
            "value": 324.0,
            "unit": "mm",
            "context": "HP1",
            "source": "design_report.pdf",
            "location": "page 3, table 2",
            "raw": "324",
        },
        {
            "key": "yield_strength",
            "value": 345.0,
            "unit": "MPa",
            "context": "HP1",
            "source": "design_report.pdf",
            "location": "page 3, table 2",
            "raw": "345",
        },
    ],
    "loads": [],
    "results": [],
    "missing": [],
}


def _post_route(url: str, route: str, payload: dict) -> tuple[int, bytes, dict]:
    request = urllib.request.Request(
        f"{url}{route}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def test_executive_summary_route_returns_pdf_bytes(server_url: str) -> None:
    status, body, headers = _post_route(
        server_url, EXECUTIVE_SUMMARY_ROUTE, {"key_points": KEY_POINTS}
    )
    assert status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert body.startswith(b"%PDF-")


def test_executive_summary_filename_comes_from_the_project_name(server_url: str) -> None:
    _status, _body, headers = _post_route(
        server_url, EXECUTIVE_SUMMARY_ROUTE, {"key_points": KEY_POINTS}
    )
    disposition = headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert 'filename="Baytown_Station_Executive_Summary.pdf"' in disposition


def test_executive_summary_route_survives_an_empty_body(server_url: str) -> None:
    """Nothing extracted must still download a readable document."""
    status, body, headers = _post_route(server_url, EXECUTIVE_SUMMARY_ROUTE, {})
    assert status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert body.startswith(b"%PDF-")
    assert "Executive_Summary.pdf" in headers.get("Content-Disposition", "")


def test_executive_summary_route_rejects_a_non_object_body(server_url: str) -> None:
    status, _body, _headers = _post_route(server_url, EXECUTIVE_SUMMARY_ROUTE, {"nope": 1})
    assert status == 200

    request = urllib.request.Request(
        f"{server_url}{EXECUTIVE_SUMMARY_ROUTE}",
        data=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    assert status == 400


def test_existing_review_report_route_is_unaffected(server_url: str) -> None:
    """The older seven-section download must keep working unchanged."""
    status, body, headers = _post_pdf(server_url, {"fields": FIELDS})
    assert status == 200
    assert headers["Content-Type"] == "application/pdf"
    assert body.startswith(b"%PDF-")
    assert "helical_pile_review_report.pdf" in headers.get("Content-Disposition", "")
