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

NOT_STATED = "Not specified"  # the single missing-value placeholder


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
    assert "7.63 in outside diameter (OD)" in text
    assert "0.375 in thick steel helical pile shaft" in text
    assert "Fy = 80 ksi" in text
    assert "three 24 in diameter helical plates" in text


def test_load_cases_are_described_per_stage():
    text = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    loads = _paragraphs(text)[2]
    assert "two governing design load cases" in loads
    assert "Stage 0 (compression) consisted of a 150 k axial compressive load" in loads
    assert "Stage 1 (uplift) consisted of a 19.975 k uplift load" in loads
    assert "10 k lateral load" in loads


def test_software_methods_and_code_are_detected_from_text():
    data = _payload(STAGE_FIELDS, KEY_POINTS, references=REFERENCES)
    text = build_executive_summary_text(data)
    assert "HelixPile (DeepFND)" in text
    assert "Vesic" in text
    assert "API p-y" in text
    assert "ANSI/AISC 360-22" in text
    assert "ICC-AC358" in text


def test_missing_values_are_reported_as_not_specified():
    text = build_executive_summary_text({})
    assert NOT_STATED in _flat(text)
    assert "fully adequate for service" not in text


def test_all_ratios_within_1_but_no_settlement_criterion_requires_review():
    fields = [*STAGE_FIELDS, _field("R/GEO", 0.907, None, "Stage 0")]
    text = build_executive_summary_text(_payload(fields, KEY_POINTS, references=REFERENCES))
    assert "All utilization ratios are below 1.0." in text
    assert "engineer review are required before the design is accepted" in text
    assert "adequate for service" not in text


def test_over_capacity_ratio_requires_review_and_names_the_check():
    fields = [*STAGE_FIELDS, _field("R/GEO", 1.34, None, "Stage 0")]
    text = build_executive_summary_text(_payload(fields, KEY_POINTS, references=REFERENCES))
    assert "exceeds the limit of 1.0" in text
    assert "engineer review are required before the design is accepted" in text
    assert "adequate" not in text
    assert "compliance" not in text


def test_fully_adequate_only_when_every_check_is_confirmed():
    fields = [
        *STAGE_FIELDS,
        _field("R/GEO", 0.907, None, "Stage 0"),
        _field("R/STR", 0.542, None, "Stage 0"),
        _field("R/CAP", 0.907, None, "Stage 0"),
        _field("Settlement @des. cap.", 0.431, "in", "Stage 0"),
        _field("Design capacity", 288.45, "k", "Stage 0"),
    ]
    data = _payload(fields, KEY_POINTS, references=REFERENCES)
    data["project_inputs"] = {
        "system_descriptor": "",
        "corrosion_category": "",
        "reviewed_by_engineer": True,
        "installation_verified": True,
    }
    text = build_executive_summary_text(data)
    assert "the proposed helical foundation design is adequate for service." in text
    assert "requires engineer review" not in text
    assert "fully adequate" not in text

    data["project_inputs"]["installation_verified"] = False
    review = build_executive_summary_text(data)
    assert "adequate for service" not in review
    assert "engineer review are required before the design is accepted" in review


def test_bucket_pile_phrase_only_when_a_bucket_pile_is_described():
    plain = build_executive_summary_text(_payload(STAGE_FIELDS, KEY_POINTS))
    assert "foundation system" in plain
    assert "bucket" not in plain

    bucket = _payload(STAGE_FIELDS, KEY_POINTS)
    bucket["project_inputs"] = {
        "system_descriptor": "combined bucket pile\u2013helical pile",
        "corrosion_category": "",
        "reviewed_by_engineer": False,
        "installation_verified": False,
    }
    text = build_executive_summary_text(bucket)
    assert "combined bucket pile\u2013helical pile foundation system" in text


TARGET_FIELDS = [
    _field("Pile diameter O.D.", 7.63, "in"),
    _field("Wall thickness", 0.375, "in"),
    _field("Yield strength (Fy)", 80, "ksi"),
    _field("Numbe", 3.0, None, "HP-1"),
    _field("Dhel", 24.0, "in", "Plate 1"),
    _field("Helix thickness", 0.5, "in", "Plate 1"),
    _field("Axial load", 150.0, "k", "Stage 0"),
    _field("Axial load", 19.975, "k", "Stage 1"),
    _field("Shear", 10.0, "k", "Stage 0"),
    _field("Shear", 10.0, "k", "Stage 1"),
    _field("R/STR", 0.542, None, "Stage 0"),
    _field("R/GEO", 0.907, None, "Stage 0"),
    _field("R/CAP", 0.907, None, "Stage 0"),
    _field("Settlement @des. cap.", 0.431, "in", "Stage 0"),
    _field("Design capacity", 288.45, "k", "Stage 0"),
]

TARGET_KEY_POINTS = {
    **KEY_POINTS,
    "pile": [
        {"key": "helix_count", "value": 3.0, "unit": None, "context": "HP-1"},
        {"key": "helix_dia", "value": 24.0, "unit": "in", "context": "Plate 1"},
        {"key": "helix_thickness", "value": 0.5, "unit": "in", "context": "Plate 1"},
    ],
    "results": [
        {"key": "result_ratio", "label": "R/STR", "value": 0.542, "unit": None, "context": "Stage 0"},
        {"key": "result_ratio", "label": "R/GEO", "value": 0.907, "unit": None, "context": "Stage 0"},
        {"key": "corrosion_thickness_loss", "value": 0.01, "unit": "in", "context": ""},
        {"key": "settlement_design", "value": 0.431, "unit": "in", "context": "Stage 0"},
        {"key": "settlement_design_capacity", "value": 288.45, "unit": "k", "context": "Stage 0"},
        {"key": "soil_model", "value": "Sand Reese", "unit": None, "context": "Soil"},
        {"key": "soil_model", "value": "Sand API", "unit": None, "context": "SM 0-1"},
    ],
}

TARGET_REFERENCES = [
    {"text": "Vesic (1974) individual plate bearing method and API p-y lateral response.", "source": "r.pdf"},
    {"text": "ANSI/AISC 360-22 Specification for Structural Steel Buildings.", "source": "r.pdf"},
    {"text": "ICC-AC358 Acceptance Criteria for Helical Pile Systems.", "source": "r.pdf"},
]

CONFIGURED = {
    "system_descriptor": "combined bucket pile\u2013helical pile",
    "corrosion_category": "non-to-mildly aggressive",
    "reviewed_by_engineer": True,
    "installation_verified": True,
}

EMPTY_CONFIG = {
    "system_descriptor": "",
    "corrosion_category": "",
    "reviewed_by_engineer": False,
    "installation_verified": False,
}


def _target_payload(config):
    data = _payload(TARGET_FIELDS, TARGET_KEY_POINTS, references=TARGET_REFERENCES)
    data["document_text"] = "HelixPile 2026 DeepFND analysis. ANSI/AISC 360-22."
    data["project_inputs"] = dict(config)
    return data


def test_complete_config_reproduces_target_template():
    text = build_executive_summary_text(_target_payload(CONFIGURED))
    assert "checks performed for PEMB (PERIMETER A)." in text
    assert "a combined bucket pile\u2013helical pile foundation system consisting of" in text
    assert "7.63 in outside diameter (OD)" in text
    assert "0.375 in thick steel helical pile shaft (Fy = 80 ksi)" in text
    assert "three 24 in diameter, 0.5 in thick helical plates" in text
    assert "two governing design load cases" in text
    assert "a 150 k axial compressive load" in text
    assert "a 10 k lateral load" in text
    assert "a 19.975 k uplift load" in text
    assert "HelixPile (DeepFND)" in text
    assert "Vesic (1974)" in text
    assert "API p-y" in text
    assert "ANSI/AISC 360-22" in text
    assert "total projected section loss of 0.01 inches" in text
    assert "maximum projected structural shaft utilization reaches 0.542" in text
    assert "excellent structural capacity reserve in compliance with ANSI/AISC 360-22" in text
    assert "code-compliant margin against soil plunge failure under the design load" in text
    assert "the proposed helical foundation design is adequate for service." in text
    assert "fully adequate" not in text


def test_empty_config_falls_back_to_foundation_system():
    text = build_executive_summary_text(_target_payload(EMPTY_CONFIG))
    assert "foundation system" in text
    assert "bucket" not in text
    assert "categorized as" not in text
    assert "All utilization ratios are below 1.0 and design settlement is 0.431 in at the " in text
    assert "288.45-k design capacity" in text
    assert "engineer review are required before the design is accepted" in text
    assert "adequate for service" not in text


def test_over_capacity_never_claims_adequate_or_compliance():
    fields = [*TARGET_FIELDS, _field("R/STR", 1.01, None, "Stage 0")]
    data = _payload(fields, TARGET_KEY_POINTS, references=TARGET_REFERENCES)
    data["project_inputs"] = dict(CONFIGURED)
    text = build_executive_summary_text(data)
    assert "adequate" not in text
    assert "compliance" not in text
    assert "exceeds the limit of 1.0" in text


def test_torque_conflict_appears_in_warnings():
    from app.web.server import _pasted_document, _run_extraction

    text = (
        "Torque rating (by Manufacturer) = 5.5 k-ft\n"
        "MODEL STAGES\n"
        "Torque rating (by Manufacturer) Tel = 197.07 k-ft\n"
    )
    payload = _run_extraction([_pasted_document(text)])
    assert any(
        "5.5" in warning and "197.07" in warning for warning in payload["warnings"]
    ), payload["warnings"]


def test_chart_and_coordinate_conflicts_are_not_warned():
    from app.web.server import _conflict_warnings

    def conflict(parameter, values):
        return {
            "parameter": parameter,
            "kind": "value",
            "fields": [
                {"value": v, "unit": "", "location": f"page {i + 1}"}
                for i, v in enumerate(values)
            ],
        }

    payload = {
        "conflicts": [
            conflict("El.", [1.0, 2.0]),               # coordinate table
            conflict("Moment", [float(i) for i in range(12)]),  # plotted series
            conflict("D.Tension", [1.0, 2.0]),          # interaction label
            conflict("Fy", [345.0, 450.0]),             # genuine spec conflict
        ]
    }
    warnings = _conflict_warnings(payload)
    assert len(warnings) == 1
    assert "Fy" in warnings[0]
    assert not any("El." in w or "Moment" in w or "D.Tension" in w for w in warnings)


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

