"""End-to-end regression tests against the real HelixPile report.

The report is not committed (it is large and client data). Point the suite at
it with the ``PEMB_PDF`` environment variable; when it is absent the whole
module is skipped.

    PEMB_PDF="/path/PEMB (PERIMETER A)_Model raw.pdf" \
        ./.venv/bin/python -m pytest tests/test_pemb_perimeter_a.py -q

Covers the extraction/summary fixes:
- only "Stage N" is a load case (section headings never are);
- capacities (GEO/Qult/Qy/Fcap/R_GEO/"capacity") are never applied loads;
- applied loads come from "Summary of loads on pile head" (p.83);
- parse_section() splits "OD x wall x Fy ksi (n)dia x thick";
- units are whitelisted and preserved (nothing bare where a unit is expected);
- corrosion, project name (.HELX), torque conflict, per-stage ratios;
- the clean summary never prints "not stated" or a bucket pile.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.extraction.document_loader import load_many
from app.extraction.field_extractor import (
    extract_fields,
    is_valid_summary_unit,
    parse_section,
)
from app.report.key_points import extract_key_points
from app.report.summary import generate_clean_summary

_DEFAULT_PDF = "/home/anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf"
PDF = Path(os.environ.get("PEMB_PDF", _DEFAULT_PDF))

pytestmark = pytest.mark.skipif(
    not PDF.exists(), reason=f"PEMB report PDF not found at {PDF}"
)


@pytest.fixture(scope="module")
def result():
    return extract_fields(load_many([str(PDF)]))


@pytest.fixture(scope="module")
def kp(result):
    docs = load_many([str(PDF)])
    return extract_key_points(result, docs).to_dict()


def points(kp, category):
    return kp[category]


def find(points_, key, context=None, label=None):
    for p in points_:
        if p["key"] != key:
            continue
        if context is not None and (p.get("context") or "") != context:
            continue
        if label is not None and (p.get("label") or "") != label:
            continue
        return p
    return None


def values(points_, key, label=None):
    return [p["value"] for p in points_
            if p["key"] == key and (label is None or p.get("label") == label)]


# --------------------------------------------------------------------------
# 1. Project name comes from the .HELX file, header stays as client_project
# --------------------------------------------------------------------------


def test_project_name_is_helx_filename(kp):
    project = {p["key"]: p["value"] for p in points(kp, "project")}
    assert project["project_name"] == "PEMB (Perimeter A)"
    assert project.get("client_project") == "KINDER MORGAN- STRUCTURE C"


# --------------------------------------------------------------------------
# 2. parse_section: "x" is a separator, never a unit
# --------------------------------------------------------------------------


def test_parse_section_splits_geometry():
    section = parse_section("7.63 x 0.375 x 80 ksi (3)24 x 0.5")
    assert section["od_in"] == pytest.approx(7.63)
    assert section["wall_in"] == pytest.approx(0.375)
    assert section["fy_ksi"] == pytest.approx(80.0)
    assert section["helix_count"] == 3
    assert section["helix_dia_in"] == pytest.approx(24.0)
    assert section["helix_thickness_in"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 3. Pile spec key points
# --------------------------------------------------------------------------


def test_pile_specs(kp):
    assert find(points(kp, "pile"), "pile_diameter")["value"] == pytest.approx(7.63)
    assert find(points(kp, "pile"), "pile_diameter")["unit"] == "in"
    assert find(points(kp, "pile"), "yield_strength")["value"] == pytest.approx(80.0)
    assert find(points(kp, "pile"), "yield_strength")["unit"] == "ksi"
    assert find(points(kp, "pile"), "helix_count")["value"] == pytest.approx(3.0)
    assert find(points(kp, "pile"), "helix_dia")["value"] == pytest.approx(24.0)
    assert find(points(kp, "pile"), "helix_dia")["unit"] == "in"
    # wall thickness from the printed section label
    wall = find(points(kp, "pile"), "pile_wall_thickness")
    assert wall is not None and wall["value"] == pytest.approx(0.375)
    assert wall["unit"] == "in"


# --------------------------------------------------------------------------
# 4. Applied loads come from "Summary of loads on pile head"
# --------------------------------------------------------------------------


def test_applied_loads(kp):
    loads = points(kp, "loads")
    comp0 = find(loads, "load_compression", context="Stage 0")
    assert comp0 is not None and comp0["value"] == pytest.approx(150.0)
    assert comp0["unit"] == "k"

    tens1 = find(loads, "load_tension", context="Stage 1")
    assert tens1 is not None and tens1["value"] == pytest.approx(-19.98)
    assert tens1["unit"] == "k"

    assert find(loads, "load_shear", context="Stage 0")["value"] == pytest.approx(10.0)
    assert find(loads, "load_shear", context="Stage 1")["value"] == pytest.approx(10.0)


def test_capacities_are_not_applied_loads(kp):
    assert all(
        p["value"] not in (138.17, 133.02, 221.0, 197.86, 165.4)
        for p in points(kp, "loads")
    )


def test_no_section_heading_is_a_load_case(kp):
    contexts = {p.get("context") for p in points(kp, "loads")}
    assert "MODEL STAGES" not in contexts
    assert "General" not in contexts
    assert contexts <= {"Stage 0", "Stage 1", ""}


# --------------------------------------------------------------------------
# 5. Corrosion / durability
# --------------------------------------------------------------------------


def test_corrosion(kp):
    results = points(kp, "results")
    life = find(results, "design_life")
    assert life["value"] == pytest.approx(30.0) and life["unit"] == "years"
    loss = find(results, "corrosion_thickness_loss")
    assert loss["value"] == pytest.approx(0.01) and loss["unit"] == "in"
    steel = find(results, "corrosion_steel_loss")
    assert steel["value"] == pytest.approx(2.39) and steel["unit"] == "%"


# --------------------------------------------------------------------------
# 6. Per-stage utilization ratios
# --------------------------------------------------------------------------


def test_utilisation_ratios_per_stage(kp):
    results = points(kp, "results")
    assert find(results, "result_ratio", context="Stage 0", label="R/CAP")[
        "value"] == pytest.approx(0.907)
    assert find(results, "result_ratio", context="Stage 0", label="R/STR")[
        "value"] == pytest.approx(0.542)
    assert find(results, "result_ratio", context="Stage 1", label="R/CAP")[
        "value"] == pytest.approx(0.205)
    assert max(values(results, "result_ratio", label="R/CAP")) == pytest.approx(0.907)


# --------------------------------------------------------------------------
# 7. Settlement / lateral
# --------------------------------------------------------------------------


def test_settlement_and_lateral(kp):
    results = points(kp, "results")
    assert find(results, "settlement_design")["value"] == pytest.approx(0.431)
    assert find(results, "settlement_design")["unit"] == "in"
    assert find(results, "settlement_ultimate")["value"] == pytest.approx(2.958)
    assert find(results, "lateral_deflection", context="Stage 0")[
        "value"] == pytest.approx(0.213)
    assert find(results, "lateral_deflection", context="Stage 1")[
        "value"] == pytest.approx(0.22)
    assert find(results, "max_moment", context="Stage 0")["value"] == pytest.approx(14.45)
    assert find(results, "max_moment", context="Stage 1")["value"] == pytest.approx(14.5)


# --------------------------------------------------------------------------
# 8. Torque conflict + soil models
# --------------------------------------------------------------------------


def test_torque_conflict_is_reported(result):
    torque = [c for c in result.conflicts if "torque" in c.parameter.lower()]
    assert torque, "torque rating conflict (5.5 k-ft p.5 vs 197.07 k-ft p.8) missing"
    numbers = {f.value for f in torque[0].fields}
    assert 5.5 in numbers and 197.07 in numbers


def test_soil_models_report_reese_and_api(kp):
    models = [p["value"] for p in points(kp, "results") if p["key"] == "soil_model"]
    joined = " ".join(str(m) for m in models)
    assert "Reese" in joined, "main soil lateral model (Reese) not reported"
    assert "API" in joined, "top SM 0-1 lateral model (API) not reported"


# --------------------------------------------------------------------------
# 9. Unit hygiene + no placeholder phrases
# --------------------------------------------------------------------------


def test_unit_whitelist_helper():
    for unit in ("in", "ft", "k", "ksi", "ksf", "psf", "pcf", "k-ft",
                 "in^2", "in^3", "in^4", "deg", "%", "years"):
        assert is_valid_summary_unit(unit), unit
    assert is_valid_summary_unit(None)  # dimensionless is allowed
    assert not is_valid_summary_unit("mm")
    assert not is_valid_summary_unit("x")


def test_no_bare_number_where_a_unit_is_expected(kp):
    unit_keys = {
        "pile_diameter", "pile_wall_thickness", "yield_strength", "helix_dia",
        "load_compression", "load_tension", "load_shear", "load_moment",
        "settlement_design", "settlement_ultimate", "settlement_design_capacity",
        "settlement_ultimate_capacity", "corrosion_thickness_loss",
        "corrosion_steel_loss", "design_life", "lateral_deflection", "max_moment",
    }
    for category in ("pile", "loads", "results"):
        for p in points(kp, category):
            if p["key"] in unit_keys and isinstance(p["value"], float):
                assert p["unit"], f"{p['key']} has a bare number {p['value']}"


def test_clean_summary_has_no_placeholders_or_bucket(kp):
    summary = generate_clean_summary(kp)
    low = summary.lower()
    assert "not stated" not in low
    assert "total projected section loss of 0" not in low
    assert "bucket" not in low
    # never call it excellent or fully adequate
    assert "excellent" not in low
    assert "fully adequate" not in low
    # keeps the standing disclaimer
    assert "not certified" in low
    # reports the whitelisted values with their units
    assert "PEMB (Perimeter A)" in summary
    assert "150" in summary and "k" in summary
