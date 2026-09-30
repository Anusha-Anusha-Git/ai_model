"""Tests for Stage 2c: app/report/key_points.py.

Run from the repo root (engineering-ai/):

    python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.document_loader import (
    ExtractedDocument,
    Table,
    TextChunk,
    load_document,
)
from app.extraction.field_extractor import extract_fields
from app.report.key_points import (
    KeyPoint,
    KeyPointsResult,
    extract_key_points,
    main,
    render_key_points,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def keypoints(*paths: str | Path) -> KeyPointsResult:
    docs = [load_document(p) for p in paths]
    return extract_key_points(extract_fields(docs), docs)


def find(section: list[KeyPoint], key: str) -> KeyPoint:
    for kp in section:
        if kp.key == key:
            return kp
    raise AssertionError(f"no {key!r} in {[k.key for k in section]}")


# --------------------------------------------------------------------------
# Project info
# --------------------------------------------------------------------------


def test_project_info_extracted():
    kp = keypoints(FIXTURES / "project_info.txt")
    names = {p.key: p for p in kp.project}

    assert names["project_name"].value == "Oak Creek Wind Farm"
    assert names["project_name"].source == "project_info.txt"
    assert names["structure"].value == "Substation Foundations"
    assert names["location"].value == "Bastrop County, Texas"
    assert names["client"].value == "Ridgeview Power Co."
    assert names["date"].value == "2026-09-23"
    assert isinstance(names["date"].value, str)  # a date, never a bare number


def test_project_info_from_full_report(selfless=0):
    kp = keypoints(FIXTURES / "full_report.txt")
    names = {p.key: p for p in kp.project}
    assert names["project_name"].value == "Greenfield Substation & Control Building"
    assert names["structure"].value == "Reinforced concrete foundations on helical piles"
    assert names["client"].value == "Meridian Utilities Co."
    assert names["project_no"].value == "G-2049"


def test_project_meta_from_pair_table():
    table = Table(location="table 1", headers=["Item", "Value"],
                  rows=[["Client", "ACME Co."], ["Project", "Demo Site"]])
    doc = ExtractedDocument(path=Path("cover.docx"), file_type="docx", tables=[table])
    kp = extract_key_points(extract_fields(doc), doc)
    assert find(kp.project, "client").value == "ACME Co."
    assert find(kp.project, "project_name").value == "Demo Site"


# --------------------------------------------------------------------------
# Pile / executive summary
# --------------------------------------------------------------------------


def test_pile_executive_summary_from_prose():
    kp = keypoints(FIXTURES / "project_info.txt")
    assert find(kp.pile, "pile_diameter").value == pytest.approx(324.0)
    assert find(kp.pile, "pile_diameter").unit == "mm"
    thickness = find(kp.pile, "pile_thickness")
    assert thickness.value == pytest.approx(12.7)
    assert thickness.unit == "mm"
    fy = find(kp.pile, "yield_strength")
    assert fy.value == pytest.approx(450.0)
    assert fy.unit == "MPa"
    assert find(kp.pile, "pile_length").value == pytest.approx(18.0)


def test_pile_from_pile_schedule_grid():
    kp = keypoints(FIXTURES / "design_report.pdf")
    diameters = [p for p in kp.pile if p.key == "pile_diameter"]
    assert {p.value for p in diameters} == {324.0, 406.0}
    assert any("HP1" in p.context for p in diameters)
    thicknesses = [p for p in kp.pile if p.key == "pile_thickness"]
    assert thicknesses[0].value == pytest.approx(12.7)


def test_helix_details_extracted():
    kp = keypoints(FIXTURES / "full_report.txt")
    assert find(kp.pile, "helix_count").value == pytest.approx(3.0)
    assert find(kp.pile, "helix_dia_1").value == pytest.approx(12.0)
    assert find(kp.pile, "helix_dia_1").unit == "in"
    assert find(kp.pile, "helix_dia_2").value == pytest.approx(10.0)
    assert find(kp.pile, "helix_dia_3").value == pytest.approx(8.0)


def test_yield_strength_with_parenthesised_qualifier():
    kp = keypoints(FIXTURES / "full_report.txt")
    fy = find(kp.pile, "yield_strength")
    assert fy.value == pytest.approx(345.0)
    assert fy.unit == "MPa"


# --------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------


def test_loads_from_text_pairs_with_case():
    kp = keypoints(FIXTURES / "project_info.txt")
    comp = [l_ for l_ in kp.loads if l_.key == "load_compression"]
    assert len(comp) == 1
    assert comp[0].value == pytest.approx(980.0)
    assert comp[0].unit == "kN"
    assert comp[0].context == "ULS"
    tension = [l_ for l_ in kp.loads if l_.key == "load_tension"]
    assert tension[0].value == pytest.approx(610.0)
    assert find(kp.loads, "load_shear").value == pytest.approx(45.0)
    assert find(kp.loads, "load_moment").value == pytest.approx(62.0)


def test_loads_from_fixed_table_per_load_case():
    kp = keypoints(FIXTURES / "design_report.xlsx")
    comp = [(l_.value, l_.context) for l_ in kp.loads if l_.key == "load_compression"]
    assert sorted(comp) == [(540.0, "SLS-1"), (720.0, "ULS-2"), (980.0, "ULS-1")]
    moment = find(kp.loads, "load_moment")
    assert moment.unit == "kN*m"

    # values must not be duplicated by the prose parser
    values = sorted(l_.value for l_ in kp.loads if l_.key == "load_compression")
    assert values == [540.0, 720.0, 980.0]


# --------------------------------------------------------------------------
# Results + evaluation
# --------------------------------------------------------------------------


def test_ratios_and_evaluation():
    kp = keypoints(FIXTURES / "full_report.txt")
    ratios = {r.context: r.value for r in kp.results}
    assert ratios == {"ULS-1": pytest.approx(0.78), "ULS-2": pytest.approx(0.65)}
    assert all("OK (demand within capacity)" in line for line in kp.evaluation
               if line.startswith("D/C ratio"))
    assert any("All reported D/C ratios are within capacity (<= 1.0)" in line
               for line in kp.evaluation)
    assert any("not certified" in line for line in kp.evaluation)


def test_over_capacity_ratio_flagged():
    doc = ExtractedDocument(
        path=Path("check.txt"), file_type="txt",
        text_chunks=[TextChunk("line 1", "ULS-1 capacity ratio: 1.35")],
    )
    kp = extract_key_points(extract_fields(doc), doc)
    assert find(kp.results, "result_ratio").value == pytest.approx(1.35)
    assert any("OVER CAPACITY" in line for line in kp.evaluation)


def test_load_without_ratio_noted():
    doc = ExtractedDocument(
        path=Path("solo.txt"), file_type="txt",
        text_chunks=[TextChunk("line 1", "ULS-1: compression 500 kN, tension 300 kN")],
    )
    kp = extract_key_points(extract_fields(doc), doc)
    assert any("ULS-1" in line and "no D/C ratio" in line for line in kp.evaluation)


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


def test_references_extracted():
    kp = keypoints(FIXTURES / "full_report.txt")
    assert len(kp.references) == 3
    assert kp.references[0].index == 1
    assert "ASCE 7-22" in kp.references[0].text


def test_no_references_section_reported():
    kp = keypoints(FIXTURES / "project_info.txt")
    assert kp.references == []
    assert any("No references/standards section found" in line for line in kp.evaluation)


# --------------------------------------------------------------------------
# Missing values -- never invented
# --------------------------------------------------------------------------


def test_missing_values_reported_not_guessed():
    doc = ExtractedDocument(
        path=Path("emptyish.txt"), file_type="txt",
        text_chunks=[TextChunk("line 1", "Some prose.")],
    )
    kp = extract_key_points(extract_fields(doc), doc)
    assert "Pile diameter" in kp.missing
    assert "Client" in kp.missing
    assert "D/C ratio" in kp.missing
    assert any("never invented" in line for line in kp.evaluation)


# --------------------------------------------------------------------------
# Serialization + CLI
# --------------------------------------------------------------------------


def test_to_dict_shape():
    kp = keypoints(FIXTURES / "full_report.txt")
    payload = kp.to_dict()
    assert set(payload) == {"project", "pile", "loads", "results",
                            "evaluation", "references", "missing"}
    assert payload["project"][0]["key"] == "project_name"
    assert payload["references"][0]["text"]


def test_render_text_contains_sections(capsys):
    assert "KEY POINTS" in render_key_points(keypoints(FIXTURES / "project_info.txt"))
    rc = main([str(FIXTURES / "full_report.txt")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PILE (EXECUTIVE SUMMARY)" in out
    assert "RESULTS (D/C RATIOS)" in out
    assert "REFERENCES" in out
    assert "ENGINEERING EVALUATION" in out


def test_cli_usage_without_args():
    assert main([]) == 2


def test_traceable_never_bare():
    kp = keypoints(FIXTURES / "full_report.txt", FIXTURES / "project_info.txt")
    for section in (kp.project, kp.pile, kp.loads, kp.results):
        for p in section:
            assert p.source and p.location
            assert p.raw