"""Tests for Stage 2b: app/extraction/field_extractor.py.

Run from the repo root (engineering-ai/):

    python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.document_loader import ExtractedDocument, Table, TextChunk, load_document
from app.extraction.field_extractor import (
    Conflict,
    ExtractionResult,
    ExtractedField,
    UnparseableValueError,
    extract_fields,
    find_conflicts,
    parse_number_strict,
    split_value_unit,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def field_by(result: ExtractionResult, parameter: str, context: str = "") -> ExtractedField:
    for f in result.fields:
        if f.parameter == parameter and f.context == context:
            return f
    raise AssertionError(
        f"no field {parameter!r} (context={context!r}); got "
        f"{[(f.parameter, f.context) for f in result.fields]}"
    )


# --------------------------------------------------------------------------
# Number / unit parsing
# --------------------------------------------------------------------------


def test_split_value_unit_with_unit():
    value, unit, raw = split_value_unit("7.63 in")
    assert value == pytest.approx(7.63)
    assert unit == "in"
    assert raw == "7.63 in"


def test_split_value_unit_bare_number():
    value, unit, raw = split_value_unit("450")
    assert value == 450.0
    assert unit is None
    assert raw == "450"


def test_split_value_unit_keeps_non_numeric_raw():
    value, unit, raw = split_value_unit("F355NH")
    assert value == "F355NH"  # a string, never coerced to a number
    assert unit is None


def test_split_value_unit_thousands_separator():
    value, unit, _ = split_value_unit("1,980 kN")
    assert value == pytest.approx(1980.0)
    assert unit == "kN"


def test_split_value_unit_complex_unit():
    value, unit, _ = split_value_unit("62 kN*m")
    assert value == pytest.approx(62.0)
    assert unit == "kN*m"


def test_split_value_unit_no_space_glued_unit():
    # "12.7mm" has no separating space but is still number + unit
    value, unit, raw = split_value_unit("12.7mm")
    assert value == pytest.approx(12.7)
    assert unit == "mm"
    assert raw == "12.7mm"


def test_float_never_invented():
    assert split_value_unit("not a number")[0] == "not a number"
    with pytest.raises(UnparseableValueError):
        parse_number_strict("TBD")


# --------------------------------------------------------------------------
# Pattern 1: label -> value pairs
# --------------------------------------------------------------------------


def test_label_value_pair_table():
    table = Table(
        location="page 7, table 1",
        headers=["Parameter", "Value"],
        rows=[
            ["Pile diameter O.D.", "7.63 in"],
            ["Pile wall thickness tP", "0.500 in"],
            ["Steel yield strength Fy", "50 ksi"],
            ["Helix diameter", "18 in"],
        ],
    )
    doc = ExtractedDocument(path=Path("software_export.pdf"), file_type="pdf",
                            tables=[table])
    result = extract_fields(doc)

    diameter = field_by(result, "Pile diameter O.D.")
    assert diameter.value == pytest.approx(7.63)
    assert diameter.unit == "in"
    assert diameter.source == "software_export.pdf"
    assert diameter.location == "page 7, table 1, row 2"
    assert diameter.raw == "7.63 in"

    fy = field_by(result, "Steel yield strength Fy")
    assert fy.value == pytest.approx(50.0)
    assert fy.unit == "ksi"


def test_pair_table_without_parameter_header():
    table = Table(
        location="page 7, table 1",
        headers=["Item", "Value"],
        rows=[["Pile diameter O.D.", "7.63 in"]],
    )
    doc = ExtractedDocument(path=Path("export.txt"), file_type="txt", tables=[table])
    result = extract_fields(doc)
    assert field_by(result, "Pile diameter O.D.").unit == "in"


# --------------------------------------------------------------------------
# Pattern 2: fixed-format engineering tables
# --------------------------------------------------------------------------


def test_pile_schedule_grid_table():
    table = Table(
        location="sheet: Pile Schedule",
        headers=["Pile", "Diameter (mm)", "Thickness (mm)", "Fy (MPa)", "Length (m)"],
        rows=[
            ["HP1", "324", "12.7", "450", "18.0"],
            ["HP2", "406", "12.7", "450", "18.0"],
            ["HP3", "324", "9.5", "450", "15.0"],
        ],
    )
    doc = ExtractedDocument(path=Path("design_report.xlsx"), file_type="xlsx",
                            tables=[table])
    result = extract_fields(doc)

    d = field_by(result, "Diameter (mm)", context="HP1")
    assert d.value == 324.0
    assert d.unit == "mm"
    t = field_by(result, "Thickness (mm)", context="HP3")
    assert t.value == pytest.approx(9.5)
    assert t.unit == "mm"
    fy = field_by(result, "Fy (MPa)", context="HP2")
    assert fy.value == 450.0
    # row-specific values must not be flagged as conflicts
    assert result.conflicts == []


def test_header_unit_used_when_cell_has_none():
    table = Table(
        location="page 3, table 1",
        headers=["Section", "Area (mm2)", "Ix (mm4)"],
        rows=[["PIPE 324x12.7", "12400", "1.19e8"]],
    )
    doc = ExtractedDocument(path=Path("sections.pdf"), file_type="pdf", tables=[table])
    result = extract_fields(doc)
    area = field_by(result, "Area (mm2)", context="PIPE 324x12.7")
    assert area.value == pytest.approx(12400.0)
    assert area.unit == "mm2"
    ix = field_by(result, "Ix (mm4)", context="PIPE 324x12.7")
    assert ix.value == pytest.approx(1.19e8)


def test_load_capacity_ratio_table():
    table = Table(
        location="page 9, table 3",
        headers=["Load case", "Load (kN)", "Capacity (kN)", "Ratio"],
        rows=[["ULS-1", "980", "1250", "0.78"]],
    )
    doc = ExtractedDocument(path=Path("check.pdf"), file_type="pdf", tables=[table])
    result = extract_fields(doc)
    ratio = field_by(result, "Ratio", context="ULS-1")
    assert ratio.value == pytest.approx(0.78)
    assert ratio.unit is None  # ratioless numbers keep unit None, never guessed


def test_non_numeric_cell_kept_as_raw_and_warned():
    table = Table(
        location="page 4, table 1",
        headers=["Pile", "Length (m)"],
        rows=[["HP9", "TBD"]],
    )
    doc = ExtractedDocument(path=Path("draft.pdf"), file_type="pdf", tables=[table])
    result = extract_fields(doc)
    length = field_by(result, "Length (m)", context="HP9")
    assert length.value == "TBD"
    assert any("not a number" in w for w in result.warnings)


@pytest.mark.parametrize("placeholder", ["N/A", "n/a", "-", "--", "\u2014"])
def test_not_applicable_placeholder_is_kept_but_not_warned(placeholder):
    table = Table(
        location="page 4, table 1",
        headers=["Pile", "Nspt"],
        rows=[["HP9", placeholder]],
    )
    doc = ExtractedDocument(path=Path("draft.pdf"), file_type="pdf", tables=[table])
    result = extract_fields(doc)
    assert field_by(result, "Nspt", context="HP9").value == placeholder
    assert not any("not a number" in w for w in result.warnings)


def test_value_misread_as_numeric_header_is_not_warned():
    # A two-column settings table: "8 k" is the value, not a parameter name.
    table = Table(
        location="page 6, table 1",
        headers=["Lateral load", "8 k"],
        rows=[["Pile section", "Elastic"], ["Cyclic loading", "No"]],
    )
    doc = ExtractedDocument(path=Path("draft.pdf"), file_type="pdf", tables=[table])
    result = extract_fields(doc)
    assert not any("not a number" in w for w in result.warnings)


def test_formula_cell_skipped_with_warning_not_guessed():
    table = Table(
        location="sheet: Loads",
        headers=["Load Case", "Resistance check"],
        rows=[["ULS-1", "=C2+D2"]],
    )
    doc = ExtractedDocument(path=Path("report.xlsx"), file_type="xlsx", tables=[table])
    result = extract_fields(doc)
    assert not any(f.context == "ULS-1" for f in result.fields)
    assert any("formula" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Free text label = value lines
# --------------------------------------------------------------------------


def test_text_lines_extracted():
    doc = load_document(FIXTURES / "project_info.txt")
    result = extract_fields(doc)
    fy = next(f for f in result.fields if f.parameter == "Fy")
    assert fy.value == 450.0
    assert fy.unit == "MPa"
    length = next(f for f in result.fields if "pile length" in f.parameter.lower())
    assert length.value == pytest.approx(18.0)
    assert length.unit == "m"


def test_date_not_truncated_to_number():
    doc = ExtractedDocument(
        path=Path("report.txt"),
        file_type="txt",
        text_chunks=[
            TextChunk("line 1..2", "SUBMISSION DATE: 2026-09-23\nRevd 12:30 PM")
        ],
    )
    result = extract_fields(doc)
    assert not any(field.value == 2026.0 for field in result.fields)
    assert not any("SUBMISSION" in field.parameter for field in result.fields)
    assert result.fields == []  # dates/times are not numeric fields, never truncated


def test_parenthesised_label_text_pair():
    doc = ExtractedDocument(
        path=Path("report.txt"),
        file_type="txt",
        text_chunks=[
            TextChunk("line 1..12", "Yield strength (Fy): 345 MPa")
        ],
    )
    result = extract_fields(doc)
    fy = field_by(result, "Yield strength (Fy)")
    assert fy.value == pytest.approx(345.0)
    assert fy.unit == "MPa"
    assert fy.raw == "Yield strength (Fy): 345 MPa"


# --------------------------------------------------------------------------
# Section awareness: a value is never a bare word, it carries the section
# --------------------------------------------------------------------------


def test_fields_tied_to_their_section():
    doc = ExtractedDocument(
        path=Path("report.txt"),
        file_type="txt",
        text_chunks=[
            TextChunk("line 1..29", (
                "EXECUTIVE SUMMARY\n"
                "Pile diameter: 324 mm\n"
                "Pile wall thickness: 12.7 mm\n"
                "\n"
                "LOADS\n"
                "ULS-1: compression 980 kN, tension 610 kN\n"
                "\n"
                "RESULTS\n"
                "ULS-1 capacity ratio: 0.78\n"
            ))
        ],
    )
    result = extract_fields(doc)
    diameter = field_by(result, "Pile diameter")
    assert diameter.section == "EXECUTIVE SUMMARY"
    ratio = field_by(result, "ULS-1 capacity ratio")
    assert ratio.section == "RESULTS"


def test_meta_line_is_not_mistaken_for_a_section_heading():
    doc = ExtractedDocument(
        path=Path("report.txt"),
        file_type="txt",
        text_chunks=[
            TextChunk("line 1..12", (
                "PROJECT: Greenfield Substation\n"
                "Pile diameter: 324 mm\n"
            ))
        ],
    )
    result = extract_fields(doc)
    diameter = field_by(result, "Pile diameter")
    assert diameter.section == ""  # PROJECT: line is metadata, not a heading


def test_section_carried_across_chunks():
    doc = ExtractedDocument(
        path=Path("report.txt"),
        file_type="txt",
        text_chunks=[
            TextChunk("page 1", "DESIGN BASIS\nWall thickness 12.7 mm"),
            TextChunk("page 2", "Pile diameter: 324 mm"),
        ],
    )
    result = extract_fields(doc)
    for f in result.fields:
        assert f.section == "DESIGN BASIS"


def test_full_fixture_run_extracts_and_traces():
    docs = load_document(FIXTURES / "design_report.xlsx"), load_document(FIXTURES / "pile_schedule.csv")
    result = extract_fields(list(docs))
    sources = {f.source for f in result.fields}
    assert sources == {"design_report.xlsx", "pile_schedule.csv"}
    assert all(f.source and f.location for f in result.fields)


# --------------------------------------------------------------------------
# Conflicts: flag, never resolve
# --------------------------------------------------------------------------


def test_conflicting_values_flagged_not_resolved():
    t1 = Table(
        location="page 2, table 4",
        headers=["Item", "Value"],
        rows=[["Fy", "450 MPa"], ["Dp", "7.63 in"]],
    )
    t2 = Table(
        location="page 7, table 4",
        headers=["Parameter", "Value"],
        rows=[["Fy", "420 MPa"], ["Dp", "7.63 in"]],
    )
    doc = ExtractedDocument(path=Path("report.pdf"), file_type="pdf", tables=[t1, t2])
    result = extract_fields(doc)

    # BOTH originals kept -- nothing merged or dropped.
    fy_fields = [f for f in result.fields if _norm(f.parameter) == "fy"]
    assert sorted(f.value for f in fy_fields) == [420.0, 450.0]

    conflicts = [c for c in result.conflicts if _norm(c.parameter) == "fy"]
    assert len(conflicts) == 1
    assert conflicts[0].kind == "value"
    assert {f.location for f in conflicts[0].fields} == {
        "page 2, table 4, row 2",
        "page 7, table 4, row 2",
    }
    # Dp agrees -> no conflict for it.
    assert not any(_norm(c.parameter) == "dp" for c in result.conflicts)


def test_same_number_different_units_flagged():
    t1 = Table(location="page 2, table 4", headers=["Item", "Value"],
               rows=[["Pile diameter", "324 mm"]])
    t2 = Table(location="page 7, table 4", headers=["Item", "Value"],
               rows=[["Pile diameter", "324 in"]])
    doc = ExtractedDocument(path=Path("report.pdf"), file_type="pdf", tables=[t1, t2])
    result = extract_fields(doc)
    conflict = next(c for c in result.conflicts if c.kind == "unit")
    assert conflict.parameter.lower() == "pile diameter"
    assert {f.unit for f in conflict.fields} == {"mm", "in"}


def test_agreeing_sources_produce_no_conflict():
    fields = [
        ExtractedField("Fy", 450.0, "MPa", "a.pdf", "page 1", "450 MPa"),
        ExtractedField("Fy", 450.0, "MPa", "b.pdf", "page 3, table 1", "450 MPa"),
    ]
    assert find_conflicts(fields) == []


def test_conflict_never_resolves_by_averaging():
    t1 = Table(location="page 2, table 4", headers=["Item", "Value"],
               rows=[["tP", "0.500 in"]])
    t2 = Table(location="page 7, table 4", headers=["Item", "Value"],
               rows=[["tP", "0.450 in"]])
    doc = ExtractedDocument(path=Path("report.pdf"), file_type="pdf", tables=[t1, t2])
    result = extract_fields(doc)
    values = [f.value for f in result.fields if _norm(f.parameter) == "tp"]
    assert values == [0.5, 0.45]  # no average, no winner, both preserved
    conflict = next(c for c in result.conflicts if _norm(c.parameter) == "tp")
    assert isinstance(conflict, Conflict)


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


# --------------------------------------------------------------------------
# Records always traceable; serialization
# --------------------------------------------------------------------------


def test_every_field_is_traceable_never_bare_number():
    docs = [
        load_document(FIXTURES / "design_report.xlsx"),
        load_document(FIXTURES / "pile_schedule.csv"),
        load_document(FIXTURES / "project_info.txt"),
        load_document(FIXTURES / "design_report.docx"),
    ]
    result = extract_fields(docs)
    assert result.fields
    for f in result.fields:
        assert f.parameter
        assert f.source
        assert f.location
        if isinstance(f.value, float):
            assert f.raw  # number always carries the printed text it came from


def test_to_dict_round_trip_shape():
    table = Table(location="page 7, table 1", headers=["Parameter", "Value"],
                  rows=[["Pile diameter O.D.", "7.63 in"]])
    doc = ExtractedDocument(path=Path("export.pdf"), file_type="pdf", tables=[table])
    payload = extract_fields(doc).to_dict()
    assert set(payload) == {"fields", "conflicts", "warnings"}
    f = payload["fields"][0]
    assert f["parameter"] == "Pile diameter O.D."
    assert f["value"] == pytest.approx(7.63)
    assert f["unit"] == "in"
    assert f["source"] == "export.pdf"


def test_cli_runs_on_fixtures(capsys):
    from app.extraction.field_extractor import main

    rc = main([str(FIXTURES / "pile_schedule.csv"), str(FIXTURES / "project_info.txt")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fields" in out
    assert "conflicts" in out


def test_cli_usage_without_args():
    from app.extraction.field_extractor import main

    assert main([]) == 2
