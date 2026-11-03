"""Regression tests: engineering units survive extraction and summarization.

Covers the failure modes reported for HelixPile-style reports:
- a value and its unit printed in the same cell, in adjacent cells, in the
  column header, or in a separate units row under the header;
- compound hyphenated units ("k-ft", "kip-ft", "kN-m") that used to be
  truncated to the first token;
- values that genuinely carry no unit (dimensionless ratios);
- the Executive Summary reporting each value with the unit the source printed,
  and flagging any value/unit it could not trace.

Run from the repo root (engineering-ai/):

    ./.venv/bin/python -m pytest tests/test_unit_preservation.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extraction.document_loader import ExtractedDocument, Table
from app.extraction.field_extractor import (
    ExtractedField,
    extract_fields,
    split_value_unit,
)
from app.report.executive_summary import (
    build_executive_summary_text,
    validate_executive_summary,
)


def field_by(fields, parameter, context=""):
    for f in fields:
        if f.parameter == parameter and f.context == context:
            return f
    raise AssertionError(f"no field {parameter!r} (context={context!r})")


def _doc(tables=None, path="report.pdf", file_type="pdf"):
    return ExtractedDocument(path=Path(path), file_type=file_type, tables=tables or [])


def _payload(fields=None, key_points=None):
    return {
        "fields": fields or [],
        "key_points": key_points or {},
        "conflicts": [],
        "warnings": [],
        "clean_engineering_summary": "",
    }


def _field(parameter, value, unit=None, context="", raw=None):
    return {
        "parameter": parameter,
        "value": value,
        "unit": unit,
        "context": context,
        "section": "",
        "source": "report.pdf",
        "location": "page 2",
        "raw": raw if raw is not None else f"{parameter}: {value} {unit or ''}".strip(),
    }


# --------------------------------------------------------------------------
# 1. Value and unit in the same cell
# --------------------------------------------------------------------------


def test_value_and_unit_in_one_cell_kept_together():
    value, unit, raw = split_value_unit("7.63 in")
    assert value == pytest.approx(7.63)
    assert unit == "in"
    assert raw == "7.63 in"


def test_dimensionless_value_in_one_cell_gets_no_unit():
    value, unit, _ = split_value_unit("0.682")
    assert value == pytest.approx(0.682)
    assert unit is None


# --------------------------------------------------------------------------
# 2. Compound hyphenated units are never truncated
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "printed, number, unit",
    [
        ("126 k-ft", 126.0, "k-ft"),
        ("141.6 k-ft", 141.6, "k-ft"),
        ("40 kip-ft", 40.0, "kip-ft"),
        ("62 kN-m", 62.0, "kN-m"),
        ("120 lb-ft", 120.0, "lb-ft"),
        ("10 kN*m", 10.0, "kN*m"),
    ],
)
def test_compound_units_not_truncated(printed, number, unit):
    value, parsed_unit, _ = split_value_unit(printed)
    assert value == pytest.approx(number)
    assert parsed_unit == unit


def test_compound_unit_survives_a_grid_table():
    table = Table(
        location="page 4, table 1",
        headers=["Load case", "Moment", "Shear"],
        rows=[["ULS-1", "126 k-ft", "12 kip-ft"]],
    )
    result = extract_fields(_doc([table]))
    assert field_by(result.fields, "Moment", "ULS-1").unit == "k-ft"
    assert field_by(result.fields, "Shear", "ULS-1").unit == "kip-ft"


# --------------------------------------------------------------------------
# 3. Units in a separate row under the header (HelixPile-style)
# --------------------------------------------------------------------------


def test_unit_row_associated_with_its_column():
    table = Table(
        location="page 2, table 1",
        headers=["Section", "Fy", "Dp", "tP"],
        rows=[
            ["Name", "(ksi)", "(in)", "(in)"],
            ["5: 7.63 x 0.375 x 80", "70", "18", "0.375"],
        ],
    )
    result = extract_fields(_doc([table], path="helixpile.pdf"))
    fy = field_by(result.fields, "Fy", "5: 7.63 x 0.375 x 80")
    dp = field_by(result.fields, "Dp", "5: 7.63 x 0.375 x 80")
    tp = field_by(result.fields, "tP", "5: 7.63 x 0.375 x 80")
    assert (fy.value, fy.unit) == (70.0, "ksi")
    assert (dp.value, dp.unit) == (18.0, "in")
    assert (tp.value, tp.unit) == (0.375, "in")
    assert fy.unit_status == dp.unit_status == tp.unit_status == "unit_row"
    # the units row itself must never become a field
    assert not any(isinstance(f.value, str) for f in result.fields)


def test_unit_row_with_exponents_and_labels():
    table = Table(
        location="page 3, table 1",
        headers=["Plate", "Dhel", "Ahel", "Qhel"],
        rows=[["Name", "(in)", "(in^2)", "(k)"], ["1", "24", "452", "60"]],
    )
    result = extract_fields(_doc([table], path="helixpile.pdf"))
    dhel = field_by(result.fields, "Dhel")
    ahel = field_by(result.fields, "Ahel")
    assert (dhel.value, dhel.unit) == (24.0, "in")
    assert (ahel.value, ahel.unit) == (452.0, "in^2")
    assert dhel.unit_status == ahel.unit_status == "unit_row"


# --------------------------------------------------------------------------
# 4. Units in an adjacent cell / column
# --------------------------------------------------------------------------


def test_adjacent_unit_column_used_when_value_has_none():
    table = Table(
        location="page 1, table 1",
        headers=["Parameter", "Value", "Unit"],
        rows=[["Yield strength (Fy)", "70", "ksi"], ["Moment", "126", "k-ft"]],
    )
    result = extract_fields(_doc([table]))
    assert field_by(result.fields, "Yield strength (Fy)").unit == "ksi"
    assert field_by(result.fields, "Moment").unit == "k-ft"


# --------------------------------------------------------------------------
# 5. Units in the column header
# --------------------------------------------------------------------------


def test_header_unit_parenthesised_used_for_bare_values():
    table = Table(
        location="page 3, table 1",
        headers=["Section", "Area (mm2)", "Ix (mm4)"],
        rows=[["PIPE 324", "12400", "1.19e8"]],
    )
    result = extract_fields(_doc([table]))
    area = field_by(result.fields, "Area (mm2)", "PIPE 324")
    ix = field_by(result.fields, "Ix (mm4)", "PIPE 324")
    assert (area.value, area.unit) == (12400.0, "mm2")
    assert (ix.value, ix.unit) == (1.19e8, "mm4")  # scientific notation preserved
    assert area.unit_status == ix.unit_status == "from_header"


def test_header_unit_trailing_separator_used():
    table = Table(
        location="page 1, table 1",
        headers=["Item", "Yield strength, ksi", "Diameter in"],
        rows=[["Steel", "70", "18"]],
    )
    result = extract_fields(_doc([table]))
    assert field_by(result.fields, "Yield strength, ksi", "Steel").unit == "ksi"
    assert field_by(result.fields, "Diameter in", "Steel").unit == "in"


# --------------------------------------------------------------------------
# 6. Decimals, negatives, percentages, dimensionless
# --------------------------------------------------------------------------


def test_decimal_negative_and_percent_units_preserved():
    assert split_value_unit("-12.5 mm")[0:2] == (-12.5, "mm")
    value, unit, _ = split_value_unit("55 %")
    assert value == pytest.approx(55.0)
    assert unit == "%"


def test_dimensionless_ratio_keeps_no_unit():
    table = Table(
        location="page 9, table 3",
        headers=["Load case", "Load (kN)", "Ratio"],
        rows=[["ULS-1", "980", "0.682"]],
    )
    result = extract_fields(_doc([table]))
    ratio = field_by(result.fields, "Ratio", "ULS-1")
    assert ratio.value == pytest.approx(0.682)
    assert ratio.unit is None
    assert ratio.unit_status == "none"


# --------------------------------------------------------------------------
# 7. A genuinely missing unit is never invented
# --------------------------------------------------------------------------


def test_value_with_no_unit_anywhere_stays_unit_none():
    table = Table(
        location="page 1, table 1",
        headers=["Item", "Count"],
        rows=[["Helices", "3"]],
    )
    result = extract_fields(_doc([table]))
    count = field_by(result.fields, "Helices")
    assert count.unit is None
    assert count.unit_status == "none"


def test_existing_source_unit_not_overwritten_by_header():
    table = Table(
        location="page 1, table 1",
        headers=["Section", "Load (kN)", "Other", "X"],
        rows=[["A", "980 lb", "1", "2"]],
    )
    result = extract_fields(_doc([table]))
    load = field_by(result.fields, "Load (kN)", "A")
    assert load.unit == "lb"  # the value's own printed unit wins
    assert load.unit_status == "value"


# --------------------------------------------------------------------------
# 8. Conflicting / ambiguous source units are never silently resolved
# --------------------------------------------------------------------------


def test_conflicting_units_are_flagged_in_extraction():
    t1 = Table(location="page 1, table 1", headers=["Item", "Value"],
               rows=[["Dp", "18 in"]])
    t2 = Table(location="page 2, table 1", headers=["Item", "Value"],
               rows=[["Dp", "457 mm"]])
    result = extract_fields(_doc([t1, t2]))
    conflicts = [c for c in result.conflicts if c.parameter.lower() == "dp"]
    assert len(conflicts) == 1
    assert conflicts[0].kind == "unit"
    assert {f.unit for f in conflicts[0].fields} == {"in", "mm"}


# --------------------------------------------------------------------------
# 9. Executive Summary keeps each value with its unit
# --------------------------------------------------------------------------


EXAMPLE_FIELDS = [
    _field("Dp", 18.0, "in"),
    _field("Fy", 70.0, "ksi", "HP1"),
    _field("tP", 0.375, "in"),
    _field("Axial load", 126.0, "k", "Stage 0"),
    _field("Uplift load", 141.6, "k", "Stage 1"),
    _field("R/GEO", 0.682, None, "Stage 0"),
]

EXAMPLE_KEY_POINTS = {
    "project": [{"key": "project_name", "value": "Kinder Morgan", "unit": None, "context": ""}],
    "pile": [{"key": "helix_count", "value": 3.0, "unit": None, "context": "HP1"}],
    "loads": [
        {"key": "load_compression", "value": 126.0, "unit": "k", "context": "Stage 0"},
        {"key": "load_tension", "value": 141.6, "unit": "k", "context": "Stage 1"},
    ],
    "results": [],
}


def test_summary_reports_each_value_with_its_source_unit():
    text = build_executive_summary_text(_payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS))
    for fragment in ("18 in", "0.375 in", "Fy = 70 ksi", "126 k", "141.6 k"):
        assert fragment in " ".join(text.split()), fragment


def test_summary_does_not_put_a_unit_on_a_dimensionless_ratio():
    text = build_executive_summary_text(_payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS))
    assert "0.682" in text
    assert "0.682 in" not in text
    assert "0.682 k" not in text


def test_missing_dimension_does_not_produce_a_broken_clause():
    payload = _payload(
        fields=[_field("Fy", 70.0, "ksi", "HP1")],
        key_points=EXAMPLE_KEY_POINTS,
    )
    text = build_executive_summary_text(payload)
    assert "not stated in the report, thick" not in text
    assert "equipped with not stated in the report" not in text


# --------------------------------------------------------------------------
# 10. Validation flags lost, changed, and invented values
# --------------------------------------------------------------------------


def test_clean_summary_has_no_validation_flags():
    data = _payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS)
    text = build_executive_summary_text(data)
    assert validate_executive_summary(text, data) == []


def test_lost_unit_is_flagged():
    data = _payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS)
    text = build_executive_summary_text(data).replace("18 in outside", "18 outside")
    flags = validate_executive_summary(text, data)
    assert any("18" in flag and "in" in flag for flag in flags)


def test_changed_unit_is_flagged():
    data = _payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS)
    text = build_executive_summary_text(data).replace("70 ksi", "70 MPa")
    flags = validate_executive_summary(text, data)
    assert any("70" in flag and "ksi" in flag for flag in flags)


def test_invented_value_is_flagged():
    data = _payload(EXAMPLE_FIELDS, EXAMPLE_KEY_POINTS)
    text = build_executive_summary_text(data).replace("0.682", "0.812")
    flags = validate_executive_summary(text, data)
    assert any("0.812" in flag for flag in flags)


def test_same_number_in_another_field_does_not_flag_a_correct_unit():
    # "24 in" (helix diameter) is correct even though unrelated fields share
    # the number 24 with other units (a 24 k load, etc.).
    data = _payload(
        fields=[
            _field("Dhel", 24.0, "in"),
            _field("Some load", 24.0, "k", "Stage 0"),
        ],
        key_points={},
    )
    text = "The pile is equipped with a 24 in diameter helix."
    assert validate_executive_summary(text, data) == []


def test_a_number_with_no_agreeing_source_unit_is_flagged():
    data = _payload(fields=[_field("Dhel", 24.0, "in")], key_points={})
    text = "The pile is equipped with a 24 k diameter helix."
    flags = validate_executive_summary(text, data)
    assert any("24" in flag and "in" in flag for flag in flags)


# --------------------------------------------------------------------------
# 11. Provenance and serialization
# --------------------------------------------------------------------------


def test_unit_status_records_provenance():
    assert ExtractedField("Dp", 18.0, "in", "r.pdf", "p1", "18 in").unit_status == ""
    table = Table(
        location="page 1, table 1",
        headers=["Item", "Value (kN)", "Unit"],
        rows=[["A", "1", "kN"], ["B", "2", "kN"]],
    )
    result = extract_fields(_doc([table]))
    assert field_by(result.fields, "Value (kN)", "A").unit_status == "from_header"


def test_to_dict_includes_unit_status():
    table = Table(location="page 1, table 1", headers=["Parameter", "Value"],
                  rows=[["Fy", "70 ksi"]])
    payload = extract_fields(_doc([table])).to_dict()
    assert payload["fields"][0]["unit_status"] == "value"


def test_label_value_unit_table_preserves_unit():
    """Structural label|value|unit tables with generic headers yield values+units."""
    table = Table(
        location="page 2, table 1",
        headers=["Item", "Illustrative input / result", "Unit / note"],
        rows=[
            ["Pile shaft outside diameter", "8.00", "in"],
            ["Shaft wall thickness", "0.375", "in"],
            ["Steel yield strength, Fy", "65", "ksi"],
            ["Applied axial compression", "135", "kip (k)"],
            ["Pile-head moment", "0", "kip-ft"],
            ["Assumed analysis design life", "30", "years"],
        ],
    )
    result = extract_fields(_doc([table]))
    f = field_by(result.fields, "Pile shaft outside diameter")
    assert f.value == pytest.approx(8.0)
    assert f.unit == "in"
    f = field_by(result.fields, "Steel yield strength, Fy")
    assert f.value == pytest.approx(65.0)
    assert f.unit == "ksi"
    f = field_by(result.fields, "Applied axial compression")
    assert f.unit == "kip (k)" or f.unit == "k"
    f = field_by(result.fields, "Pile-head moment")
    assert f.unit == "kip-ft"
    f = field_by(result.fields, "Assumed analysis design life")
    assert f.unit == "years"
