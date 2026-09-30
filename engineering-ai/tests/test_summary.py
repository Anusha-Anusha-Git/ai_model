"""Tests for Stage 6: app/report/summary.py (noise filter + LLM prompt)."""

from __future__ import annotations

from pathlib import Path

from app.extraction.document_loader import Table
from app.extraction.field_extractor import ExtractedField
from app.report.summary import (
    build_prompt_payload,
    build_summary_prompt,
    filter_noise_for_llm,
    generate_clean_summary,
    graph_table_locations,
    validate_summary_format,
    validate_summary_values,
)


def row(parameter, value, unit=None, context=""):
    return {"parameter": parameter, "value": value, "unit": unit,
            "context": context, "location": "page 7, table 1"}


def f(parameter, value, unit=None, context="", location="page 1, table 1, row 2"):
    return ExtractedField(parameter=parameter, value=value, unit=unit, source="r.pdf",
                          location=location, raw=str(value), context=context)


def test_graph_table_ignored():
    tables = [
        Table(location="page 4, table 1",
              headers=["El.", "Moment", "Shear", "Slope", "Settlement y"],
              rows=[[str(i)] * 5 for i in range(15)]),  # 15 rows -> graph
    ]
    locs = graph_table_locations(tables)
    assert locs == {"page 4, table 1"}
    fields = [
        f("Moment", 12.0, location="page 4, table 1, row 3"),
        f("Pile diameter", 7.63, "in"),
        f("v", 0.3, context="18.5", location="page 4, table 1, row 5"),  # coordinate ctx
    ]
    cleaned = filter_noise_for_llm(fields, tables)
    assert [c.parameter for c in cleaned] == ["Pile diameter"]


def test_short_table_is_not_noise():
    # a few short rows are manageable; >10-row coordinate tables are the noise
    tables = [
        Table(location="page 2, table 1",
              headers=["El.", "Moment"],
              rows=[["0", "1"], ["1", "2"]]),
    ]
    assert graph_table_locations(tables) == set()


def test_keyword_and_numeric_context_filtered():
    fields = [
        f("Spring Pres.", 3.2, "k/ft2"),
        f("Shear", 45.0, "kN"),
        f("Moment", 62.0, "kN*m"),
        f("Compression", 980.0, "kN", context="ULS-1"),
        f("Pile diameter", 7.63, "in", context=""),
    ]
    cleaned = filter_noise_for_llm(fields)
    assert [c.parameter for c in cleaned] == [
        "Shear", "Moment", "Compression", "Pile diameter"
    ]


def test_numeric_stage_context_is_preserved():
    fields = [
        f("Axial Load", 120.0, "k", context="0"),
        f("Pile diameter", 12.0, "in", context="12.5"),
    ]
    cleaned = filter_noise_for_llm(fields)
    assert [field.parameter for field in cleaned] == ["Axial Load", "Pile diameter"]


def test_prompt_contains_exact_contract_and_clean_data():
    fields = [f("Pile diameter", 7.63, "in")]
    prompt = build_summary_prompt(fields, raw_count=12)
    for heading in (
        "1. PROJECT INFO:",
        "2. PILE SPECS:",
        "3. SOIL PARAMS:",
        "4. APPLIED LOADS:",
        "5. RESULTS:",
        "6. SETTLEMENT & LATERAL:",
        "7. DURABILITY:",
        "8. ENGINEERING EVALUATION:",
        "9. REFERENCES:",
    ):
        assert heading in prompt
    assert "Analysis summary" in prompt
    assert "Not specified" in prompt
    assert '"total_raw_fields_extracted": 12' in prompt
    assert '"parameter": "Pile diameter"' in prompt
    assert "Final shear and moment values" in prompt
    assert "largest explicitly reported R/CAP" in prompt
    assert "Never substitute an ULTIMATE capacity" in prompt
    assert "📄" not in prompt


def test_prompt_includes_source_paths_and_table_headers():
    fields = [f("Pile diameter", 12.0, "in")]
    prompt = build_summary_prompt(
        fields,
        raw_count=1,
        source_paths=["/project/site-a/report.pdf"],
        table_references=[{
            "source": "report.pdf",
            "source_path": "/project/site-a/report.pdf",
            "location": "page 7, table 1",
            "headers": ["Stage", "R/CAP"],
        }],
    )
    assert '"source_path": "/project/site-a/report.pdf"' in prompt
    assert '"location": "page 7, table 1"' in prompt
    assert '"R/CAP"' in prompt


def valid_summary() -> str:
    return """1. PROJECT INFO: Project Name: Test; Structure: Not specified; Location: Not specified; Client: Not specified; Prepared By: Not specified; Date: Not specified; Software: Not specified.
2. PILE SPECS: Type: Helical pile; O.D.: 12 in; Thickness: initial 0.375 in, effective after corrosion Not specified; Fy: 50 ksi; # of Helices: 3; Helix Diameters: 12 in, 10 in, 8 in; Spacing: 3 in; Tip Elevation: -30 ft.
3. SOIL PARAMS: Model: Reese; Friction Angle (Φ): 30 deg; Cohesion (c'): 1 ksf; Unit Weight (γ): 110 pcf.
4. APPLIED LOADS: Stage 0 (Axial: 120 k; Shear: 20 k; Moment: 40 k-ft); Stage 1 (Axial: 60 k; Shear: 10 k; Moment: 25 k-ft).
5. RESULTS: Governing Stage: Stage 0; Geo Comp Capacity: 200 k; Geo Tension Capacity: 150 k; Structural Capacity: 180 k; R/GEO: 0.6 (dimensionless); R/STR: 0.7 (dimensionless); R/CAP: 0.8 (dimensionless); Governing Failure Mode: Plate bearing.
6. SETTLEMENT & LATERAL: Settlement at Design Load: 0.5 in; Max Lateral Deflection: 0.4 in; Max Internal Moment: 35 k-ft; Pile Head Condition: Fixed.
7. DURABILITY: Design Life: 75 years; Corrosion Thickness Loss: 0.047 in.
8. ENGINEERING EVALUATION:
- Overall Adequacy / Governing Limit State: PASS — Stage 0 has R/CAP below 1.0.
- Utilization: Stage 0 has the largest reported R/CAP at 0.8 (dimensionless).
- Settlement: The report does not state an acceptance criterion, so tolerance cannot be assessed.
- Red Flags to Verify: Effective thickness and acceptance criterion are Not specified.
9. REFERENCES: report.pdf, page 7, table 1, columns Stage and R/CAP."""


def test_summary_format_accepts_exact_nine_section_contract():
    assert validate_summary_format(valid_summary()) == []


def test_summary_format_rejects_missing_or_extra_bullets():
    missing_reference = valid_summary().replace("9. REFERENCES:", "10. REFERENCES:")
    assert validate_summary_format(missing_reference)

    extra_bullet = valid_summary().replace(
        "- Red Flags to Verify:",
        "- Red Flags to Verify:\n- Extra:",
    )
    errors = validate_summary_format(extra_bullet)
    assert "section 8 must contain exactly four bullet lines" in errors


def test_summary_values_reject_numbers_not_in_source():
    source = build_prompt_payload(
        [f("R/CAP", 0.8, context="Stage 0", location="page 7, table 1")],
        raw_count=1,
    )
    assert validate_summary_values(valid_summary(), source)
    assert "numeric value '0.9' is not present in source data" in validate_summary_values(
        valid_summary().replace("R/CAP: 0.8", "R/CAP: 0.9"), source
    )


# --------------------------------------------------------------------------
# Stage 2b clean engineering summary: deterministic, and never invents a value
# --------------------------------------------------------------------------

HELIX_FIELDS = [
    row("Pile diameter O.D.", 7.63, "in", "HP-1"),
    row("tP", 0.375, "in", "HP-1"),
    row("Fy", 80.0, "ksi", "HP-1"),
    row("Numbe", 3.0, None, "HP-1"),
    row("Dhel", 24.0, "in", "Plate 1"),
    row("LOAD/CAP", 0.907, None, "Critical"),
    row("R/GEO", 0.907, None, "Stage 0"),
    row("R/STR", 0.542, None, "Stage 0"),
    row("Settlement @des. cap.", 0.431, "in", "Stage 0"),
    row("Mx.max", 14.45, "k-ft", "Stage 0"),
    row("Factored load", 150.0, "k", "Stage 0"),
    row("Factored load", 19.98, "k", "Stage 1"),
    row("Design life", 30.0, "years"),
]


def test_clean_summary_reports_only_printed_values():
    summary = generate_clean_summary(HELIX_FIELDS)
    assert "**Pile Diameter (O.D.):** 7.63 in" in summary
    assert "**Pile Thickness (initial):** 0.375 in" in summary
    assert "**R/CAP (Overall Critical Ratio):** 0.907" in summary
    assert "**Estimated Settlement:** 0.431 in" not in summary  # old fabricated label
    assert "**Settlement at Design Load:** 0.431 in" in summary
    assert "**# of Helices:** 3" in summary


def test_clean_summary_never_invents_a_missing_value():
    """A null unit must never render as the string 'None'."""
    summary = generate_clean_summary([])
    for label in ("**Project Name:**", "**Pile Diameter (O.D.):**",
                  "**R/CAP (Overall Critical Ratio):**", "**Design Life:**"):
        assert f"{label} Not specified" in summary
    assert "None" not in summary
    # values that were never extracted are not conjured up
    for invented in ("7.63", "165.4", "375.64", "0.213", "30 years",
                     "KINDER MORGAN", "ICC AC358", "Free head", "6 ft"):
        assert invented not in summary


def test_clean_summary_adequacy_requires_a_printed_ratio():
    no_ratio = generate_clean_summary([row("Pile diameter O.D.", 7.63, "in")])
    assert "**FAIL** \u2014 adequacy cannot be demonstrated" in no_ratio

    under = generate_clean_summary([row("LOAD/CAP", 0.907, None, "Critical")])
    assert "**PASS**" in under and "0.907" in under

    over = generate_clean_summary([row("LOAD/CAP", 1.34, None, "Critical")])
    assert "**FAIL** \u2014 explicitly reported R/CAP = 1.34" in over


def test_clean_summary_governing_ratio_is_the_largest_printed_one():
    summary = generate_clean_summary([
        row("LOAD/CAP", 0.907, None, "Stage 0"),
        row("LOAD/CAP", 1.24, None, "Stage 1"),
    ])
    assert "**R/CAP (Overall Critical Ratio):** 1.24" in summary
    assert "**Governing Stage:** Stage 1" in summary


def test_clean_summary_flags_missing_critical_values_and_is_not_certified():
    summary = generate_clean_summary([row("Pile diameter O.D.", 7.63, "in")])
    assert "Red Flags to Verify" in summary
    assert "never assumed" in summary
    assert "Settlement" in summary
    assert "not certified or approved engineering work" in summary


def test_prompt_lookup_guide_is_hints_not_expected_values():
    """The HelixPile label guide must never seed an answer into the output."""
    prompt = build_summary_prompt([f("R/CAP", 0.8, context="Stage 0")], raw_count=1)
    assert "### FIELD LOOKUP GUIDE ###" in prompt
    for label in ("Dp", "tP", "T new", "Numbe", "Dhel", "Shel",
                  "LOAD/CAP GEOTECH", "Settlement @des. cap.", "Thickness loss Steel (in)"):
        assert label in prompt
    assert "SEARCH HINTS ONLY" in prompt
    # a guide that names a standard or a soil value would let the model invent it
    assert "ICC AC358" not in prompt
    assert "40°" not in prompt
    assert "4000 psf" not in prompt
    assert "not printed in the source data" in prompt


# --------------------------------------------------------------------------
# Project title block: a title line holds no number, so Stage 2b never emits it
# as a field. The summary must read it from the Stage-2c project key points.
# --------------------------------------------------------------------------

TITLE_BLOCK = [
    {"key": "project_name", "value": "Baytown Compressor Station - Structure C",
     "unit": None, "context": "", "source": "report.pdf", "location": "page 1", "raw": "..."},
    {"key": "structure", "value": "PEMB", "unit": None, "context": "",
     "source": "report.pdf", "location": "page 1", "raw": "..."},
    {"key": "location", "value": "Karnes County, Texas", "unit": None, "context": "",
     "source": "report.pdf", "location": "page 1", "raw": "..."},
    {"key": "client", "value": "Kinder Morgan", "unit": None, "context": "",
     "source": "report.pdf", "location": "page 1", "raw": "..."},
    {"key": "prepared_by", "value": "J. Alvarez, P.E.", "unit": None, "context": "",
     "source": "report.pdf", "location": "page 1", "raw": "..."},
    {"key": "date", "value": "8/27/2026", "unit": None, "context": "",
     "source": "report.pdf", "location": "page 1", "raw": "..."},
]


def test_clean_summary_reads_project_name_from_title_block():
    summary = generate_clean_summary([], project=TITLE_BLOCK)
    assert "**Project Name:** Baytown Compressor Station - Structure C" in summary
    assert "**Structure:** PEMB" in summary
    assert "**Location:** Karnes County, Texas" in summary
    assert "**Client:** Kinder Morgan" in summary
    assert "**Prepared By:** J. Alvarez, P.E." in summary
    assert "**Date:** 8/27/2026" in summary


def test_clean_summary_project_name_never_inferred_without_source():
    """No title block and no field means 'Not specified', never a reconstruction."""
    summary = generate_clean_summary([row("Pile diameter O.D.", 7.63, "in", "HP-1")])
    assert "**Project Name:** Not specified" in summary
    assert "**Location:** Not specified" in summary


def test_prompt_carries_the_title_block():
    prompt = build_summary_prompt(
        [f("Pile diameter", 7.63, "in")], raw_count=1, project=TITLE_BLOCK
    )
    assert '"project_title_block"' in prompt
    assert '"Baytown Compressor Station - Structure C"' in prompt
    assert "do not reconstruct a project name from a file path" in prompt
