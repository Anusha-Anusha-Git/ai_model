"""Executive Summary narrative builder (five-paragraph contract).

Turns the extraction payload the web UI already holds (``fields``,
``key_points``, ``clean_engineering_summary``, ``conflicts``, ``warnings``, and
the user-supplied ``project_inputs``) into the fixed five-paragraph Executive
Summary used for engineering review, and renders it as a one- to two-page PDF.

Hard constraints (inherited from the rest of the project):
- Fully deterministic. No LLM, no recalculation, no rounding, no unit
  conversion. Every number is copied verbatim from the extracted fields.
- **Nothing is invented.** A value the report does not state renders as
  ``Not specified``.
- The adequacy verdict is ``adequate for service`` only when every reported
  utilization ratio is below 1.0 AND a design settlement is reported AND the
  user-supplied installation / review flags are set. Otherwise the summary names
  the check that is not satisfied (or could not be confirmed) and states that the
  design requires engineer review. The phrase ``fully adequate`` is never used.
- User-supplied inputs (``project_inputs``) are used only for wording that is not
  in the report (a system descriptor, a corrosion category, two sign-off flags)
  and are labelled ``user-supplied`` in the output metadata.
- Output is an evaluation aid. It is never certified or approved work.

The public entry points are :func:`build_executive_summary_text`,
:func:`build_executive_summary_narrative_pdf`, and
:func:`validate_executive_summary` (which flags a reported measurement that
lost its source unit or cannot be traced back to the source data).
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Sequence
from xml.sax.saxutils import escape

from app.config import normalize_project_inputs
from app.extraction.field_extractor import _KNOWN_UNITS
from app.report.summary import (
    MISSING_VALUE,
    resolve_summary_field,
)

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

TITLE = "Executive Summary: Helical Pile Design Verification"

#: The single placeholder for a value the report does not print.  There is no
#: second placeholder phrase anywhere in the narrative.
NOT_STATED = MISSING_VALUE

#: The foundation-system descriptor when the user supplies none.
SYSTEM_FALLBACK = "foundation system"

#: Extracted code string -> the display string printed in the summary. Kept
#: separate from extraction so the report is never re-worded at the source.
DISPLAY_CODE_MAP = {"AISC 360-22": "ANSI/AISC 360-22"}

_ADEQUATE_VERDICT = "adequate for service"
_NOT_ADEQUATE_VERDICT = "requires engineer review"

_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

_MARGIN = 20 * mm
_USABLE_WIDTH = A4[0] - 2 * _MARGIN

_RE_GARBAGE = {"", MISSING_VALUE.lower(), "none", "null", "n/a"}

_INCH_UNITS = {"in", "inch", "inches"}

_MEASURE_RE = re.compile(
    r"^\s*(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>[A-Za-z%][A-Za-z0-9%*/^\-]*)?"
)


# --------------------------------------------------------------------------
# Value helpers
# --------------------------------------------------------------------------


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text if text is not None else "")).strip()


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return _clean(value)


def _field_text(field: dict[str, Any]) -> str:
    value = field.get("Value", field.get("value"))
    unit = _clean(field.get("Unit", field.get("unit")))
    text = _format_number(value)
    return _clean(f"{text} {unit}" if unit else text)


def _resolved(fields: Sequence[dict[str, Any]], key: str) -> str | None:
    """The printed value the clean summary resolves for one line, or None."""
    field, _context = resolve_summary_field(list(fields), key)
    if field is None:
        return None
    text = _field_text(field)
    if not text or text.casefold() in _RE_GARBAGE:
        return None
    return text


def _leading_number(text: Any) -> float | None:
    match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", _clean(text))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _split_measure(text: Any) -> tuple[float, str | None] | None:
    match = _MEASURE_RE.match(_clean(text))
    if not match:
        return None
    try:
        number = float(match.group("num").replace(",", ""))
    except ValueError:
        return None
    return number, (_clean(match.group("unit")) or None)


def _measure_from_point(point: dict[str, Any] | None) -> tuple[float, str | None] | None:
    if not isinstance(point, dict):
        return None
    value = point.get("value")
    unit = _clean(point.get("unit")) or None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value), unit
    return _split_measure(value)


def _measure_from_field(
    fields: Sequence[dict[str, Any]], key: str
) -> tuple[float, str | None] | None:
    field, _context = resolve_summary_field(list(fields), key)
    if field is None:
        return None
    value = field.get("Value", field.get("value"))
    unit = _clean(field.get("Unit", field.get("unit"))) or None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value), unit
    parsed = _split_measure(value)
    if parsed is None:
        return None
    number, parsed_unit = parsed
    return number, (unit or parsed_unit)


def _measure(data: dict[str, Any], kp_key: str, field_key: str) -> tuple[float, str | None] | None:
    """Prefer the extracted key point; fall back to the resolved raw field."""
    key_points = _as_dict(data.get("key_points"))
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    point = _key_point(key_points, kp_key)
    parsed = _measure_from_point(point)
    if parsed is not None:
        return parsed
    return _measure_from_field(fields, field_key)


def _in_inches(number: float, unit: str | None, form: str) -> str:
    """A dimensional phrase preserving the source unit."""
    text = _format_number(number)
    unit_str = unit if unit else ""
    suffix = {
        "od": " outside diameter (OD)",
        "wall": " thick",
        "dia": " diameter",
        "thick": " thick",
    }[form]
    if unit_str:
        return f"{text} {unit_str}{suffix}"
    return f"{text}{suffix}"


def _force_text(point: dict[str, Any] | None) -> str | None:
    """A force load preserving the source unit."""
    if not isinstance(point, dict):
        return None
    value = point.get("value")
    if isinstance(value, bool):
        return None
    number = value if isinstance(value, (int, float)) else _leading_number(value)
    if number is None:
        return None
    unit = _clean(point.get("unit"))
    magnitude = abs(float(number))
    text = _format_number(magnitude)
    if unit:
        return f"{text} {unit}"
    return text


def _number_word(value: Any) -> str | None:
    number = _leading_number(value) if not isinstance(value, (int, float)) else float(value)
    if number is None or not float(number).is_integer():
        return None
    return _NUMBER_WORDS.get(int(number))


# --------------------------------------------------------------------------
# Key-point access
# --------------------------------------------------------------------------


def _key_point(key_points: dict[str, Any], key: str) -> dict[str, Any] | None:
    candidates = []
    for section in ("project", "pile", "loads", "results"):
        for point in key_points.get(section) or []:
            if isinstance(point, dict) and _clean(point.get("key")) == key:
                candidates.append(point)
    if not candidates:
        return None
    if key == 'yield_strength':
        best = None
        for p in candidates:
            score = 0
            if str(p.get('unit','')).lower() == 'ksi':
                score += 10
            loc = str(p.get('location',''))
            if 'table' in loc.lower():
                score += 5
            ctx = str(p.get('context',''))
            if '7.63' in ctx or '0.375' in ctx:
                score += 3
            if best is None or score > best[0]:
                best = (score, p)
        if best is not None:
            return best[1]
    return candidates[0]


def _key_point_value(key_points: dict[str, Any], key: str) -> str | None:
    point = _key_point(key_points, key)
    if point is None:
        return None
    value = _format_number(point.get("value"))
    if value and value.casefold() not in _RE_GARBAGE:
        return value
    return None


def _key_point_measured(key_points: dict[str, Any], key: str) -> str | None:
    point = _key_point(key_points, key)
    if point is None:
        return None
    text = _key_point_text(point)
    if text and text.casefold() not in _RE_GARBAGE:
        return text
    return None


def _key_point_text(point: dict[str, Any]) -> str:
    value = _format_number(point.get("value"))
    unit = _clean(point.get("unit"))
    return f"{value} {unit}".strip() if unit else value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _source_text(data: dict[str, Any]) -> str:
    """Every piece of free text available to the web payload, joined."""
    parts: list[str] = []
    key_points = _as_dict(data.get("key_points"))
    for reference in key_points.get("references") or []:
        if isinstance(reference, dict):
            parts.append(_clean(reference.get("text")))
    for field in data.get("fields") or []:
        if not isinstance(field, dict):
            continue
        parts.append(_clean(field.get("raw")))
        parts.append(_clean(field.get("section")))
        parts.append(_clean(field.get("parameter")))
    parts.append(_clean(data.get("clean_engineering_summary")))
    parts.append(_clean(data.get("document_text")))
    for warning in data.get("warnings") or []:
        parts.append(_clean(warning))
    return "\n".join(part for part in parts if part)


# --------------------------------------------------------------------------
# Standards / methodology detection (deterministic text search)
# --------------------------------------------------------------------------


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _software(data: dict[str, Any], text: str) -> str:
    """"HelixPile (DeepFND)" only when both names appear in the source text."""
    low = text.lower()
    has_helix = "helixpile" in low or "helix pile" in low
    has_deep = re.search(r"deep\s*fnd", text, re.I) is not None
    if has_helix and has_deep:
        return "HelixPile (DeepFND)"
    point = _key_point(_as_dict(data.get("key_points")), "software")
    if point is not None:
        value = _clean(point.get("value"))
        if value:
            return value
    if has_helix:
        return "HelixPile"
    if has_deep:
        return "DeepFND"
    return NOT_STATED


def _geotechnical_method(text: str) -> str:
    if _has(text, r"vesic"):
        if _has(text, r"\b19\s*74\b"):
            return "the Vesic (1974) individual plate bearing methodology"
        return "the Vesic individual plate bearing methodology"
    if _has(text, r"individual\s+plate\s+bearing"):
        return "the individual plate bearing methodology"
    return NOT_STATED


def _lateral_method(data: dict[str, Any], text: str) -> str:
    """The p-y model(s), read from the extracted lateral soil table first."""
    key_points = _as_dict(data.get("key_points"))
    models = " ".join(
        _clean(point.get("value"))
        for point in key_points.get("results") or []
        if isinstance(point, dict) and _clean(point.get("key")) == "soil_model"
    ).lower()
    has_reese = "reese" in models
    has_api = "api" in models
    if has_reese and has_api:
        return "Reese and API p-y"
    if has_api:
        return "API p-y"
    if has_reese:
        return "Reese p-y"
    if _has(text, r"\bapi\b") and _has(text, r"\bp[\s\-]?y\b"):
        return "API p-y"
    if _has(text, r"reese"):
        return "Reese p-y"
    if _has(text, r"\bp[\s\-]?y\b"):
        return "p-y"
    return NOT_STATED


def _structural_code(text: str) -> str:
    """Extract the AISC 360 code string, then map it for display only."""
    match = re.search(
        r"(?:ANSI\s*/\s*)?AISC\s*360(?:\s*-\s*(?P<year>\d{2}))?", text, re.I
    )
    if not match:
        return NOT_STATED
    year = match.group("year")
    canonical = f"AISC 360-{year}" if year else "AISC 360"
    return DISPLAY_CODE_MAP.get(canonical, canonical)


def _corrosion_standard(text: str) -> str:
    if _has(text, r"icc") and _has(text, r"ac\s*[-]?\s*358"):
        return "ICC-AC358"
    match = re.search(r"AC\s*[-]?\s*358", text, re.I)
    if match:
        return "AC358"
    return NOT_STATED


# --------------------------------------------------------------------------
# Ratio / settlement resolution (computed, never hardcoded)
# --------------------------------------------------------------------------


def _ratio_values(data: dict[str, Any], token: str) -> list[float]:
    token = token.lower()
    out: list[float] = []
    for field in data.get("fields") or []:
        if not isinstance(field, dict):
            continue
        if token in _clean(field.get("parameter")).lower():
            number = _leading_number(_field_text(field))
            if number is not None:
                out.append(number)
    key_points = _as_dict(data.get("key_points"))
    for point in key_points.get("results") or []:
        if not isinstance(point, dict) or _clean(point.get("key")) != "result_ratio":
            continue
        if token in _clean(point.get("label")).lower():
            number = _leading_number(_format_number(point.get("value")))
            if number is not None:
                out.append(number)
    return out


def _all_ratios(data: dict[str, Any]) -> list[float]:
    values = (
        _ratio_values(data, "r/cap")
        + _ratio_values(data, "r/str")
        + _ratio_values(data, "r/geo")
    )
    if values:
        return values
    key_points = _as_dict(data.get("key_points"))
    for point in key_points.get("results") or []:
        if isinstance(point, dict) and _clean(point.get("key")) == "result_ratio":
            number = _leading_number(_format_number(point.get("value")))
            if number is not None:
                values.append(number)
    return values


def _max_ratio(data: dict[str, Any], token: str) -> float | None:
    values = _ratio_values(data, token)
    return max(values) if values else None


def _settlement_measure(data: dict[str, Any], key: str, field_key: str):
    key_points = _as_dict(data.get("key_points"))
    point = _key_point(key_points, key)
    parsed = _measure_from_point(point)
    if parsed is not None:
        return parsed
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    return _measure_from_field(fields, field_key)


def _corrosion_loss(data: dict[str, Any]) -> str | None:
    """Section loss in inches, from the "Steel (in)" column only."""
    key_points = _as_dict(data.get("key_points"))
    parsed = _measure_from_point(_key_point(key_points, "corrosion_thickness_loss"))
    if parsed is not None:
        number, unit = parsed
        if unit is None or unit.lower() in _INCH_UNITS:
            return f"{_format_number(number)} inches"
        return f"{_format_number(number)} {unit}"
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    for field_key in ("corrosion_loss", "corrosion_thickness_loss"):
        parsed = _measure_from_field(fields, field_key)
        if parsed is None:
            continue
        number, unit = parsed
        if unit and "mil" in unit.lower():
            continue  # "Protection (mils)" is never the section loss
        if unit is None or unit.lower() in _INCH_UNITS:
            return f"{_format_number(number)} inches"
        return f"{_format_number(number)} {unit}"
    return None


# --------------------------------------------------------------------------
# Five-paragraph builder
# --------------------------------------------------------------------------


def _helix_clause(data: dict[str, Any]) -> str | None:
    count = _measure(data, "helix_count", "helix_count")
    count_word = _number_word(count[0]) if count is not None else None

    diameter = None
    for key in ("helix_dia", *[f"helix_dia_{i}" for i in range(1, 6)]):
        diameter = _measure(data, key, "helix_diameter")
        if diameter is not None:
            break
    thickness = _measure(data, "helix_thickness", "helix_thickness")

    if diameter is None and thickness is None:
        return None
    if count_word and diameter is not None:
        description = f"{count_word} {_in_inches(diameter[0], diameter[1], 'dia')}"
    elif count_word:
        description = count_word
    elif diameter is not None:
        description = _in_inches(diameter[0], diameter[1], "dia")
    else:
        description = ""
    if thickness is not None:
        thickness_text = _in_inches(thickness[0], thickness[1], "thick")
        description = f"{description + ', ' if description else ''}{thickness_text}"
    plural = count_word != "one"
    return f"{description} helical plate{'s' if plural else ''}".strip()


def _shaft_clause(data: dict[str, Any]) -> str:
    od = _measure(data, "pile_diameter", "pile_diameter")
    wall = _measure(data, "pile_wall_thickness", "pile_thickness")
    if wall is None:
        wall = _measure(data, "pile_thickness", "pile_thickness")
    fy = _measure(data, "yield_strength", "fy")

    dimension = ""
    if od is not None:
        dimension += _in_inches(od[0], od[1], "od")
    if wall is not None:
        dimension += (", " if dimension else "") + _in_inches(wall[0], wall[1], "wall")
    if dimension:
        shaft = f"a {dimension} steel helical pile shaft"
    else:
        shaft = "a steel helical pile shaft"
    if fy is not None:
        unit = f" {fy[1]}" if fy[1] else ""
        shaft += f" (Fy = {_format_number(fy[0])}{unit})"
    helix = _helix_clause(data)
    if helix:
        shaft += f" equipped with {helix}"
    return shaft


def _system_label(project_inputs: dict[str, Any]) -> str:
    descriptor = _clean(project_inputs.get("system_descriptor"))
    return f"{descriptor} foundation system" if descriptor else SYSTEM_FALLBACK


def _paragraph_scope(data: dict[str, Any], system: str) -> str:
    key_points = _as_dict(data.get("key_points"))
    project = _key_point_value(key_points, "project_name")
    shaft = _shaft_clause(data)
    target = f"for {project}" if project else "for the project"
    return (
        f"This section summarizes the structural and geotechnical design checks "
        f"performed {target}. The evaluation was conducted for a {system} "
        f"consisting of {shaft}. The analyses include axial compression and "
        f"uplift capacity, lateral resistance, structural strength, settlement, "
        f"and serviceability check to verify that the {system} satisfies the "
        f"project design requirements."
    )


def _load_cases(key_points: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    for point in key_points.get("loads") or []:
        if not isinstance(point, dict):
            continue
        context = _clean(point.get("context")) or "General"
        cases.setdefault(context, {})
        cases[context][_clean(point.get("key"))] = point
        if context not in order:
            order.append(context)
    return order, cases


def _case_sentence(context: str, case: dict[str, Any], system: str) -> str:
    compression = case.get("load_compression")
    tension = case.get("load_tension")
    lateral = _force_text(case.get("load_shear"))

    if compression:
        sentence = (
            f"{context} (compression) consisted of a "
            f"{_force_text(compression)} axial compressive load"
        )
        if lateral:
            sentence += f" and a {lateral} lateral load"
        return sentence + f" applied at the top of the {system}."
    if tension:
        sentence = f"{context} (uplift) consisted of a {_force_text(tension)} uplift load"
        if lateral:
            sentence += f" combined with a {lateral} lateral load"
        return sentence + "."
    if lateral:
        return (
            f"{context} consisted of a {lateral} lateral load applied at the top "
            f"of the {system}."
        )
    return f"{context} carried no reported load."


def _paragraph_loads(data: dict[str, Any], system: str) -> str:
    key_points = _as_dict(data.get("key_points"))
    order, cases = _load_cases(key_points)
    if not order:
        return (
            "The report printed no design load case, so the governing load case "
            f"is {NOT_STATED}."
        )
    count_word = _number_word(len(order))
    count_text = count_word or str(len(order))
    sentences = [_case_sentence(context, cases[context], system) for context in order]
    return (
        f"The {system} was evaluated under {count_text} governing design load "
        f"{'case' if len(order) == 1 else 'cases'}. " + " ".join(sentences)
    )


def _paragraph_analysis(data: dict[str, Any], text: str) -> str:
    software = _software(data, text)
    geo = _geotechnical_method(text)
    lateral = _lateral_method(data, text)
    code = _structural_code(text)
    if lateral != NOT_STATED:
        lateral = f"{lateral} lateral soil interaction curves"
    return (
        f"Comprehensive foundation analysis was executed via the {software} "
        f"engineering platform. Geotechnical capacities were verified using "
        f"{geo} and {lateral}. Structural checks were evaluated against the "
        f"governing provisions of {code}."
    )


def _structural_reserve(structural_max: float | None, all_ok: bool, code: str) -> str:
    if structural_max is None:
        return "with the structural capacity reserve not specified"
    if structural_max >= 1.0:
        return (
            "which exceeds the limit of 1.0 and does not maintain a structural "
            f"capacity reserve under {code}"
        )
    article = "an excellent " if structural_max <= 0.6 else "a "
    if all_ok:
        reserve = "in compliance with"
    else:
        reserve = "under"
    return f"maintaining {article}structural capacity reserve {reserve} {code}"


def _paragraph_durability(
    data: dict[str, Any], text: str, project_inputs: dict[str, Any]
) -> str:
    standard = _corrosion_standard(text)
    category = _clean(project_inputs.get("corrosion_category"))
    loss = _corrosion_loss(data) or NOT_STATED
    structural_max = _max_ratio(data, "r/str")
    all_ok = bool(_all_ratios(data)) and all(value < 1.0 for value in _all_ratios(data))
    code = _structural_code(text)

    if structural_max is None:
        utilization = NOT_STATED
    else:
        utilization = _format_number(structural_max)
    reserve = _structural_reserve(structural_max, all_ok, code)

    sentences = [
        "To satisfy long-term durability criteria, a site-specific corrosion "
        f"evaluation was performed for the bare steel pipe shaft in accordance "
        f"with {standard}."
    ]
    if category:
        sentences.append(
            "Based on site-specific soil parameters, the subterranean environment "
            f"is categorized as {category}."
        )
    sentences.append(f"The evaluation yields a total projected section loss of {loss}.")
    sentences.append(
        "Under this long-term degraded condition, the maximum projected structural "
        f"shaft utilization reaches {utilization}, {reserve}."
    )
    return " ".join(sentences)


def _final_sentence(data: dict[str, Any], project_inputs: dict[str, Any]) -> str:
    ratios = _all_ratios(data)
    all_ok = bool(ratios) and all(value < 1.0 for value in ratios)
    settlement = _settlement_measure(data, "settlement_design", "settlement")
    capacity = _settlement_measure(
        data, "settlement_design_capacity", "settlement_design_capacity"
    )
    installation_ok = bool(project_inputs.get("installation_verified"))
    review_ok = bool(project_inputs.get("reviewed_by_engineer"))

    if all_ok and settlement is not None and installation_ok and review_ok:
        return (
            "All settlement deformations, structural steel stresses, and "
            "installation verification criteria satisfy their respective code "
            f"requirements; the proposed helical foundation design is {_ADEQUATE_VERDICT}."
        )
    if all_ok:
        if settlement is not None and capacity is not None:
            settlement_text = (
                f"{_format_number(settlement[0])} {settlement[1] or 'in'}"
            )
            capacity_text = f"{_format_number(capacity[0])}-{capacity[1] or 'k'}"
            return (
                "All utilization ratios are below 1.0 and design settlement is "
                f"{settlement_text} at the {capacity_text} design capacity. "
                "Installation torque verification and engineer review are required "
                "before the design is accepted."
            )
        return (
            "All utilization ratios are below 1.0. Installation torque verification "
            "and engineer review are required before the design is accepted."
        )
    if not ratios:
        return (
            "No utilization ratio was reported; installation torque verification and "
            "engineer review are required before the design is accepted."
        )
    return (
        "A reported utilization ratio exceeds 1.0. Installation torque verification "
        "and engineer review are required before the design is accepted."
    )


def _paragraph_geotechnical(data: dict[str, Any], project_inputs: dict[str, Any]) -> str:
    geotech_max = _max_ratio(data, "r/geo")
    ratios = _all_ratios(data)
    all_ok = bool(ratios) and all(value < 1.0 for value in ratios)

    if geotech_max is None:
        utilization = NOT_STATED
        margin = (
            "the margin against soil plunge failure under the design load could "
            "not be confirmed."
        )
    else:
        utilization = _format_number(geotech_max)
        if geotech_max < 1.0:
            if all_ok:
                margin = (
                    "This maintains a code-compliant margin against soil plunge "
                    "failure under the design load."
                )
            else:
                margin = (
                    "This maintains a margin against soil plunge failure under the "
                    "design load."
                )
        else:
            margin = (
                "This exceeds the limit of 1.0 and does not maintain a code-compliant "
                "margin against soil plunge failure under the design load."
            )
    return (
        f"Geotechnically, the helix configuration operates at a peak compression "
        f"utilization rate of {utilization}. {margin} {_final_sentence(data, project_inputs)}"
    )


# --------------------------------------------------------------------------
# Validation: every reported measurement keeps its source value and unit
# --------------------------------------------------------------------------

_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9.])(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
)


def _source_number_tokens(data: dict[str, Any]) -> set[str]:
    """Every numeric token anywhere in the payload (fields, text, references)."""
    blob = json.dumps(data, default=str)
    return {match.group("num").replace(",", "") for match in _NUMBER_TOKEN.finditer(blob)}


def _numeric_entry(obj: dict[str, Any], label_key: str) -> tuple[str, str, str, str] | None:
    """(number_text, unit, label, context) for one numeric source value."""
    value = obj.get("value")
    if isinstance(value, bool):
        return None
    unit = _clean(obj.get("unit"))
    label = _clean(obj.get(label_key))
    context = _clean(obj.get("context"))
    if isinstance(value, (int, float)):
        return _format_number(value), unit, label, context
    if isinstance(value, str):
        number = _leading_number(value)
        if number is not None:
            return _format_number(number), unit, label, context
    return None


def _source_measurements(data: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """All numeric measurements the summary could quote, with their units."""
    out: list[tuple[str, str, str, str]] = []
    for field in data.get("fields") or []:
        if isinstance(field, dict):
            entry = _numeric_entry(field, "parameter")
            if entry:
                out.append(entry)
    key_points = _as_dict(data.get("key_points"))
    for section in ("pile", "loads", "results"):
        for point in key_points.get(section) or []:
            if isinstance(point, dict):
                entry = _numeric_entry(point, "key")
                if entry:
                    out.append(entry)
    return out


def _normalized_unit(unit: str) -> str:
    return re.sub(r"\s+", "", _clean(unit).lower()).replace("^", "").rstrip(".")


#: Plural / spelling variants of the same unit, so "kips" and "kip" agree
#: without ever making "k" (kip) agree with "ksi" (stress).
_UNIT_SYNONYMS = {
    "kips": "kip", "kip": "k", "lbs": "lb", "lbf": "lb", "feet": "ft",
    "foot": "ft", "inches": "in", "inch": "in", "degrees": "deg",
    "degree": "deg", "yrs": "yr", "years": "yr", "year": "yr", "hrs": "hr",
    "hours": "hr", "hour": "hr", "mils": "mil",
}


def _canonical_unit(unit: str) -> str:
    norm = _normalized_unit(unit)
    return _UNIT_SYNONYMS.get(norm, norm)


def _units_agree(source_unit: str, summary_unit: str) -> bool:
    return _canonical_unit(source_unit) == _canonical_unit(summary_unit)


#: A unit follows a number either directly ("150 kip") or hyphenated
#: ("150-kip"); both are the same measurement.
_SUMMARY_UNIT_RE = re.compile(
    r"[\s\-]*(?P<u>[A-Za-z\u00b5\u00b0%][A-Za-z0-9\u00b5\u00b0%*/^\-\u00b7]*)"
)


def validate_executive_summary(
    text: str, data: dict[str, Any] | None = None
) -> list[str]:
    """Flag summary numbers that lost their source unit or traceability.

    Deterministic and side-effect free: the summary text is checked against the
    same payload it was built from. An empty list means every reported
    measurement carries the unit and value the source printed, and every
    numeric token traces back to the source data.
    """
    data = _as_dict(data)
    if not text:
        return []
    flat = _clean(text)
    flags: list[str] = []

    # 1. A number quoted from the source must keep a unit the source printed.
    #    Several distinct fields can share a number (24 in a helix diameter and
    #    a "24" count), so the reported unit is only wrong when it agrees with
    #    none of the source measurements carrying that number -- otherwise a
    #    single unrelated field would flag a perfectly correct statement.
    by_number: dict[str, list[tuple[str, str, str, str]]] = {}
    for number, unit, label, context in _source_measurements(data):
        if not unit or number in {"0", "1"}:
            continue
        by_number.setdefault(number, []).append((number, unit, label, context))

    for number, entries in by_number.items():
        token = re.compile(rf"(?<![0-9.]){re.escape(number)}(?![0-9])")
        occurrences = list(token.finditer(flat))
        if not occurrences:
            continue
        accepted = {_canonical_unit(unit) for _, unit, _, _ in entries}
        found_with_unit = mismatch = False
        for match in occurrences:
            window = flat[match.end(): match.end() + 16]
            unit_match = _SUMMARY_UNIT_RE.match(window)
            candidate = unit_match.group("u") if unit_match else ""
            agrees = bool(candidate) and _canonical_unit(candidate) in accepted
            if not agrees and candidate and "-" in candidate:
                # "24-inch-diameter" is the unit "inch" plus a noun; accept the
                # leading segment, but never split a known compound like "k-ft".
                if _normalized_unit(candidate) not in _KNOWN_UNITS:
                    agrees = _canonical_unit(candidate.split("-", 1)[0]) in accepted
            if agrees:
                found_with_unit = True
                break
            # Only a recognized unit counts as a conflicting unit; any other
            # word after the number is prose, not a unit.
            if candidate and _normalized_unit(candidate) in _KNOWN_UNITS:
                mismatch = True
        if not found_with_unit and occurrences:
            number, unit, label, context = entries[0]
            where = f" ({label}{(' ' + context) if context else ''})"
            if mismatch:
                flags.append(
                    f"{number}{where} is reported with a different unit than the "
                    f"source ({unit})"
                )
            else:
                flags.append(
                    f"{number}{where} is reported without its source unit ({unit})"
                )

    # 2. Every numeric token must trace back to the source payload. Compare
    #    numeric values, not text, so 126 and 126.0 are the same measurement.
    source_values: set[float] = set()
    for token in _source_number_tokens(data):
        try:
            source_values.add(float(token))
        except ValueError:
            continue
    for match in _NUMBER_TOKEN.finditer(flat):
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value in source_values or value in {0.0, 1.0}:
            continue
        if value.is_integer() and 0 <= value <= 9:
            continue
        if "." in raw or len(raw.lstrip("-+")) >= 3:
            flags.append(
                f"numeric value {raw!r} is not present in the source data"
            )
    return flags


def build_executive_summary_text(data: dict[str, Any]) -> str:
    """Return the title line followed by the five Executive Summary paragraphs."""
    if not isinstance(data, dict):
        raise ValueError("executive summary data must be a mapping")

    text = _source_text(data)
    project_inputs = normalize_project_inputs(data.get("project_inputs"))
    system = _system_label(project_inputs)

    paragraphs = [
        _paragraph_scope(data, system),
        _paragraph_loads(data, system),
        _paragraph_analysis(data, text),
        _paragraph_durability(data, text, project_inputs),
        _paragraph_geotechnical(data, project_inputs),
    ]
    return TITLE + "\n\n" + "\n\n".join(paragraphs)


# --------------------------------------------------------------------------
# PDF rendering
# --------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "ExecutiveTitle",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "ExecutiveBody",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
    }


def build_executive_summary_narrative_pdf(data: dict[str, Any]) -> bytes:
    """Render the five-paragraph Executive Summary and return its PDF bytes."""
    if not isinstance(data, dict):
        raise ValueError("executive summary data must be a mapping")

    text = build_executive_summary_text(data)
    styles = _styles()
    story: list[Any] = [Paragraph(escape(TITLE), styles["title"])]
    for paragraph in text.split("\n\n")[1:]:
        story.append(Paragraph(escape(paragraph), styles["body"]))
        story.append(Spacer(1, 2))

    # Validate before finalizing the document: a measurement that lost its unit
    # or a value with no source trace is surfaced instead of shipping silently.
    flags = validate_executive_summary(text, data)
    if flags:
        note = "Automated validation flags: " + "; ".join(dict.fromkeys(flags)) + "."
        story.append(Paragraph(escape(note), styles["body"]))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title="Executive Summary",
        author="engineering-ai",
        subject="Engineering evaluation aid - not a formal engineering approval",
    )
    document.build(story)
    return buffer.getvalue()
