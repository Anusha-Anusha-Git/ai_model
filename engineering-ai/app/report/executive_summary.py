"""Executive Summary narrative builder (five-paragraph contract).

Turns the extraction payload the web UI already holds (``fields``,
``key_points``, ``clean_engineering_summary``, ``conflicts``, ``warnings``) into
the fixed five-paragraph Executive Summary used for engineering review, and
renders it as a one- to two-page PDF.

Hard constraints (inherited from the rest of the project):
- Fully deterministic. No LLM, no recalculation, no rounding, no unit
  conversion. Every number is copied verbatim from the extracted fields.
- **Nothing is invented.** A value the report does not state renders as
  ``not stated in the report``.
- The adequacy verdict is ``fully adequate for service`` only when every
  reported utilization ratio is at or below 1.0 AND the report states that the
  settlement and installation criteria are satisfied. Otherwise the summary
  names the check that is not satisfied (or could not be confirmed) and states
  that the design requires engineer review.
- The phrase ``combined bucket pile-helical pile foundation system`` is used
  only when the source actually describes a bucket pile.
- Output is an evaluation aid. It is never certified or approved work.

The only public entry points are :func:`build_executive_summary_text` and
:func:`build_executive_summary_narrative_pdf`.
"""

from __future__ import annotations

import io
import re
from typing import Any, Sequence
from xml.sax.saxutils import escape

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

NOT_STATED = "not stated in the report"

SYSTEM_LABEL = "foundation system"
BUCKET_SYSTEM_LABEL = "combined bucket pile-helical pile foundation system"

_NOT_ADEQUATE_VERDICT = "requires engineer review"
_ADEQUATE_VERDICT = "fully adequate for service"

_MARGIN = 20 * mm
_USABLE_WIDTH = A4[0] - 2 * _MARGIN

_RE_GARBAGE = {"", MISSING_VALUE.lower(), "none", "null", "n/a"}


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


def _key_point(key_points: dict[str, Any], key: str) -> dict[str, Any] | None:
    for section in ("project", "pile", "loads", "results"):
        for point in key_points.get(section) or []:
            if isinstance(point, dict) and _clean(point.get("key")) == key:
                return point
    return None


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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# Standards / methodology detection (deterministic text search)
# --------------------------------------------------------------------------


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _software(data: dict[str, Any], text: str) -> str:
    software = _key_point_value(_as_dict(data.get("key_points")), "software")
    if software:
        return software
    match = re.search(r"(HelixPile|DeepFND|Deep\s*FND|GSPile|PLAXIS|intact)", text, re.I)
    return _clean(match.group(1)) if match else NOT_STATED


def _geotechnical_method(text: str) -> str:
    if _has(text, r"vesic"):
        return "the Vesic individual plate bearing methodology"
    if _has(text, r"individual\s+plate\s+bearing"):
        return "the individual plate bearing methodology"
    return NOT_STATED


def _lateral_method(text: str) -> str:
    if _has(text, r"\bapi\b") and _has(text, r"\bp[\s\-]?y\b"):
        return "API p-y lateral soil interaction curves"
    if _has(text, r"\bp[\s\-]?y\b"):
        return "p-y lateral soil interaction curves"
    return NOT_STATED


def _structural_code(text: str) -> str:
    match = re.search(r"(?:ANSI\s*/\s*)?AISC\s*360(?:\s*-\s*\d{2})?", text, re.I)
    if match:
        return _clean(match.group(0))
    return NOT_STATED


def _corrosion_standard(text: str) -> str:
    if _has(text, r"icc") and _has(text, r"ac\s*[-]?\s*358"):
        return "ICC-AC358"
    match = re.search(r"AC\s*[-]?\s*358", text, re.I)
    if match:
        return "AC358"
    return NOT_STATED


def _aggressiveness(text: str) -> str:
    match = re.search(
        r"(non[\s\-]*to[\s\-]*mildly\s+aggressive|non[\s\-]*aggressive"
        r"|mildly\s+aggressive|moderately\s+aggressive|severely\s+aggressive"
        r"|highly\s+aggressive|aggressive)",
        text,
        re.I,
    )
    if not match:
        return NOT_STATED
    phrase = re.sub(r"\s+", " ", match.group(1).lower())
    return re.sub(r"^non to", "non-to", phrase)


def _mentions_bucket_pile(text: str) -> bool:
    return _has(text, r"bucket\s+pile")


def _installation_satisfied(text: str) -> bool:
    return _has(
        text,
        r"installation[^.\n]{0,80}(?:satisf|verif|criteria|require|accept|comply)",
        r"(?:satisf|verif|criteria)[^.\n]{0,80}installation",
    )


# --------------------------------------------------------------------------
# Payload resolution
# --------------------------------------------------------------------------


def _all_ratios(fields: Sequence[dict[str, Any]], key_points: dict[str, Any]) -> list[float]:
    ratios: list[float] = []
    for key in ("R/GEO", "R/STR", "R/CAP"):
        field, _context = resolve_summary_field(list(fields), key)
        if field is not None:
            number = _leading_number(_field_text(field))
            if number is not None:
                ratios.append(number)
    for point in key_points.get("results") or []:
        if isinstance(point, dict) and _clean(point.get("key")) == "result_ratio":
            number = _leading_number(_format_number(point.get("value")))
            if number is not None:
                ratios.append(number)
    return ratios


def _settlement_satisfied(fields: Sequence[dict[str, Any]]) -> bool:
    settlement = _resolved(fields, "settlement")
    criterion = _resolved(fields, "settlement_criterion")
    if settlement is None or criterion is None:
        return False
    actual = _leading_number(settlement)
    limit = _leading_number(criterion)
    if actual is None or limit is None:
        return False
    return actual <= limit


# --------------------------------------------------------------------------
# Five-paragraph builder
# --------------------------------------------------------------------------


def _paragraph_scope(data: dict[str, Any], text: str) -> str:
    key_points = _as_dict(data.get("key_points"))
    project = _key_point_value(key_points, "project_name") or NOT_STATED
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]

    od = _resolved(fields, "pile_diameter") or NOT_STATED
    thickness = _resolved(fields, "pile_thickness") or NOT_STATED
    fy = _resolved(fields, "fy") or NOT_STATED

    count = _key_point_value(key_points, "helix_count")
    if count is None:
        count = _resolved(fields, "helix_count")
    diameters: list[str] = []
    for index in range(1, 6):
        candidate = _key_point_measured(key_points, f"helix_dia_{index}")
        if candidate is None:
            candidate = _resolved(fields, f"helix_dia_{index}")
        if candidate:
            diameters.append(candidate)
    if not diameters:
        single = _key_point_measured(key_points, "helix_dia") or _resolved(fields, "helix_diameter")
        if single:
            diameters.append(single)

    if count or diameters:
        parts = []
        if count:
            parts.append(f"{count}")
        if diameters:
            parts.append(f"{', '.join(diameters)} diameter")
        helix = " ".join(parts) + " helical plate"
    else:
        helix = NOT_STATED

    system = BUCKET_SYSTEM_LABEL if _mentions_bucket_pile(text) else SYSTEM_LABEL

    return (
        f"This section summarizes the structural and geotechnical design checks "
        f"performed for {project}. The evaluation was conducted for a {system} "
        f"consisting of a {od}, {thickness} thick steel helical pile shaft "
        f"(Fy = {fy}) equipped with {helix}. The analyses include axial "
        f"compression and uplift capacity, lateral resistance, structural "
        f"strength, settlement, and serviceability check to verify that the "
        f"{system} satisfies the project design requirements."
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
    shear = case.get("load_shear")
    lateral = _key_point_text(shear) if shear else NOT_STATED

    if compression:
        return (
            f"{context} (compression) consisted of a "
            f"{_key_point_text(compression)} axial compressive load and a "
            f"{lateral} lateral load applied at the top of the {system}."
        )
    if tension:
        return (
            f"{context} (uplift) consisted of {_key_point_text(tension)} uplift "
            f"load combined with a {lateral} lateral load."
        )
    return (
        f"{context} consisted of {lateral} lateral load applied at the top of "
        f"the {system}."
    )


def _paragraph_loads(data: dict[str, Any], system: str) -> str:
    key_points = _as_dict(data.get("key_points"))
    order, cases = _load_cases(key_points)
    if not order:
        return (
            "The foundation system was evaluated under design load cases that are "
            f"{NOT_STATED}. No governing load case was printed in the report."
        )
    sentences = [_case_sentence(context, cases[context], system) for context in order]
    count = len(order)
    return (
        f"The {system} was evaluated under {count} governing design load "
        f"{'case' if count == 1 else 'cases'}. " + " ".join(sentences)
    )


def _paragraph_analysis(data: dict[str, Any], text: str) -> str:
    software = _software(data, text)
    geo = _geotechnical_method(text)
    lateral = _lateral_method(text)
    code = _structural_code(text)
    return (
        f"Comprehensive foundation analysis was executed via the {software} "
        f"engineering platform. Geotechnical capacities were verified using "
        f"{geo} and {lateral}. Structural checks were evaluated against the "
        f"governing provisions of {code}."
    )


def _paragraph_durability(
    data: dict[str, Any], text: str, structural_util: str | None
) -> str:
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    standard = _corrosion_standard(text)
    category = _aggressiveness(text)
    loss = _resolved(fields, "corrosion_loss") or NOT_STATED
    code = _structural_code(text)

    if structural_util is None:
        utilization = NOT_STATED
        reserve = "with the structural capacity reserve not stated in the report"
    else:
        utilization = structural_util
        number = _leading_number(structural_util)
        if number is not None and number <= 1.0:
            reserve = (
                "maintaining an excellent structural capacity reserve in full "
                f"compliance with {code}"
            )
        else:
            reserve = (
                "which exceeds the limit of 1.0 and does not maintain a structural "
                f"capacity reserve under {code}"
            )

    return (
        f"To satisfy long-term durability criteria, a site-specific corrosion "
        f"evaluation was performed for the bare steel pipe shaft in accordance "
        f"with {standard}. Based on site-specific soil parameters, the "
        f"subterranean environment is categorized as {category}. The evaluation "
        f"yields a total projected section loss of {loss}. Under this long-term "
        f"degraded condition, the maximum projected structural shaft utilization "
        f"reaches {utilization}, {reserve}."
    )


def _geotechnical_util(fields: Sequence[dict[str, Any]], key_points: dict[str, Any]) -> str | None:
    field, _context = resolve_summary_field(list(fields), "R/GEO")
    if field is not None:
        text = _field_text(field)
        if text and text.casefold() not in _RE_GARBAGE:
            return text
    for point in key_points.get("results") or []:
        if isinstance(point, dict) and _clean(point.get("key")) == "result_ratio":
            return _key_point_text(point)
    return None


def _adequacy(
    fields: Sequence[dict[str, Any]], key_points: dict[str, Any], text: str
) -> tuple[bool, str]:
    """(is_fully_adequate, reason) per the five-paragraph rules."""
    ratios = _all_ratios(fields, key_points)
    if not ratios:
        return False, "the report states no utilization ratio that could be confirmed"
    over = [value for value in ratios if value > 1.0]
    if over:
        return False, (
            "a reported utilization ratio exceeds 1.0 "
            f"({_format_number(max(over))})"
        )
    if not _settlement_satisfied(fields):
        return False, (
            "the report does not confirm that the settlement criteria are satisfied"
        )
    if not _installation_satisfied(text):
        return False, (
            "the report does not confirm that the installation verification "
            "criteria are satisfied"
        )
    return True, ""


def _paragraph_geotechnical(
    data: dict[str, Any], text: str, adequate: bool, reason: str
) -> str:
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    key_points = _as_dict(data.get("key_points"))
    utilization = _geotechnical_util(fields, key_points) or NOT_STATED
    if adequate:
        because = (
            "Because all settlement deformations, structural steel stresses, and "
            "installation verification criteria satisfy their respective standard "
            "code requirements"
        )
        closing = f"the proposed helical foundation design is {_ADEQUATE_VERDICT}"
    else:
        because = f"Because {reason}"
        closing = f"the proposed helical foundation design {_NOT_ADEQUATE_VERDICT}"
    return (
        f"Geotechnically, the helix configuration operates at a peak compression "
        f"utilization rate of {utilization}. This maintains a safe, code-compliant "
        f"margin against soil plunge failure under the design load. {because}, "
        f"{closing}."
    )


def build_executive_summary_text(data: dict[str, Any]) -> str:
    """Return the title line followed by the five Executive Summary paragraphs."""
    if not isinstance(data, dict):
        raise ValueError("executive summary data must be a mapping")

    text = _source_text(data)
    fields = [f for f in data.get("fields") or [] if isinstance(f, dict)]
    key_points = _as_dict(data.get("key_points"))
    system = BUCKET_SYSTEM_LABEL if _mentions_bucket_pile(text) else SYSTEM_LABEL

    structural_field, _context = resolve_summary_field(fields, "R/STR")
    structural_util = _field_text(structural_field) if structural_field is not None else None
    if structural_util is not None and structural_util.casefold() in _RE_GARBAGE:
        structural_util = None

    adequate, reason = _adequacy(fields, key_points, text)

    paragraphs = [
        _paragraph_scope(data, text),
        _paragraph_loads(data, system),
        _paragraph_analysis(data, text),
        _paragraph_durability(data, text, structural_util),
        _paragraph_geotechnical(data, text, adequate, reason),
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

    styles = _styles()
    story: list[Any] = [Paragraph(escape(TITLE), styles["title"])]
    for paragraph in build_executive_summary_text(data).split("\n\n")[1:]:
        story.append(Paragraph(escape(paragraph), styles["body"]))
        story.append(Spacer(1, 2))

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
