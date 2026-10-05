"""Stage 1: fixed-layout PDF report generator.

Takes the JSON produced by ``app.extraction.field_extractor`` (the shape of
``ExtractionResult.to_dict()``) and renders a PDF with a FIXED layout:

    1. Project Information
    2. Executive Summary
    3. Load Summary
    4. Engineering Calculations
    5. Results
    6. Engineering Evaluation
    7. References

The section list is a module-level constant and never changes with the input
document. Only the *content* of each section changes.

Design rules (Hard Constraints, inherited from the rest of the project):
- No LLM is involved. Every number printed is interpolated from the supplied
  JSON, so the model can never write a value it did not read.
- Nothing is invented. A row with no matching field prints NOT_FOUND.
- A parameter extracted twice with disagreeing values is printed as a flagged
  conflict with both sources -- never silently resolved, never averaged.
- Every printed value carries its unit and its source location (page when the
  source location names a page).
- Output is a review aid. It is never certified or approved engineering work.

Usage:
    python -m app.report.pdf_report extracted_fields.json out.pdf
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------
# Fixed contract
# --------------------------------------------------------------------------

#: Exact text printed for any value absent from the document. Never blank,
#: never a zero, never a default -- absence is reported as absence.
NOT_FOUND = "Not found in document"

#: Marker shown next to a value whose sources disagree.
CONFLICT_FLAG = "CONFLICT"

#: The section order is part of the output contract and is immutable.
SECTION_ORDER: tuple[str, ...] = (
    "Project Information",
    "Executive Summary",
    "Load Summary",
    "Engineering Calculations",
    "Results",
    "Engineering Evaluation",
    "References",
)

DISCLAIMER = (
    "This report is an automated review aid generated from the extracted "
    "fields of a single source document. It is not certified or approved "
    "engineering work. Values are reproduced as printed in the source; any "
    "item marked \"Not found in document\" or flagged as a conflict requires "
    "manual verification by a qualified engineer."
)

_STATUS_OK = "ok"
_STATUS_MISSING = "missing"
_STATUS_CONFLICT = "conflict"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _normalize(text: str | None) -> str:
    """Case/punctuation-insensitive key, matching the rest of the codebase."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _format_number(value: object) -> str:
    """Print a number the way the extractor stores it: no trailing '.0'."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return _clean(str(value))


def _page_of(location: str | None) -> str:
    """Extract 'page N' from a location string; '' when it names no page."""
    match = re.search(r"page\s+(\d+)", location or "", re.I)
    return f"page {match.group(1)}" if match else ""


def _cite(source: str | None, location: str | None) -> str:
    """Human-readable traceability for one extracted field."""
    source = _clean(source) or "unknown source"
    page = _page_of(location)
    location = _clean(location)
    if location:
        return f"{source} ({location})"
    if page:
        return f"{source} ({page})"
    return source


# --------------------------------------------------------------------------
# Row model
# --------------------------------------------------------------------------


@dataclass
class Row:
    """One rendered line: a label, its value, and where the value came from."""

    label: str
    value: str
    citation: str = ""
    status: str = _STATUS_OK

    @property
    def is_missing(self) -> bool:
        return self.status == _STATUS_MISSING

    @property
    def is_conflict(self) -> bool:
        return self.status == _STATUS_CONFLICT

    def display_value(self) -> str:
        if self.is_missing:
            return NOT_FOUND
        return f"{self.value} [{CONFLICT_FLAG}]" if self.is_conflict else self.value


@dataclass
class ExtractionPayload:
    """The parsed extraction JSON, validated only as far as we need."""

    fields: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionPayload":
        if not isinstance(data, dict):
            raise ValueError("extraction payload must be a JSON object")
        return cls(
            fields=[item for item in data.get("fields") or [] if isinstance(item, dict)],
            conflicts=[item for item in data.get("conflicts") or [] if isinstance(item, dict)],
            warnings=[str(item) for item in data.get("warnings") or []],
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExtractionPayload":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def source_names(self) -> list[str]:
        seen: list[str] = []
        for item in self.fields:
            name = _clean(item.get("source"))
            if name and name not in seen:
                seen.append(name)
        return seen


def _conflicted_parameters(payload: ExtractionPayload) -> dict[str, str]:
    """Map a normalized parameter name to a human reason, for flagged rows."""
    reasons: dict[str, str] = {}
    for conflict in payload.conflicts:
        parameter = _normalize(conflict.get("parameter"))
        if not parameter:
            continue
        kind = _clean(conflict.get("kind")) or "value"
        reasons[parameter] = (
            f"{kind} conflict: the source prints this parameter more than once "
            "with different values"
        )
    return reasons


def _field_matches(item: dict[str, Any], keywords: Sequence[str]) -> bool:
    """True when a field's parameter matches any keyword.

    Matching is exact-on-normalized-form first, then substring, so a short
    keyword such as 'fy' matches 'fy' and 'fyultimate' but never 'safety'.
    """
    parameter = _normalize(item.get("parameter"))
    if not parameter:
        return False
    for keyword in keywords:
        needle = _normalize(keyword)
        if not needle:
            continue
        if parameter == needle or parameter.startswith(needle) or parameter.endswith(needle):
            return True
        if needle in parameter:
            return True
    return False


def resolve_row(
    label: str,
    fields: Iterable[dict[str, Any]],
    payload: ExtractionPayload,
    keywords: Sequence[str],
) -> Row:
    """Find the single best field for one output row.

    Preference order: an exact parameter match, then the first matching field
    in document order. Returns a missing Row when nothing matches, and a
    conflict Row when the matched parameter is flagged in ``payload``.
    """
    matches = [item for item in fields if _field_matches(item, keywords)]
    if not matches:
        return Row(label=label, value="", status=_STATUS_MISSING)

    chosen = min(
        matches,
        key=lambda item: 0 if _normalize(item.get("parameter")) in {_normalize(k) for k in keywords} else 1,
    )
    value = chosen.get("value")
    if value is None or _clean(str(value)) == "":
        return Row(label=label, value="", status=_STATUS_MISSING)

    unit = _clean(chosen.get("unit"))
    rendered = _format_number(value)
    if unit:
        rendered = f"{rendered} {unit}"
    if isinstance(value, str):
        # A non-numeric raw string is reproduced verbatim, still with its source.
        rendered = f"{value}{(' ' + unit) if unit else ''}"

    reasons = _conflicted_parameters(payload)
    status = _STATUS_CONFLICT if _normalize(chosen.get("parameter")) in reasons else _STATUS_OK
    return Row(
        label=label,
        value=rendered,
        citation=_cite(chosen.get("source"), chosen.get("location")),
        status=status,
    )


# --------------------------------------------------------------------------
# Keyword sets -- the only place a field name is bound to a report row
# --------------------------------------------------------------------------

_KW = {
    "project_name": ("projectname", "nameofproject", "project"),
    "project_no": ("projectno", "projectnumber", "jobno", "jobnumber", "documentno"),
    "structure": ("structure", "projectstructure", "substructure", "descriptionofwork"),
    "location": ("projectlocation", "sitelocation", "location", "site"),
    "client": ("client", "owner", "customer"),
    "engineer": ("engineer", "preparedby", "checkedby", "designer"),
    "submission_date": ("submissiondate", "dateofsubmission", "dateofissue", "issuedate"),
    "company": ("company", "organization", "organisation"),
    "pile_diameter": ("pilediameter", "diameter", "memberdiameter", "outerdiameter", "dp"),
    "wall_thickness": ("pilethickness", "wallthickness", "thickness", "tp"),
    "yield_strength": ("yieldstrength", "fy"),
    "ultimate_strength": ("ultimatestrength", "fu"),
    "pile_length": ("pilelength", "memberlength", "length"),
    "helix_count": ("numberofhelices", "numberofplates", "helixcount", "platecount", "numbe"),
    "helix_diameter": ("helixdiameter", "platediameter", "dhel", "helixdia"),
    "helix_spacing": ("helixspacing", "platespacing", "shel", "spacing"),
    "compression": ("compression", "compressiveload", "axialcompression", "axialcomp", "axc"),
    "tension": ("tension", "tensileload", "axiallytension", "uplift", "upliftload", "pullout"),
    "shear": ("shear", "shearfx", "lateralfx", "fx", "fxx"),
    "moment": ("moment", "bending", "mxx", "mxmax", "myy", "mymax"),
    "ratio": ("rcap", "rgeo", "rstr", "ratio", "dcratio", "demandcapacity"),
    "capacity": ("capacity", "qult", "qult", "qall", "qdesign"),
    "settlement": ("settlement", "predictedsettlement", "verticalsettlement", "smax"),
    "deflection": ("deflection", "maxdeflection", "lateraldisplacement", "displacement", "rotation"),
    "design_life": ("designlife",),
    "corrosion_loss": ("corrosionloss", "thicknessloss", "allowableloss", "steelloss"),
    "phi": ("phi", "frictionangle", "angleoffriction"),
    "cohesion": ("cohesion", "cohesive"),
    "su": ("su", "undrainedshearstrength", "undrainedstrength"),
    "unit_weight": ("unitweight", "gamma", "density"),
}

#: Rows rendered in the Project Information table, in order.
_PROJECT_ROWS: tuple[tuple[str, str], ...] = (
    ("Project Name", "project_name"),
    ("Project Number", "project_no"),
    ("Structure", "structure"),
    ("Project Location", "location"),
    ("Client", "client"),
    ("Prepared By", "engineer"),
    ("Submission Date", "submission_date"),
    ("Company", "company"),
)

#: Rows rendered in the Engineering Calculations table, in order.
_CALC_ROWS: tuple[tuple[str, str], ...] = (
    ("Pile Diameter (O.D.)", "pile_diameter"),
    ("Wall Thickness", "wall_thickness"),
    ("Yield Strength (Fy)", "yield_strength"),
    ("Ultimate Strength (Fu)", "ultimate_strength"),
    ("Pile Length", "pile_length"),
    ("Number of Helices", "helix_count"),
    ("Helix Diameter", "helix_diameter"),
    ("Helix Spacing", "helix_spacing"),
    ("Friction Angle (phi)", "phi"),
    ("Cohesion (c)", "cohesion"),
    ("Undrained Strength (Su)", "su"),
    ("Unit Weight", "unit_weight"),
    ("Design Life", "design_life"),
    ("Corrosion Loss", "corrosion_loss"),
)


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------


def build_project_rows(payload: ExtractionPayload) -> list[Row]:
    return [resolve_row(label, payload.fields, payload, _KW[key]) for label, key in _PROJECT_ROWS]


def build_calculation_rows(payload: ExtractionPayload) -> list[Row]:
    return [resolve_row(label, payload.fields, payload, _KW[key]) for label, key in _CALC_ROWS]


def _group_by_context(
    fields: Iterable[dict[str, Any]], keywords: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Group matching fields by their row context (pile id / load case)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in fields:
        if not _field_matches(item, keywords):
            continue
        key = _clean(item.get("context")) or _clean(item.get("parameter")) or "Unlabelled"
        groups.setdefault(key, []).append(item)
    return groups


def _first_in(groups: dict[str, list[dict[str, Any]]], context: str) -> tuple[str, str]:
    """Return (rendered value, citation) for one cell of a load/ratio table."""
    for item in groups.get(context, []):
        value = item.get("value")
        if value is None or _clean(str(value)) == "":
            continue
        unit = _clean(item.get("unit"))
        rendered = _clean(str(value)) if isinstance(value, str) else _format_number(value)
        if unit:
            rendered = f"{rendered} {unit}"
        return rendered, _cite(item.get("source"), item.get("location"))
    return NOT_FOUND, ""


def _load_table(payload: ExtractionPayload) -> list[list[str]]:
    """Load Summary: one row per load case, one column per action."""
    columns = ("Load Case", "Compression", "Tension", "Shear", "Moment")
    action_keys = ("compression", "tension", "shear", "moment")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in payload.fields:
        context = _clean(item.get("context"))
        if not context:
            continue
        if any(_field_matches(item, _KW[key]) for key in action_keys):
            grouped.setdefault(context, []).append(item)

    if not grouped:
        return [list(columns), [NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND]]

    table: list[list[str]] = [list(columns)]
    for context in grouped:
        cells: list[str] = [context]
        for key in action_keys:
            subset = {
                name: [i for i in items if _field_matches(i, _KW[key])]
                for name, items in grouped.items()
            }
            value, _citation = _first_in(subset, context)
            cells.append(value)
        table.append(cells)
    return table


def _result_table(payload: ExtractionPayload) -> list[list[str]]:
    """Results: one row per case with its reported ratio, settlement, deflection."""
    columns = ("Case", "Ratio (R/CAP)", "Settlement", "Lateral Deflection")
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in payload.fields:
        context = _clean(item.get("context")) or _clean(item.get("parameter"))
        if not context:
            continue
        if any(
            _field_matches(item, _KW[key])
            for key in ("ratio", "settlement", "deflection", "capacity")
        ):
            groups.setdefault(context, []).append(item)

    if not groups:
        return [list(columns), [NOT_FOUND, NOT_FOUND, NOT_FOUND, NOT_FOUND]]

    table: list[list[str]] = [list(columns)]
    for context, items in groups.items():
        subset = {
            key: [i for i in items if _field_matches(i, _KW[k])]
            for key, k in (("ratio", "ratio"), ("settlement", "settlement"), ("deflection", "deflection"))
        }
        cells = [context]
        for key in ("ratio", "settlement", "deflection"):
            value, _citation = _first_in(subset, context)
            cells.append(value)
        table.append(cells)
    return table


def build_evaluation_rows(payload: ExtractionPayload) -> list[Row]:
    """Deterministic ratio-vs-1.00 comparison.

    This is an arithmetic comparison of numbers the document already printed.
    It is explicitly NOT a design check and NOT a certification.
    """
    rows: list[Row] = []
    seen: set[str] = set()
    for item in payload.fields:
        if not _field_matches(item, _KW["ratio"]):
            continue
        value = item.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        context = _clean(item.get("context")) or _clean(item.get("parameter")) or "Unlabelled"
        if context in seen:
            continue
        seen.add(context)
        verdict = "within 1.00" if value <= 1.0 else "exceeds 1.00"
        rows.append(
            Row(
                label=context,
                value=f"{_format_number(value)} - {verdict}",
                citation=_cite(item.get("source"), item.get("location")),
                status=_STATUS_OK,
            )
        )

    if not rows:
        return [Row(label="Ratio comparison", value="", status=_STATUS_MISSING)]
    rows.append(
        Row(
            label="Governing ratio",
            value=_format_number(
                max(
                    item.get("value")
                    for item in payload.fields
                    if _field_matches(item, _KW["ratio"])
                    and isinstance(item.get("value"), (int, float))
                    and not isinstance(item.get("value"), bool)
                )
            ),
            citation="largest ratio reported in the source document",
        )
    )
    return rows


def build_summary_paragraphs(payload: ExtractionPayload) -> tuple[list[str], list[str]]:
    """Build the Executive Summary deterministically.

    Every clause is a fixed template with its numbers interpolated from the
    JSON. A clause whose value is absent is dropped and listed under
    ``not_found`` instead -- the sentence is never completed with a guess.
    """
    clauses: list[str] = []
    not_found: list[str] = []
    reasons = _conflicted_parameters(payload)

    def clause(template: str, label: str, key: str) -> None:
        row = resolve_row(label, payload.fields, payload, _KW[key])
        if row.is_missing:
            not_found.append(f"{label} ({NOT_FOUND})")
            return
        rendered = row.value + (f" - {CONFLICT_FLAG}, see note" if row.is_conflict else "")
        clauses.append(template.format(value=rendered))

    sources = payload.source_names()
    if sources:
        clauses.append(
            "This review was generated from the following source document(s): "
            + ", ".join(sources)
            + "."
        )
    else:
        not_found.append(f"Source document ({NOT_FOUND})")

    clause("The pile section recorded in the source has an outer diameter of {value}.", "Pile Diameter (O.D.)", "pile_diameter")
    clause("The pile wall thickness recorded in the source is {value}.", "Wall Thickness", "wall_thickness")
    clause("The steel yield strength (Fy) recorded in the source is {value}.", "Yield Strength (Fy)", "yield_strength")
    clause("The pile length recorded in the source is {value}.", "Pile Length", "pile_length")
    clause("The source records {value} for the number of helix plates.", "Number of Helices", "helix_count")
    clause("The helix plate diameter recorded in the source is {value}.", "Helix Diameter", "helix_diameter")

    ratios = [
        item.get("value")
        for item in payload.fields
        if _field_matches(item, _KW["ratio"])
        and isinstance(item.get("value"), (int, float))
        and not isinstance(item.get("value"), bool)
    ]
    if ratios:
        clauses.append(
            f"The largest load-to-capacity ratio explicitly reported in the source is "
            f"{_format_number(max(ratios))}."
        )
    else:
        not_found.append(f"Load-to-capacity ratio ({NOT_FOUND})")

    if reasons:
        clauses.append(
            f"{len(reasons)} parameter(s) appear more than once in the source with "
            "differing values. These are flagged for manual review and were not resolved."
        )
    return clauses, not_found


def build_reference_rows(payload: ExtractionPayload) -> list[Row]:
    """References: the source document(s) plus every warning the extractor raised."""
    rows: list[Row] = []
    names = payload.source_names()
    if not names:
        rows.append(Row(label="Source document", value="", status=_STATUS_MISSING))
    for name in names:
        pages = sorted(
            {
                _page_of(item.get("location"))
                for item in payload.fields
                if _clean(item.get("source")) == name and _page_of(item.get("location"))
            },
            key=lambda item: int(item.split()[1]),
        )
        detail = ", ".join(pages) if pages else "no page reference available"
        rows.append(Row(label=f"Source document: {name}", value=detail))
    rows.append(
        Row(
            label="Extracted fields used",
            value=str(len(payload.fields)),
            citation="count of records in the supplied extraction JSON",
        )
    )
    if payload.conflicts:
        rows.append(
            Row(
                label="Flagged conflicts",
                value=str(len(payload.conflicts)),
                citation="parameters printed more than once with differing values",
            )
        )
    for warning in payload.warnings:
        rows.append(Row(label="Extractor warning", value=_clean(warning)))
    return rows


# --------------------------------------------------------------------------
# PDF rendering
# --------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=16, leading=19, spaceAfter=2, alignment=TA_LEFT
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#444444")
        ),
        "heading": ParagraphStyle(
            "SectionHeading", parent=base["Heading2"], fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=5
        ),
        "body": ParagraphStyle("BodyText", parent=base["BodyText"], fontSize=9, leading=12.5, spaceAfter=4),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["BodyText"], fontSize=7.6, leading=10, textColor=colors.HexColor("#444444")
        ),
        "cell": ParagraphStyle("Cell", parent=base["Normal"], fontSize=8.2, leading=10.4),
        "cellHead": ParagraphStyle("CellHead", parent=base["Normal"], fontSize=8.2, leading=10.4, fontName="Helvetica-Bold"),
        "cellMissing": ParagraphStyle(
            "CellMissing", parent=base["Normal"], fontSize=8.2, leading=10.4, textColor=colors.HexColor("#8a6d00")
        ),
        "cellConflict": ParagraphStyle(
            "CellConflict", parent=base["Normal"], fontSize=8.2, leading=10.4, textColor=colors.HexColor("#a00000")
        ),
    }


_TABLE_STYLE = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
)


def _cell(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def _rows_table(rows: list[Row], styles: dict[str, ParagraphStyle], width: float) -> Table:
    data: list[list[Paragraph]] = [
        [_cell("Item", styles["cellHead"]), _cell("Value", styles["cellHead"]), _cell("Source", styles["cellHead"])]
    ]
    for row in rows:
        if row.is_missing:
            style = styles["cellMissing"]
        elif row.is_conflict:
            style = styles["cellConflict"]
        else:
            style = styles["cell"]
        data.append(
            [
                _cell(row.label, styles["cell"]),
                _cell(row.display_value(), style),
                _cell(row.citation or "-", styles["cell"]),
            ]
        )
    table = Table(data, colWidths=[width * 0.30, width * 0.34, width * 0.36], repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    return table


def _grid_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], width: float) -> Table:
    header = [[_cell(text, styles["cellHead"]) for text in rows[0]]]
    body: list[list[Paragraph]] = []
    for line in rows[1:]:
        style = styles["cellMissing"] if NOT_FOUND in line else styles["cell"]
        body.append([_cell(text, style) for text in line])
    count = len(rows[0])
    table = Table(header + body, colWidths=[width / count] * count, repeatRows=1)
    table.setStyle(_TABLE_STYLE)
    return table


def _footer(canvas: Any, document: Any) -> None:
    """Page number and provenance on every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(document.leftMargin, 10 * mm, "Automated review aid - not certified engineering work")
    canvas.drawRightString(A4[0] - document.rightMargin, 10 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_story(payload: ExtractionPayload) -> list[Any]:
    """Assemble the full flowable list. The section order here IS the contract."""
    styles = _styles()
    width = A4[0] - 25 * mm
    story: list[Any] = []

    sources = ", ".join(payload.source_names()) or NOT_FOUND
    story.append(_cell("Helical Pile Design Review Report", styles["title"]))
    story.append(
        _cell(
            f"Source: {sources}  |  Fields: {len(payload.fields)}  |  "
            f"Conflicts: {len(payload.conflicts)}",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(_cell(DISCLAIMER, styles["disclaimer"]))

    # 1. Project Information
    story.append(_cell(f"1. {SECTION_ORDER[0]}", styles["heading"]))
    story.append(_rows_table(build_project_rows(payload), styles, width))

    # 2. Executive Summary
    story.append(_cell(f"2. {SECTION_ORDER[1]}", styles["heading"]))
    clauses, not_found = build_summary_paragraphs(payload)
    for clause in clauses:
        story.append(_cell(clause, styles["body"]))
    if not clauses:
        story.append(_cell(NOT_FOUND, styles["cellMissing"]))
    if not_found:
        story.append(Spacer(1, 2))
        story.append(_cell("The following items could not be stated from the source document:", styles["body"]))
        for item in not_found:
            story.append(_cell(f"- {escape(item)}", styles["cellMissing"]))

    # 3. Load Summary
    story.append(_cell(f"3. {SECTION_ORDER[2]}", styles["heading"]))
    story.append(_grid_table(_load_table(payload), styles, width))

    # 4. Engineering Calculations
    story.append(_cell(f"4. {SECTION_ORDER[3]}", styles["heading"]))
    story.append(_rows_table(build_calculation_rows(payload), styles, width))

    # 5. Results
    story.append(_cell(f"5. {SECTION_ORDER[4]}", styles["heading"]))
    story.append(_grid_table(_result_table(payload), styles, width))

    # 6. Engineering Evaluation
    story.append(_cell(f"6. {SECTION_ORDER[5]}", styles["heading"]))
    story.append(
        _cell(
            "Comparison of ratios explicitly reported in the source against 1.00. "
            "This is an arithmetic comparison only - it is not a design check, and "
            "it is not certification.",
            styles["disclaimer"],
        )
    )
    story.append(Spacer(1, 3))
    story.append(_rows_table(build_evaluation_rows(payload), styles, width))

    # 7. References
    story.append(PageBreak())
    story.append(_cell(f"7. {SECTION_ORDER[6]}", styles["heading"]))
    story.append(_rows_table(build_reference_rows(payload), styles, width))

    return story


def generate_pdf(payload: ExtractionPayload | dict[str, Any], output_path: str | Path) -> Path:
    """Render the fixed-layout PDF. Returns the written path."""
    if isinstance(payload, dict):
        payload = ExtractionPayload.from_dict(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=12.5 * mm,
        rightMargin=12.5 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title="Helical Pile Design Review Report",
        author="engineering-ai",
    )
    document.build(build_story(payload), onFirstPage=_footer, onLaterPages=_footer)
    return output_path


# --------------------------------------------------------------------------
# CLI: python -m app.report.pdf_report <extraction.json> <out.pdf>
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m app.report.pdf_report <extraction.json> <out.pdf>")
        return 2
    source, destination = args
    try:
        payload = ExtractionPayload.load(source)
    except FileNotFoundError:
        print(f"!! no such file: {source}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"!! {source} is not valid JSON: {exc}")
        return 1
    except ValueError as exc:
        print(f"!! {exc}")
        return 1

    written = generate_pdf(payload, destination)
    print(f"wrote {written}")
    print(f"  sections : {len(SECTION_ORDER)} (fixed layout)")
    print(f"  fields   : {len(payload.fields)}")
    print(f"  conflicts: {len(payload.conflicts)}")
    print(f"  warnings : {len(payload.warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
