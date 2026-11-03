"""Executive Summary PDF export.

Renders a tight, professional 1-2 page A4 "Executive Summary" from data that
has ALREADY been extracted. This module never runs the model and never
re-extracts anything: it is a pure renderer over the dict it is handed, so it
can never write a value that was not read from the source document.

Design rules (inherited from the rest of the project):
- Nothing is invented. An absent value prints ``Not found`` and is repeated in
  "Notes and Flags".
- A key reported more than once with different values prints BOTH sides with
  their pile-id context and is flagged. It is never silently resolved,
  averaged, or reduced to one number.
- A value that is not a plain number is printed verbatim and flagged as
  unclear, never coerced into a number.
- Every printed value carries a unit when the source printed one, and a source
  citation (document + page/section/row).
- Output is an evaluation aid. It is never certified or approved work.

The only public entry point is :func:`build_executive_summary_pdf`; the web
layer also uses :func:`executive_summary_filename` for the download name.

Expected input shape (the ``key_points`` block of ``KeyPointsResult.to_dict()``,
see ``app/report/key_points.py``)::

    {
      "key_points": {
        "project":    [{"key": ..., "value": ..., "unit": ..., "context": ...,
                        "source": ..., "location": ..., "raw": ...}, ...],
        "pile":       [...],
        "loads":      [...],
        "results":    [...],
        "evaluation": [...],
        "references": [...],
        "missing":    [...],
      },
      "conflicts": [...],
      "warnings":  [...],
    }

Every part of that structure is treated as optional.

When the payload also carries the raw ``fields`` the browser holds (the same
list the web UI's clean summary is built from), the Load Summary and Key
Results cells are resolved through :mod:`app.report.summary` so the PDF prints
the same single value the UI shows, rather than every raw key-point side.
Without ``fields`` the key points are rendered directly.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

from app.report.key_points import is_key_point_parameter
from app.report.summary import (
    SUMMARY_RATIO_LINES,
    field_text as _summary_field_text,
    resolve_summary_field,
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# --------------------------------------------------------------------------
# Contract constants
# --------------------------------------------------------------------------

#: Exact text printed for any value absent from the source. Never blank, never
#: a zero, never a default -- absence is reported as absence.
NOT_FOUND = "Not found"

#: Marker shown next to a value whose reported sources disagree.
CONFLICT_FLAG = "CONFLICT"

#: Marker shown when a key was reported more than once with different values.
MULTI_FLAG = "MULTIPLE VALUES"

#: Printed on every page, without exception.
FOOTER_TEXT = (
    "Generated automatically from the source report. This is an engineering "
    "evaluation aid, not a formal engineering approval."
)

#: Row caps that keep the summary inside its 1-2 page budget. Truncation is
#: never silent: every cap that bites adds a "Notes and Flags" entry.
MAX_LOAD_CASES = 12
MAX_RATIOS = 12

#: Per-conflict caps. A conflict with more distinct sides than
#: :data:`CHART_CONFLICT_MIN_SIDES` is a chart/analysis data series (dozens of
#: plotted coordinates), not a specification disagreement, and is omitted.
#: :data:`MAX_CONFLICT_SIDES` caps how many sides an ordinary conflict prints.
MAX_CONFLICT_SIDES = 6
CHART_CONFLICT_MIN_SIDES = 10

#: Row label used for a resolved load whose source states no load case (the
#: context-free shear/moment lines the web UI also prints without a case).
_GENERAL_LOAD_CASE = "General"

_MARGIN = 12 * mm
_BOTTOM_MARGIN = 17 * mm
_USABLE_WIDTH = A4[0] - 2 * _MARGIN

_STATUS_OK = "ok"
_STATUS_MISSING = "missing"
_STATUS_CONFLICT = "conflict"

# --------------------------------------------------------------------------
# Small helpers (mirrors app/report/pdf_report.py conventions)
# --------------------------------------------------------------------------


def _clean(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _normalize(text: Any) -> str:
    """Case/punctuation-insensitive key, matching the rest of the codebase."""
    return re.sub(r"[^a-z0-9]+", "", _clean(text).lower())


def _format_number(value: Any) -> str:
    """Print a number the way the extractor stores it: no trailing '.0'."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return _clean(value)


def _page_of(location: Any) -> str:
    """Extract 'page N' from a location string; '' when it names no page."""
    match = re.search(r"page\s+(\d+)", _clean(location), re.I)
    return f"page {match.group(1)}" if match else ""


def _cite(source: Any, location: Any) -> str:
    """Human-readable traceability for one extracted record."""
    source = _clean(source)
    location = _clean(location)
    if not source and not location:
        return NOT_FOUND
    if not source:
        return location
    return f"{source} ({location})" if location else source


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_records(value: Any) -> list[dict[str, Any]]:
    """Any list of dicts, ignoring anything that is not a dict."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean(item) for item in value if _clean(item)]


def _section(data: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return _as_records(_as_dict(data.get("key_points")).get(name))


def _records_for(records: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Every record filed under one canonical key, in document order."""
    return [rec for rec in records if _clean(rec.get("key")) == key]


# --------------------------------------------------------------------------
# Value resolution -- the never-guess core
# --------------------------------------------------------------------------

#: Keys whose value must be a plain number. Project keys are excluded: a client
#: name or a date is a string by nature, so flagging it would be false noise.
_NUMERIC_KEYS = frozenset(
    {
        "pile_diameter",
        "pile_thickness",
        "yield_strength",
        "pile_length",
        "helix_count",
        "result_ratio",
        "load_compression",
        "load_tension",
        "load_shear",
        "load_moment",
    }
) | frozenset(f"helix_dia_{n}" for n in range(1, 6)) | {"helix_dia"}

#: Keys whose value is a physical quantity, so a missing unit is worth
#: reporting. Deliberately narrower than _NUMERIC_KEYS: a count or a ratio has
#: no unit, and merging these two sets would flag those as errors.
_UNIT_EXPECTED = frozenset(
    {
        "pile_diameter",
        "pile_thickness",
        "yield_strength",
        "pile_length",
        "load_compression",
        "load_tension",
        "load_shear",
        "load_moment",
    }
) | frozenset(f"helix_dia_{n}" for n in range(1, 6)) | {"helix_dia"}


@dataclass
class Resolved:
    """One rendered cell: its text, its citation, and everything worth flagging."""

    text: str
    citation: str = ""
    status: str = _STATUS_OK
    notes: list[str] = field(default_factory=list)


def _render_one(record: dict[str, Any]) -> tuple[str, str, bool]:
    """(rendered text, unit, is_numeric) for a single record."""
    raw_value = record.get("value")
    unit = _clean(record.get("unit"))

    if isinstance(raw_value, bool):
        return _format_number(raw_value), unit, False

    if isinstance(raw_value, (int, float)):
        return f"{_format_number(raw_value)} {unit}".strip(), unit, True

    # A non-numeric cell is reproduced verbatim. It is never parsed, coerced,
    # or dropped -- an unreadable cell is data to be flagged, not to guess at.
    text = _clean(raw_value)
    return (f"{text} {unit}".strip() if unit else text), unit, False


def _resolve(label: str, records: Sequence[dict[str, Any]], key: str) -> Resolved:
    """Resolve the records filed under ``key`` into one printable cell.

    Handles the three ways real extraction goes wrong, and never papers over
    any of them:
      * nothing reported          -> "Not found" + a note
      * several distinct values   -> every value shown, flagged, never reduced
      * a value that is not a
        plain number, or a quantity with no unit -> flagged as unclear
    """
    matches = _records_for(records, key)

    if not matches:
        return Resolved(
            text=NOT_FOUND,
            citation=NOT_FOUND,
            status=_STATUS_MISSING,
            notes=[f"{label}: {NOT_FOUND}. This item was not present in the source report."],
        )

    distinct: list[tuple[str, str, str, bool]] = []
    for record in matches:
        raw_value = record.get("value")
        if raw_value is None or _clean(raw_value) == "":
            continue
        rendered, unit, is_numeric = _render_one(record)
        entry = (rendered, unit, _clean(record.get("context")), is_numeric)
        if entry not in distinct:
            distinct.append(entry)

    if not distinct:
        return Resolved(
            text=NOT_FOUND,
            citation=_cite(matches[0].get("source"), matches[0].get("location")),
            status=_STATUS_MISSING,
            notes=[
                f"{label}: {NOT_FOUND}. The source report names this item but "
                "prints no readable value."
            ],
        )

    citation = _cite(matches[0].get("source"), matches[0].get("location"))
    notes: list[str] = []

    # Several different values for one item: show them all, flag, never pick.
    if len(distinct) > 1:
        sides = []
        for rendered, _unit, context, _numeric in distinct:
            sides.append(f"{rendered} ({context})" if context else rendered)
        places = sorted(
            {
                _cite(rec.get("source"), rec.get("location"))
                for rec in matches
                if rec.get("value") is not None
            }
        )
        notes.append(
            f"{label}: the source report gives more than one value "
            f"({' / '.join(sides)}). Every value is shown and none was selected. "
            f"Source: {'; '.join(places)}."
        )
        if key in _NUMERIC_KEYS:
            for rendered, unit, context, is_numeric in distinct:
                where = f" for {context}" if context else ""
                if not is_numeric:
                    notes.append(
                        f"{label}{where}: {rendered!r} is not a plain number. It is "
                        "shown exactly as printed and needs manual checking."
                    )
                elif key in _UNIT_EXPECTED and not unit:
                    notes.append(
                        f"{label}{where}: {rendered} was reported with no unit. The "
                        "unit was not assumed."
                    )
        return Resolved(
            text=f"{MULTI_FLAG}: " + "  /  ".join(sides),
            citation=citation,
            status=_STATUS_CONFLICT,
            notes=notes,
        )

    rendered, unit, _context, is_numeric = distinct[0]
    numeric_expected = key in _NUMERIC_KEYS
    unit_expected = key in _UNIT_EXPECTED

    if numeric_expected and not is_numeric:
        notes.append(
            f"{label}: the source prints {rendered!r} here, which is not a plain "
            "number. It is shown exactly as printed and needs manual checking."
        )
    if unit_expected and not unit and is_numeric:
        notes.append(
            f"{label}: a value of {rendered} was reported with no unit. The unit "
            "was not assumed."
        )

    status = _STATUS_OK
    if (numeric_expected and not is_numeric) or (unit_expected and not unit and is_numeric):
        status = _STATUS_CONFLICT

    return Resolved(text=rendered, citation=citation, status=status, notes=notes)


# --------------------------------------------------------------------------
# Section content
# --------------------------------------------------------------------------

#: Project Information rows, in printed order.
_PROJECT_ROWS: tuple[tuple[str, str], ...] = (
    ("Project name", "project_name"),
    ("Structure", "structure"),
    ("Project location", "location"),
    ("Client", "client"),
    ("Date of submission", "date"),
)

#: Pile Details rows that always appear, in printed order.
_PILE_ROWS: tuple[tuple[str, str], ...] = (
    ("Pile diameter", "pile_diameter"),
    ("Pile wall thickness", "pile_thickness"),
    ("Yield strength (Fy)", "yield_strength"),
    ("Pile length", "pile_length"),
    ("Number of helixes", "helix_count"),
)

#: Load Summary columns, in printed order.
_LOAD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Compression", "load_compression"),
    ("Tension", "load_tension"),
    ("Shear", "load_shear"),
    ("Moment", "load_moment"),
)

_LOAD_HEADERS = ("Load case",) + tuple(title for title, _ in _LOAD_COLUMNS)

#: Load column -> clean-summary line key (app.report.summary._SUMMARY_SEARCHES).
_LOAD_SUMMARY_KEY = {
    "load_compression": "compression",
    "load_tension": "tension",
    "load_shear": "shear",
    "load_moment": "moment",
}


def _payload_fields(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The raw fields the browser sent, if any, used for UI-aligned resolution."""
    return [item for item in data.get("fields") or [] if isinstance(item, dict)]


def _value_number(value: Any) -> float | None:
    """First number of a value, or None (used to pick the governing ratio)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", _clean(value))
    return float(match.group(0)) if match else None


@dataclass
class Row:
    """One rendered line: a label, its value, where it came from, its status."""

    label: str
    resolved: Resolved

    @property
    def status(self) -> str:
        return self.resolved.status


def _project_name(data: dict[str, Any]) -> str:
    records = _section(data, "project")
    resolved = _resolve("Project name", records, "project_name")
    if resolved.status == _STATUS_MISSING:
        return NOT_FOUND
    return resolved.text.split(": ", 1)[-1].strip() or NOT_FOUND


def _helix_rows(data: dict[str, Any]) -> list[Row]:
    """Numbered helix diameters when present, else the unnumbered one."""
    records = _section(data, "pile")
    rows: list[Row] = []
    for number in range(1, 6):
        key = f"helix_dia_{number}"
        if _records_for(records, key):
            rows.append(Row(f"Helix diameter {number}", _resolve(f"Helix diameter {number}", records, key)))
    if not rows and _records_for(records, "helix_dia"):
        rows.append(Row("Helix diameter", _resolve("Helix diameter", records, "helix_dia")))
    if not rows:
        rows.append(
            Row(
                "Helix diameter",
                Resolved(
                    text=NOT_FOUND,
                    citation=NOT_FOUND,
                    status=_STATUS_MISSING,
                    notes=[
                        "Helix diameter: Not found. No helix diameter was printed "
                        "in the source report."
                    ],
                ),
            )
        )
    return rows


def _project_rows(data: dict[str, Any]) -> list[Row]:
    records = _section(data, "project")
    return [Row(label, _resolve(label, records, key)) for label, key in _PROJECT_ROWS]


def _pile_rows(data: dict[str, Any]) -> list[Row]:
    records = _section(data, "pile")
    rows = [Row(label, _resolve(label, records, key)) for label, key in _PILE_ROWS]
    rows.extend(_helix_rows(data))
    return rows


def _load_table(data: dict[str, Any]) -> tuple[list[list[str]], list[str], list[str]]:
    """(cells, footnotes, notes) for the Load Summary table.

    When the payload carries the raw ``fields`` the browser holds, each load
    cell is resolved exactly as the web UI's clean summary resolves it (same
    keyword, same context scoping), so the PDF and the UI print one value.
    Otherwise the key points are read directly, as from a headless caller.
    """
    fields = _payload_fields(data)
    if fields:
        return _load_table_resolved(fields)
    return _load_table_from_keypoints(data)


def _load_table_resolved(
    fields: list[dict[str, Any]]
) -> tuple[list[list[str]], list[str], list[str]]:
    """Load Summary built from the UI's resolved applied-load lines.

    Every column is resolved to one value by ``resolve_summary_field`` and is
    placed on the row of the load case that value belongs to; a line whose
    source states no case (shear/moment) sits on the "General" row. A column
    the UI also cannot resolve prints "Not found".
    """
    columns: list[tuple[str, str, str, str]] = []
    for title, key in _LOAD_COLUMNS:
        field, context = resolve_summary_field(fields, _LOAD_SUMMARY_KEY[key])
        if field is None:
            columns.append((title, "", _GENERAL_LOAD_CASE, ""))
            continue
        columns.append((
            title,
            _summary_field_text(field),
            context or _GENERAL_LOAD_CASE,
            _cite(field.get("source"), field.get("location")),
        ))

    order: list[str] = []
    for _title, _text, context, _citation in columns:
        if context not in order:
            order.append(context)

    notes: list[str] = []
    body: list[list[str]] = []
    footnotes: list[str] = []
    for context in order:
        cells = [context]
        sources: list[str] = []
        for title, text, column_context, citation in columns:
            if column_context != context:
                cells.append("")
                continue
            if not text:
                cells.append(NOT_FOUND)
                notes.append(
                    f"{title} load: {NOT_FOUND}. The source report prints no such "
                    "value."
                )
                continue
            cells.append(text)
            if citation and citation != NOT_FOUND:
                sources.append(citation)
        body.append(cells)
        if sources:
            footnotes.append(f"{context}: {', '.join(dict.fromkeys(sources))}")

    return [list(_LOAD_HEADERS), *body], footnotes, notes


def _load_table_from_keypoints(
    data: dict[str, Any]
) -> tuple[list[list[str]], list[str], list[str]]:
    """(cells, footnotes, notes) for the Load Summary from key points."""
    records = _section(data, "loads")
    notes: list[str] = []

    contexts: list[str] = []
    for record in records:
        context = _clean(record.get("context"))
        if context and context not in contexts:
            contexts.append(context)

    if not contexts:
        row = [NOT_FOUND] * len(_LOAD_HEADERS)
        notes.append(
            "Loads: Not found. No factored loads were printed in the source report."
        )
        return [list(_LOAD_HEADERS), row], [], notes

    shown = contexts[:MAX_LOAD_CASES]
    if len(contexts) > MAX_LOAD_CASES:
        hidden = len(contexts) - MAX_LOAD_CASES
        notes.append(
            f"Load cases: only the first {MAX_LOAD_CASES} of {len(contexts)} load "
            f"cases are shown on this page; {hidden} more are in the source report."
        )

    body: list[list[str]] = []
    footnotes: list[str] = []
    for context in shown:
        cells = [context]
        sources: list[str] = []
        for _title, key in _LOAD_COLUMNS:
            subset = [
                record
                for record in _records_for(records, key)
                if _clean(record.get("context")) == context
            ]
            if not subset:
                cells.append(NOT_FOUND)
                notes.append(
                    f"{_title} load for {context}: {NOT_FOUND}. The source report "
                    "does not print this value for that load case."
                )
                continue
            resolved = _resolve(f"{_title} load for {context}", subset, key)
            cells.append(resolved.text)
            notes.extend(resolved.notes)
            if resolved.citation and resolved.citation != NOT_FOUND:
                sources.append(resolved.citation)
        body.append(cells)
        if sources:
            footnotes.append(f"{context}: {', '.join(dict.fromkeys(sources))}")

    return [list(_LOAD_HEADERS), *body], footnotes, notes


def _result_rows(data: dict[str, Any]) -> tuple[list[Row], list[str], str]:
    """(rows, notes, governing-case sentence) for the Key Results section.

    With the browser's raw ``fields`` present, the three ratio lines the web UI
    reports (R/GEO, R/STR, R/CAP) are resolved the same way the UI resolves
    them. Otherwise the key points are read directly.
    """
    fields = _payload_fields(data)
    if fields:
        return _result_rows_resolved(fields)
    return _result_rows_from_keypoints(data)


def _result_rows_resolved(
    fields: list[dict[str, Any]]
) -> tuple[list[Row], list[str], str]:
    """Key Results from the UI's resolved ratio lines."""
    notes: list[str] = []
    rows: list[Row] = []
    governing: tuple[float, str, str] | None = None
    for label, key in SUMMARY_RATIO_LINES:
        field, _context = resolve_summary_field(fields, key)
        if field is None:
            resolved = Resolved(
                text=NOT_FOUND,
                citation=NOT_FOUND,
                status=_STATUS_MISSING,
                notes=[
                    f"{label}: {NOT_FOUND}. The source report prints no such ratio."
                ],
            )
            rows.append(Row(label, resolved))
            notes.extend(resolved.notes)
            continue
        text = _summary_field_text(field)
        rows.append(
            Row(
                label,
                Resolved(
                    text=text,
                    citation=_cite(field.get("source"), field.get("location")),
                ),
            )
        )
        number = _value_number(field.get("value"))
        if number is not None and (governing is None or number > governing[0]):
            governing = (number, label, text)

    if governing is None:
        sentence = (
            "Governing ratio: Not found. The source report prints no plain ratio, "
            "so no governing limit state could be identified."
        )
    else:
        sentence = (
            f"Governing ratio: {governing[1]} = {governing[2]} (largest reported "
            "ratio)."
        )
    return rows, notes, sentence


def _result_rows_from_keypoints(
    data: dict[str, Any]
) -> tuple[list[Row], list[str], str]:
    """(rows, notes, governing-case sentence) from key points."""
    records = _section(data, "results")
    notes: list[str] = []

    contexts: list[str] = []
    for record in records:
        context = _clean(record.get("context")) or "Load case not specified"
        if context not in contexts:
            contexts.append(context)

    if not contexts:
        return (
            [
                Row(
                    "D/C ratio",
                    Resolved(
                        text=NOT_FOUND,
                        citation=NOT_FOUND,
                        status=_STATUS_MISSING,
                        notes=[
                            "D/C ratio: Not found. The source report prints no "
                            "demand/capacity ratio, so no governing load case can "
                            "be identified."
                        ],
                    ),
                )
            ],
            notes,
            "Governing load case: Not found. No demand/capacity ratio was printed "
            "in the source report, so the governing load case cannot be identified.",
        )

    shown = contexts[:MAX_RATIOS]
    if len(contexts) > MAX_RATIOS:
        hidden = len(contexts) - MAX_RATIOS
        notes.append(
            f"D/C ratios: only the first {MAX_RATIOS} of {len(contexts)} load cases "
            f"are shown on this page; {hidden} more are in the source report."
        )

    rows: list[Row] = []
    governing: tuple[float, str] | None = None
    for context in shown:
        subset = [
            record
            for record in _records_for(records, "result_ratio")
            if _clean(record.get("context")) == context
        ]
        resolved = _resolve(f"D/C ratio for {context}", subset, "result_ratio")
        value = next(
            (
                float(record["value"])
                for record in subset
                if isinstance(record.get("value"), (int, float))
                and not isinstance(record.get("value"), bool)
            ),
            None,
        )
        rows.append(Row(f"{context} -- D/C ratio", resolved))
        notes.extend(resolved.notes)
        if value is not None and (governing is None or value > governing[0]):
            governing = (value, context)

    if governing is None:
        sentence = (
            "Governing load case: Not found. The ratios in the source report are "
            "not plain numbers, so no governing load case could be identified."
        )
    else:
        sentence = (
            f"Governing load case: {governing[1]}, with the largest reported "
            f"D/C ratio of {_format_number(governing[0])}."
        )
    return rows, notes, sentence


# --------------------------------------------------------------------------
# Notes and Flags
# --------------------------------------------------------------------------


def _collect_notes(
    data: dict[str, Any],
    collected: Sequence[str],
    conflicts: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> list[str]:
    """Every missing, unclear, and conflicting value, de-duplicated, in order."""
    notes: list[str] = []
    seen: set[str] = set()

    def add(note: str) -> None:
        note = _clean(note)
        if note and note not in seen:
            seen.add(note)
            notes.append(note)

    for note in collected:
        add(note)

    # Items the extractor itself reported as absent.
    for item in _as_texts(_as_dict(data.get("key_points")).get("missing")):
        add(f"{item}: {NOT_FOUND}. This item was not present in the source report.")

    # Disagreements the field extractor recorded. Only conflicts that name a
    # value the summary reports are listed: the extractor also flags every
    # chart/soil-profile column, and those series are drawing data, not
    # specification disagreements. Ordinary conflicts keep every side, capped.
    chart_series: list[str] = []
    for conflict in conflicts:
        parameter = _clean(conflict.get("parameter")) or "Parameter"
        if not is_key_point_parameter(parameter):
            continue
        kind = _clean(conflict.get("kind")) or "value"
        sides: list[str] = []
        seen_sides: set[str] = set()
        for record in _as_records(conflict.get("fields")):
            rendered, _unit, _numeric = _render_one(record)
            side = f"{rendered} ({_cite(record.get('source'), record.get('location'))})"
            if side in seen_sides:
                continue
            seen_sides.add(side)
            sides.append(side)
        if len(sides) >= CHART_CONFLICT_MIN_SIDES:
            chart_series.append(parameter)
            continue
        if not sides:
            add(f"{parameter}: flagged as a {kind} conflict in the source report.")
            continue
        shown = sides[:MAX_CONFLICT_SIDES]
        hidden = len(sides) - len(shown)
        listing = " vs ".join(shown)
        if hidden:
            listing += f" and {hidden} more"
        add(
            f"{parameter}: the source report prints conflicting {kind}s "
            f"({listing}). Every side is shown; none was selected."
        )

    if chart_series:
        names = ", ".join(dict.fromkeys(chart_series))
        add(
            f"Chart/analysis data series ({names}) report dozens of plotted "
            "values, not a specification conflict; they are omitted here. See "
            "the source figures."
        )

    for warning in _as_texts(warnings):
        add(f"Extraction note: {warning}")

    return notes


# --------------------------------------------------------------------------
# PDF rendering
# --------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "EsTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceAfter=1,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "EsSubtitle",
            parent=base["Normal"],
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#1e5a86"),
        ),
        "heading": ParagraphStyle(
            "EsHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12.5,
            spaceBefore=7,
            spaceAfter=3,
            textColor=colors.HexColor("#14456b"),
        ),
        "body": ParagraphStyle(
            "EsBody", parent=base["BodyText"], fontSize=8.4, leading=11, spaceAfter=3
        ),
        "note": ParagraphStyle(
            "EsNote", parent=base["BodyText"], fontSize=7.6, leading=9.6, leftIndent=8
        ),
        "footnote": ParagraphStyle(
            "EsFootnote",
            parent=base["BodyText"],
            fontSize=6.8,
            leading=8.4,
            textColor=colors.HexColor("#555555"),
        ),
        "cell": ParagraphStyle("EsCell", parent=base["Normal"], fontSize=7.6, leading=9.4),
        "cellHead": ParagraphStyle(
            "EsCellHead",
            parent=base["Normal"],
            fontSize=7.6,
            leading=9.4,
            fontName="Helvetica-Bold",
        ),
        "cellMissing": ParagraphStyle(
            "EsCellMissing",
            parent=base["Normal"],
            fontSize=7.6,
            leading=9.4,
            textColor=colors.HexColor("#8a6d00"),
        ),
        "cellFlag": ParagraphStyle(
            "EsCellFlag",
            parent=base["Normal"],
            fontSize=7.6,
            leading=9.4,
            textColor=colors.HexColor("#a00000"),
        ),
    }


_GRID = TableStyle(
    [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4cf")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]
)


def _cell(text: Any, style: ParagraphStyle) -> Paragraph:
    """Every value reaches the page through escape(): source text is arbitrary."""
    return Paragraph(escape(_clean(text)), style)


def _rows_table(rows: Sequence[Row], styles: dict[str, ParagraphStyle]) -> Table:
    body: list[list[Paragraph]] = [
        [
            _cell("Item", styles["cellHead"]),
            _cell("Value", styles["cellHead"]),
            _cell("Source", styles["cellHead"]),
        ]
    ]
    for row in rows:
        if row.status == _STATUS_MISSING:
            style = styles["cellMissing"]
        elif row.status == _STATUS_CONFLICT:
            style = styles["cellFlag"]
        else:
            style = styles["cell"]
        body.append(
            [
                _cell(row.label, styles["cell"]),
                _cell(row.resolved.text, style),
                _cell(row.resolved.citation, styles["cell"]),
            ]
        )
    table = Table(
        body,
        colWidths=[_USABLE_WIDTH * 0.30, _USABLE_WIDTH * 0.38, _USABLE_WIDTH * 0.32],
        repeatRows=1,
    )
    table.setStyle(_GRID)
    return table


def _load_summary_table(
    cells: Sequence[Sequence[str]], styles: dict[str, ParagraphStyle]
) -> Table:
    rendered: list[list[Paragraph]] = []
    for row_index, row in enumerate(cells):
        style = styles["cellHead"] if row_index == 0 else styles["cell"]
        rendered.append(
            [
                _cell(value, styles["cellMissing"] if value == NOT_FOUND and row_index else style)
                for value in row
            ]
        )
    widths = [_USABLE_WIDTH * 0.16] + [_USABLE_WIDTH * 0.21] * (len(cells[0]) - 1)
    table = Table(rendered, colWidths=widths, repeatRows=1)
    table.setStyle(_GRID)
    return table


def _build_story(data: dict[str, Any]) -> list[Any]:
    styles = _styles()
    collected: list[str] = []

    project_records = _section(data, "project")
    title_subtitle = _project_name(data)

    story: list[Any] = [_cell("Executive Summary", styles["title"])]
    story.append(_cell(title_subtitle, styles["subtitle"]))
    story.append(Spacer(1, 3))

    # 1. Project Information
    project_rows = _project_rows(data)
    for row in project_rows:
        collected.extend(row.resolved.notes)
    story.append(_cell("Project Information", styles["heading"]))
    story.append(_rows_table(project_rows, styles))

    # 2. Pile Details
    pile_rows = _pile_rows(data)
    for row in pile_rows:
        collected.extend(row.resolved.notes)
    story.append(_cell("Pile Details", styles["heading"]))
    story.append(_rows_table(pile_rows, styles))

    # 3. Load Summary
    load_cells, load_footnotes, load_notes = _load_table(data)
    collected.extend(load_notes)
    story.append(_cell("Load Summary", styles["heading"]))
    story.append(_load_summary_table(load_cells, styles))
    if load_footnotes:
        story.append(Spacer(1, 1.5))
        story.append(_cell("Sources: " + ". ".join(load_footnotes), styles["footnote"]))

    # 4. Key Results
    result_rows, result_notes, governing = _result_rows(data)
    collected.extend(result_notes)
    story.append(_cell("Key Results", styles["heading"]))
    story.append(_cell(governing, styles["body"]))
    story.append(
        _cell(
            "Ratios above are the demand/capacity values printed in the source "
            "report. They are reproduced as printed and were not recalculated.",
            styles["footnote"],
        )
    )
    story.append(Spacer(1, 2))
    story.append(_rows_table(result_rows, styles))

    # 5. Notes and Flags
    notes = _collect_notes(
        data,
        collected,
        _as_records(data.get("conflicts")),
        _as_texts(data.get("warnings")),
    )
    story.append(_cell("Notes and Flags", styles["heading"]))
    if notes:
        for note in notes:
            story.append(_cell(f"- {note}", styles["note"]))
    else:
        story.append(
            _cell(
                "No missing, unclear, or conflicting values were identified in the "
                "extracted data.",
                styles["body"],
            )
        )

    return story


class _NumberedCanvas(pdfcanvas.Canvas):
    """Adds 'Page N of M' by deferring each page until the total is known."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pages: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self._draw_footer(total)
            super().showPage()
        super().save()

    def _draw_footer(self, total: int) -> None:
        self.saveState()
        self.setFont("Helvetica", 6.4)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawString(_MARGIN, 8 * mm, FOOTER_TEXT)
        self.setFont("Helvetica", 6.8)
        self.drawRightString(
            A4[0] - _MARGIN, 4.6 * mm, f"Page {self._pageNumber} of {total}"
        )
        self.restoreState()


def build_executive_summary_pdf(data: dict) -> bytes:
    """Render the Executive Summary PDF and return its bytes.

    ``data`` is the extraction payload, optionally carrying its ``key_points``
    block. Every part is optional: an empty dict yields a valid PDF in which
    every value reads "Not found" and every absent item is listed under
    "Notes and Flags". No model is loaded and no value is guessed.
    """
    if not isinstance(data, dict):
        raise ValueError("executive summary data must be a mapping")

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=12 * mm,
        bottomMargin=_BOTTOM_MARGIN,
        title="Executive Summary",
        author="engineering-ai",
        subject="Engineering evaluation aid - not a formal engineering approval",
    )
    document.build(
        _build_story(data),
        canvasmaker=_NumberedCanvas,
    )
    return buffer.getvalue()


def executive_summary_filename(data: dict) -> str:
    """Download name for the Executive Summary PDF.

    Derived from the extracted project name so the file is recognisable, and
    sanitised because it is written into an HTTP Content-Disposition header.
    Falls back to a constant name rather than ever returning something unsafe.
    """
    fallback = "Executive_Summary.pdf"
    if not isinstance(data, dict):
        return fallback

    name = _project_name(data)
    if name == NOT_FOUND:
        return fallback

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    safe = safe[:60].strip("._-")
    return f"{safe}_Executive_Summary.pdf" if safe else fallback