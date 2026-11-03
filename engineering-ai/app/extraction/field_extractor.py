"""Stage 2b: field extraction.

Turns Table objects (and simple label=value text lines) from Stage 2 into
structured records:

    {parameter, value, unit, source, context, raw}

Design rules (Hard Constraints):
- Fully deterministic. No LLM involved.
- Numbers are parsed with Python's float() from the printed text only --
  never inferred, never rounded, never converted.
- Nothing is invented: a cell that does not parse becomes a raw string
  value with unit=None, never a guess.
- The same parameter extracted from two different places is compared; any
  disagreement (different number, or same number with different units) is
  reported as a Conflict. Conflicting values are NEVER silently resolved --
  every original record is kept and the conflict is flagged for review.

Recognized table patterns:
1. label -> value pairs   (e.g. "Pile diameter O.D." | "7.63 in")
2. fixed-format engineering tables (headers carry the unit, each data row
   is one specimen/section/load case -- e.g. pile section, helix, load
   and capacity ratio tables).
3. HelixPile-style summary tables:
   - pile-section table  (Section, Fy, Fu, Dp, tP, Ap, I, ... Qult, Numbe)
   - helix table         (Plate, Dhel, Shel, Ahel, Qhel)
   - load/capacity ratio table (multi-row header carrying "LOAD/CAP")
   - "Helical pile analysis summary for all stages (Design values)"
   - "Helical pile analysis summary for all stages (Ultimate values)"
   Design and Ultimate are never conflated: the kind is carried in the
   parameter name and every value keeps its printed source.

Title-page label/value lines (Project, Company, Prepared by engineer,
File number, Time, Date, ...) are kept verbatim, string values included.

Raw model-input dumps (soil depth profiles, p-y/lateral profiles, plate
bearing results, load-settlement curves, load factor tables, ...) are
recognised model DATA, not engineering FIELDS, and are skipped entirely.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.extraction.document_loader import ExtractedDocument, Table, load_many

# A number: optional sign, digits with optional thousands commas, optional
# decimal part, optional exponent.  float() does the actual parsing/validation.
_NUMBER_RE = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$")

# Unit token attached to a number: letters, %, degree, *, /, ^, digits (for
# ft2, m3, kN*m, mm^2, %, degC, ...) and the hyphen inside compound units
# (k-ft, kip-ft, kN-m, lb-ft).  Stops at parentheses/whitespace.
_UNIT_RE = re.compile(r"^[A-Za-zµ°%][A-Za-z0-9µ°%*/^·\-]*")

# Number glued directly to its unit with no space: "12.7mm", "324mm", "62kn*m".
_NUMBER_UNIT_RE = re.compile(
    r"^(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·\-]*)$"
)

# A helix plate label like "24S24S24S" -- a code, never a number+unit.
_PLATE_LABEL_RE = re.compile(r"^\d+(?:S\d+)+$")

# HelixPile pipe/helix section label, e.g.
#   "7.63 x 0.375 x 80 ksi (3)24 x 0.5"
# "x" is a separator between the shaft O.D., wall, Fy, and the helix count /
# diameter / thickness -- it is never a unit.
_SECTION_SPEC_RE = re.compile(
    r"(?P<od>\d+(?:\.\d+)?)\s*[xX]\s*(?P<wall>\d+(?:\.\d+)?)\s*[xX]\s*"
    r"(?P<fy>\d+(?:\.\d+)?)\s*ksi\s*"
    r"\((?P<n>\d+)\)\s*(?P<dia>\d+(?:\.\d+)?)\s*[xX]\s*(?P<thick>\d+(?:\.\d+)?)"
)

# "Label = 450 MPa"  /  "Label: 7.63 in"  in free text (label and value follow
# straight after the start, e.g. the whole line is a defining sentence).
# Parentheses are allowed so "Yield strength (Fy): 345 MPa" keeps its label.
_TEXT_PAIR_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9_ ./°^*+\-()]{0,60}?)"
    r"\s*[=:]\s*"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·\-]*)?"
    # number must be maximal, and ":23"/"-09"/".." tails mark dates/times.
    r"(?![0-9]|[:\\-][0-9])"
)

# "Steel grade: Fy = 450 MPa" -- a short identifier followed by "= <number>"
# embedded anywhere in a prose line.
_EMBEDDED_PAIR_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9_.°/]{0,10}?)\s*=\s*"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·\-]*)?"
    r"(?![0-9]|[:\\-][0-9])"  # maximal number, not a date/time tail
)  # noqa: W605

_VALUE_HEADERS = {"value", "values", "val"}
_UNIT_HEADERS = {"unit", "units", "uom"}
_PARAM_HEADERS = {"parameter", "parameters", "property", "properties",
                  "description", "item", "name", "label"}

# Section headings in free text: a short, standalone ALL-CAPS line (or a
# numbered heading like "1.0 LOADS").  Every field extracted after such a
# line is tied to the section it appeared in, so a value is never a bare
# word -- it always carries "EXECUTIVE SUMMARY", "LOADS", "RESULTS", ...
_SECTION_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 ().&/_\-\x27]{2,45}"
    r"|[0-9]{1,2}[.)]\s*[A-Z][A-Z0-9 ().&/_\-\x27]{2,40})$"
)


def _section_heading(line: str) -> str:
    """Return the section heading a standalone line declares, else ''.

    A heading is short, has no ':'/'=' (so 'PROJECT: ...' or 'Fy = 450 MPa'
    can never be mistaken for one), and sports a genuine all-caps title.
    """
    s = line.strip()
    if not s or len(s) > 60 or ":" in s or "=" in s:
        return ""
    if not re.search(r"[A-Za-z]{4,}", s):
        return ""
    if _SECTION_HEADING_RE.match(s):
        return s
    return ""


class UnparseableValueError(ValueError):
    """Raised when a cell that must be numeric cannot be parsed by float()."""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class ExtractedField:
    """One extracted parameter with full traceability.

    value is a float when the printed text parsed as a number (via float()),
    otherwise the exact printed string (unit may then be None).
    Never a bare, untraceable number: source/location always travel with it.
    """

    parameter: str
    value: float | str | None
    unit: str | None
    source: str  # file name
    location: str  # e.g. "page 7, table 1", "row 3", "line 1..14"
    raw: str  # exact printed text the value came from
    context: str = ""  # row label (pile id, load case, ...) when present
    section: str = ""  # report section heading the value came from, if any
    #: where the unit came from: "value" (printed with the number),
    #: "unit_row" (a separate units row), "from_header" (the column header),
    #: "invalid" (a unit token was printed but is not a recognized unit), or
    #: "none" (no unit was found for this value).
    unit_status: str = ""

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "location": self.location,
            "context": self.context,
            "section": self.section,
            "raw": self.raw,
            "unit_status": self.unit_status,
        }


@dataclass
class Conflict:
    """The same parameter extracted from two places with disagreeing values.

    Never resolved automatically -- both sides are kept, flagged for review.
    """

    parameter: str
    kind: str  # "value" (numbers differ) or "unit" (units differ)
    fields: list[ExtractedField]

    def describe(self) -> str:
        sides = "; ".join(
            f"{f.raw!r} ({f.unit or 'no unit'} @ {f.source}, {f.location})"
            for f in self.fields
        )
        return f"{self.kind} conflict for {self.parameter!r}: {sides}"

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "kind": self.kind,
            "fields": [f.to_dict() for f in self.fields],
            "detail": self.describe(),
        }


@dataclass
class ExtractionResult:
    """All fields extracted from one or more documents, plus conflicts."""

    fields: list[ExtractedField] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": [f.to_dict() for f in self.fields],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Number / unit parsing (deterministic, float() only)
# --------------------------------------------------------------------------


def _try_number(text: str) -> float | None:
    """Parse a printed number with float(). Return None if it is not one."""
    candidate = text.strip()
    if not candidate or not _NUMBER_RE.match(candidate):
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


#: Tokens that appear in a HelixPile-style units row but are labels, not units
#: ("Name", "Number", "-").  Used only to decide whether a row is a units row.
_UNIT_ROW_LABELS = frozenset({
    "name", "number", "no", "value", "values", "val", "unit", "units",
    "id", "item", "label", "type", "index", "count", "total", "ratio",
    "critical", "active", "yes", "none", "n/a", "-",
})

#: Recognized engineering units.  A separate units row is only treated as one
#: when its cells name these tokens (so a row of spec text is never swallowed).
_KNOWN_UNITS = frozenset({
    # length
    "in", "inch", "inches", "ft", "foot", "feet", "mm", "cm", "m", "km",
    "mil", "mils", "yd", "yds",
    # force
    "n", "kn", "k", "kip", "kips", "lb", "lbs", "lbf", "kg", "g", "ton",
    "tons", "tonne", "tonnes", "mn",
    # stress / pressure / density
    "ksi", "psi", "ksf", "psf", "pcf", "pci", "plf", "klf", "pa", "kpa",
    "mpa", "gpa", "bar",
    # area / second moment / volume
    "in2", "in3", "in4", "mm2", "mm3", "mm4", "cm2", "cm3", "cm4",
    "m2", "m3", "m4", "ft2", "ft3", "ft4", "gal", "l",
    # moment / energy
    "k-ft", "kft", "kip-ft", "kipft", "kn-m", "knm", "kn*m", "nm",
    "ft-kip", "ft-lb", "ftlb", "lb-ft", "lbft", "in-lb", "inlb",
    # angle / time / misc
    "deg", "degree", "degrees", "rad", "radian", "radians", "\u00b0", "%",
    "percent", "pct", "yr", "yrs", "year", "years", "day", "days", "hr",
    "hrs", "hour", "hours", "week", "weeks", "month", "months",
})


def _norm_unit(text: str) -> str:
    """Case/space-insensitive unit key ("kN * m" -> "kn*m", "in^2" -> "in2")."""
    key = re.sub(r"\s+", "", (text or "").strip().lower()).rstrip(".")
    return key.replace("^", "")


#: Cells that explicitly mean "no value here". They are kept as raw text like
#: any other non-numeric cell, but they are expected in these reports, so they
#: are not worth an extraction warning (an unexpected "TBD" still is).
_NON_VALUE_PLACEHOLDERS = frozenset({
    "n/a", "na", "n.a.", "-", "--", "---", "\u2013", "\u2014", ".", "none",
})


#: Strict whitelist of units allowed to appear in the clean engineering
#: summary.  Deliberately narrower than ``_KNOWN_UNITS``: recognition stays
#: broad (so a printed unit is never invented or silently dropped), but the
#: summary only ever repeats units on this list.
_SUMMARY_UNITS = frozenset({
    "in", "ft", "k", "ksi", "ksf", "psf", "pcf", "k-ft",
    "in2", "in3", "in4", "deg", "%", "years",
})


def is_valid_summary_unit(unit: str | None) -> bool:
    """Whether a unit may appear in the clean summary.

    ``None`` (a dimensionless / non-engineering value) is allowed; every other
    unit must be in the strict whitelist.
    """
    if unit is None:
        return True
    return _norm_unit(unit) in _SUMMARY_UNITS


def is_known_unit(unit: str | None) -> bool:
    """Whether ``unit`` is a recognized engineering unit.

    Broader than :func:`is_valid_summary_unit`: used where a printed unit must
    be accepted as real, without narrowing it to the clean-summary whitelist.
    """
    return unit is not None and _norm_unit(unit) in _KNOWN_UNITS


def _unit_token(cell: str) -> str | None:
    """The unit a table cell names, else None.

    "(ksi)" -> "ksi"; "[kN*m]" -> "kN*m"; "in" -> "in"; "Name" -> None.
    """
    text = (cell or "").strip()
    if not text:
        return None
    if (text[0] == "(" and text[-1] == ")") or (text[0] == "[" and text[-1] == "]"):
        text = text[1:-1].strip()
    if not text or _try_number(text) is not None:
        return None
    return text if _norm_unit(text) in _KNOWN_UNITS else None


def split_value_unit(cell: str) -> tuple[float | str | None, str | None, str]:
    """Split a printed cell into (value, unit, raw).

    - "7.63 in"   -> (7.63, "in", "7.63 in")
    - "450"       -> (450.0, None, "450")
    - "40 k-ft"   -> (40.0, "k-ft", "40 k-ft")
    - "F355NH"    -> ("F355NH", None, "F355NH")   # never forced to a number
    - "24S24S24S" -> ("24S24S24S", None, ...)     # plate labels are not units

    A trailing token is only kept as the unit when it is a recognized unit;
    "x" (a section separator) and unknown tokens are never units.
    """
    raw = cell.strip()
    if not raw:
        return None, None, raw

    # A helix plate label such as "24S24S24S" is a code, not a number+unit.
    if _PLATE_LABEL_RE.match(raw):
        return raw, None, raw

    parts = raw.split(None, 1)
    number = _try_number(parts[0])
    if number is not None:
        unit = None
        rest = parts[1].strip() if len(parts) > 1 else ""
        if rest:
            match = _UNIT_RE.match(rest)
            if match:
                token = match.group(0).rstrip(".-")
                if _norm_unit(token) in _KNOWN_UNITS:
                    unit = token or None
                    rest = rest[match.end():].strip()
        # Trailing commentary in parentheses is context, not unit.
        if rest.startswith("("):
            rest = rest[1:].split(")", 1)[0].strip()
        return number, unit, raw

    # Whole-cell number (no separating space), e.g. "12.7mm"
    no_space = _NUMBER_UNIT_RE.match(raw)
    if no_space:
        number = _try_number(no_space.group("num"))
        if number is not None:
            token = no_space.group("unit").rstrip(".-")
            unit = token if _norm_unit(token) in _KNOWN_UNITS else None
            return number, unit or None, raw
    number = _try_number(raw)
    if number is not None:
        return number, None, raw
    return raw, None, raw


#: A leading unit followed by a parenthesized abbreviation/note, e.g.
#: "kip (k)", "in (OD \u00d7 wall)", "ft (example only)".
_UNIT_LEAD_RE = re.compile(r"^(?P<unit>[A-Za-z%\u00b0][A-Za-z0-9%.*^\-]*)\s*\((?P<note>[^)]*)\)")


def _unit_from_cell(cell: str) -> str | None:
    """The recognized unit named by a dedicated unit/note cell, else None.

    "in" -> "in"; "kip (k)" -> "k"; "in (OD \\u00d7 wall)" -> "in";
    "ft (example only)" -> "ft"; "years" -> "years";
    "\u2014" / "Must be supplied/verified" / "assumed configuration" -> None.

    A parenthesized abbreviation wins when it is itself a unit ("kip (k)" ->
    "k", the printed short form), otherwise the leading token is used and any
    parenthesized note is ignored (it is commentary, not a unit).
    """
    text = (cell or "").strip()
    if not text:
        return None
    match = _UNIT_LEAD_RE.match(text)
    if match:
        note = match.group("note").strip()
        if _norm_unit(note) in _KNOWN_UNITS:
            return note
    for token in re.split(r"[\s,;/]+", text):
        token = token.strip().strip("()[]").rstrip(".-")
        if token and _norm_unit(token) in _KNOWN_UNITS:
            return token
    return None


def parse_number_strict(text: str) -> float:
    """Parse a cell that is required to be numeric; never invents a value."""
    value, _, raw = split_value_unit(text)
    if isinstance(value, float):
        return value
    raise UnparseableValueError(f"cannot parse numeric value from {raw!r}")


def parse_section(text: str) -> dict:
    """Parse a HelixPile pipe/helix section label (deterministic).

    "7.63 x 0.375 x 80 ksi (3)24 x 0.5" ->
        {od_in: 7.63, wall_in: 0.375, fy_ksi: 80.0,
         helix_count: 3, helix_dia_in: 24.0, helix_thickness_in: 0.5}

    The "x" is a separator, never a unit.  Returns ``{}`` when the label does
    not match -- nothing is guessed.
    """
    match = _SECTION_SPEC_RE.search(text or "")
    if not match:
        return {}
    return {
        "od_in": float(match.group("od")),
        "wall_in": float(match.group("wall")),
        "fy_ksi": float(match.group("fy")),
        "helix_count": int(match.group("n")),
        "helix_dia_in": float(match.group("dia")),
        "helix_thickness_in": float(match.group("thick")),
    }


_HEADER_TRAILING_UNIT_RE = re.compile(
    r"^(?P<name>.+?)[\s,;/]+(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·\-]*)\s*$"
)


def _header_unit(header: str) -> tuple[str, str | None]:
    """'Diameter (mm)' -> ('Diameter', 'mm');  'Fy' -> ('Fy', None).

    The unit may be parenthesised/bracketed ("Fy (ksi)") or trailing after a
    separator ("Yield strength, ksi").  A trailing word is only treated as a
    unit when it is a recognized unit token, so "Pile head condition" is left
    alone.
    """
    text = header.strip()
    match = re.search(r"\(([^)]*)\)\s*$", text)
    if match:
        name = text[: match.start()].strip() or text
        unit = match.group(1).strip() or None
        return name, unit
    match = re.search(r"\[([^\]]*)\]\s*$", text)
    if match:
        name = text[: match.start()].strip() or text
        unit = match.group(1).strip() or None
        return name, unit
    match = _HEADER_TRAILING_UNIT_RE.match(text)
    if match and _norm_unit(match.group("unit")) in _KNOWN_UNITS:
        return match.group("name").strip() or text, match.group("unit").strip()
    return text, None


def _split_unit_row(table: Table) -> tuple[list[str | None] | None, list[list[str]]]:
    """Detect a HelixPile-style units row directly under the header.

    Returns ``(units_by_column, data_rows)``.  ``units_by_column`` is ``None``
    when the first data row is not a units row, and ``data_rows`` never
    includes it when it is -- so "(in)"/"(ksi)" cells never become values and
    every column keeps the unit its own header row printed.
    """
    rows = list(table.rows)
    if not rows:
        return None, rows
    first = rows[0]
    if not first:
        return None, rows

    units: list[str | None] = []
    saw_unit = False
    for cell in first:
        text = (cell or "").strip()
        if not text:
            units.append(None)
            continue
        if _try_number(text) is not None:
            return None, rows  # a numeric cell -> this is a data row
        unit = _unit_token(text)
        if unit is not None:
            saw_unit = True
        elif (
            _norm_unit(text) not in _UNIT_ROW_LABELS
            and not re.fullmatch(r"[A-Za-z][A-Za-z .\-/']*", text)
        ):
            # A real spec/description cell (digits, symbols) -> data row.
            return None, rows
        units.append(unit)

    if not saw_unit:
        return None, rows
    return units, rows[1:]


def _is_number_cell(text: str) -> bool:
    return _try_number(text.split(None, 1)[0]) is not None if text.strip() else False


def _trailing_unit_token(raw: str) -> str | None:
    """The token printed right after the number in ``raw`` (may be invalid)."""
    parts = raw.split(None, 1)
    if len(parts) < 2:
        return None
    match = _UNIT_RE.match(parts[1].strip())
    return match.group(0).rstrip(".-") if match else None


def _normalize(name: str) -> str:
    """Case/punctuation-insensitive key for conflict grouping (no fuzzy matching)."""
    return re.sub(r"[\s_\-./()]+", " ", name.strip().lower()).strip()


#: Printed labels that all denote the same parameter.  A prefix match folds the
#: p.3 "Torque rating (by Manufacturer) = 5.5 k-ft" and the p.8
#: "... (by Manufacturer) Tel = 197.07 k-ft" into one conflicting rating.
_CONFLICT_ALIASES = (
    ("torque rating", "torque_rating"),
    ("maximum torque", "torque_rating"),
    ("max torque", "torque_rating"),
)


def _canonical_parameter(normalized: str) -> str | None:
    for prefix, canonical in _CONFLICT_ALIASES:
        if normalized.startswith(prefix):
            return canonical
    return None


# --------------------------------------------------------------------------
# Table pattern 1: label -> value pairs
# --------------------------------------------------------------------------


def _looks_like_label_value_table(table: Table) -> bool:
    """True for a label | value [| unit/note] table with generic headers.

    Some reports (and hand-built exports) print the same label -> value
    columns under non-standard headers such as "Item", "Example value",
    "Illustrative input / result" or "Property".  Detect them structurally:
    a narrow table whose first column is mostly non-numeric labels, whose
    second column is mostly numbers, and which has exactly one numeric data
    column.  A table with two or more numeric columns (e.g. Stage | load |
    settlement) is an engineering grid, not a label/value list.
    """
    if not 2 <= len(table.headers) <= 4:
        return False
    # A value header that itself names a unit ("Value (kN)") is handled by the
    # engineering-grid path, which records that unit as "from_header".
    if _header_unit(table.headers[1])[1] is not None:
        return False
    rows = [r for r in table.rows if len(r) >= 2 and r[0].strip() and r[1].strip()]
    if len(rows) < 2:
        return False
    threshold = max(1, int(0.6 * len(rows)))
    labels = sum(1 for r in rows if not _is_number_cell(r[0]))
    if labels < threshold:
        return False
    numeric = sum(1 for r in rows if _is_number_cell(r[1]))
    if numeric < threshold:
        return False
    # Exactly one numeric column: a units / note column never counts, but a
    # second measured column turns the table back into a grid.
    numeric_columns = 0
    for column in range(1, len(table.headers)):
        if any(len(r) > column and _is_number_cell(r[column]) for r in rows):
            numeric_columns += 1
    return numeric_columns <= 1


def _looks_like_pair_table(table: Table) -> bool:
    """True for narrow label | value tables (2-3 columns), not wide grids."""
    if not table.headers:
        return False
    if _looks_like_label_value_table(table):
        return True
    header_names = [_normalize(h) for h in table.headers[:2]]
    if (
        len(table.headers) <= 3
        and header_names[0] in _PARAM_HEADERS
        and header_names[1] in _VALUE_HEADERS
    ):
        return True
    if len(table.headers) > 3:  # engineering grid (pile/load/ratio table)
        return False
    if len(table.headers) > 2 and not (
        header_names[0] in _PARAM_HEADERS and header_names[1] in _VALUE_HEADERS
    ):
        return False
    rows = [r for r in table.rows if len(r) >= 2 and r[0].strip() and r[1].strip()]
    if not rows:
        return False
    # A pair row is a non-numeric label followed by a value that parses as a
    # number -- never guess from a bare description cell like "TBD".
    hits = sum(1 for r in rows
               if not _is_number_cell(r[0]) and _is_number_cell(r[1]))
    return hits >= max(1, int(0.6 * len(rows)))


def _unit_column_index(headers: list[str]) -> int | None:
    for index, header in enumerate(headers):
        normalized = _normalize(header)
        if normalized in _UNIT_HEADERS or normalized.startswith("unit"):
            return index
    return None


def _extract_pair_table(table: Table, source: str, result: ExtractionResult) -> None:
    unit_col = _unit_column_index(table.headers)
    for row_index, row in enumerate(table.rows, start=2):  # row 1 = header
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue
        parameter = row[0].strip()
        if _is_number_cell(parameter) and len(row) >= 3:
            # first cell is an index, parameter is the middle label
            context, parameter = parameter, row[1].strip()
            value_cell = row[2]
        else:
            context, value_cell = "", row[1]
        if _normalize(parameter) in _PARAM_HEADERS:
            continue
        value, unit, raw = split_value_unit(value_cell)
        # A sibling "Unit" column carries the unit when the value cell has none.
        if unit is None and unit_col is not None and unit_col < len(row):
            unit = _unit_from_cell(row[unit_col])
        result.fields.append(
            ExtractedField(
                parameter=parameter,
                value=value,
                unit=unit,
                source=source,
                location=f"{table.location}, row {row_index}",
                raw=raw,
                context=context,
                unit_status="value" if unit else "none",
            )
        )


# --------------------------------------------------------------------------
# Table pattern 2: fixed-format engineering table
# --------------------------------------------------------------------------


def _extract_grid_table(table: Table, source: str, result: ExtractionResult) -> None:
    """One record per numeric cell: parameter = row label + header name."""
    if not table.headers:
        return
    parsed_headers = [_header_unit(h) for h in table.headers]
    row_units, data_rows = _split_unit_row(table)
    # The header occupies row 1; keep row numbers stable when a units row is
    # dropped from the data.
    offset = len(table.rows) - len(data_rows)
    for local_index, row in enumerate(data_rows):
        row_index = local_index + 2 + offset
        cells = list(row) + [""] * (len(table.headers) - len(row))
        context = ""
        first = cells[0].strip() if cells else ""
        start_col = 0
        if first and not _is_number_cell(first):
            context = first
            start_col = 1
        for col in range(start_col, len(parsed_headers)):
            cell = cells[col].strip() if col < len(cells) else ""
            if not cell or cell.startswith("="):
                if cell.startswith("="):
                    result.warnings.append(
                        f"{source}, {table.location}, row {row_index}, "
                        f"column {table.headers[col]!r}: formula {cell!r} has no "
                        "parsed value; skipped (never guessed)."
                    )
                continue
            raw_header = table.headers[col]
            name, header_unit = parsed_headers[col]
            value, cell_unit, raw = split_value_unit(cell)
            column_unit = (
                row_units[col] if row_units and col < len(row_units) else None
            )
            unit = cell_unit or column_unit or header_unit
            if cell_unit:
                unit_status = "value"
            elif column_unit:
                unit_status = "unit_row"
            elif header_unit:
                unit_status = "from_header"
            else:
                unit_status = "none"
                token = _trailing_unit_token(raw)
                if token and _norm_unit(token) not in _KNOWN_UNITS:
                    unit_status = "invalid"
            if isinstance(value, str) and header_unit:
                # Non-numeric cell under a numeric header: keep raw, flag it --
                # unless it is an explicit "not applicable" placeholder, or the
                # "header" is itself a bare number (a value misread as a header,
                # as in a two-column settings table), which names no parameter.
                if (
                    raw.strip().lower() not in _NON_VALUE_PLACEHOLDERS
                    and re.search(r"[A-Za-z]", name)
                ):
                    result.warnings.append(
                        f"{source}, {table.location}, row {row_index}: cell {raw!r} "
                        f"for {name!r} is not a number; kept as raw text."
                    )
            parameter = raw_header
            result.fields.append(
                ExtractedField(
                    parameter=parameter,
                    value=value,
                    unit=unit,
                    source=source,
                    location=f"{table.location}, row {row_index}",
                    raw=raw,
                    context=context,
                    unit_status=unit_status,
                )
            )


def _extract_table(table: Table, source: str, result: ExtractionResult) -> None:
    if _looks_like_pair_table(table):
        _extract_pair_table(table, source, result)
    else:
        _extract_grid_table(table, source, result)


# --------------------------------------------------------------------------
# Free text: "Fy = 450 MPa", "Design pile length: 18.0 m embedded depth"
# --------------------------------------------------------------------------

# Dates and times are information, not numeric fields.  They are masked out
# before matching so "SUBMISSION DATE: 2026-09-23" or "Revised 12:30 PM" can
# never be truncated into a bare number (e.g. 2026) -- that would be a
# silently invented value.
_DATE_TIME_RE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"        # 2026-09-23, 2026/09/23
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"          # 09/23/2026
    r"|\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?\b"  # 12:30 / 12:30 PM
)


def _mask_datetime(line: str) -> str:
    return _DATE_TIME_RE.sub(" ", line)


def _extract_text(document: ExtractedDocument, result: ExtractionResult) -> None:
    source = document.path.name
    section = ""  # current report section heading, carried across all chunks
    for chunk in document.text_chunks:
        for line in chunk.text.splitlines():
            line = line.strip()
            if not line:
                continue
            heading = _section_heading(line)
            if heading:
                section = heading
            masked = _mask_datetime(line)
            # A prose line can hold several values ("Fy = 450 MPa, tP = 12.7 mm"):
            # anchor the label=value pair, then keep scanning the remainder for
            # embedded "short = number" pairs. Never re-emit the same value.
            pos = 0
            while True:
                tail = masked[pos:]
                match = _TEXT_PAIR_RE.match(tail) if pos == 0 else None
                if not match:
                    match = _EMBEDDED_PAIR_RE.search(tail)
                if not match:
                    break
                parameter = match.group("label").strip(" .:;,-")
                if len(parameter) >= 2:
                    number = _try_number(match.group("num"))
                    if number is not None:
                        unit = match.group("unit")
                        unit_status = "value" if unit else "none"
                        if unit and _norm_unit(unit) not in _KNOWN_UNITS:
                            unit, unit_status = None, "invalid"
                        result.fields.append(
                            ExtractedField(
                                parameter=parameter,
                                value=number,
                                unit=unit,
                                source=source,
                                location=chunk.location,
                                raw=line,
                                section=section,
                                unit_status=unit_status,
                            )
                        )
                pos += match.end()


# --------------------------------------------------------------------------
# Conflict detection -- flag, never resolve
# --------------------------------------------------------------------------


def _values_agree(a: ExtractedField, b: ExtractedField) -> bool:
    if isinstance(a.value, float) and isinstance(b.value, float):
        return math.isclose(a.value, b.value, rel_tol=1e-12, abs_tol=0.0)
    return a.value == b.value


def find_conflicts(fields: list[ExtractedField]) -> list[Conflict]:
    """Group by (parameter, context) and flag disagreements.

    - numbers differ           -> Conflict(kind="value")
    - same number, other units -> Conflict(kind="unit")
    Every field is kept as-is; nothing is merged or averaged.
    """
    groups: dict[tuple[str, str], list[ExtractedField]] = {}
    display: dict[tuple[str, str], str] = {}
    for f in fields:
        parameter = (f.parameter or "").strip()
        # A plate label such as "24S24S24S" is a code, not a parameter.
        if _PLATE_LABEL_RE.match(parameter):
            continue
        norm = _normalize(f.parameter)
        canonical = _canonical_parameter(norm)
        if canonical is not None:
            # Rate ratings are the same parameter regardless of surrounding text.
            key = (canonical, "")
            display.setdefault(key, canonical)
        else:
            key = (norm, _normalize(f.context))
        groups.setdefault(key, []).append(f)

    conflicts: list[Conflict] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        units = {f.unit for f in members if f.unit is not None}
        numbers = [f for f in members if isinstance(f.value, float)]
        distinct_numbers = {
            f.value for f in numbers
        }
        name = display.get(key, members[0].parameter)
        if len(units) > 1:
            conflicts.append(
                Conflict(parameter=name, kind="unit", fields=list(members))
            )
        elif len(distinct_numbers) > 1:
            conflicts.append(
                Conflict(parameter=name, kind="value", fields=list(members))
            )
        elif len(numbers) < len(members):
            # At least one non-numeric rendition of the same parameter.
            seen = {repr(f.value) for f in members}
            if len(seen) > 1:
                conflicts.append(
                    Conflict(parameter=name, kind="value", fields=list(members))
                )
    conflicts.sort(key=lambda c: (_normalize(c.parameter), c.kind))
    return conflicts


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def extract_fields(documents: ExtractedDocument | list[ExtractedDocument]) -> ExtractionResult:
    """Extract structured fields from Stage-2 documents (deterministic)."""
    docs = [documents] if isinstance(documents, ExtractedDocument) else list(documents)
    result = ExtractionResult()
    for document in docs:
        source = document.path.name
        for table in document.tables:
            _extract_table(table, source, result)
        _extract_text(document, result)
        result.warnings.extend(f"{source}: {w}" for w in document.warnings)
    result.conflicts = find_conflicts(result.fields)
    return result


# --------------------------------------------------------------------------
# CLI: python -m app.extraction.field_extractor <path-or-dir> [...]
# --------------------------------------------------------------------------


def _print_result(result: ExtractionResult) -> None:
    print(f"fields    : {len(result.fields)}")
    for f in result.fields:
        unit = f" {f.unit}" if f.unit else ""
        ctx = f" [{f.context}]" if f.context else ""
        sec = f" <{f.section}>" if f.section else ""
        print(f"  - {f.parameter}{ctx}{sec}: {f.value}{unit}"
              f"   ({f.source}, {f.location}; raw={f.raw!r})")
    print(f"conflicts : {len(result.conflicts)}")
    for c in result.conflicts:
        print(f"  ! {c.describe()}")
    if result.warnings:
        print(f"warnings  : {len(result.warnings)}")
        for w in result.warnings[:10]:
            print(f"  - {w}")
        if len(result.warnings) > 10:
            print(f"  ... and {len(result.warnings) - 10} more")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m app.extraction.field_extractor <path-or-dir> [...]")
        return 2
    try:
        docs = load_many(args)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        print(f"!! {exc}")
        return 1
    result = extract_fields(docs)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
