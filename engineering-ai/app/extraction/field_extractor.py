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
# ft2, m3, kN*m, mm^2, %, degC, ...).  Stops at parentheses/whitespace.
_UNIT_RE = re.compile(r"^[A-Za-zµ°%][A-Za-z0-9µ°%*/^·]*")

# Number glued directly to its unit with no space: "12.7mm", "324mm", "62kn*m".
_NUMBER_UNIT_RE = re.compile(
    r"^(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·]*)$"
)

# "Label = 450 MPa"  /  "Label: 7.63 in"  in free text (label and value follow
# straight after the start, e.g. the whole line is a defining sentence).
# Parentheses are allowed so "Yield strength (Fy): 345 MPa" keeps its label.
_TEXT_PAIR_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z0-9_ ./°^*+\-()]{0,60}?)"
    r"\s*[=:]\s*"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·]*)?"
    # number must be maximal, and ":23"/"-09"/".." tails mark dates/times.
    r"(?![0-9]|[:\\-][0-9])"
)

# "Steel grade: Fy = 450 MPa" -- a short identifier followed by "= <number>"
# embedded anywhere in a prose line.
_EMBEDDED_PAIR_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9_.°/]{0,10}?)\s*=\s*"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*(?P<unit>[A-Za-zµ°%][A-Za-z0-9µ°%*/^·]*)?"
    r"(?![0-9]|[:\\-][0-9])"  # maximal number, not a date/time tail
)  # noqa: W605

_VALUE_HEADERS = {"value", "values", "val"}
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


def split_value_unit(cell: str) -> tuple[float | str | None, str | None, str]:
    """Split a printed cell into (value, unit, raw).

    - "7.63 in"   -> (7.63, "in", "7.63 in")
    - "450"       -> (450.0, None, "450")
    - "F355NH"    -> ("F355NH", None, "F355NH")   # never forced to a number
    """
    raw = cell.strip()
    if not raw:
        return None, None, raw

    parts = raw.split(None, 1)
    number = _try_number(parts[0])
    if number is not None:
        unit = None
        rest = parts[1].strip() if len(parts) > 1 else ""
        if rest:
            match = _UNIT_RE.match(rest)
            if match:
                unit = match.group(0)
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
            return number, no_space.group("unit") or None, raw
    number = _try_number(raw)
    if number is not None:
        return number, None, raw
    return raw, None, raw


def parse_number_strict(text: str) -> float:
    """Parse a cell that is required to be numeric; never invents a value."""
    value, _, raw = split_value_unit(text)
    if isinstance(value, float):
        return value
    raise UnparseableValueError(f"cannot parse numeric value from {raw!r}")


def _header_unit(header: str) -> tuple[str, str | None]:
    """'Diameter (mm)' -> ('Diameter', 'mm');  'Fy' -> ('Fy', None)."""
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
    return text, None


def _is_number_cell(text: str) -> bool:
    return _try_number(text.split(None, 1)[0]) is not None if text.strip() else False


def _normalize(name: str) -> str:
    """Case/punctuation-insensitive key for conflict grouping (no fuzzy matching)."""
    return re.sub(r"[\s_\-./()]+", " ", name.strip().lower()).strip()


# --------------------------------------------------------------------------
# Table pattern 1: label -> value pairs
# --------------------------------------------------------------------------


def _looks_like_pair_table(table: Table) -> bool:
    """True for narrow label | value tables (2-3 columns), not wide grids."""
    if not table.headers:
        return False
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


def _extract_pair_table(table: Table, source: str, result: ExtractionResult) -> None:
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
        result.fields.append(
            ExtractedField(
                parameter=parameter,
                value=value,
                unit=unit,
                source=source,
                location=f"{table.location}, row {row_index}",
                raw=raw,
                context=context,
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
    for row_index, row in enumerate(table.rows, start=2):
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
            unit = cell_unit or header_unit
            if isinstance(value, str) and header_unit:
                # Non-numeric cell under a numeric header: keep raw, flag it.
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
                        result.fields.append(
                            ExtractedField(
                                parameter=parameter,
                                value=number,
                                unit=match.group("unit"),
                                source=source,
                                location=chunk.location,
                                raw=line,
                                section=section,
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
    for f in fields:
        key = (_normalize(f.parameter), _normalize(f.context))
        groups.setdefault(key, []).append(f)

    conflicts: list[Conflict] = []
    for (_param, _ctx), members in groups.items():
        if len(members) < 2:
            continue
        units = {f.unit for f in members if f.unit is not None}
        numbers = [f for f in members if isinstance(f.value, float)]
        distinct_numbers = {
            f.value for f in numbers
        }
        if len(units) > 1:
            conflicts.append(
                Conflict(parameter=members[0].parameter, kind="unit", fields=list(members))
            )
        elif len(distinct_numbers) > 1:
            conflicts.append(
                Conflict(parameter=members[0].parameter, kind="value", fields=list(members))
            )
        elif len(numbers) < len(members):
            # At least one non-numeric rendition of the same parameter.
            seen = {repr(f.value) for f in members}
            if len(seen) > 1:
                conflicts.append(
                    Conflict(
                        parameter=members[0].parameter, kind="value", fields=list(members)
                    )
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
