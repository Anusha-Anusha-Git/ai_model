"""Stage 2c: key-point summary of an engineering report.

Turns the Stage 2b ExtractionResult (plus the Stage-2 documents for raw text)
into a structured, deterministic key-point summary:

    project   -- name, structure, location, client, submission date
    pile      -- pile dia, wall thickness, Fy, helix no, helix dia 1/2, ...
    loads     -- compression / tension / shear / moment per load case
    results   -- printed D/C ratios per load case
    evaluation-- deterministic verdicts on the printed ratios (no certification)
    references-- the report's references / codes & standards section

Design rules (Hard Constraints):
- Fully deterministic. No LLM involved. Everything traces to a source
  (file + location + raw printed text).
- Nothing is invented: a key point that is not found is reported as
  *missing*, never guessed.
- Conflicting renditions of the same value are both kept -- never silently
  resolved, averaged, or merged.
- The engineering evaluation only restates printed ratios against 1.0 and
  explicitly states it is not certified/approved engineering work.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.extraction.document_loader import ExtractedDocument, load_many
from app.extraction.field_extractor import ExtractionResult, extract_fields, split_value_unit

# a number, validated by float() (same shape as Stage 2b)
_NUMBER_RE = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def _compact(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _try_number(text: str) -> float | None:
    candidate = (text or "").strip()
    if not candidate or not _NUMBER_RE.match(candidate):
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class KeyPoint:
    """One structured key point with full traceability."""

    category: str  # "project" | "pile" | "loads" | "results"
    key: str       # canonical id, e.g. "pile_diameter" / "load_compression"
    label: str     # human label, e.g. "Pile diameter" / "Compression load"
    value: float | str | None
    unit: str | None
    context: str = ""  # load case ("ULS-1") or specimen ("HP1")
    source: str = ""
    location: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "context": self.context,
            "source": self.source,
            "location": self.location,
            "raw": self.raw,
        }


@dataclass
class Reference:
    index: int
    text: str
    source: str = ""
    location: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text,
                "source": self.source, "location": self.location}


@dataclass
class KeyPointsResult:
    project: list[KeyPoint] = field(default_factory=list)
    pile: list[KeyPoint] = field(default_factory=list)
    loads: list[KeyPoint] = field(default_factory=list)
    results: list[KeyPoint] = field(default_factory=list)
    evaluation: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "project": [p.to_dict() for p in self.project],
            "pile": [p.to_dict() for p in self.pile],
            "loads": [l_.to_dict() for l_ in self.loads],
            "results": [r.to_dict() for r in self.results],
            "evaluation": list(self.evaluation),
            "references": [r.to_dict() for r in self.references],
            "missing": list(self.missing),
        }


# --------------------------------------------------------------------------
# Field classification: parameter name -> canonical key point
# --------------------------------------------------------------------------


#: Table labels that HelixPile/DeepFND print as data cells: unit headers,
#: unit fragments, and row names. A real printed measurement always carries a
#: digit and is never one of these, so they are rejected as values.
_LABEL_VALUES = frozenset({
    "name", "number", "helix", "stage", "stage 0", "stage 1", "critical",
    "ratio", "structural", "geotech", "warning", "active", "yes", "no",
    "response", "elastic", "cylinder", "plate", "tip", "design", "ultimate",
    "value", "unit", "n/a", "none",
})


def _usable_value(value) -> bool:
    """False for a header label or unit fragment mistaken for a reading.

    The field extractor keeps every cell of an engineering table, including the
    unit-header row ("(ksi)", "(in)", "Cylinder", "Stage 0"). Those are labels,
    not measurements, and must never be promoted to a key-point value.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value or "").strip()
    if not text:
        return False
    if text[0] in "([":
        return False  # unit fragment lifted from a header row, e.g. "(ksi)"
    if not re.search(r"\d", text):
        return False  # a printed measurement always contains a digit
    return text.lower().strip(" .:") not in _LABEL_VALUES


#: Tokens marking a cell as a capacity or ultimate value. Capacities are never
#: reclassified as applied loads.
_CAPACITY_TOKENS = ("qult", "qhel", "qy", "cap", "capac")


def _is_capacity_parameter(n: str) -> bool:
    return any(tok in n for tok in _CAPACITY_TOKENS) or n.endswith("ult")


def _is_ratio_parameter(n: str) -> bool:
    if ("ratio" in n or "demand" in n or "utilization" in n or n.endswith("ur")
            or n in ("dc", "adc", "dcratio")):
        return True
    # HelixPile/DeepFND ratio columns: R/CAP, R/STR, R/GEO, LOAD/CAP.
    return n in ("rcap", "rstr", "rgeo") or n.startswith("loadcap")


#: Chart/drawing-data row labels, e.g. "Lateral soil data points along pile
#: Design section". These name a plotted point, not an applied load or a spec.
_DRAWING_PARAM_RE = re.compile(
    r"\bpoints?\s+along\b|\bdata\s+points?\b|\bresults?\s+points?\b"
    r"|\banalysis\s+results\b",
    re.I,
)


def _classify(parameter: str, context: str) -> tuple[str, str, str] | None:
    """Return (key, label, category) for a field, or None if not a key point."""
    # a trailing parenthesised/bracketed unit ("Diameter (mm)") is not part
    # of the parameter name
    stripped = re.sub(r"[\[\(].*$", "", parameter.strip())
    n = _compact(stripped)
    c = _compact(context)
    if not n:
        return None
    if _DRAWING_PARAM_RE.search(parameter):
        return None

    pile_like = "pile" in c or c.startswith("hp") or "pipe" in c

    # --- helix geometry (HelixPile prints "Dhel" / "Numbe") ---------------
    if n == "dhel" or (("helix" in n or "plate" in n) and "dia" in n):
        numbered = re.search(r"helixdia(?P<num>[1-5])", n)
        if numbered:
            num = numbered.group("num")
            return f"helix_dia_{num}", f"Helix diameter {num}", "pile"
        return "helix_dia", "Helix diameter", "pile"
    if n == "numbe" or (
        "helix" in n and any(tok in n for tok in ("no", "num", "count", "qty"))
    ):
        return "helix_count", "Number of helixes", "pile"

    # --- yield strength (HelixPile prints "Fy"; "Lat. Force Fy" is not Fy) --
    if n == "fy" or "yield" in n:
        return "yield_strength", "Yield strength (Fy)", "pile"

    # --- ratios (checked before bare "moment"/"shear" word matches) --------
    if _is_ratio_parameter(n):
        return "result_ratio", "D/C ratio", "results"

    # --- capacities are strength/capacity data, never applied loads --------
    if _is_capacity_parameter(n):
        return None

    # --- pile geometry ------------------------------------------------------
    if n in ("dp", "od", "outerdia", "outsidediameter", "diameterod"):
        return "pile_diameter", "Pile diameter", "pile"
    if "pile" in n and "dia" in n:
        return "pile_diameter", "Pile diameter", "pile"
    if n == "diameter" and pile_like:
        return "pile_diameter", "Pile diameter", "pile"
    # "Pile section: 324 mm OD" is a pile line, but a longer title that merely
    # mentions a pile section ("Helical pile section", or a "...points along
    # pile Design section" chart row) is not a diameter.
    if "pile" in n and "section" in n and len(n) <= 16:
        return "pile_diameter", "Pile diameter", "pile"
    if n == "tp" or (n == "t" and pile_like) or (
        "thick" in n and ("pile" in n or "wall" in n or "pipe" in n
                          or "section" in n or pile_like)
    ):
        return "pile_thickness", "Pile wall thickness", "pile"
    if ("pile" in n and "length" in n) or ("section" in n and "length" in n) or (
        "embed" in n
    ) or ((n in ("length", "l")) and pile_like):
        return "pile_length", "Pile length", "pile"

    # --- loads --------------------------------------------------------------
    if "compress" in n or n in ("axial", "axialload", "axialforce"):
        return "load_compression", "Compression load", "loads"
    if "tension" in n or "uplift" in n or "pullout" in n:
        return "load_tension", "Tension load", "loads"
    if "shear" in n or "lateral" in n:
        return "load_shear", "Shear load", "loads"
    if "moment" in n or "bending" in n:
        return "load_moment", "Moment", "loads"

    return None


def is_key_point_parameter(parameter: str, context: str = "") -> bool:
    """True when a parameter name classifies to a pile/load/result key point.

    The field extractor records a "conflict" for every parameter column whose
    values disagree, including the hundreds of coordinate/soil-profile columns
    of the report's chart tables. Those are drawing data, not engineering
    fields. The Executive Summary PDF uses this to keep only conflicts that
    name a value it reports.
    """
    if not isinstance(parameter, str):
        return False
    return _classify(parameter, context) is not None


_RATIO_CTX_RE = re.compile(
    r"^(?P<ctx>.*?\S)\s+(?:(?:capacity\s+)?(?:ratio|d/c)|demand[-\s]?capacity|utilization(?: ratio)?)$",
    re.I,
)
_GENERIC_CTX = {"capacity", "result", "results", "demand", "check", "utilization",
                "loadcase", "load", "governing", "max", "maximum", "worst"}
_CASE_CODES = {"ULS", "SLS", "STR", "SER", "EQ", "LSD", "LC"}


def _ratio_context(parameter: str) -> str:
    """Derive a load-case context from a ratio label like 'ULS-1 capacity ratio'."""
    m = _RATIO_CTX_RE.match(parameter.strip().lower())
    if not m:
        return ""
    ctx = m.group("ctx").strip(" .:;-")
    if not ctx or ctx in _GENERIC_CTX:
        return ""
    # a genuine load-case token contains a digit or is a known short code
    if not re.search(r"\d", ctx) and ctx.upper() not in _CASE_CODES:
        return ""
    return ctx.upper()


_RATIO_SUB_RE = re.compile(r"(compression|tension|shear|moment|bending|axial)", re.I)


def _ratio_subtype(parameter: str) -> str:
    m = _RATIO_SUB_RE.search(parameter)
    if not m:
        return ""
    word = m.group(1).lower()
    if "moment" in word or "bend" in word:
        return "Moment"
    if "shear" in word:
        return "Shear"
    if "tension" in word:
        return "Tension"
    if "compress" in word or "axial" in word:
        return "Compression"
    return ""


# --------------------------------------------------------------------------
# Field -> key points (piles, loads, results)
# --------------------------------------------------------------------------


def _fields_to_keypoints(fields) -> tuple[list[KeyPoint], list[KeyPoint], list[KeyPoint]]:
    pile: list[KeyPoint] = []
    loads: list[KeyPoint] = []
    results: list[KeyPoint] = []
    for f in fields:
        cls = _classify(f.parameter, f.context)
        if cls is None:
            continue
        if not _usable_value(f.value):
            continue
        key, label, category = cls
        if key == "result_ratio":
            context = _ratio_context(f.parameter) or f.context or f.section
            sub = _ratio_subtype(f.parameter)
            if sub:
                label = f"{sub} D/C ratio"
        else:
            context = f.context or f.section
        kp = KeyPoint(category=category, key=key, label=label, value=f.value,
                      unit=f.unit, context=context, source=f.source,
                      location=f.location, raw=f.raw)
        {"pile": pile, "loads": loads, "results": results}[category].append(kp)
    return pile, loads, results


# --------------------------------------------------------------------------
# Project metadata (free text: "CLIENT: ...", pair tables)
# --------------------------------------------------------------------------

_PROJECT_META = [
    ("project_name", "Project name",
     re.compile(r"^(?:project|name of project|project name)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("project_no", "Project no.",
     re.compile(r"^(?:project\s*(?:no\.?|number)|job\s*(?:no\.?|number)"
                r"|document\s*(?:no\.?|number))\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("structure", "Structure",
     re.compile(r"^(?:structure|project structure|sub[-\s]structure"
                r"|description\s+of\s+work)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("location", "Project location",
     re.compile(r"^(?:location|project location|site location|site)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("client", "Client",
     re.compile(r"^(?:client|owner|customer|company)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("prepared_by", "Prepared by",
     re.compile(r"^(?:prepared\s+by(?:\s+engineer)?|designed\s+by|checked\s+by"
                r"|engineer(?:\s+of\s+record)?)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
    ("date", "Date of submission",
     re.compile(r"^(?:submission\s+date|date\s+of\s+(?:submission|issue|construction)"
                r"|date\s+(?:submitted|prepared|issued)|issue\s+date"
                r"|date|time)\b\s*[:\-]\s*(?P<value>.+)$", re.I)),
]

# Software is not read from a "Software:" label: HelixPile/DeepFND title pages
# print the program on a bare line ("HelixPile 2026: Report Output"), and PDF
# text order is unreliable. A labelled line is only trusted when the separator
# is present and the value is short -- a long sentence after the word
# "program" is body prose, not a program name.
_SOFTWARE_LABELLED_RE = re.compile(
    r"^(?:software|program(?:\s+name)?|analysed\s+by|analyzed\s+by)\b"
    r"\s*[:\-]\s*(?P<value>[^.\n]{1,40})$", re.I)
_SOFTWARE_TOKEN_RE = re.compile(
    r"^(?P<value>(?:HelixPile|DeepFND|Deep\s*FND|GSPile|PLAXIS|intact)"
    r"\b[^:\n]{0,20})", re.I)

_META_LABEL = {
    "project": ("project_name", "Project name"),
    "projectname": ("project_name", "Project name"),
    "nameofproject": ("project_name", "Project name"),
    "projectno": ("project_no", "Project no."),
    "projectnumber": ("project_no", "Project no."),
    "jobno": ("project_no", "Project no."),
    "structure": ("structure", "Structure"),
    "projectstructure": ("structure", "Structure"),
    "substructure": ("structure", "Structure"),
    "descriptionofwork": ("structure", "Structure"),
    "location": ("location", "Project location"),
    "projectlocation": ("location", "Project location"),
    "sitelocation": ("location", "Project location"),
    "site": ("location", "Project location"),
    "client": ("client", "Client"),
    "owner": ("client", "Client"),
    "customer": ("client", "Client"),
    "company": ("client", "Client"),
    "preparedby": ("prepared_by", "Prepared by"),
    "preparedbyengineer": ("prepared_by", "Prepared by"),
    "designedby": ("prepared_by", "Prepared by"),
    "checkedby": ("prepared_by", "Prepared by"),
    "engineer": ("prepared_by", "Prepared by"),
    # A bare "Project"/"Company"/"Date" line, used when the value sits on the
    # following line (common in PDF title pages). Looked up in _META_LABEL too.
    "structure": ("structure", "Structure"),
    "location": ("location", "Project location"),
    "date": ("date", "Date of submission"),
    "time": ("date", "Date of submission"),
    "filename": ("project_no", "Project no."),
    "filenumber": ("project_no", "Project no."),
    "submissiondate": ("date", "Date of submission"),
    "dateofsubmission": ("date", "Date of submission"),
    "issuedate": ("date", "Date of submission"),
    "dateofissue": ("date", "Date of submission"),
    "date": ("date", "Date of submission"),
}


def _meta_from_label(label: str) -> tuple[str, str] | None:
    return _META_LABEL.get(_compact(label))


def _clean_meta_value(value: str) -> str:
    return value.strip().strip("\u2022\u2013\u2014:;-").strip()


def _is_bare_meta_label(line: str) -> tuple[str, str] | None:
    """True when a line is only a label, so the value is on the next line.

    A PDF title page often extracts as "Project" / "Baytown Station" on two
    consecutive lines, which a same-line pattern can never see.
    """
    text = line.strip().rstrip(":").rstrip()
    if len(text) > 40 or _try_number(text) is not None:
        return None
    return _meta_from_label(text)


def _extract_project_info(documents: list[ExtractedDocument]) -> list[KeyPoint]:
    found: list[KeyPoint] = []
    for doc in documents:
        source = doc.path.name
        for chunk in doc.text_chunks:
            lines = [line.strip() for line in chunk.text.splitlines()]
            for index, line in enumerate(lines):
                if not line:
                    continue
                for key, label, rx in _PROJECT_META:
                    m = rx.match(line)
                    if not m:
                        continue
                    value = _clean_meta_value(m.group("value"))
                    if not value:
                        continue
                    found.append(KeyPoint(
                        category="project", key=key, label=label,
                        value=value, unit=None, context="",
                        source=source, location=chunk.location, raw=line,
                    ))
                    break  # one meta label per line

                else:
                    # same-line pattern found nothing: is this a bare label whose
                    # value is on one of the next few lines?
                    matched = _is_bare_meta_label(line)
                    if not matched:
                        continue
                    key, label = matched
                    for ahead in lines[index + 1:index + 4]:
                        if not ahead or _is_bare_meta_label(ahead):
                            break
                        if len(ahead) > 80 or _try_number(ahead) is not None:
                            break
                        found.append(KeyPoint(
                            category="project", key=key, label=label,
                            value=_clean_meta_value(ahead), unit=None, context="",
                            source=source, location=chunk.location,
                            raw=f"{line} / {ahead}",
                        ))
                        break
        for table in doc.tables:
            for row_index, row in enumerate(table.rows, start=2):
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                matched = _meta_from_label(row[0])
                if not matched:
                    continue
                key, label = matched
                value = _clean_meta_value(row[1])
                if not value:
                    continue
                found.append(KeyPoint(
                    category="project", key=key, label=label, value=value,
                    unit=None, context="", source=source,
                    location=f"{table.location}, row {row_index}",
                    raw=f"{row[0].strip()} = {row[1].strip()}",
                ))

    # structure is often appended to the project name ("A -- B")
    if not any(p.key == "structure" for p in found):
        for p in found:
            if p.key != "project_name" or not isinstance(p.value, str):
                continue
            for sep in (" -- ", " - ", "\u2014", "/"):
                if sep in p.value:
                    name, _, structure = p.value.partition(sep)
                    name = _clean_meta_value(name)
                    structure = _clean_meta_value(structure)
                    if name and structure:
                        p.value = name
                        found.append(KeyPoint(
                            category="project", key="structure", label="Structure",
                            value=structure, unit=None, context="",
                            source=p.source, location=p.location, raw=p.raw,
                        ))
                    break
            if any(x.key == "structure" for x in found):
                break
    return found


# --------------------------------------------------------------------------
# Prose fallback: label-aware keyword -> number parsing on raw lines.
# Covers prose that the field parser cannot see deterministically yet, e.g.
#   "ULS-1: compression 980 kN, ..."      (value consumed by the case token)
#   "wall thickness 12.7 mm"              (label without '=' or ':')
#   "Yield strength (Fy): 345 MPa"        (parenthesised qualifier)
# Never guesses a number; duplicates of a captured field value collapse.
# --------------------------------------------------------------------------

_KW_VALUE = re.compile(
    r"(?P<kw>"
    r"axial\s+compression(?:load)?|compression(?:load)?"
    r"|tension(?:load)?|uplift|pullout"
    r"|shear(?:load)?|lateral(?:\s+load)?"
    r"|bending\s+moment|moment"
    r"|pile\s+wall\s+thickness|pile\s+thickness|wall\s+thickness"
    r"|design\s+pile\s+length|pile\s+length"
    r"|pile\s+diameter|pile\s+dia"
    r"|yield\s+strength|fy"
    r"|helix\s+(?:no\.?|number|count)|helix\s+dia\s*[1-5]"
    r"|helix\s+diameter|helix\s+dia(?!\s*[0-9])"
    r")\b"
    r"\s*(?:\([^)]*\))?\s*[=:]?\s*"
    r"(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>[A-Za-z\u00b5%\u00b0][A-Za-z0-9\u00b5%\u00b0*/\u00b7^]*)?",
    re.I,
)
_RATIO_KW = re.compile(
    r"(?P<sub>(?:compression|tension|shear|moment|bending|axial)\s+)?"
    r"(?P<kw>demand[-\s]?capacity|capacity\s*ratio|utilization(?: ratio)?"
    r"|d\s*/\s*c|d/?c|ratio)"
    r"\s*[=:]?\s*(?P<num>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\b",
    re.I,
)
_CASE_FROM_START = re.compile(r"^\s*(?P<case>[A-Za-z]{2,8}[-_\s]\d{1,3}|[A-Za-z]{2,8}\d{1,3})")
_CASE_PAREN = re.compile(r"\((?:load\s+)?case\s*[:\-]?\s*(?P<case>[A-Za-z0-9\-_\s]+?)\)", re.I)
_CASE_BARE = re.compile(r"\((?P<case>(?:ULS|SLS|SER|STR|EQ|LSD|LC)[-_\s]?\d*)\)", re.I)
_CASE_WORD = re.compile(
    r"\b(?:load\s+case|case|comb(?:ination)?)\s*(?:no\.?|#)?\s*[:=\-]?\s*"
    r"(?P<case>[A-Za-z0-9\-_]+)\b", re.I)
_CASE_CODE = re.compile(r"\b(?P<case>(?:ULS|SLS|SER|STR|EQ|LSD|LC)[-_\s]?\d*)\b", re.I)


def _line_load_case(line: str, head: int = 60) -> str:
    """Best-effort, deterministic load-case token for a prose line."""
    head = line[:head]
    m = _CASE_FROM_START.match(head)
    if m:
        cand = m.group("case").strip()
        if re.fullmatch(r"[A-Za-z]+[-_\s]?\d+", cand):
            return cand.upper()
    m = _CASE_PAREN.search(head)
    if m:
        cand = _clean_meta_value(m.group("case"))
        if cand:
            return cand.strip().upper()
    m = _CASE_BARE.search(head)
    if m:
        return m.group("case").strip().upper()
    m = _CASE_WORD.search(head)
    if m:
        return m.group("case").strip().upper()
    m = _CASE_CODE.search(head)
    if m:
        return m.group("case").strip().upper()
    return ""


def _prose_values(documents: list[ExtractedDocument]) -> list[KeyPoint]:
    out: list[KeyPoint] = []
    for doc in documents:
        source = doc.path.name
        for chunk in doc.text_chunks:
            for line in chunk.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                case = _line_load_case(line)
                for m in _KW_VALUE.finditer(line):
                    kw = m.group("kw")
                    cls = _classify(kw, "")
                    if cls is None:
                        continue
                    key, label, category = cls
                    value, unit, raw = split_value_unit(
                        f"{m.group('num')} {m.group('unit')}".strip()
                    )
                    if not isinstance(value, float):
                        continue
                    out.append(KeyPoint(
                        category=category, key=key, label=label, value=value,
                        unit=unit, context=case, source=source,
                        location=chunk.location, raw=line,
                    ))
    return out


def _prose_ratios(documents: list[ExtractedDocument]) -> list[KeyPoint]:
    out: list[KeyPoint] = []
    for doc in documents:
        source = doc.path.name
        for chunk in doc.text_chunks:
            for line in chunk.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                case = _line_load_case(line)
                for m in _RATIO_KW.finditer(line):
                    value = _try_number(m.group("num"))
                    if value is None:
                        continue
                    sub = _ratio_subtype(m.group("sub") or "")
                    label = f"{sub} D/C ratio" if sub else "D/C ratio"
                    out.append(KeyPoint(
                        category="results", key="result_ratio", label=label,
                        value=value, unit=None, context=case, source=source,
                        location=chunk.location, raw=line,
                    ))
    return out


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

_REFERENCE_HEADING = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited|references?\s*and\s*codes"
    r"|codes?\s*and\s*standards|standards?)\s*[:.]?\s*$", re.I)
_NEW_HEADING = re.compile(
    r"^\s*(?:executive\s+summary|design\s+(?:criteria|basis|input|assumptions)"
    r"|loads?\b|results?\b|conclusions?\b|appendix|introduction|project\s+description"
    r"|methodology)\b.*$", re.I)
_SOFT_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 .:/&()_\-]{0,30}$")


def _extract_references(documents: list[ExtractedDocument]) -> list[Reference]:
    refs: list[Reference] = []
    for doc in documents:
        source = doc.path.name
        lines: list[tuple[str, str]] = []
        for chunk in doc.text_chunks:
            for line in chunk.text.splitlines():
                lines.append((chunk.location, line.strip()))
        collecting = False
        blanks = 0
        for loc, line in lines:
            if not collecting:
                if _REFERENCE_HEADING.match(line):
                    collecting = True
                continue
            if not line:
                blanks += 1
                if blanks >= 2:
                    break
                continue
            blanks = 0
            if (_NEW_HEADING.match(line) or
                    (_SOFT_HEADING.match(line) and len(line) <= 34
                     and not re.match(r"^[\d\[\(]", line))):
                break
            if len(refs) >= 30:
                break
            refs.append(Reference(index=len(refs) + 1, text=line,
                                  source=source, location=loc))
    return refs


# --------------------------------------------------------------------------
# Missing keys + engineering evaluation
# --------------------------------------------------------------------------

_REQUIRED = [
    ("project_name", "Project name"),
    ("structure", "Structure"),
    ("location", "Project location"),
    ("client", "Client"),
    ("date", "Date of submission"),
    ("pile_diameter", "Pile diameter"),
    ("pile_thickness", "Pile wall thickness"),
    ("yield_strength", "Yield strength (Fy)"),
    ("helix", "Helix details"),
    ("pile_length", "Pile length"),
    ("load_compression", "Compression load"),
    ("load_tension", "Tension load"),
    ("load_shear", "Shear load"),
    ("load_moment", "Moment"),
    ("result_ratio", "D/C ratio"),
]


def _missing_keys(kp: KeyPointsResult) -> list[str]:
    present: set[str] = set()
    for section in (kp.project, kp.pile, kp.loads, kp.results):
        for p in section:
            present.add(p.key)
    helix_present = any(k in present for k in
                        ("helix_count", "helix_dia",
                         *(f"helix_dia_{i}" for i in range(1, 6))))
    missing = []
    for key, label in _REQUIRED:
        if key == "helix":
            if not helix_present:
                missing.append(label)
        elif key not in present:
            missing.append(label)
    return missing


def _build_evaluation(kp: KeyPointsResult) -> list[str]:
    lines: list[str] = []

    for r in kp.results:
        ctx = f" [{r.context}]" if r.context else ""
        if isinstance(r.value, float):
            if r.value <= 1.0:
                lines.append(f"{r.label}{ctx} = {r.value:.2f} <= 1.00 -> OK "
                             "(demand within capacity)")
            else:
                lines.append(f"{r.label}{ctx} = {r.value:.2f} > 1.00 -> OVER CAPACITY "
                             "(check governs)")
        else:
            lines.append(f"{r.label}{ctx} = {r.raw!r} cannot be evaluated "
                         "(value is not numeric)")

    load_ctx = {l_.context for l_ in kp.loads if l_.context}
    result_ctx = {r.context for r in kp.results if r.context}
    for ctx in sorted(load_ctx - result_ctx):
        lines.append(f"Load case {ctx!r} has applied loads but no D/C ratio "
                     "in the report.")

    if kp.results and all(
        isinstance(r.value, float) and r.value <= 1.0 for r in kp.results
    ):
        lines.append("All reported D/C ratios are within capacity (<= 1.0).")
    elif not kp.results:
        lines.append("No D/C ratio found in the report; capacity cannot be evaluated.")

    if not kp.references:
        lines.append("No references/standards section found.")

    if kp.missing:
        lines.append("Missing (not found, never invented): "
                     + ", ".join(kp.missing) + ".")

    lines.append("Deterministic comparison only; this is not certified or "
                 "approved engineering work.")
    return lines


# --------------------------------------------------------------------------
# Merge / dedupe helpers
# --------------------------------------------------------------------------


def _same_demand(a: KeyPoint, b: KeyPoint) -> bool:
    """Same canonical key with equal value and a compatible unit."""
    if a.key != b.key:
        return False
    if isinstance(a.value, float) and isinstance(b.value, float):
        if not math.isclose(a.value, b.value, rel_tol=1e-9, abs_tol=1e-9):
            return False
    elif a.value != b.value:
        return False
    if a.unit == b.unit:
        return True
    if not a.unit or not b.unit:
        return True
    return a.unit.startswith(b.unit) or b.unit.startswith(a.unit)


def _merge(fields: list[KeyPoint], prose: list[KeyPoint]) -> list[KeyPoint]:
    """Fields win (richer unit/provenance); prose fills gaps and adds load-case
    contexts the field parser could not see. Never duplicates the same demand."""
    out = list(fields)
    for p in prose:
        idx = None
        for i, f in enumerate(out):
            if _same_demand(f, p):
                idx = i
                break
        if idx is None:
            out.append(KeyPoint(
                category=p.category, key=p.key, label=p.label, value=p.value,
                unit=p.unit, context=p.context, source=p.source,
                location=p.location, raw=p.raw,
            ))
        else:
            f = out[idx]
            if p.context and not f.context:
                f.context = p.context
    return out


def _collapse_identical(points: list[KeyPoint]) -> list[KeyPoint]:
    """Collapse identical (key, value, unit) pile records, listing contexts
    (e.g. the same 12.7 mm wall across HP1/HP2)."""
    by_sig: dict[tuple, list[KeyPoint]] = {}
    for m in points:
        by_sig.setdefault((m.key, m.value, m.unit), []).append(m)
    out: list[KeyPoint] = []
    for members in by_sig.values():
        contexts = sorted({m.context for m in members if m.context})
        rep = members[0]
        out.append(KeyPoint(
            category=rep.category, key=rep.key, label=rep.label, value=rep.value,
            unit=rep.unit, context=", ".join(contexts), source=rep.source,
            location=rep.location, raw=rep.raw,
        ))
    return out


_CANONICAL_PILE_ORDER = ["pile_diameter", "pile_thickness", "yield_strength",
                         "helix_count", *[f"helix_dia_{i}" for i in range(1, 6)],
                         "helix_dia", "pile_length"]
_CANONICAL_LOAD_ORDER = ["load_compression", "load_tension", "load_shear", "load_moment"]


def _ordered(points: list[KeyPoint], order: list[str]) -> list[KeyPoint]:
    rank = {key: i for i, key in enumerate(order)}
    return sorted(points, key=lambda p: (rank.get(p.key, len(order)), p.context, p.label))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _extract_software(documents: list[ExtractedDocument]) -> list[KeyPoint]:
    """The program name, from a labelled line or a bare program identifier."""
    found: list[KeyPoint] = []
    for doc in documents:
        source = doc.path.name
        for chunk in doc.text_chunks:
            for line in chunk.text.splitlines():
                line = line.strip()
                if not line or len(line) > 70:
                    continue
                match = _SOFTWARE_LABELLED_RE.match(line) or _SOFTWARE_TOKEN_RE.match(line)
                if not match:
                    continue
                value = _clean_meta_value(match.group("value"))
                if not value or any(kp.value == value for kp in found):
                    continue
                if match.re is _SOFTWARE_TOKEN_RE and len(value.split()) > 3:
                    continue  # prose that merely starts with a program name
                found.append(KeyPoint(
                    category="project", key="software", label="Software",
                    value=value, unit=None, context="", source=source,
                    location=chunk.location, raw=line,
                ))
    return found


def _clean_fields(fields, docs: list[ExtractedDocument]):
    """Drop graph/analysis-table noise before key-point classification.

    The field extractor keeps every numeric cell, including the hundreds of
    coordinate rows of moment/shear/settlement chart tables. Those are drawing
    data, not engineering fields, and the report summary already discards them
    via ``summary.filter_noise_for_llm``. Reuse that exact filter so key points
    agree with the web UI. Imported lazily: ``summary`` imports this module.
    """
    try:
        from app.report.summary import filter_noise_for_llm
    except Exception:  # pragma: no cover - defensive: never lose all fields
        return list(fields)
    tables = [t for doc in docs for t in getattr(doc, "tables", []) or []]
    try:
        return list(filter_noise_for_llm(list(fields), tables))
    except Exception:  # pragma: no cover - defensive
        return list(fields)


def extract_key_points(result: ExtractionResult,
                       documents: ExtractedDocument | list[ExtractedDocument]) -> KeyPointsResult:
    """Structured key-point summary from an ExtractionResult + Stage-2 docs."""
    docs = [documents] if isinstance(documents, ExtractedDocument) else list(documents)
    kp = KeyPointsResult()

    kp.project = _extract_project_info(docs)
    if not any(p.key == "software" for p in kp.project):
        kp.project.extend(_extract_software(docs))

    pile, loads, results = _fields_to_keypoints(_clean_fields(result.fields, docs))
    prose = _prose_values(docs)

    loads = _merge(loads, [p for p in prose if p.category == "loads"])
    results = _merge(results, [p for p in prose if p.category == "results"])
    pile = _merge(pile, [p for p in prose if p.category == "pile"])

    kp.pile = _ordered(_collapse_identical(pile), _CANONICAL_PILE_ORDER)
    kp.loads = _ordered(loads, _CANONICAL_LOAD_ORDER)
    kp.results = _ordered(_merge(results, _prose_ratios(docs)), [])

    kp.references = _extract_references(docs)
    kp.missing = _missing_keys(kp)
    kp.evaluation = _build_evaluation(kp)
    return kp


# --------------------------------------------------------------------------
# CLI: python -m app.report.key_points <path-or-dir> [...]
# --------------------------------------------------------------------------


def _fmt(kp: KeyPoint) -> str:
    unit = f" {kp.unit}" if kp.unit else ""
    ctx = f"  [{kp.context}]" if kp.context else ""
    return f"{kp.label}{ctx}: {kp.value}{unit}"


def render_key_points(kp: KeyPointsResult) -> str:
    lines = ["KEY POINTS", "----------", ""]
    for title, section in (("PROJECT", kp.project),
                           ("PILE (EXECUTIVE SUMMARY)", kp.pile),
                           ("LOADS", kp.loads),
                           ("RESULTS (D/C RATIOS)", kp.results)):
        lines.append(title)
        if not section:
            lines.append("  (none found)")
        for p in section:
            src = f"   <- {p.source} ({p.location})" if p.source else ""
            lines.append(f"  - {_fmt(p)}{src}")
        lines.append("")
    lines.append("ENGINEERING EVALUATION")
    if not kp.evaluation:
        lines.append("  (none)")
    for note in kp.evaluation:
        lines.append(f"  - {note}")
    lines.append("")
    lines.append("REFERENCES")
    if not kp.references:
        lines.append("  (none found)")
    for ref in kp.references:
        lines.append(f"  [{ref.index}] {ref.text}")
    lines.append("")
    if kp.missing:
        lines.append("MISSING / NOT FOUND")
        for name in kp.missing:
            lines.append(f"  - {name}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m app.report.key_points <path-or-dir> [...]")
        return 2
    try:
        docs = load_many(args)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        print(f"!! {exc}")
        return 1
    result = extract_fields(docs)
    kp = extract_key_points(result, docs)
    print(render_key_points(kp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())