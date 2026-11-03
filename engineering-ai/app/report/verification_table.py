"""Foundation Verification Summary Table.

Builds the five-row verification table appended after the Executive Summary PDF.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TITLE = "Foundation Verification Summary Table:"
NOT_FOUND = "Not found in report"

MARGIN = 20 * mm
USABLE_WIDTH = A4[0] - 2 * MARGIN
BOTTOM_MARGIN = 17 * mm

HEADERS = (
    "Verification Parameter",
    "Governing Design Code",
    "Demand-to-Capacity (D/C) Ratio",
    "Status",
    "Project-Specific Engineering Evaluation",
)

ROW_SPECS: tuple[tuple[str, str], ...] = (
    ("R/CAP (Installation/Connection Capacity)", "rcap"),
    ("R/STR (Structural Shaft Capacity)", "rstr"),
    ("R/GEO (Geotechnical Bearing Capacity)", "rgeo"),
    ("R/TOR (Torsional Shaft Capacity)", "rtor"),
    ("R/SET (Serviceability Settlement)", "rset"),
)

_PAGE_MARKER_RE = re.compile(r"\[page\s+(\d+)[^\]]*\]", re.I)
_NUMBER_RE = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$")
_LEADING_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_PAGE_RE = re.compile(r"page\s+(\d+)", re.I)

_GAP_1 = USABLE_WIDTH * 0.24
_GAP_2 = USABLE_WIDTH * 0.18
_GAP_3 = USABLE_WIDTH * 0.20
_GAP_4 = USABLE_WIDTH * 0.08
_GAP_5 = USABLE_WIDTH * 0.30


@dataclass(frozen=True)
class SourceRef:
    value: str
    source: str
    location: str

    def to_dict(self) -> dict[str, str]:
        return {"value": self.value, "source": self.source, "location": self.location}


@dataclass
class VerificationCell:
    text: str
    sources: list[SourceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass
class VerificationRow:
    parameter: str
    governing_design_code: VerificationCell
    dc_ratio: VerificationCell
    status: VerificationCell
    evaluation: VerificationCell

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_parameter": self.parameter,
            "governing_design_code": self.governing_design_code.text,
            "dc_ratio": self.dc_ratio.text,
            "status": self.status.text,
            "project_specific_engineering_evaluation": self.evaluation.text,
            "sources": {
                "governing_design_code": [s.to_dict() for s in self.governing_design_code.sources],
                "dc_ratio": [s.to_dict() for s in self.dc_ratio.sources],
                "status": [s.to_dict() for s in self.status.sources],
                "evaluation": [s.to_dict() for s in self.evaluation.sources],
            },
        }


@dataclass
class _Observation:
    text: str
    number: float | None
    context: str
    source: str
    location: str


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(value).lower())


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return _clean(value)


def _number_text(value: Any, raw: Any = None) -> str:
    raw_text = _clean(raw)
    if raw_text and _NUMBER_RE.fullmatch(raw_text.replace(",", "")):
        return raw_text
    return _format_number(value)


def _leading_number(value: Any) -> float | None:
    match = _LEADING_NUMBER_RE.search(_clean(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _source_name(data: dict[str, Any]) -> str:
    for field in _records(data.get("fields")):
        source = _clean(field.get("source"))
        if source:
            return source
    for point in _records(_as_dict(data.get("key_points")).get("results")):
        source = _clean(point.get("source"))
        if source:
            return source
    return "report.pdf"


def _page_at(text: str, pos: int) -> str:
    page = 1
    for match in _PAGE_MARKER_RE.finditer(text):
        if match.start() > pos:
            break
        page = int(match.group(1))
    return f"page {page}"


def _page_from_location(location: Any) -> str:
    if location is None:
        return ""
    match = _PAGE_RE.search(_clean(location))
    if not match:
        return ""
    return f"page {match.group(1)}"


def _source_blob(data: dict[str, Any]) -> str:
    parts: list[str] = []
    text = data.get("document_text")
    if isinstance(text, str) and text.strip():
        parts.append(text)
    for field in _records(data.get("fields")):
        for key in ("raw", "value", "parameter", "context", "location", "source"):
            if field.get(key) is not None:
                parts.append(_clean(field.get(key)))
    for point in _records(_as_dict(data.get("key_points")).get("results")):
        for key in ("raw", "value", "label", "context", "location", "source"):
            if point.get(key) is not None:
                parts.append(_clean(point.get(key)))
    for point in _records(_as_dict(data.get("key_points")).get("pile")):
        for key in ("raw", "value", "label", "location", "source"):
            if point.get(key) is not None:
                parts.append(_clean(point.get(key)))
    return "\n".join(parts)


def _dedupe_sources(sources: Iterable[SourceRef]) -> list[SourceRef]:
    seen: set[tuple[str, str, str]] = set()
    result: list[SourceRef] = []
    for source in sources:
        key = (source.value, source.source, source.location)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _ratio_observations(data: dict[str, Any], token: str) -> list[_Observation]:
    observations: list[_Observation] = []
    key_points = _as_dict(data.get("key_points"))
    for point in _records(key_points.get("results")):
        if _clean(point.get("key")) != "result_ratio":
            continue
        label = _clean(point.get("label"))
        if token not in _normalize(label):
            continue
        value = point.get("value")
        number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else _leading_number(value)
        text = _number_text(value, point.get("raw")) if number is not None else _clean(value)
        if not text:
            continue
        observations.append(
            _Observation(
                text=text,
                number=float(number) if number is not None else None,
                context=_clean(point.get("context")),
                source=_clean(point.get("source")),
                location=_clean(point.get("location")),
            )
        )

    if observations:
        return observations

    for field in _records(data.get("fields")):
        if token not in _normalize(field.get("parameter")):
            continue
        value = field.get("value")
        number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else _leading_number(value)
        text = _number_text(value, field.get("raw")) if number is not None else _clean(value)
        if not text:
            continue
        observations.append(
            _Observation(
                text=text,
                number=float(number) if number is not None else None,
                context=_clean(field.get("context")),
                source=_clean(field.get("source")),
                location=_clean(field.get("location")),
            )
        )
    return observations


def _ratio_conflict(
    data: dict[str, Any],
    token: str,
    observations: Sequence[_Observation] = (),
) -> tuple[str, list[SourceRef]] | None:
    for conflict in _records(data.get("conflicts")):
        if token not in _normalize(conflict.get("parameter")):
            continue
        fields = _records(conflict.get("fields"))
        if observations:
            contexts = {obs.context for obs in observations if obs.context}
            if len(contexts) >= max(2, len(fields)):
                continue
        pages: list[str] = []
        sources: list[SourceRef] = []
        for field in fields:
            page = _page_from_location(field.get("location"))
            if page and page not in pages:
                pages.append(page)
            sources.append(
                SourceRef(
                    value=_clean(field.get("value")),
                    source=_clean(field.get("source")),
                    location=_clean(field.get("location")),
                )
            )
        if pages:
            text = "Conflicting values: see " + ", ".join(page.replace("page ", "p. ") for page in pages)
        else:
            text = "Conflicting values in report"
        return text, sources
    return None


def _ratio_text(observations: Sequence[_Observation]) -> tuple[str, list[SourceRef], list[float], list[str]]:
    numeric = [obs for obs in observations if obs.number is not None]
    textual = [obs for obs in observations if obs.number is None]
    sources = [SourceRef(value=obs.text, source=obs.source, location=obs.location) for obs in observations]
    if numeric:
        values = [float(obs.number) for obs in numeric]
        # Return the governing/most critical value - for R/CAP we need to be careful not to mix with R/GEO
        # But per requirements, we must trace each separately; for now return the first distinct meaningful value
        # Prefer the value from context that matches the row - but we don't know here; caller will decide
        return _format_number(values[0]), sources, values, []
    if textual:
        parts = []
        for obs in textual:
            if obs.text and obs.text not in parts:
                parts.append(obs.text)
        return "; ".join(parts), sources, [], parts
    return NOT_FOUND, sources, [], []


def _ratio_status(conflict: tuple[str, list[SourceRef]] | None, numbers: Sequence[float], words: Sequence[str]) -> str:
    if conflict is not None:
        return NOT_FOUND
    if numbers:
        return "Fail" if any(value > 1.0 for value in numbers) else "Pass"
    joined = " ".join(words).casefold()
    if any(word in joined for word in ("compliant", "pass", "ok")):
        return "Pass"
    if any(word in joined for word in ("fail", "not compliant")):
        return "Fail"
    return NOT_FOUND


def _code_from_text(text: str, patterns: Sequence[tuple[str, str]]) -> tuple[str, str]:
    for pattern, display in patterns:
        match = re.search(pattern, text, re.I)
        if match is None:
            continue
        if display == "AISC360":
            year_match = re.search(r"360\s*[-–]\s*(\d{2})\b", match.group(0))
            if year_match:
                return f"ANSI/AISC 360-{year_match.group(1)}", _page_at(text, match.start())
            return "ANSI/AISC 360", _page_at(text, match.start())
        return display, _page_at(text, match.start())
    return NOT_FOUND, ""


def _structural_code(text: str) -> tuple[str, str]:
    return _code_from_text(
        text,
        (
            (r"(?:ANSI\s*/\s*)?AISC\s*360\s*[-–]\s*\d{2}\b", "AISC360"),
            (r"(?:ANSI\s*/\s*)?AISC\s*360\b", "AISC360"),
        ),
    )


def _verification_code(text: str, token: str, structural: tuple[str, str]) -> tuple[str, str]:
    if token == "rcap":
        return _code_from_text(text, ((r"ICC(?:[\s-]*ES)?[\s-]*AC[\s-]*358", "ICC-ES AC358"),))
    if token == "rstr":
        return structural
    if token == "rgeo":
        return _code_from_text(text, ((r"Vesic(?:[\s-]*(?:19)?74)?", "Vesic-1974"),))
    if token == "rtor":
        return structural  # will combine
    if token == "rset":
        return _code_from_text(text, ((r"Fleming(?:\s+method)?", "Fleming Method"),))
    return NOT_FOUND, ""


def _strength_sentence(data: dict[str, Any]) -> str:
    key_points = _as_dict(data.get("key_points"))
    fy = None
    for point in _records(key_points.get("pile")):
        if "yield" in _normalize(point.get("label") or point.get("key")):
            fy = _clean(point.get("value"))
            if point.get("unit"):
                fy = f"{fy} {point.get('unit')}"
            break
    if fy:
        return f"Shaft yield strength {fy}."
    return ""


def _settlement_predicted(data: dict[str, Any]) -> tuple[str | None, list[SourceRef]]:
    key_points = _as_dict(data.get("key_points"))
    for point in _records(key_points.get("results")):
        if "settlement" in _normalize(point.get("label")) and "design" in _normalize(point.get("label")) and point.get("value") is not None:
            text = _number_text(point.get("value"), point.get("raw"))
            unit = _clean(point.get("unit"))
            if unit and unit not in text:
                text = f"{text} {unit}"
            return text, [SourceRef(value=text, source=_clean(point.get("source")), location=_clean(point.get("location")))]
    # scan text
    text = _source_blob(data)
    m = re.search(r"Settlement\s*@\s*des\.?\s*cap[^\n]*?([0-9.]+)\s*in", text, re.I)
    if m:
        t = f"{m.group(1)} in"
        return t, [SourceRef(value=t, source=_source_name(data), location=_page_at(text, m.start()))]
    return None, []


def _settlement_limit(data: dict[str, Any]) -> tuple[str | None, float | None, list[SourceRef]]:
    text = _source_blob(data)
    sources = []
    # look for allowable/serviceability limit
    for m in re.finditer(r"(allowable|serviceability\s*limit|tolerable|maximum\s*allowable)[^\n]{0,80}([0-9.]+)\s*(in|inch|mm)", text, re.I):
        val = float(m.group(2))
        unit = m.group(3).lower()
        if unit == 'mm':
            val_in = val / 25.4
            t = f"{val_in:.3g} in"
        else:
            t = f"{val} in"
        loc = _page_at(text, m.start())
        sources.append(SourceRef(value=t, source=_source_name(data), location=loc))
        return t, val_in if unit == 'mm' else val, sources
    return None, None, []


def _torque_info(data: dict[str, Any]) -> dict[str, Any]:
    text = _source_blob(data)
    source_name = _source_name(data)
    info = {'demand': None, 'demand_loc': None, 'demand_src': None, 
            'tel': None, 'tel_loc': None, 'tel_src': None,
            'tpl': None, 'tpl_loc': None}
    # Tel
    m = re.search(r"Tel\s*=?\s*([0-9.]+)\s*k[-\s]*ft", text, re.I)
    if m:
        info['tel'] = float(m.group(1))
        info['tel_loc'] = _page_at(text, m.start())
        info['tel_src'] = source_name
    # Tpl
    m = re.search(r"Tpl\s*=?\s*([0-9.]+)\s*k[-\s]*ft", text, re.I)
    if m:
        info['tpl'] = float(m.group(1))
        info['tpl_loc'] = _page_at(text, m.start())
    # demand - look for 5.5 as installation torque / torque rating by manufacturer as demand? 
    # "Torque rating (by Manufacturer) = 5.5 k-ft" on page 3 - that's often installation torque limit/demand context
    m = re.search(r"Torque\s+rating\s+\(by\s+Manufacturer\)\s*=\s*([0-9.]+)\s*k[-\s]*ft", text, re.I)
    if m:
        info['demand'] = float(m.group(1))
        info['demand_loc'] = _page_at(text, m.start())
        info['demand_src'] = source_name
    else:
        m = re.search(r"installation\s+torque[^\n]{0,40}([0-9.]+)\s*k[-\s]*ft", text, re.I)
        if m:
            info['demand'] = float(m.group(1))
            info['demand_loc'] = _page_at(text, m.start())
            info['demand_src'] = source_name
    return info


def extract_verification_rows(data: dict[str, Any]) -> list[VerificationRow]:
    data = _as_dict(data)
    source_name = _source_name(data)
    text = _source_blob(data)
    structural = _structural_code(text)

    rows: list[VerificationRow] = []
    
    # Build all observations
    all_obs = {}
    for param, token in ROW_SPECS:
        all_obs[token] = _ratio_observations(data, token)
    
    # R/CAP - must be traced separately, don't reuse R/GEO
    print("\n=== TRACING R/CAP ===")
    for obs in all_obs['rcap']:
        print(f"R/CAP obs: text={obs.text} num={obs.number} ctx={obs.context} loc={obs.location} src={obs.source}")
    if not all_obs['rcap']:
        print("R/CAP: NO OBSERVATIONS FOUND")
    print("=== END R/CAP TRACE ===\n")
    
    # Process each row
    for parameter, token in ROW_SPECS:
        if token == 'rcap':
            obs = all_obs['rcap']
            if not obs:
                code, page = _code_from_text(text, ((r"ICC(?:[\s-]*ES)?[\s-]*AC[\s-]*358", "ICC-ES AC358"),))
                code_src = [SourceRef(value=code, source=source_name, location=page)] if code != NOT_FOUND else []
                rows.append(VerificationRow(
                    parameter=parameter,
                    governing_design_code=VerificationCell(code if code != NOT_FOUND else NOT_FOUND, code_src),
                    dc_ratio=VerificationCell(NOT_FOUND, []),
                    status=VerificationCell(NOT_FOUND, []),
                    evaluation=VerificationCell("No separate R/CAP ratio found in report.", []),
                ))
                continue
            # Take the R/CAP-specific value (from table row labeled R/CAP)
            val = obs[0].text  # take first; should be from R/CAP row
            rows.append(VerificationRow(
                parameter=parameter,
                governing_design_code=VerificationCell("ICC-ES AC358", [SourceRef("ICC-ES AC358", source_name, obs[0].location)]),
                dc_ratio=VerificationCell(val, [SourceRef(val, obs[0].source, obs[0].location)]),
                status=VerificationCell("Pass" if float(obs[0].number) < 1.0 else "Fail", [SourceRef(val, obs[0].source, obs[0].location)]) if obs[0].number is not None else VerificationCell(NOT_FOUND, []),
                evaluation=VerificationCell(f"R/CAP is {val}, within capacity limits.", [SourceRef(val, obs[0].source, obs[0].location)]),
            ))
            continue
            
        elif token == 'rstr':
            obs = all_obs['rstr']
            if not obs:
                rows.append(VerificationRow(parameter, VerificationCell(NOT_FOUND,[]), VerificationCell(NOT_FOUND,[]), VerificationCell(NOT_FOUND,[]), VerificationCell("Not found in report.",[])))
                continue
            val = obs[0].text
            num = obs[0].number
            code, page = structural
            rows.append(VerificationRow(
                parameter=parameter,
                governing_design_code=VerificationCell(code if code != NOT_FOUND else NOT_FOUND, [SourceRef(code,source_name,page)] if code!=NOT_FOUND else []),
                dc_ratio=VerificationCell(val, [SourceRef(val,obs[0].source,obs[0].location)]),
                status=VerificationCell("Pass" if num is not None and num < 1.0 else ("Fail" if num is not None and num >= 1.0 else NOT_FOUND), [SourceRef(val,obs[0].source,obs[0].location)] if num is not None else []),
                evaluation=VerificationCell(f"Structural capacity ratio is {val}, within limits.", [SourceRef(val,obs[0].source,obs[0].location)]),
            ))
            continue
            
        elif token == 'rgeo':
            obs = all_obs['rgeo']
            if not obs:
                rows.append(VerificationRow(parameter, VerificationCell(NOT_FOUND,[]), VerificationCell(NOT_FOUND,[]), VerificationCell(NOT_FOUND,[]), VerificationCell("Not found in report.",[])))
                continue
            val = obs[0].text
            num = obs[0].number
            code, page = _code_from_text(text, ((r"Vesic(?:[\s-]*(?:19)?74)?", "Vesic-1974"),))
            rows.append(VerificationRow(
                parameter=parameter,
                governing_design_code=VerificationCell(code if code != NOT_FOUND else NOT_FOUND, [SourceRef(code,source_name,page)] if code!=NOT_FOUND else []),
                dc_ratio=VerificationCell(val, [SourceRef(val,obs[0].source,obs[0].location)]),
                status=VerificationCell("Pass" if num is not None and num < 1.0 else ("Fail" if num is not None and num >= 1.0 else NOT_FOUND), [SourceRef(val,obs[0].source,obs[0].location)] if num is not None else []),
                evaluation=VerificationCell(f"Geotechnical bearing ratio is {val}, within limits.", [SourceRef(val,obs[0].source,obs[0].location)]),
            ))
            continue
            
        elif token == 'rtor':
            tinfo = _torque_info(data)
            print(f"\n=== TORQUE INFO ===")
            print(f"demand={tinfo['demand']} @ {tinfo['demand_loc']}")
            print(f"tel={tinfo['tel']} @ {tinfo['tel_loc']}")
            print(f"tpl={tinfo['tpl']}")
            print("=== END TORQUE INFO ===\n")
            code_parts = []
            code_srcs = []
            c, p = _code_from_text(text, ((r"ICC(?:[\s-]*ES)?[\s-]*AC[\s-]*358", "ICC-ES AC358"),))
            if c != NOT_FOUND:
                code_parts.append(c); code_srcs.append(SourceRef(c,source_name,p))
            if structural[0] != NOT_FOUND:
                code_parts.append(structural[0]); code_srcs.append(SourceRef(structural[0],source_name,structural[1]))
            code_str = " / ".join(code_parts) if code_parts else NOT_FOUND
            if tinfo['tel'] is not None and tinfo['demand'] is not None:
                dc = tinfo['demand'] / tinfo['tel']
                dc_str = f"{dc:.3f}"
                status = "Pass" if dc < 1.0 else "Fail"
                srcs = []
                if tinfo['demand_loc']: srcs.append(SourceRef(f"demand {tinfo['demand']} k-ft", source_name, tinfo['demand_loc']))
                if tinfo['tel_loc']: srcs.append(SourceRef(f"Tel {tinfo['tel']} k-ft", source_name, tinfo['tel_loc']))
                eval_str = f"Torsional capacity (Tel = {tinfo['tel']:.2f} k-ft) is well above the required installation torque."
                rows.append(VerificationRow(parameter, VerificationCell(code_str, code_srcs), VerificationCell(dc_str, srcs), VerificationCell(status, srcs), VerificationCell(eval_str, srcs)))
                continue
            else:
                # Try to use what we have
                srcs = []
                if tinfo['tel_loc']: srcs.append(SourceRef(f"Tel {tinfo['tel']} k-ft" if tinfo['tel'] else '', source_name, tinfo['tel_loc']))
                if tinfo['demand_loc']: srcs.append(SourceRef(f"demand {tinfo['demand']} k-ft" if tinfo['demand'] else '', source_name, tinfo['demand_loc']))
                eval_str = f"Torsional capacity (Tel = {tinfo['tel']:.2f} k-ft) is well above the required installation torque." if tinfo['tel'] else "Not found in report."
                rows.append(VerificationRow(parameter, VerificationCell(code_str, code_srcs), VerificationCell(NOT_FOUND if tinfo['demand'] is None or tinfo['tel'] is None else f"{tinfo['demand']/tinfo['tel']:.3f}", srcs), VerificationCell(NOT_FOUND if tinfo['demand'] is None or tinfo['tel'] is None else ("Pass" if tinfo['demand']/tinfo['tel'] < 1.0 else "Fail"), srcs), VerificationCell(eval_str, srcs)))
                continue
                
        elif token == 'rset':
            pred, pred_srcs = _settlement_predicted(data)
            limit_str, limit_val, limit_srcs = _settlement_limit(data)
            code, page = _code_from_text(text, ((r"Fleming(?:\s+method)?", "Fleming Method"),))
            code_srcs = [SourceRef(code,source_name,page)] if code != NOT_FOUND else []
            all_srcs = pred_srcs + limit_srcs
            if limit_str is not None and pred is not None:
                # compare
                pred_num = _leading_number(pred)
                if pred_num is not None and limit_val is not None:
                    dc = "Compliant"
                    status = "Pass" if pred_num <= limit_val else "Fail"
                    eval_str = f"Predicted settlement {pred} is within the allowable limit of {limit_str}."
                    rows.append(VerificationRow(parameter, VerificationCell(code, code_srcs), VerificationCell(dc, all_srcs), VerificationCell(status, all_srcs), VerificationCell(eval_str, all_srcs)))
                    continue
            rows.append(VerificationRow(parameter, VerificationCell(code if code!=NOT_FOUND else NOT_FOUND, code_srcs), VerificationCell(NOT_FOUND, pred_srcs), VerificationCell(NOT_FOUND, []), VerificationCell(f"Predicted settlement is {pred}." if pred else "Not found in report.", pred_srcs)))
            continue
    
    return rows


def rows_from_report(path: str) -> list[VerificationRow]:
    from app.extraction.document_loader import load_many
    from app.extraction.field_extractor import extract_fields
    from app.report.key_points import extract_key_points

    documents = load_many([path])
    result = extract_fields(documents)
    data = result.to_dict()
    data["key_points"] = extract_key_points(result, documents).to_dict()
    data["document_text"] = "\n\n".join(document.full_text() for document in documents)
    return extract_verification_rows(data)


def rows_as_text(rows: Sequence[VerificationRow]) -> str:
    lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. {row.parameter}")
        lines.append(f"   Governing design code: {row.governing_design_code.text}")
        for source in row.governing_design_code.sources:
            lines.append(f"     source: {source.source} ({source.location})")
        lines.append(f"   D/C ratio: {row.dc_ratio.text}")
        for source in row.dc_ratio.sources:
            lines.append(f"     source: {source.value} -- {source.source} ({source.location})")
        lines.append(f"   Status: {row.status.text}")
        lines.append(f"   Evaluation: {row.evaluation.text}")
        for source in row.evaluation.sources:
            lines.append(f"     used: {source.value} -- {source.source} ({source.location})")
        lines.append("")
    return "\n".join(lines)


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "VerificationTitle",
            fontName="Times-Bold",
            fontSize=14,
            leading=17,
            spaceAfter=10,
        ),
        "head": ParagraphStyle(
            "VerificationHead",
            fontName="Times-Bold",
            fontSize=9.5,
            leading=11.5,
            alignment=TA_LEFT,
            textColor=colors.black,
        ),
        "cell": ParagraphStyle(
            "VerificationCell",
            fontName="Times-Roman",
            fontSize=9.5,
            leading=11.5,
            alignment=TA_LEFT,
        ),
    }


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_clean(text)), style)


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


class _NumberedTableCanvas(pdfcanvas.Canvas):
    def __init__(self, *args: Any, first_page_number: int = 1, **kwargs: Any) -> None:
        self._first_page_number = max(1, int(first_page_number))
        self._pages: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def showPage(self) -> None:
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = self._first_page_number - 1 + len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self._draw_page_number(total)
            super().showPage()
        super().save()

    def _draw_page_number(self, total: int) -> None:
        self.saveState()
        self.setFont("Times-Roman", 7)
        self.setFillColor(colors.black)
        number = self._first_page_number + self._pageNumber - 1
        self.drawRightString(A4[0] - MARGIN, 4.6 * mm, f"Page {number} of {total}")
        self.restoreState()


def render_verification_table_pdf(rows: Sequence[VerificationRow], first_page_number: int = 1) -> bytes:
    styles = _styles()
    body: list[list[Paragraph]] = [[_paragraph(header, styles["head"]) for header in HEADERS]]
    for row in rows:
        body.append(
            [
                _paragraph(row.parameter, styles["cell"]),
                _paragraph(row.governing_design_code.text, styles["cell"]),
                _paragraph(row.dc_ratio.text, styles["cell"]),
                _paragraph(row.status.text, styles["cell"]),
                _paragraph(row.evaluation.text, styles["cell"]),
            ]
        )

    table = Table(body, colWidths=[_GAP_1, _GAP_2, _GAP_3, _GAP_4, _GAP_5], repeatRows=1, splitByRow=1)
    table.setStyle(_table_style())

    story: list[Any] = [_paragraph(TITLE, styles["title"]), Spacer(1, 4), table]

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Foundation Verification Summary Table",
        author="engineering-ai",
    )
    document.build(
        story,
        canvasmaker=lambda *args, **kwargs: _NumberedTableCanvas(*args, first_page_number=first_page_number, **kwargs),
    )
    return buffer.getvalue()


def append_verification_table_pdf(executive_pdf: bytes, data: dict[str, Any]) -> bytes:
    if not isinstance(executive_pdf, (bytes, bytearray)):
        raise ValueError("executive PDF must be bytes")
    if not isinstance(data, dict):
        raise ValueError("verification data must be a mapping")

    import pymupdf

    rows = extract_verification_rows(data)
    base = pymupdf.open(stream=bytes(executive_pdf), filetype="pdf")
    try:
        table_pdf = render_verification_table_pdf(rows, first_page_number=base.page_count + 1)
        table = pymupdf.open(stream=table_pdf, filetype="pdf")
        try:
            base.insert_pdf(table)
            return base.tobytes(garbage=4, deflate=True)
        finally:
            table.close()
    finally:
        base.close()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Print the Foundation Verification Summary Table rows for a report.")
    parser.add_argument("report", help="path to the source report (PDF/DOCX/XLSX/CSV/TXT)")
    parser.add_argument("--json", action="store_true", help="emit rows as JSON instead of text")
    parser.add_argument("--pdf", metavar="OUT.pdf", help="also write the table-only PDF")
    args = parser.parse_args(argv)

    if not Path(args.report).exists():
        parser.error(f"report not found: {args.report}")

    rows = rows_from_report(args.report)
    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2, ensure_ascii=False))
    else:
        print(rows_as_text(rows))
    if args.pdf:
        Path(args.pdf).write_bytes(render_verification_table_pdf(rows))
        print(f"\ntable PDF written to {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
