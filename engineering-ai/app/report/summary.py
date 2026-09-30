"""Stage 6: raw extraction -> strict nine-section engineering summary.

The deterministic stages remove graph-coordinate noise and preserve source
traceability. The local model receives only filtered fields and must return the
fixed output contract defined in this module. Generated section structure and
numeric provenance are validated before the summary is returned.

Run from the repo root (engineering-ai/):

    python -m app.report.summary <path-or-dir> [...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from app.extraction.document_loader import ExtractedDocument, load_many
from app.extraction.field_extractor import ExtractedField, Table, extract_fields
from app.report.key_points import extract_key_points

# --------------------------------------------------------------------------
# System prompt (Senior Geotechnical Engineer -> Construction Manager)
# --------------------------------------------------------------------------

# A real HelixPile report yields ~4,100 extracted fields, ~1,100 of which
# survive the noise filter, and their JSON runs to ~330,000 characters -- far
# past the 32k context of the local model, which then returns "Not specified"
# for everything. The prompt is therefore budgeted: the document text (what a
# reader actually reads) is included, and the field set is cut down to the
# parameters the summary needs.
MAX_PROMPT_CHARS = 46_000
MAX_DOCUMENT_TEXT_CHARS = 18_000
MAX_RELEVANT_FIELDS = 400

MISSING_VALUE = "Not specified"
MAX_SUMMARY_TOKENS = 1800
MAX_FORMAT_ATTEMPTS = 2

# Appended to every generated report: the project's standing hard constraint.
_NOT_CERTIFIED = (
    "- **Status:** Deterministic extraction of printed values only. This is "
    "not certified or approved engineering work; verify against the source "
    "report and the site geotechnical data."
)

REQUIRED_SECTION_LABELS = {
    1: (
        "Project Name:",
        "Structure:",
        "Location:",
        "Client:",
        "Prepared By:",
        "Date:",
        "Software:",
    ),
    2: (
        "Type:",
        "O.D.:",
        "Thickness:",
        "Fy:",
        "# of Helices:",
        "Helix Diameters:",
        "Spacing:",
        "Tip Elevation:",
    ),
    3: ("Model:", "Friction Angle (Φ):", "Cohesion (c'):", "Unit Weight (γ):"),
    4: ("Stage 0", "Stage 1", "Axial:", "Shear:", "Moment:"),
    5: (
        "Governing Stage:",
        "Geo Comp Capacity:",
        "Geo Tension Capacity:",
        "Structural Capacity:",
        "R/GEO:",
        "R/STR:",
        "R/CAP:",
        "Governing Failure Mode:",
    ),
    6: (
        "Settlement at Design Load:",
        "Max Lateral Deflection:",
        "Max Internal Moment:",
        "Pile Head Condition:",
    ),
    7: ("Design Life:", "Corrosion Thickness Loss:"),
    8: (),
    9: (),
}

REQUIRED_SECTION_HEADINGS = {
    1: "1. PROJECT INFO:",
    2: "2. PILE SPECS:",
    3: "3. SOIL PARAMS:",
    4: "4. APPLIED LOADS:",
    5: "5. RESULTS:",
    6: "6. SETTLEMENT & LATERAL:",
    7: "7. DURABILITY:",
    8: "8. ENGINEERING EVALUATION:",
    9: "9. REFERENCES:",
}

EVALUATION_BULLET_PREFIXES = (
    "- Overall Adequacy / Governing Limit State:",
    "- Utilization:",
    "- Settlement:",
    "- Red Flags to Verify:",
)

SYSTEM_INSTRUCTIONS = f"""You are a Senior Geotechnical and Structural Engineer. Convert a noisy HelixPile, DeepFND, or similar foundation-report extraction into one concise engineering summary.

PROCESSING RULES:
1. Use only final design values and rows from sections or tables explicitly identified as "Analysis summary", "Analysis Summary", "Executive Summary", "Design Summary", or final design results.
2. Ignore graph-coordinate rows, repeated calculation blocks, equations, iteration logs, OCR fragments, and intermediate values. Final shear and moment values from short load-summary rows must be retained.
3. Prefer the governing or final-summary row over repeated narrative or calculation rows. Preserve explicitly printed values exactly; do not calculate, interpolate, average, or convert them.
4. Use only DESIGN capacities in the requested capacity fields. Never substitute an ULTIMATE capacity. If design capacity is absent, write "{MISSING_VALUE}".
5. Governing stage: use an explicitly printed governing stage; otherwise select the stage with the largest explicitly reported R/CAP in the final analysis-summary rows. If no valid comparison exists, write "{MISSING_VALUE}".
6. Overall adequacy is PASS only when the governing R/CAP is explicitly reported and less than 1.0. A ratio greater than or equal to 1.0 is FAIL. If no governing ratio is available, state FAIL — adequacy cannot be demonstrated from the reported ratios.
7. Preserve source units. If a printed number has no unit, write "value (unit {MISSING_VALUE})". Ratios are dimensionless and may be written with "(dimensionless)".
8. If a required value is absent, write exactly "{MISSING_VALUE}". If final values conflict and the governing value cannot be identified, write "{MISSING_VALUE}" and identify the conflict under Red Flags to Verify.
9. Infer Location only when the supplied source path contains an explicit place name. Do not treat generic folder names, usernames, or filenames as a project location.
10. References must cite the exact supplied source filename and page, sheet, table, or line location. Cite exact table column headers when no printed table title exists. Never invent a table name or page. If no source location is available, write "{MISSING_VALUE}".
11. Do not add a title, introduction, conclusion, disclaimer, Markdown heading, code fence, or extra section.
12. Never name a code, standard, or assumed soil parameter that is not printed in the source data. Do not write a specific friction angle, cohesion, unit weight, corrosion allowance, or design life unless that number appears in the source; otherwise write "{MISSING_VALUE}". "Verify against the geotechnical report" is acceptable only as generic advice and must not assert a value.
13. Section 1 project lines come from "project_title_block" first, then the field data. That block holds the printed title-page lines verbatim; a project name, structure, location, client, prepared-by, or date printed there is a source value, not an inference. If the block has no such line and the field data has none either, write "{MISSING_VALUE}" -- do not reconstruct a project name from a file path or a pile schedule.
14. "document_text" is the report itself, in reading order. Read it before answering. A title page, a heading, a note, or a value the field extractor did not turn into a structured field is still there and may be copied from it. Prefer a value you can see in "document_text" over a field you cannot place.
15. "fields" is a filtered subset, not the whole report. A value absent from "fields" is not evidence that the report omits it -- check "document_text" before writing "{MISSING_VALUE}", and cite the document_text location you used.

OUTPUT RULES:
- Output exactly sections 1 through 9 in the order shown below.
- Keep sections 1 through 7 and 9 on one line each.
- Section 8 must contain exactly four bullet lines.
- Combine overall adequacy and governing limit state in the first bullet.

### FIELD LOOKUP GUIDE ###
HelixPile/DeepFND print abbreviated column and row labels. Use these mappings to find each requested value. They are SEARCH HINTS ONLY -- never expected outputs, never defaults. If none of the listed labels is present in the source data, write "{MISSING_VALUE}".
- Project Name: "Project:", "Project Name:", or "Name of Project".
- Structure: "Structure:", or a structure named inside the project name.
- Location: "Location:"; otherwise "{MISSING_VALUE}".
- Client: "Company:", "Client:", "Owner:".
- Prepared By: "Prepared by engineer:", "Prepared by:".
- Date: "Time:", "Date:", "Submission date:".
- Software: "HelixPile", "DeepFND", or a printed program/version string.
- O.D.: "Dp", "Pile diameter O.D.", "Diameter".
- Thickness: initial "tP", "Pipe thickness"; after corrosion "T new", "effective pile thickness".
- Fy: "Fy", "Yield strength".
- # of Helices: "Numbe", or the helix-plate count in the pile-section table.
- Helix Diameters: "Dhel", "Plate dia", "Helix dia".
- Spacing: "Shel", "Helix spacing".
- Stage 0 axial: "Axial load" or "Factored load" on the Stage 0 row.
- Stage 1 axial: "Axial load" on the Stage 1 row (keep a printed negative sign as printed).
- Shear: "Shear force at pile head", "Lat. Force Fx", "Lateral load".
- Moment: "Moment force at pile head", "Mx.max", "M.xx".
- Geo capacity: "CAP. compressi GEO.", "Geotechnical capacity" (DESIGN rows only).
- Structural capacity: "Structural cap." (DESIGN rows only).
- R/GEO: "LOAD/CAP GEOTECH", "R/GEO".
- R/STR: "LOAD/CAP STRUCTURAL", "R/STR".
- R/CAP: "LOAD/CAP RATIO", "R/CAP".
- Settlement: "Settlement @des. cap.".
- Max lateral deflection: "δ.x", "x.max", "Lat. Defl.".
- Max internal moment: "Mx.max", "Max internal moment".
- Pile head condition: "Pile head (rotation)", "Top of pile".
- Design life: "Design life", "Service life".
- Corrosion loss: "Thickness loss Steel (in)", "Thickness loss".
- Friction angle and cohesion: only the printed "Phi"/"Friction angle" and "Cohesion"/"c'" soil values.

### SOURCE DATA ###
{{raw_extracted_data}}

### REQUIRED OUTPUT FORMAT ###
1. PROJECT INFO: Project Name: [value or {MISSING_VALUE}]; Structure: [value or {MISSING_VALUE}]; Location: [value or {MISSING_VALUE}]; Client: [value or {MISSING_VALUE}]; Prepared By: [value or {MISSING_VALUE}]; Date: [value or {MISSING_VALUE}]; Software: [value or {MISSING_VALUE}].
2. PILE SPECS: Type: [value or {MISSING_VALUE}]; O.D.: [value or {MISSING_VALUE}]; Thickness: initial [value or {MISSING_VALUE}], effective after corrosion [value or {MISSING_VALUE}]; Fy: [value or {MISSING_VALUE}]; # of Helices: [value or {MISSING_VALUE}]; Helix Diameters: [values or {MISSING_VALUE}]; Spacing: [value or {MISSING_VALUE}]; Tip Elevation: [value or {MISSING_VALUE}].
3. SOIL PARAMS: Model: [value or {MISSING_VALUE}]; Friction Angle (Φ): [value or {MISSING_VALUE}]; Cohesion (c'): [value or {MISSING_VALUE}]; Unit Weight (γ): [value or {MISSING_VALUE}].
4. APPLIED LOADS: Stage 0 (Axial: [value or {MISSING_VALUE}]; Shear: [value or {MISSING_VALUE}]; Moment: [value or {MISSING_VALUE}]); Stage 1 (Axial: [value or {MISSING_VALUE}]; Shear: [value or {MISSING_VALUE}]; Moment: [value or {MISSING_VALUE}]).
5. RESULTS: Governing Stage: [value or {MISSING_VALUE}]; Geo Comp Capacity: [value or {MISSING_VALUE}]; Geo Tension Capacity: [value or {MISSING_VALUE}]; Structural Capacity: [value or {MISSING_VALUE}]; R/GEO: [value or {MISSING_VALUE}]; R/STR: [value or {MISSING_VALUE}]; R/CAP: [value or {MISSING_VALUE}]; Governing Failure Mode: [value or {MISSING_VALUE}].
6. SETTLEMENT & LATERAL: Settlement at Design Load: [value or {MISSING_VALUE}]; Max Lateral Deflection: [value or {MISSING_VALUE}]; Max Internal Moment: [value or {MISSING_VALUE}]; Pile Head Condition: [value or {MISSING_VALUE}].
7. DURABILITY: Design Life: [value or {MISSING_VALUE}]; Corrosion Thickness Loss: [value or {MISSING_VALUE}].
8. ENGINEERING EVALUATION:
- Overall Adequacy / Governing Limit State: [PASS or FAIL] — [concise source-based reason].
- Utilization: [governing ratio and source-based headroom statement; do not calculate a new margin].
- Settlement: [state whether the reported settlement satisfies an explicitly reported acceptance criterion; otherwise state that it cannot be assessed].
- Red Flags to Verify: [missing, conflicting, or assumption-dependent inputs requiring verification].
9. REFERENCES: [exact source and page/table/line locations for governing values and maximum loads, or {MISSING_VALUE}].
"""

# --------------------------------------------------------------------------
# Noise filtering (deterministic, run before the LLM)
# --------------------------------------------------------------------------

# Parameter names that almost always belong to long coordinate/graph tables
# (elevation/moment/shear/slope depth profiles) or intermediate spring model
# internals -- not engineering parameters for a construction summary.
NOISE_KEYWORDS = (
    "el.",
    "slope",
    "spring pres.",
    "soil dist. loa",
    "spring r",
    "settlement y",
)
COORDINATE_KEYWORDS = ("depth", "elevation", "ordinate", "settlement y", "el.")
_GRAPH_HEADER_HINTS = ("el.", "moment", "shear", "slope", "settlement y",
                       "p-y", "spring r")
_GRAPH_MIN_ROWS = 10


def _numeric_context(context: str) -> bool:
    """True when the context is a bare coordinate (depth/elevation row label)."""
    if not context:
        return False
    return context.replace(".", "").replace("-", "").replace("+", "").isdigit()


def graph_table_locations(tables: list[Table] | None) -> set[str]:
    """Locations of wide, many-row coordinate/graph tables (drawing data)."""
    locations: set[str] = set()
    for table in tables or []:
        header = " ".join(table.headers).lower()
        if len(table.rows) > _GRAPH_MIN_ROWS and any(
            hint in header for hint in _GRAPH_HEADER_HINTS
        ):
            locations.add(table.location)
    return locations


def filter_noise_for_llm(
    fields: list[ExtractedField], tables: list[Table] | None = None
) -> list[ExtractedField]:
    """Removes graph-coordinate and intermediate-calculation noise.

    Fields from long graph tables and coordinate-only rows are removed. Final
    shear and moment fields from short load-summary rows remain available to the
    model.
    """
    graph_locations = graph_table_locations(tables)

    filtered: list[ExtractedField] = []
    for field in fields:
        if field.location in graph_locations or any(
            field.location.startswith(loc + ", row") for loc in graph_locations
        ):
            continue
        parameter = field.parameter.lower()
        if any(keyword in parameter for keyword in NOISE_KEYWORDS):
            continue
        if _numeric_context(field.context) and any(
            keyword in parameter for keyword in COORDINATE_KEYWORDS
        ):
            continue
        filtered.append(field)
    return filtered


# --------------------------------------------------------------------------
# Prompt building (deterministic)
# --------------------------------------------------------------------------


def _source_path_for(source: str, source_paths: list[str] | None) -> str:
    for source_path in source_paths or []:
        if Path(source_path).name == source:
            return source_path
    return source


def build_raw_data_text(
    fields: list[ExtractedField],
    source_paths: list[str] | None = None,
) -> str:
    """Filtered fields rendered as the JSON injected into the system prompt."""
    payload = [
        {
            "parameter": field.parameter,
            "value": field.value,
            "unit": field.unit,
            "context": field.context,
            "section": field.section,
            "source": field.source,
            "source_path": _source_path_for(field.source, source_paths),
            "location": field.location,
            "raw": field.raw,
        }
        for field in fields
    ]
    return json.dumps(payload, indent=2)


def _truncate(text: str, limit: int) -> str:
    """Cut text to a character budget at a line boundary, marking the cut."""
    if len(text) <= limit:
        return text
    kept = text[:limit]
    newline = kept.rfind("\n")
    if newline > limit * 0.5:
        kept = kept[:newline]
    return kept + f"\n[...truncated at {limit} characters...]"


def _summarizer_relevant_fields(fields: list[ExtractedField]) -> list[ExtractedField]:
    """Fields whose parameter matches a summary line, capped for the context.

    Everything else is an intermediate calculation or profile row the summary
    never cites, and sending it only pushes the real values out of the window.
    """
    keywords: set[str] = set()
    for searches in _SUMMARY_SEARCHES.values():
        keywords.update(param for param, _ in searches)
    for keywords_for_ratio in _RATIO_KEYWORDS.values():
        keywords.update(keywords_for_ratio)

    relevant = [
        field for field in fields
        if any(_smart_contains(field.parameter, k) for k in keywords)
    ]
    return relevant[:MAX_RELEVANT_FIELDS]


def build_prompt_payload(
    fields: list[ExtractedField],
    raw_count: int,
    source_paths: list[str] | None = None,
    table_references: list[dict] | None = None,
    project: list[dict] | None = None,
    document_text: str = "",
) -> dict:
    return {
        "total_raw_fields_extracted": raw_count,
        "source_paths": source_paths or sorted({field.source for field in fields}),
        "project_title_block": project or [],
        "document_text": _truncate(document_text, MAX_DOCUMENT_TEXT_CHARS),
        "table_references": table_references or [],
        "fields": json.loads(build_raw_data_text(fields, source_paths)),
    }


def build_summary_prompt(
    fields: list[ExtractedField],
    raw_count: int,
    source_paths: list[str] | None = None,
    table_references: list[dict] | None = None,
    project: list[dict] | None = None,
    document_text: str = "",
) -> str:
    """Build the strict engineering prompt with traceable source data."""
    payload = build_prompt_payload(
        fields,
        raw_count,
        source_paths=source_paths,
        table_references=table_references,
        project=project,
        document_text=document_text,
    )
    prompt = SYSTEM_INSTRUCTIONS.replace(
        "{raw_extracted_data}",
        json.dumps(payload, indent=2),
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        # Last resort: shrink the field set until the prompt fits the context.
        while fields and len(prompt) > MAX_PROMPT_CHARS:
            fields = fields[:int(len(fields) * 0.7) or 1]
            payload["fields"] = json.loads(build_raw_data_text(fields, source_paths))
            prompt = SYSTEM_INSTRUCTIONS.replace(
                "{raw_extracted_data}", json.dumps(payload, indent=2)
            )
    return prompt


def _section_matches(markdown: str) -> tuple[list[tuple[int, re.Match]], list[str]]:
    matches: list[tuple[int, re.Match]] = []
    errors: list[str] = []
    for number, heading in REQUIRED_SECTION_HEADINGS.items():
        found = list(re.finditer(rf"(?m)^{re.escape(heading)}.*$", markdown))
        if len(found) != 1:
            errors.append(
                f"section {number} must appear exactly once as '{heading}'"
            )
            continue
        matches.append((number, found[0]))
    return matches, errors


def validate_summary_format(markdown: str) -> list[str]:
    """Return output-contract errors for a generated engineering summary."""
    matches, errors = _section_matches(markdown)
    if len(matches) != len(REQUIRED_SECTION_HEADINGS):
        return errors

    ordered = [number for number, _ in matches]
    positions = [match.start() for _, match in matches]
    if ordered != sorted(ordered) or positions != sorted(positions):
        errors.append("sections must appear exactly once in numeric order")
        return errors

    for index, (number, match) in enumerate(matches):
        end = matches[index + 1][1].start() if index + 1 < len(matches) else len(markdown)
        section_text = markdown[match.start():end]
        for label in REQUIRED_SECTION_LABELS[number]:
            if label not in section_text:
                errors.append(f"section {number} is missing label '{label}'")

    if 8 in ordered:
        start = dict(matches)[8].end()
        end_evaluation = dict(matches)[9].start()
        evaluation_text = markdown[start:end_evaluation]
        bullet_lines = [
            line.strip() for line in evaluation_text.splitlines()
            if line.strip().startswith("- ")
        ]
        if len(bullet_lines) != len(EVALUATION_BULLET_PREFIXES):
            errors.append("section 8 must contain exactly four bullet lines")
        for prefix in EVALUATION_BULLET_PREFIXES:
            if sum(line.startswith(prefix) for line in bullet_lines) != 1:
                errors.append(f"section 8 is missing bullet '{prefix}'")
    return errors


_NUMBER_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
_ALLOWED_OUTPUT_NUMBERS = {str(number) for number in range(10)} | {"1.0"}


def validate_summary_values(markdown: str, source_payload: dict | str) -> list[str]:
    """Reject numeric values that do not occur in the supplied source data."""
    source_text = (
        source_payload if isinstance(source_payload, str)
        else json.dumps(source_payload)
    )
    source_numbers = {
        match.group(0).replace(",", "")
        for match in _NUMBER_TOKEN_PATTERN.finditer(source_text)
    }
    unknown: list[str] = []
    for match in _NUMBER_TOKEN_PATTERN.finditer(markdown):
        number = match.group(0).replace(",", "")
        if number in source_numbers or number in _ALLOWED_OUTPUT_NUMBERS:
            continue
        if number not in unknown:
            unknown.append(number)
    return [f"numeric value '{number}' is not present in source data" for number in unknown]


# --------------------------------------------------------------------------
# End-to-end summary (LLM step; model loaded lazily)
# --------------------------------------------------------------------------


def _build_repair_prompt(
    prompt: str,
    markdown: str,
    errors: list[str],
) -> str:
    return (
        f"{prompt}\n\n"
        "FORMAT VALIDATION FAILED\n"
        "Correct every listed error and return the complete nine-section report. "
        "Do not explain the correction.\n"
        f"ERRORS:\n- {'; '.join(errors)}\n\n"
        f"PREVIOUS RESPONSE:\n{markdown}"
    )


def generate_markdown_summary(
    documents: ExtractedDocument | list[ExtractedDocument],
) -> tuple[str, dict]:
    """Generate and validate the strict nine-section engineering summary."""
    docs = [documents] if isinstance(documents, ExtractedDocument) else list(documents)
    tables = [table for doc in docs for table in doc.tables]
    source_paths = [str(doc.path.resolve()) for doc in docs]
    table_references = [
        {
            "source": doc.path.name,
            "source_path": str(doc.path.resolve()),
            "location": table.location,
            "headers": table.headers,
        }
        for doc in docs
        for table in doc.tables
    ]

    result = extract_fields(docs)
    raw_fields = list(result.fields)
    clean_fields = filter_noise_for_llm(raw_fields, tables)
    # A title block holds no number, so Stage 2b never emits it as a field.
    # Stage 2c reads it from the document text; hand it to the model explicitly.
    project = [p.to_dict() for p in extract_key_points(result, docs).project]
    # Read the document as a person would: the model gets the report text, not
    # only the extracted numbers, so title pages, notes, and labels it was never
    # turned into fields for are still visible to it.
    document_text = "\n\n".join(doc.full_text() for doc in docs)
    relevant_fields = _summarizer_relevant_fields(clean_fields)
    payload = build_prompt_payload(
        relevant_fields,
        len(raw_fields),
        source_paths=source_paths,
        table_references=table_references,
        project=project,
        document_text=document_text,
    )
    prompt = build_summary_prompt(
        relevant_fields,
        len(raw_fields),
        source_paths=source_paths,
        table_references=table_references,
        project=project,
        document_text=document_text,
    )

    from app.models.llm import generate_response, load_model

    model, tokenizer = load_model(quiet=True)
    markdown = ""
    validation_errors: list[str] = []
    attempts = 0
    for attempts in range(1, MAX_FORMAT_ATTEMPTS + 1):
        response_prompt = (
            prompt if attempts == 1
            else _build_repair_prompt(prompt, markdown, validation_errors)
        )
        markdown = generate_response(
            model,
            tokenizer,
            response_prompt,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            quiet=True,
        ).strip()
        validation_errors = validate_summary_format(markdown)
        validation_errors.extend(validate_summary_values(markdown, payload))
        if not validation_errors:
            break

    if validation_errors:
        detail = "; ".join(dict.fromkeys(validation_errors))
        raise ValueError(f"LLM summary failed output validation: {detail}")

    stats = {
        "raw_fields": len(raw_fields),
        "fields_after_noise_filter": len(clean_fields),
        "fields_sent_to_model": len(relevant_fields),
        "document_text_chars": len(document_text),
        "dropped_as_graph_or_noise": len(raw_fields) - len(clean_fields),
        "conflicts": len(result.conflicts),
        "format_attempts": attempts,
    }
    return markdown, stats


def _to_document_text(documents: list[ExtractedDocument]) -> str:
    """Debug helper: dump what the LLM was actually sent (off by default)."""
    return "\n\n".join(doc.full_text() for doc in documents)


def filter_critical_fields(fields: list) -> list:
    """Filter out intermediate calculations and non-summary report noise."""
    summary_pages = {'1', '2', '3', '5', '7', '8', '9', '12', '82', '83'}
    critical_params = {
        'R/CAP', 'R/STR', 'R/GEO', 'Factored load', 'CAP. compressi',
        'CAP. tension', 'Structural cap.', 'LOAD/CAP', 'Settlement @des. cap.',
        'Mx.max', 'δ.x', 'Lat. Force Fx', 'Pile diameter O.D.', 'Helical plates',
        'Design life', 'Thickness loss', 'Project name', 'Time', 'Axial load',
        'F.xx', 'M.xx',
    }

    filtered = []
    for field in fields:
        location = str(field.get('Location', field.get('location', '')))
        parameter = str(field.get('Parameter', ''))

        page_number = None
        if 'page ' in location:
            try:
                page_number = location.split('page ')[1].split(',')[0].strip()
            except Exception:
                pass

        if page_number in summary_pages or parameter in critical_params:
            if parameter != 'El.':
                filtered.append(field)

    return filtered


_SMART_GARBAGE_VALUES = {
    "name",
    "(ksi)",
    "(in)",
    "(k)",
    "stage 0",
    "stage 1",
    "critical",
    "compress. (k)",
    "tension (k)",
}


def _smart_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _clean(text) -> str:
    """Stringify an extracted value, treating a null unit/value as empty."""
    return "" if text is None else str(text).strip()


def _smart_contains(value, keyword) -> bool:
    text = _smart_text(value)
    needle = _smart_text(keyword)
    if not needle:
        return True
    if len(needle.replace(" ", "")) <= 4:
        return re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text
        ) is not None
    return needle in text


def find_smart(fields, param_keyword, context_keyword=None):
    """Search noisy extracted fields for a usable value."""
    for field in fields:
        parameter = field.get("Parameter", field.get("parameter", ""))
        context = _clean(field.get("Context", field.get("context")))
        value = _clean(field.get("Value", field.get("value")))
        unit = _clean(field.get("Unit", field.get("unit")))
        if not _smart_contains(parameter, param_keyword):
            continue
        if context_keyword is not None and not (
            _smart_contains(context, context_keyword)
            or _smart_contains(value, context_keyword)
        ):
            continue
        if not value or value.casefold() in _SMART_GARBAGE_VALUES:
            continue
        return f"{value} {unit}".strip()
    return MISSING_VALUE


def _first_smart(fields, searches) -> str:
    for param_keyword, context_keyword in searches:
        value = find_smart(fields, param_keyword, context_keyword)
        if value != "Not specified":
            return value
    return "Not specified"


def _leading_number(text) -> float | None:
    """First number of a formatted value ("0.907 dimensionless" -> 0.907)."""
    match = _NUMBER_TOKEN_PATTERN.search(str(text if text is not None else ""))
    return float(match.group(0).replace(",", "")) if match else None


def _usable(field) -> bool:
    value = _clean(field.get("Value", field.get("value")))
    return bool(value) and value.casefold() not in _SMART_GARBAGE_VALUES


def _field_text(field) -> str:
    value, unit = _clean(field.get("Value", field.get("value"))), _clean(
        field.get("Unit", field.get("unit")))
    return f"{value} {unit}".strip() if unit else value


# A pile-schedule row label ("HP1", "P-3"). A bare "Diameter" or "Thickness"
# header only identifies a pile inside such a row, so those lines are resolved
# against these rows before falling back to a context-free match.
_PILE_ROW_RE = re.compile(r"^(?:HP|P)[-_ ]?\d+$", re.I)


def _find_pile_row(fields, param_keywords) -> str:
    """First value whose parameter matches and whose row label is a pile id."""
    for keyword in param_keywords:
        for field in fields:
            parameter = field.get("Parameter", field.get("parameter", ""))
            context = _clean(field.get("Context", field.get("context")))
            if not _smart_contains(parameter, keyword) or not _usable(field):
                continue
            if not _PILE_ROW_RE.match(context):
                continue
            return _field_text(field)
    return MISSING_VALUE


def _find_governing(fields, param_keywords) -> tuple[str, str]:
    """Largest explicitly reported value for these parameters.

    Returns (printed value + unit, row context) so the governing row keeps its
    provenance. Stage 6 rule 5: the governing row is the one with the largest
    explicitly reported ratio -- never an assumed or hardcoded one. Returns
    (MISSING_VALUE, "") when the report prints no such value.
    """
    best_text, best_context, best_number = MISSING_VALUE, "", None
    for keyword in param_keywords:
        for field in fields:
            parameter = field.get("Parameter", field.get("parameter", ""))
            if not _smart_contains(parameter, keyword) or not _usable(field):
                continue
            number = _leading_number(field.get("Value", field.get("value")))
            if number is None:
                continue
            if best_number is None or number > best_number:
                best_number = number
                best_text = _field_text(field)
                best_context = _clean(field.get("Context", field.get("context")))
    return best_text, best_context.strip()


# Parameter keywords per summary line, most specific first. A bare "thickness"
# or "moment" fallback would also match the effective/corrosion rows, so the
# printed HelixPile names come first.
_SUMMARY_SEARCHES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "project_name": (("project name", None), ("project", None)),
    "structure": (("structure", None), ("substructure", None), ("description of work", None)),
    "location": (("location", None), ("project location", None), ("site", None)),
    "client": (("client", None), ("company", None), ("owner", None)),
    "prepared_by": (("prepared by", None), ("designed by", None), ("checked by", None)),
    "date": (("date", None), ("time", None), ("issued", None)),
    "software": (("software", None), ("program", None), ("version", None)),
    "pile_diameter": (("pile diameter", None), ("diameter o d", None), ("dp", None),
                      ("diameter", None)),
    "pile_thickness": (("tp", None), ("wall thickness", None), ("thickness", None)),
    "pile_thickness": (("tp", None), ("wall thickness", None), ("thickness", None)),
    "effective_thickness": (("effective thickness", None),
                            ("thickness after corrosion", None),
                            ("net thickness", None)),
    "fy": (("fy", None), ("yield", None)),
    "helix_count": (("numbe", None), ("number of helix", None), ("helix", None)),
    "helix_diameter": (("dhel", None), ("helix dia", None), ("plate dia", None)),
    "helix_spacing": (("helix spacing", None), ("spacing", None)),
    "compression": (("factored load", "stage 0"), ("axial force", "stage 0"),
                    ("compression", "stage 0"), ("axial load", "stage 0")),
    "tension": (("factored load", "stage 1"), ("axial force", "stage 1"),
                ("tension", "stage 1"), ("axial load", "stage 1")),
    "shear": (("shear force", None), ("lateral load", None), ("lat force", None),
              ("shear", None)),
    "moment": (("moment force", None), ("moment", "stage"), ("moment", None)),
    "geo_capacity": (("cap compressi", None), ("geotechnical capacity", None),
                     ("geo capacity", None)),
    "str_capacity": (("structural cap", None), ("cap structur", None),
                     ("structural capacity", None)),
    "settlement": (("settlement @des. cap.", None), ("settlement", None)),
    "settlement_criterion": (("allowable settlement", None), ("settlement limit", None),
                             ("settlement criteria", None)),
    "lateral_deflection": (("max lateral", None), ("x max", None),
                           ("deflection", None)),
    "max_moment": (("mx max", None), ("max internal moment", None)),
    "pile_head": (),  # resolved with an anchored match, see below
    "design_life": (("design life", None), ("service life", None)),
    "corrosion_loss": (("thickness loss", None), ("corrosion", None)),
}

# Ratio columns as printed in HelixPile/DeepFND analysis summaries. Only a
# parameter that names the ratio counts: a HelixPile ratio table prints one
# "LOAD/CAP" header over a RATIO/STRUCTURAL/GEOTECH sub-header row, which Stage
# 2b collapses into three identically named fields. Taking the largest of those
# would report the geotechnical ratio as the structural one, so an
# indistinguishable column yields "Not specified" rather than a wrong number.
_RATIO_KEYWORDS: dict[str, tuple[str, ...]] = {
    "R/GEO": ("r/geo", "load/cap geotech", "geotechnical ratio"),
    "R/STR": ("r/str", "load/cap struct", "structural ratio"),
    "R/CAP": ("r/cap", "load/cap"),
}

# Lines that must resolve to a number. A non-numeric match here is a header
# label or a bare unit fragment the extractor kept ("Helix", "(k-ft)",
# "Protection (mils)"), never a measurement.
_NUMERIC_LINES = frozenset({
    "pile_diameter", "pile_thickness", "effective_thickness", "fy",
    "helix_count", "helix_diameter", "helix_spacing",
    "compression", "tension", "shear", "moment",
    "geo_capacity", "str_capacity", "settlement", "settlement_criterion",
    "lateral_deflection", "max_moment", "design_life", "corrosion_loss",
})

# Dimensionless counts: no unit is expected, so no "(unit ...)" marker.
_DIMENSIONLESS_LINES = frozenset({"helix_count"})


def _first_numeric(fields, searches) -> str:
    """First search hit whose value actually parses as a number."""
    for param_keyword, context_keyword in searches:
        for field in fields:
            parameter = field.get("Parameter", field.get("parameter", ""))
            context = _clean(field.get("Context", field.get("context")))
            if not _smart_contains(parameter, param_keyword):
                continue
            if context_keyword is not None and not _smart_contains(
                context, context_keyword
            ):
                continue
            if not _usable(field):
                continue
            if _leading_number(field.get("Value", field.get("value"))) is None:
                continue
            return _field_text(field)
    return MISSING_VALUE


def _format_measured(value: str, key: str) -> str:
    """Keep the printed unit; say so explicitly when the report printed none."""
    if value == MISSING_VALUE or key in _DIMENSIONLESS_LINES:
        return value
    if _leading_number(value) is not None and not re.search(r"[A-Za-z]", value):
        return f"{value} (unit {MISSING_VALUE})"
    return value


def _first_param_startswith(fields, keywords) -> str:
    """First hit whose parameter *begins* with the keyword.

    Anchored so "Pile head" cannot match "Shear force at pile head".
    """
    for keyword in keywords:
        for field in fields:
            parameter = _clean(field.get("Parameter", field.get("parameter")))
            if not parameter.lower().startswith(keyword) or not _usable(field):
                continue
            return _field_text(field)
    return MISSING_VALUE

_EVALUATION_CRITICAL = (
    ("Project name", "project_name"),
    ("Pile diameter", "pile_diameter"),
    ("Pile thickness", "pile_thickness"),
    ("Yield strength (Fy)", "fy"),
    ("Compression load (Stage 0)", "compression"),
    ("Tension load (Stage 1)", "tension"),
    ("Shear", "shear"),
    ("Moment", "moment"),
    ("R/CAP", "r_cap"),
    ("Settlement", "settlement"),
    ("Design life", "design_life"),
)


def _adequacy_statement(r_cap: str) -> str:
    """PASS only on an explicitly reported R/CAP below 1.0 (Stage 6 rule 6)."""
    number = _leading_number(r_cap)
    if number is None:
        return (
            "**FAIL** \u2014 adequacy cannot be demonstrated: the report prints no "
            f"numeric R/CAP ({MISSING_VALUE})."
        )
    verdict = "**PASS**" if number < 1.0 else "**FAIL**"
    return f"{verdict} \u2014 explicitly reported R/CAP = {r_cap} against a limit of 1.0."


def _governing_statement(ratios: dict[str, str]) -> str:
    """Name the governing limit state from the reported ratios only."""
    reported = {
        label: value for label, value in ratios.items()
        if value != MISSING_VALUE and _leading_number(value) is not None
    }
    if not reported:
        return MISSING_VALUE
    label, value = max(reported.items(), key=lambda item: _leading_number(item[1]))
    return f"{label} governs at {value} (largest reported ratio against a limit of 1.0)."


def _serviceability_statement(settlement: str, criterion: str) -> str:
    if settlement == MISSING_VALUE:
        return f"Reported settlement is {MISSING_VALUE}; serviceability cannot be assessed."
    if criterion == MISSING_VALUE:
        return (
            f"Reported settlement is {settlement}, but the report states no acceptance "
            f"criterion ({MISSING_VALUE}), so tolerance compliance cannot be assessed."
        )
    limit = _leading_number(criterion)
    actual = _leading_number(settlement)
    if limit is None or actual is None:
        return f"Reported settlement {settlement} vs criterion {criterion}; not comparable."
    if actual <= limit:
        return f"Reported settlement {settlement} does not exceed the stated criterion {criterion}."
    return f"Reported settlement {settlement} EXCEEDS the stated criterion {criterion}."


def _project_meta(project) -> dict[str, str]:
    """Index Stage-2c project key points by key -> printed value.

    A title block ("Project: Baytown Station - Structure C") carries no number,
    so Stage 2b never turns it into a field. Stage 2c reads it from the document
    text, so the summary takes the project lines from there and falls back to a
    field keyword search only when nothing was supplied.
    """
    if isinstance(project, dict):
        project = [{"key": k, "value": v} for k, v in project.items()]
    found: dict[str, str] = {}
    for point in project or []:
        key = str(point.get("key", "")).strip().lower()
        value = _clean(point.get("value", point.get("Value")))
        if key and value and key not in found:
            found[key] = value
    return found


# project line -> Stage-2c project key
_PROJECT_LINES = {
    "project_name": "project_name",
    "structure": "structure",
    "location": "location",
    "client": "client",
    "prepared_by": "prepared_by",
    "date": "date",
    "software": "software",
}


def generate_clean_summary(fields, project=None) -> str:
    """Build a deterministic engineering summary from extracted fields.

    Hard constraints (all stages):
    - No LLM. Pure keyword lookup over Stage-2b fields.
    - **Nothing is invented.** A value that is not present in the supplied
      fields is rendered as "Not specified" -- there is no default, fallback,
      or assumed soil/pile parameter anywhere in this function.
    - Project/structure/client/prepared-by/date are read from `project` (the
      Stage-2c title-block key points) first, because a title line holds no
      number and so never becomes a Stage-2b field.
    - Adequacy is PASS only when the report explicitly prints an R/CAP below
      1.0; otherwise the report is shown as FAIL with the reason.
    - Output is never certified or approved engineering work.
    """
    values = {key: _first_smart(fields, searches)
              for key, searches in _SUMMARY_SEARCHES.items() if searches}
    # A measurement line only accepts a numeric value: a text match is a header
    # label or unit fragment the extractor kept, not a reading.
    for key in _NUMERIC_LINES:
        if key in values and values[key] != MISSING_VALUE:
            values[key] = _first_numeric(fields, _SUMMARY_SEARCHES[key])
    values["pile_head"] = _first_param_startswith(
        fields, ("pile head", "head condition", "top of pile", "fixity")
    )
    meta = _project_meta(project)
    for line, meta_key in _PROJECT_LINES.items():
        if meta.get(meta_key):
            values[line] = meta[meta_key]
    # Geometry first from a pile-schedule row, so a bare "Thickness (mm)" header
    # can never be confused with an effective/corrosion thickness line.
    for key in ("pile_diameter", "pile_thickness"):
        scoped = _find_pile_row(fields, _SUMMARY_SEARCHES[key])
        if scoped != MISSING_VALUE:
            values[key] = scoped
    ratios = {
        label: _find_governing(fields, keywords)[0]
        for label, keywords in _RATIO_KEYWORDS.items()
    }
    r_cap_text, r_cap_context = _find_governing(fields, _RATIO_KEYWORDS["R/CAP"])

    missing = [label for label, key in _EVALUATION_CRITICAL
               if (r_cap_text if key == "r_cap" else values[key]) == MISSING_VALUE]
    red_flags = (
        "Not found in the source report (never assumed): " + ", ".join(missing) + "."
        if missing else
        "All critical values above were found in the source report."
    )
    governing = _governing_statement(ratios)
    adequacy = _adequacy_statement(r_cap_text)
    if governing != MISSING_VALUE:
        adequacy = f"{adequacy} {governing}"

    return f"""
### 1. PROJECT INFORMATION
- **Project Name:** {values['project_name']}
- **Structure:** {values['structure']}
- **Location:** {values['location']}
- **Client:** {values['client']}
- **Prepared By:** {values['prepared_by']}
- **Date:** {values['date']}
- **Software:** {values['software']}

### 2. PILE SPECIFICATIONS
- **Pile Diameter (O.D.):** {_format_measured(values['pile_diameter'], 'pile_diameter')}
- **Pile Thickness (initial):** {_format_measured(values['pile_thickness'], 'pile_thickness')}
- **Effective Thickness (after corrosion):** {_format_measured(values['effective_thickness'], 'effective_thickness')}
- **Yield Strength (Fy):** {_format_measured(values['fy'], 'fy')}
- **# of Helices:** {_format_measured(values['helix_count'], 'helix_count')}
- **Helix Diameters:** {_format_measured(values['helix_diameter'], 'helix_diameter')}
- **Helix Spacing:** {_format_measured(values['helix_spacing'], 'helix_spacing')}

### 3. APPLIED LOADS (FACTORED)
- **Compression (Stage 0):** {_format_measured(values['compression'], 'compression')}
- **Tension (Stage 1):** {_format_measured(values['tension'], 'tension')}
- **Shear:** {_format_measured(values['shear'], 'shear')}
- **Moment:** {_format_measured(values['moment'], 'moment')}

### 4. RESULTS (RATIOS)
- **Governing Stage:** {r_cap_context or MISSING_VALUE}
- **Geotechnical Capacity:** {_format_measured(values['geo_capacity'], 'geo_capacity')}
- **Structural Capacity:** {_format_measured(values['str_capacity'], 'str_capacity')}
- **R/GEO (Geotechnical Ratio):** {ratios['R/GEO']}
- **R/STR (Structural Ratio):** {ratios['R/STR']}
- **R/CAP (Overall Critical Ratio):** {r_cap_text}

### 5. SETTLEMENT & LATERAL
- **Settlement at Design Load:** {_format_measured(values['settlement'], 'settlement')}
- **Max Lateral Deflection:** {_format_measured(values['lateral_deflection'], 'lateral_deflection')}
- **Max Internal Moment:** {_format_measured(values['max_moment'], 'max_moment')}
- **Pile Head Condition:** {values['pile_head']}

### 6. DURABILITY
- **Design Life:** {_format_measured(values['design_life'], 'design_life')}
- **Corrosion Thickness Loss:** {_format_measured(values['corrosion_loss'], 'corrosion_loss')}

### 7. ENGINEERING EVALUATION
- **Overall Adequacy / Governing Limit State:** {adequacy}
- **Utilization:** R/CAP = {r_cap_text} against a limit of 1.0; no margin is calculated from unprinted values.
- **Settlement:** {_serviceability_statement(values['settlement'], values['settlement_criterion'])}
- **Red Flags to Verify:** {red_flags} Confirm all soil parameters against the site geotechnical report.
{_NOT_CERTIFIED}"""


def generate_llm_summary(fields: list, llm_client) -> str:
    """Generate a focused summary from critical fields and a strict prompt."""
    critical_fields = filter_critical_fields(fields)
    data_string = "\n".join(
        f"{field.get('Parameter')} ({field.get('Context')}): "
        f"{field.get('Value')} {field.get('Unit')}"
        for field in critical_fields
    )

    prompt = f"""You are a Senior Geotechnical Engineer.
IGNORE all intermediate calculations, depths, and repetitive rows.
Extract ONLY the final governing values and output EXACTLY this format:

### 1. PROJECT INFORMATION
- Project Name:
- Structure:
- Location:
- Client:
- Date:

### 2. PILE SPECIFICATIONS
- Pile Diameter (O.D.):
- Pile Thickness:
- Yield Strength (Fy):
- Helix No:
- Helix Diameters:

### 3. APPLIED LOADS
- Compression (Stage 0):
- Tension (Stage 1):
- Shear:
- Moment:

### 4. RESULTS (RATIOS)
- Governing Stage:
- Geotechnical Capacity:
- Structural Capacity:
- R/GEO:
- R/STR:
- R/CAP:

### 5. ENGINEERING EVALUATION
- Adequacy (PASS/FAIL):
- Governing Limit State:
- Serviceability:
- Red Flags:

### 6. REFERENCES
- Cite table/pages for R/CAP and max loads.

RULES:
1. DO NOT output raw text or intermediate math.
2. If a value is missing, write "Not specified".
3. ALWAYS include units.

DATA:
{data_string}
"""

    try:
        return llm_client.generate(prompt)
    except Exception as exc:
        return f"LLM Summary Generation Failed: {str(exc)}"


# --------------------------------------------------------------------------
# CLI: python -m app.report.summary <path-or-dir> [...]
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m app.report.summary <path-or-dir> [...]")
        return 2
    try:
        docs = load_many(args)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError) as exc:
        print(f"!! {exc}")
        return 1
    try:
        markdown, stats = generate_markdown_summary(docs)
    except ValueError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 1
    print(markdown)
    print(f"<!-- stats: {stats} -->", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())