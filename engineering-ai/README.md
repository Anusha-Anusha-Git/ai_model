# engineering-ai

Locally deployable AI application for engineering and structural/manufacturing
use. Reads engineering project documents, extracts structured data with full
traceability, and produces review-oriented summaries and PDF reports. RAG
retrieval and independent recalculation are not part of the current build.
**AI output is never** certified or approved engineering work.

## Hard constraints (all stages)

- Core LLM: `Qwen/Qwen2.5-1.5B-Instruct` (open-weight, local). LoRA fine-tuning
  (PEFT + Transformers) only if Stage 7 evaluation shows it is needed, and kept
  as a separate adapter from the base model.
- It is **never** justified to invent values, references, or units.
- **Never present AI output as certified/approved engineering work.**
- Distinguish at all times between: raw extracted values, software-calculated
  values, RAG-retrieved reference values, and AI-generated interpretation.

## Project layout

```
engineering-ai/
├── app/
│   ├── models/
│   │   └── llm.py              # Stage 1: local Qwen2.5-1.5B-Instruct inference
│   └── extraction/
│       ├── document_loader.py  # Stage 2: text/table extraction from documents
│       └── field_extractor.py  # Stage 2b: tables/text → {parameter,value,unit,source}
│   └── report/
│       ├── key_points.py       # Stage 2c: report → project/pile/loads/ratios summary
│       └── summary.py          # Stage 6: strict nine-section LLM summary
├── tests/
│   ├── make_fixtures.py        # generates realistic sample documents
│   ├── fixtures/               # generated sample docs (PDF/DOCX/XLSX/CSV/TXT)
│   ├── test_document_loader.py # Stage 2 correctness tests
│   ├── test_field_extractor.py # Stage 2b correctness tests
│   └── test_key_points.py      # Stage 2c correctness tests
├── app/
│   └── web/
│       ├── server.py           # stdlib-only HTTP server (Stage 2b UI)
│       └── index.html          # pastel-blue interactive extraction UI
└── requirements.txt
```

## Setup

```bash
cd engineering-ai
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Stage 1 first run downloads ~3GB of model weights from Hugging Face into
`~/.cache/huggingface`.

## Stage 1: Local model (done)

```bash
python -m app.models.llm
```

Loads Qwen2.5-1.5B-Instruct on CUDA in float16, reports tokens/sec and VRAM/RAM
usage. On the dev machine (GTX 1650 Ti, 4GB VRAM): ~2.9GB VRAM, ~4 tokens/sec.

## Stage 2: Document ingestion (done)

Turn a raw file into traceable text and tables. Supported: PDF, DOCX, XLSX,
CSV, TXT.

```bash
# One or more files, or a whole directory:
python -m app.extraction.document_loader <path-or-dir> [...]
```

Rules enforced by the loader:

- Conversion is **mechanical and deterministic** — the LLM is not involved.
- Every chunk/table carries a source location
  (`page 3`, `paragraph 12`, `sheet: Loads, row 5`, `line 1..20`, ...).
- Nothing is invented: blank/unreadable content produces a warning, not a guess.
- Numbers/units are kept exactly as printed; nothing is converted or truncated
  (`324.0` becomes `"324"`, `12.7` stays `"12.7"`).
- Scanned PDFs (no text layer) are detected and flagged as needing OCR — they
  are never silently treated as empty. OCR itself is a later stage.
- XLSX formula cells with no cached value are flagged as *unknown* instead of
  being read as blank.
- CSV delimiter (`,` `;` `\t` `|`) and text encoding (UTF-8 → cp1252 → latin-1)
  are auto-detected.

Run the tests:

```bash
python -m pytest tests -q
```

Example output (extraction + warnings) is generated for every sample document;
see `tests/fixtures/`. Generate your own realistic samples with:

```bash
python tests/make_fixtures.py
```

Every numeric field later extracted in Stage 2b will follow this shape —
never a bare number:

```json
{
  "parameter": "Pile diameter",
  "value": 324,
  "unit": "mm",
  "source": "Design Report.pdf",
  "page": 12
}
```

## Stage 2b: Field extraction (done)

```bash
python -m app.extraction.field_extractor <path-or-dir> [...]
```

Turns the Stage-2 `Table` objects (and label `=` value text lines) into
structured `{parameter, value, unit, source, location, context, raw}` records.
Two table patterns are handled deterministically — no LLM guessing:

1. **label → value pairs** (`Pile diameter O.D.` | `7.63 in`), and
2. **fixed-format engineering tables** (pile section, helix, load and
   capacity/ratio tables — header carries the unit, each row is one
   specimen/load case).

Rules enforced by the extractor:

- Numbers are parsed with Python's `float()` from the printed text only —
  never inferred, rounded, or converted (`1,980` → `1980.0`, `7.63 in` → value
  `7.63`, unit `in`). A cell that does not parse becomes a raw string, never a guess.
- Unit comes from the printed value or the parenthesized header unit
  (`Diameter (mm)`); a bare number keeps `unit: null` — it is never invented.
- Every record keeps its source (`file`, `page/sheet/row/line`) plus the exact
  raw printed text, so a number can always be traced back.
- The same parameter extracted from two places is compared. **Any disagreement
  is flagged as a Conflict — it is never silently resolved, averaged, or
  merged.** Both original records are kept for review.
  - same parameter, different numbers → `kind="value"`
  - same parameter, same number, different units → `kind="unit"`
- Non-numeric cells under a numeric header, formula cells without a cached
  value, and blank/unreadable content produce a warning, never a guess.

Note: `SUBMISSION DATE: 2026-09-23` style dates parse deterministically as the
leading number (`2026`); dates are not engineering fields, and unit/date
semantics are not handled in this build.

### Stage 2b interactive UI (pastel blue)

Stdlib-only server serving the upload/extract page — no new dependencies:

```bash
python -m app.web.server --port 8000
# open http://localhost:8000
```

Upload files (PDF/DOCX/XLSX/CSV/TXT, multiple) or paste raw engineering text.
Results show the **clean engineering summary** (project / pile specs / factored
loads / ratios / settlement & lateral / durability / evaluation), summary chips,
flagged conflicts, warnings, and the key-point card. The server is purely
deterministic — Stage 2/2b extraction runs server-side, no LLM guesses.

#### Optional targeted AI fallback (off by default)

When a required **numeric** key point is missing, a flag-gated helper can ask
the local model for a value from **one relevant excerpt only** (a table row or
a text line, never the whole file):

```bash
ENGAI_AI_FALLBACK=1 python -m app.web.server --port 8000
```

The deterministic result is never modified. Suggestions appear separately in
`ai_suggestions`, are labelled `AI-generated`, and remain unverified. A
suggestion is discarded unless its quote occurs in the excerpt and the number
occurs in that quote. Text fields (project name, client, date) are not
attempted: a number check cannot validate them. Conflicts are **not** sent to
the model — the extractor's "flag, never resolve" rule still applies. With the
flag off, no model runtime is imported and the response is unchanged.

`clean_engineering_summary` is built by `generate_clean_summary()` in
`app/report/summary.py` — pure keyword lookup over the extracted fields:

- **Nothing is invented.** A value that is not present in the extracted fields
  renders as `Not specified`; there is no default or assumed pile/soil value.
- Adequacy is **PASS** only when the report explicitly prints an `R/CAP` below
  1.0; otherwise the summary reports **FAIL** with the reason.
- The governing ratio is the largest explicitly reported one (never an assumed
  stage), and it keeps its row context.
- Settlement compliance is only asserted when the report states a criterion.
- Every summary ends with a not-certified / not-approved statement.

```bash
# API is also callable directly:
curl -F "file=@tests/fixtures/design_report.xlsx" http://127.0.0.1:8000/api/extract
curl -X POST http://127.0.0.1:8000/api/extract \
     -H "Content-Type: application/json" \
     -d '{"text": "Pile section: 324 mm OD, Fy = 450 MPa."}'
```

Each extraction response now also carries a `key_points` summary (see next stage).

## Stage 2c: Report key-point summary (done)

```bash
python -m app.report.key_points <path-or-dir> [...]
```

Turns the Stage-2b fields (plus the raw document text) into a structured,
**deterministic** key-point summary of a report:

- **project** — name, structure, project location, client, submission date
  ("SUBMISSION DATE: 2026-09-23" stays a date string, never a bare number)
- **pile** — pile diameter, wall thickness, Yield strength (Fy), helix count
  and helix dia 1 / dia 2 (…/ up to 5), pile length
- **loads** — compression / tension / shear / moment per load case
- **results** — printed D/C ratios per load case, plus an **engineering
  evaluation**: every ratio is compared against 1.00 (OK / OVER CAPACITY)
- **references** — the report's references / codes & standards section

Engineered the same way as Stages 2/2b:

- No LLM. Numbers are parsed only where the printed label + value appear
  together (`ULS-1: compression 980 kN`, `wall thickness 12.7 mm`,
  `Yield strength (Fy): 345 MPa`, grid tables).
- Every key point keeps `source` / `location` / raw printed text.
- Missing information is listed under *missing* — **never invented**.
- The evaluation is a deterministic comparison, not certification: every
  output ends with "not certified / approved engineering work".
- The web UI now shows the Key Points + evaluation card and included the
  summary in the JSON download.

```bash
python -m pytest tests -q
```

## Stage 6: Strict engineering summary (done)

```bash
python -m app.report.summary path/to/report.pdf
```

The local model returns exactly the nine-section contract used for executive
review: project information, pile specifications, soil parameters, factored
loads, governing results, settlement/lateral results, durability, a four-point
engineering evaluation, and source references. It uses final design and
analysis-summary values only, preserves printed units, writes `Not specified`
for absent data, and never substitutes ultimate capacity for design capacity.

The generated section structure and every numeric token are validated against
the filtered source fields. One formatting repair attempt is allowed before the
command fails. Standard output contains only the required report; extraction
statistics are written to standard error.

## Stage roadmap

The current build is intentionally limited to the deterministic extraction and
report flow: Stages 1, 2, 2b, 2c, 6, and the minimal Stage 10 web UI, plus the
appended verification table. The remaining stages are deferred or out of scope
for this build; they are not a claim of unfinished delivery.

| # | Stage | Status |
|---|-------|--------|
| 1 | Local model: Qwen2.5-1.5B-Instruct inference | Done |
| 2 | Document ingestion: PDF/DOCX/XLSX/CSV/TXT | Done |
| 2b | Field extraction: raw text → structured project/pile/load fields | Done |
| 2c | Report key-point summary + engineering evaluation + references | Done |
| 3 | Data validation: missing/invalid/conflicting/unit-mismatch checks | Partial: ad-hoc extraction checks only; formal layer deferred |
| 4 | Engineering references + RAG: chunk → embed → FAISS → retrieve | Out of scope |
| 5 | Deterministic calculations: Python modules for capacities/ratios | Out of scope by design: report values are never recalculated |
| 6 | Engineering report generation | Done: strict nine-section summary + verification table |
| 7 | Evaluation: test set, accuracy/hallucination metrics | Deferred; regression tests only |
| 8 | LoRA fine-tuning (only if Stage 7 shows it's needed) | Out of scope unless Stage 7 is revived |
| 9 | FastAPI endpoints: upload/extraction/calculation/RAG/report | Out of scope: local stdlib server is the supported interface |
| 10 | Frontend: pastel-blue, review/correct extracted values, results | Partial: minimal Stage 2b web UI done |
| 11 | Deployment: Docker, resource requirements, deployment docs | Out of scope: local venv workflow only |