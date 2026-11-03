"""Flag-gated, targeted AI fallback for missing extraction fields.

Hard rules
----------
- The deterministic pipeline remains the source of truth. Nothing in this
  module mutates ``ExtractionResult.fields`` or any existing value.
- Off by default. The feature runs only when ``ENGAI_AI_FALLBACK`` is truthy.
  With the flag off, the model runtime (``torch`` / ``transformers``) is never
  imported.
- Only *missing required key points* are candidates. Conflicts are never sent
  to the model: an LLM must not choose between two printed values.
- A suggestion is accepted only when its quote occurs in the retrieved excerpt
  and the suggested number occurs inside that quote. A failed check discards
  the suggestion and the original result is preserved.

Accepted suggestions are reported separately and labelled ``AI-generated``.
They are never merged into extracted fields and remain unverified.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from app.config import PROVENANCE_AI_GENERATED
from app.extraction.document_loader import ExtractedDocument
from app.extraction.field_extractor import (
    ExtractionResult,
    is_known_unit,
    split_value_unit,
)
from app.report.key_points import required_key_points
from app.report.summary import (
    ratio_search_terms,
    smart_contains,
    summary_search_terms,
)

AI_FALLBACK_ENV = "ENGAI_AI_FALLBACK"

#: Bound the work per upload: each local generation can take tens of seconds.
MAX_GAPS = 3
MAX_EXCERPT_CHARS = 4_000

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}

#: Only numeric engineering fields can be checked by "quote contains number".
#: Text fields (project name, client, date, ...) are deliberately excluded: a
#: number check cannot validate them, and an LLM must not invent labels.
_NUMERIC_GAP_KEYS = frozenset({
    "pile_diameter",
    "pile_thickness",
    "yield_strength",
    "helix",
    "pile_length",
    "load_compression",
    "load_tension",
    "load_shear",
    "load_moment",
    "result_ratio",
})

#: Required key-point keys (see ``key_points.required_key_points``) mapped to
#: the deterministic summary lines that already know the printed synonyms.
_REQUIRED_TO_SUMMARY = {
    "project_name": ("project_name",),
    "structure": ("structure",),
    "location": ("location",),
    "client": ("client",),
    "date": ("date",),
    "pile_diameter": ("pile_diameter",),
    "pile_thickness": ("pile_thickness",),
    "yield_strength": ("fy",),
    "helix": ("helix_count", "helix_diameter", "helix_spacing"),
    "load_compression": ("compression",),
    "load_tension": ("tension",),
    "load_shear": ("shear",),
    "load_moment": ("moment",),
}

#: Required keys with no dedicated summary line. These are search hints only;
#: they do not add a new required field.
_EXTRA_SEARCH_TERMS = {
    "pile_length": (
        ("pile length", None),
        ("embedded length", None),
        ("embedded depth", None),
        ("depth", None),
    ),
    "result_ratio": (
        ("r/cap", None),
        ("r/str", None),
        ("r/geo", None),
        ("d/c", None),
        ("ratio", None),
    ),
}


@dataclass(frozen=True)
class FieldGap:
    """One required key point the deterministic pipeline could not resolve."""

    key: str
    label: str

    @property
    def search_terms(self) -> list[tuple[str, str | None]]:
        terms: list[tuple[str, str | None]] = []
        for summary_key in _REQUIRED_TO_SUMMARY.get(self.key, ()):
            terms.extend(summary_search_terms(summary_key))
        terms.extend(_EXTRA_SEARCH_TERMS.get(self.key, ()))
        return terms


@dataclass(frozen=True)
class Excerpt:
    """The targeted source excerpt sent to the model."""

    text: str
    source: str
    location: str


@dataclass(frozen=True)
class AISuggestion:
    """A validated, still-unverified suggestion for one missing field."""

    key: str
    label: str
    value: float
    unit: str | None
    quote: str
    source: str
    location: str
    provenance: str = PROVENANCE_AI_GENERATED
    status: str = "unverified"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "quote": self.quote,
            "source": self.source,
            "location": self.location,
            "provenance": self.provenance,
            "status": self.status,
        }


def is_ai_fallback_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the targeted AI fallback is switched on (default: off)."""
    source = os.environ if environ is None else environ
    return str(source.get(AI_FALLBACK_ENV, "")).strip().casefold() in _TRUE_VALUES


def find_gaps(key_points: dict | object) -> list[FieldGap]:
    """Return required key points that the deterministic pipeline reports missing.

    ``key_points`` may be the ``KeyPointsResult`` object or its ``to_dict()``
    payload. Only labels already in the project's required list are considered.
    """
    if isinstance(key_points, dict):
        missing = key_points.get("missing") or []
    else:
        missing = getattr(key_points, "missing", []) or []

    by_label = {label: key for key, label in required_key_points()}
    gaps: list[FieldGap] = []
    for label in missing:
        key = by_label.get(str(label))
        if key is not None and key in _NUMERIC_GAP_KEYS:
            gaps.append(FieldGap(key=key, label=str(label)))
    return gaps


def _documents(documents: ExtractedDocument | Iterable[ExtractedDocument]) -> list[ExtractedDocument]:
    if isinstance(documents, ExtractedDocument):
        return [documents]
    return list(documents)


def _score(text: str, terms: list[tuple[str, str | None]]) -> int:
    return sum(1 for term, _context in terms if smart_contains(text, term))


def retrieve_excerpt(
    documents: ExtractedDocument | Iterable[ExtractedDocument],
    gap: FieldGap,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> Excerpt | None:
    """Find the single most relevant table row or text line for ``gap``.

    This is deterministic label matching, not retrieval-augmented generation:
    no embeddings or vector store are involved. One table row (with its header)
    or one text line (with one line of context) is returned, never the file.
    """
    terms = gap.search_terms
    if not terms:
        return None

    best: Excerpt | None = None
    best_score = 0

    for document in _documents(documents):
        source = document.path.name
        for table in document.tables:
            header = " | ".join(table.headers)
            for row_index, row in enumerate(table.rows, start=2):
                row_text = " | ".join(row)
                candidate = f"{header}\n{row_text}".strip()
                score = _score(candidate, terms)
                if score > best_score:
                    best_score = score
                    best = Excerpt(
                        text=candidate[:max_chars],
                        source=source,
                        location=f"{table.location}, row {row_index}",
                    )

        for chunk in document.text_chunks:
            lines = chunk.text.splitlines()
            for index, line in enumerate(lines):
                score = _score(line, terms)
                if score > best_score:
                    best_score = score
                    start = max(0, index - 1)
                    end = min(len(lines), index + 2)
                    excerpt = "\n".join(lines[start:end]).strip()
                    if end - start > 1:
                        location = f"{chunk.location}, lines {start + 1}-{end}"
                    else:
                        location = f"{chunk.location}, line {index + 1}"
                    best = Excerpt(
                        text=excerpt[:max_chars],
                        source=source,
                        location=location,
                    )

    return best


def build_prompt(gap: FieldGap, excerpt: Excerpt) -> str:
    """Build a narrow extraction prompt over one excerpt only."""
    return (
        "You extract one engineering value from a source excerpt.\n"
        "Use only the excerpt. Never infer, convert, or calculate.\n\n"
        f"FIELD: {gap.label}\n"
        f"SOURCE: {excerpt.source}, {excerpt.location}\n"
        "EXCERPT:\n"
        "<<<\n"
        f"{excerpt.text}\n"
        ">>>\n\n"
        "Return ONLY a JSON object in this exact shape:\n"
        '{"value": <number or null>, "unit": <printed unit or null>, '
        '"quote": <exact substring of the excerpt>}\n'
        "Rules:\n"
        "- If the value is not printed in the excerpt, return \"value\": null.\n"
        "- \"value\" must be a number alone, without its unit.\n"
        "- \"unit\" must be the unit printed with or above that value.\n"
        "- \"quote\" must be copied exactly from the excerpt and must contain "
        "the number.\n"
        "- Do not explain, and do not return markdown.\n"
    )


def _parse_response(raw: str | None) -> dict | None:
    """Extract the first JSON object from a model response, if any."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def _number_in_quote(value: float, quote: str) -> bool:
    token = f"{value:g}"
    haystack = quote.replace(",", "")
    pattern = rf"(?<![\d.]){re.escape(token)}(?![\d.])"
    return re.search(pattern, haystack) is not None


def validate_suggestion(
    gap: FieldGap,
    excerpt: Excerpt,
    raw: str | None,
) -> AISuggestion | None:
    """Accept a suggestion only if it is traceable to the excerpt.

    This checks that the model copied text correctly; it cannot prove the
    source is correct or that the value is being read in the right engineering
    context. A rejected suggestion is discarded, never applied.
    """
    data = _parse_response(raw)
    if data is None or data.get("value") is None:
        return None

    value, embedded_unit, _raw = split_value_unit(str(data.get("value")))
    if not isinstance(value, float):
        return None

    quote = data.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return None
    if _normalize_space(quote) not in _normalize_space(excerpt.text):
        return None
    if not _number_in_quote(value, quote):
        return None

    unit = data.get("unit")
    unit = str(unit).strip() if unit is not None else ""
    unit = unit or embedded_unit
    if unit is not None and unit != "":
        if not is_known_unit(unit):
            return None
        # The unit must also be printed in the excerpt, not invented.
        if not smart_contains(excerpt.text, unit):
            return None
    else:
        unit = None

    return AISuggestion(
        key=gap.key,
        label=gap.label,
        value=value,
        unit=unit,
        quote=quote,
        source=excerpt.source,
        location=excerpt.location,
    )


def _load_default_generator() -> Callable[[str], str]:
    """Load the local model once and return a prompt -> text callable.

    Imported lazily so a disabled flag never imports ``torch``.
    """
    from app.models.llm import generate_response, load_model

    model, tokenizer = load_model(quiet=True)

    def generate(prompt: str) -> str:
        return generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=160,
            quiet=True,
        )

    return generate


def build_ai_suggestions(
    documents: ExtractedDocument | Iterable[ExtractedDocument],
    result: ExtractionResult,
    key_points: dict | object,
    generate: Callable[[str], str] | None = None,
    max_gaps: int = MAX_GAPS,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> list[dict]:
    """Suggest values for missing required fields, without touching ``result``.

    ``generate`` is injectable for tests. When omitted, the local Qwen model is
    loaded once. Every returned suggestion has passed
    :func:`validate_suggestion`. ``result`` is accepted to make the
    no-mutation contract explicit and is intentionally never modified.
    """
    work: list[tuple[FieldGap, Excerpt]] = []
    for gap in find_gaps(key_points)[:max_gaps]:
        excerpt = retrieve_excerpt(documents, gap, max_chars=max_chars)
        if excerpt is not None:
            work.append((gap, excerpt))
    if not work:
        return []

    # Load the model only after a real excerpt exists; a missing field with no
    # matching text must not pay the model-load cost.
    if generate is None:
        generate = _load_default_generator()

    suggestions: list[dict] = []
    for gap, excerpt in work:
        try:
            raw = generate(build_prompt(gap, excerpt))
        except Exception:
            continue
        suggestion = validate_suggestion(gap, excerpt, raw)
        if suggestion is not None:
            suggestions.append(suggestion.to_dict())
    return suggestions


def maybe_build_ai_suggestions(
    documents: ExtractedDocument | Iterable[ExtractedDocument],
    result: ExtractionResult,
    key_points: dict | object,
    generate: Callable[[str], str] | None = None,
) -> list[dict] | None:
    """Run the fallback only when enabled; never fail the deterministic flow.

    Returns ``None`` when the feature flag is off (the caller should then leave
    its payload unchanged). Any failure yields an empty list, preserving the
    deterministic output.
    """
    if not is_ai_fallback_enabled():
        return None
    try:
        return build_ai_suggestions(
            documents,
            result,
            key_points,
            generate=generate,
        )
    except Exception:
        return []
