"""Tests for the flag-gated, targeted AI fallback.

Run from the repo root (engineering-ai/):

    python -m pytest tests/test_ai_fallback.py -q

No model is loaded and no network access is used. The generator is injected.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ai.fallback import (
    AI_FALLBACK_ENV,
    Excerpt,
    FieldGap,
    build_ai_suggestions,
    build_prompt,
    find_gaps,
    is_ai_fallback_enabled,
    maybe_build_ai_suggestions,
    retrieve_excerpt,
    validate_suggestion,
)
from app.extraction.document_loader import ExtractedDocument, Table, TextChunk
from app.extraction.field_extractor import ExtractedField, ExtractionResult


def document(
    text: str = "",
    *,
    tables: list[Table] | None = None,
    name: str = "report.pdf",
) -> ExtractedDocument:
    chunks = [TextChunk("page 3", text)] if text else []
    return ExtractedDocument(
        path=Path(name),
        file_type="pdf",
        text_chunks=chunks,
        tables=tables or [],
    )


def json_response(value, unit, quote) -> str:
    return json.dumps({"value": value, "unit": unit, "quote": quote})


# --------------------------------------------------------------------------
# Flag + gap detection
# --------------------------------------------------------------------------


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(AI_FALLBACK_ENV, raising=False)
    assert is_ai_fallback_enabled() is False
    assert is_ai_fallback_enabled({AI_FALLBACK_ENV: "1"}) is True
    assert is_ai_fallback_enabled({AI_FALLBACK_ENV: "TRUE"}) is True
    assert is_ai_fallback_enabled({AI_FALLBACK_ENV: "0"}) is False


def test_find_gaps_only_uses_required_numeric_fields():
    gaps = find_gaps(
        {"missing": ["Pile diameter", "Project name", "Not a required field"]}
    )
    assert [gap.key for gap in gaps] == ["pile_diameter"]
    assert gaps[0].label == "Pile diameter"


def test_find_gaps_skips_text_fields_that_cannot_be_quote_validated():
    assert find_gaps({"missing": ["Project name", "Client", "Date of submission"]}) == []


def test_find_gaps_accepts_key_points_object():
    class FakeKeyPoints:
        missing = ["D/C ratio"]

    gaps = find_gaps(FakeKeyPoints())
    assert [gap.key for gap in gaps] == ["result_ratio"]


# --------------------------------------------------------------------------
# Targeted excerpt retrieval (deterministic, no embeddings)
# --------------------------------------------------------------------------


def test_retrieve_excerpt_from_table_includes_header_and_row():
    table = Table(
        location="page 12, table 1",
        headers=["Stage No", "R/CAP", "R/STR"],
        rows=[["1", "0.85", "0.42"], ["2", "1.10", "0.55"]],
    )
    excerpt = retrieve_excerpt([document(tables=[table])], FieldGap("result_ratio", "D/C ratio"))
    assert excerpt is not None
    assert "R/CAP" in excerpt.text
    assert "0.85" in excerpt.text
    assert excerpt.location.endswith("row 2")


def test_retrieve_excerpt_from_text_includes_one_line_of_context():
    doc = document("Introduction\nPile diameter O.D. = 7.63 in\nNext paragraph")
    excerpt = retrieve_excerpt([doc], FieldGap("pile_diameter", "Pile diameter"))
    assert excerpt is not None
    assert "Pile diameter O.D. = 7.63 in" in excerpt.text
    assert "Introduction" in excerpt.text
    assert excerpt.location.startswith("page 3")


# --------------------------------------------------------------------------
# Validation: copy checks only
# --------------------------------------------------------------------------


def test_valid_suggestion_is_accepted():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    suggestion = validate_suggestion(
        gap, excerpt, json_response(7.63, "in", "Pile diameter O.D. = 7.63 in")
    )
    assert suggestion is not None
    assert suggestion.value == 7.63
    assert suggestion.unit == "in"
    assert suggestion.provenance == "AI-generated"
    assert suggestion.status == "unverified"


def test_fenced_json_is_accepted():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    raw = "```json\n" + json_response(7.63, "in", "Pile diameter O.D. = 7.63 in") + "\n```"
    assert validate_suggestion(gap, excerpt, raw) is not None


def test_fabricated_quote_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(
        gap, excerpt, json_response(7.63, "in", "Pile diameter is 7.63 in")
    ) is None


def test_number_not_present_in_quote_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(
        gap, excerpt, json_response(7.63, "in", "Pile diameter O.D.")
    ) is None


def test_unknown_unit_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(
        gap, excerpt, json_response(7.63, "bananas", "Pile diameter O.D. = 7.63 in")
    ) is None


def test_unit_not_printed_in_excerpt_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(
        gap, excerpt, json_response(7.63, "mm", "Pile diameter O.D. = 7.63 in")
    ) is None


def test_null_value_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(
        gap, excerpt, json_response(None, None, "Pile diameter O.D. = 7.63 in")
    ) is None


def test_unparseable_response_is_rejected():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    assert validate_suggestion(gap, excerpt, "I could not find it.") is None


# --------------------------------------------------------------------------
# Build: never mutate deterministic results
# --------------------------------------------------------------------------


def _result_with_one_field() -> ExtractionResult:
    return ExtractionResult(
        fields=[
            ExtractedField(
                parameter="Pile thickness",
                value=0.375,
                unit="in",
                source="report.pdf",
                location="page 3",
                raw="0.375 in",
            )
        ]
    )


def test_build_suggestions_does_not_mutate_extracted_fields():
    result = _result_with_one_field()
    before = result.to_dict()
    doc = document("Pile diameter O.D. = 7.63 in")

    suggestions = build_ai_suggestions(
        [doc],
        result,
        {"missing": ["Pile diameter"]},
        generate=lambda _prompt: json_response(
            7.63, "in", "Pile diameter O.D. = 7.63 in"
        ),
    )

    assert len(suggestions) == 1
    assert suggestions[0]["provenance"] == "AI-generated"
    assert result.to_dict() == before


def test_missing_excerpt_never_loads_the_model(monkeypatch):
    import app.ai.fallback as fallback

    loaded: list[bool] = []
    monkeypatch.setattr(
        fallback, "_load_default_generator", lambda: loaded.append(True)
    )

    output = fallback.build_ai_suggestions(
        [document("Unrelated text with no matching label")],
        _result_with_one_field(),
        {"missing": ["Pile diameter"]},
    )

    assert output == []
    assert loaded == []


def test_disabled_flag_returns_none_and_never_calls_model(monkeypatch):
    monkeypatch.delenv(AI_FALLBACK_ENV, raising=False)
    calls: list[str] = []

    output = maybe_build_ai_suggestions(
        [document("Pile diameter O.D. = 7.63 in")],
        _result_with_one_field(),
        {"missing": ["Pile diameter"]},
        generate=lambda prompt: calls.append(prompt) or "",
    )

    assert output is None
    assert calls == []


def test_enabled_flag_returns_validated_suggestions(monkeypatch):
    monkeypatch.setenv(AI_FALLBACK_ENV, "1")

    output = maybe_build_ai_suggestions(
        [document("Pile diameter O.D. = 7.63 in")],
        _result_with_one_field(),
        {"missing": ["Pile diameter"]},
        generate=lambda _prompt: json_response(
            7.63, "in", "Pile diameter O.D. = 7.63 in"
        ),
    )

    assert output is not None
    assert len(output) == 1


def test_prompt_is_narrow_and_contains_the_excerpt():
    gap = FieldGap("pile_diameter", "Pile diameter")
    excerpt = Excerpt("Pile diameter O.D. = 7.63 in", "report.pdf", "page 3")
    prompt = build_prompt(gap, excerpt)

    assert "Pile diameter O.D. = 7.63 in" in prompt
    assert "Never infer" in prompt
    assert "JSON" in prompt


# --------------------------------------------------------------------------
# Server integration: off by default means no payload change
# --------------------------------------------------------------------------


def test_run_extraction_has_no_ai_key_when_flag_is_off(monkeypatch):
    from app.web.server import _run_extraction

    monkeypatch.delenv(AI_FALLBACK_ENV, raising=False)
    payload = _run_extraction([document("Pile diameter O.D. = 7.63 in")])

    assert "ai_suggestions" not in payload
    assert payload["fields"]


def test_run_extraction_adds_a_separate_suggestion_when_flag_is_on(monkeypatch):
    import app.ai.fallback as fallback
    from app.web.server import _run_extraction

    monkeypatch.setenv(AI_FALLBACK_ENV, "1")
    monkeypatch.setattr(
        fallback,
        "_load_default_generator",
        lambda: (
            lambda _prompt: json_response(
                0.375, "in", "Wall thickness tp 0.375 in"
            )
        ),
    )

    # The deterministic parser does not read this prose line, so the field is
    # genuinely missing and the fallback has a targeted excerpt to work from.
    payload = _run_extraction([document("Wall thickness tp 0.375 in")])

    assert payload["fields"] == []
    assert len(payload["ai_suggestions"]) == 1
    suggestion = payload["ai_suggestions"][0]
    assert suggestion["key"] == "pile_thickness"
    assert suggestion["provenance"] == "AI-generated"
    assert suggestion["status"] == "unverified"
