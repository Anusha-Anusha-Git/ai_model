"""AI-assisted helpers that never replace deterministic results."""

from app.ai.fallback import (
    AI_FALLBACK_ENV,
    AISuggestion,
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

__all__ = [
    "AI_FALLBACK_ENV",
    "AISuggestion",
    "Excerpt",
    "FieldGap",
    "build_ai_suggestions",
    "build_prompt",
    "find_gaps",
    "is_ai_fallback_enabled",
    "maybe_build_ai_suggestions",
    "retrieve_excerpt",
    "validate_suggestion",
]
