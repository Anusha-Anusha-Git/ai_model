"""Per-project user inputs (``project_inputs.json``).

These are the values a person supplies that are **not** in the report and are
**not** calculated:

    system_descriptor     free-text description of the foundation system
    corrosion_category    free-text corrosivity category
    reviewed_by_engineer  sign-off flag
    installation_verified sign-off flag

They are labelled ``"user-supplied"`` in the output metadata so they are never
confused with ``"extracted"``, ``"calculated"``, or ``"AI-generated"`` values.

The file is optional and lives at the repository root. A caller (for example the
web UI) may override any field; missing or malformed values fall back to the
neutral defaults (empty descriptor, empty category, both flags false), which the
Executive Summary renders as "foundation system" and omits the category clause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Provenance vocabulary. Every value in the output carries exactly one of these.
PROVENANCE_USER_SUPPLIED = "user-supplied"
PROVENANCE_EXTRACTED = "extracted"
PROVENANCE_CALCULATED = "calculated"
PROVENANCE_AI_GENERATED = "AI-generated"

#: Repository root: ``app/config.py`` -> ``app/`` -> repo root.
DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent.parent / "project_inputs.json"

DEFAULT_PROJECT_INPUTS: dict[str, Any] = {
    "system_descriptor": "",
    "corrosion_category": "",
    "reviewed_by_engineer": False,
    "installation_verified": False,
}

_STRING_FIELDS = ("system_descriptor", "corrosion_category")
_BOOL_FIELDS = ("reviewed_by_engineer", "installation_verified")
_ALL_FIELDS = _STRING_FIELDS + _BOOL_FIELDS

_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return False


def normalize_project_inputs(raw: Any) -> dict[str, Any]:
    """Return a complete, well-typed inputs mapping.

    Unknown keys are dropped; known keys are coerced to the expected type. A
    non-mapping (or ``None``) yields the neutral defaults.
    """
    inputs = dict(DEFAULT_PROJECT_INPUTS)
    if not isinstance(raw, dict):
        return inputs
    for key in _STRING_FIELDS:
        if key in raw:
            inputs[key] = str(raw.get(key) or "").strip()
    for key in _BOOL_FIELDS:
        if key in raw:
            inputs[key] = _coerce_bool(raw.get(key))
    return inputs


def load_project_inputs(
    path: str | Path | None = None, overrides: Any = None
) -> dict[str, Any]:
    """Load ``project_inputs.json`` (optional) and apply any overrides.

    A missing, unreadable, or malformed file is not an error: the neutral
    defaults are used. ``overrides`` is normally what the web UI submitted; only
    the recognised keys are honoured.
    """
    inputs = dict(DEFAULT_PROJECT_INPUTS)
    source = Path(path) if path is not None else DEFAULT_INPUTS_PATH
    try:
        if source.exists():
            inputs = normalize_project_inputs(
                json.loads(source.read_text(encoding="utf-8"))
            )
    except (OSError, ValueError):
        inputs = dict(DEFAULT_PROJECT_INPUTS)

    if isinstance(overrides, dict):
        merged = dict(inputs)
        for key in _ALL_FIELDS:
            if key in overrides:
                merged[key] = overrides[key]
        inputs = normalize_project_inputs(merged)
    return inputs


def project_inputs_provenance() -> dict[str, str]:
    """Every project input is, by definition, user-supplied."""
    return {key: PROVENANCE_USER_SUPPLIED for key in _ALL_FIELDS}
