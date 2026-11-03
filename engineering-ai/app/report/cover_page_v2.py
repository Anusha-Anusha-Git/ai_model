"""Compatibility wrapper for the cover-page builder.

Older code imports ``app.report.cover_page_v2``.  The implementation now lives
in :mod:`app.report.cover_page`; this module re-exports the same public API so
both import paths render the identical cover.
"""

from __future__ import annotations

from app.report.cover_page import (  # noqa: F401
    ASSETS_DIR,
    FIXED_PNA_LINES,
    FIXED_US,
    MARGIN,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    PROPRIETARY,
    build_cover_page,
    build_cover_page_pdf,
    prepend_cover_page_pdf,
    render_cover_page_pdf,
    resolve_cover_fields,
)

__all__ = [
    "ASSETS_DIR",
    "FIXED_PNA_LINES",
    "FIXED_US",
    "MARGIN",
    "PAGE_HEIGHT",
    "PAGE_WIDTH",
    "PROPRIETARY",
    "build_cover_page",
    "build_cover_page_pdf",
    "prepend_cover_page_pdf",
    "render_cover_page_pdf",
    "resolve_cover_fields",
]
