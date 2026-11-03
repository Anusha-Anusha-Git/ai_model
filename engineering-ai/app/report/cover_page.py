"""Cover page for the downloadable PDF (first page, no page number).

The cover is drawn directly on a ReportLab canvas so that the decorative
banner, the right-aligned title block, the company blocks, and the revision
record table all share one measured coordinate system.

Page size follows the rest of the report (A4).  All layout positions below are
expressed as named constants; nothing is placed with an unexplained magic
number in the drawing helpers.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from typing import Any, Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import Flowable

# --------------------------------------------------------------------------
# Page geometry
# --------------------------------------------------------------------------

#: The rest of this project renders A4 (see pdf_report.py and
#: executive_summary.py), so the cover must match rather than switch to Letter.
PAGE_WIDTH, PAGE_HEIGHT = A4

MARGIN = 0.75 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
CONTENT_RIGHT = PAGE_WIDTH - MARGIN

BANNER_HEIGHT = 1.2 * inch

TITLE_TOP = PAGE_HEIGHT - BANNER_HEIGHT - 0.82 * inch
TITLE_SIZE = 21.0
TITLE_LEADING = 0.37 * inch
TITLE_GAP_AFTER = 0.62 * inch

PROJECT_SIZE = 13.5
PROJECT_LEADING = 0.24 * inch
ADDRESS_SIZE = 10.5
ADDRESS_LEADING = 0.18 * inch
PROJECT_GAP_AFTER = 0.44 * inch

US_LOGO_BOX = (1.72 * inch, 0.64 * inch)      # (max width, max height)
PNA_LOGO_BOX = (1.50 * inch, 0.62 * inch)
LOGO_ADDRESS_GAP = 0.14 * inch
LOGO_BLOCK_GAP = 0.30 * inch

PNA_LINE_SIZE = 8.0
PNA_LINE_LEADING = 0.145 * inch

BOTTOM_MARGIN = 0.90 * inch
TABLE_TITLE_HEIGHT = 0.25 * inch
TABLE_HEADER_HEIGHT = 0.25 * inch
TABLE_ROW_HEIGHT = 0.30 * inch
TABLE_TO_PROPRIETARY_GAP = 0.10 * inch
PROPRIETARY_SIZE = 7.6
PROPRIETARY_LEADING = 0.14 * inch
PROPRIETARY_PADDING_X = 0.16 * inch
PROPRIETARY_PADDING_Y = 0.12 * inch
MIN_GAP_ABOVE_TABLE = 0.30 * inch

#: Column proportions. The Description column is intentionally widest.
REVISION_COLUMNS: tuple[tuple[str, float], ...] = (
    ("Rev", 0.070),
    ("Date", 0.145),
    ("Description", 0.405),
    ("Originated by", 0.126),
    ("Checked by", 0.126),
    ("Approved by", 0.128),
)

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

TITLE_BLUE = colors.HexColor("#146C8F")
PROJECT_MAROON = colors.HexColor("#7A1B3C")
BANNER_TEAL = colors.HexColor("#1F7F9E")
BANNER_LIGHT_BLUE = colors.HexColor("#58B4D2")
BANNER_CYAN = colors.HexColor("#8BD5E3")
BANNER_NAVY = colors.HexColor("#123F63")
BANNER_BLUE = colors.HexColor("#2472AE")
TABLE_HEADER = colors.HexColor("#0F6C86")
TABLE_BORDER = colors.HexColor("#2CA6BE")
TABLE_ROW = colors.HexColor("#DDF0F7")
PROPRIETARY_FILL = colors.HexColor("#DFF1F8")
TEXT_DARK = colors.HexColor("#1F2933")
WHITE = colors.white

# --------------------------------------------------------------------------
# Fixed company wording
# --------------------------------------------------------------------------

FIXED_US = "202 S Dixon St, Callisburg, TX 76240, United States"
FIXED_PNA_LINES = (
    "PNA ASSOCIATES Inc.",
    "(dba PNA TECHNICAL SERVICES)",
    "(Texas Firm# 26780)",
    "HOUSTON, TEXAS",
)
PROPRIETARY = (
    "This document is the property of PNA Technical Services. It must NOT be "
    "reproduced or transmitted to others without authorization from the COMPANY."
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

_US_LOGO_NAMES = (
    "logo_us_helicals.png",
    "logo_us_helicals.jpg",
    "logo_us_helicals.jpeg",
    "us_helicals.png",
    "us_helicals_logo.png",
    "us-helicals.png",
)
_PNA_LOGO_NAMES = (
    "logo_pna.png",
    "logo_pna.jpg",
    "logo_pna.jpeg",
    "pna_logo.png",
    "pna-associates.png",
    "pna.png",
)

_MISSING_TOKENS = {"none", "null", "n/a", "na", "not specified"}
_ASSET_WARNINGS: set[str] = set()


# --------------------------------------------------------------------------
# Small value helpers
# --------------------------------------------------------------------------


def _clean(text: Any) -> str:
    """Return printable text, collapsing whitespace and hiding null markers."""
    if text is None:
        return ""
    value = re.sub(r"\s+", " ", str(text)).strip()
    return "" if value.casefold() in _MISSING_TOKENS else value


def _value_of(mapping: dict[str, Any]) -> str:
    for key in ("value", "Value"):
        if key in mapping:
            return _clean(mapping.get(key))
    return ""


def _key_of(mapping: dict[str, Any]) -> str:
    for key in ("key", "Key", "parameter", "Parameter", "name", "field"):
        value = _clean(mapping.get(key))
        if value:
            return value
    return ""


def _field_records(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Collect record-shaped dictionaries from every supported payload shape."""
    if not isinstance(data, dict):
        return []
    records: list[dict[str, Any]] = []

    # Key points are the curated summary, so they take precedence over the
    # generic extraction list when both carry the same project field.
    key_points = data.get("key_points")
    if isinstance(key_points, dict):
        for section in ("project", "pile", "results"):
            value = key_points.get(section)
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, dict))

    for name in ("fields", "project_inputs"):
        value = data.get(name)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.append(value)
    return records


def _extract_from_data(data: dict[str, Any] | None) -> dict[str, str]:
    """Best-effort fallback values from report metadata, never overwriting input."""
    extracted: dict[str, str] = {}
    aliases = {
        "structure_name": {
            "structure", "structure name", "structure_name", "substructure",
            "project structure", "project_structure", "title", "title line",
        },
        "project_name": {
            "project", "project name", "project_name", "projectname",
            "name of project",
        },
        "project_address": {
            "address", "project address", "project_address", "location",
            "project location", "site", "site location", "site_location",
        },
    }
    for record in _field_records(data):
        key = re.sub(r"\s+", " ", _key_of(record).strip().lower())
        value = _value_of(record)
        if not key or not value:
            continue
        for target, names in aliases.items():
            if target not in extracted and key in names:
                extracted[target] = value
    return extracted


def resolve_cover_fields(
    user_fields: dict[str, Any] | None, data: dict[str, Any] | None = None
) -> dict[str, str]:
    """Merge user-supplied cover values with report metadata.

    User input always wins.  A blank user value may be filled from the
    extraction payload; a genuinely absent value stays blank and is omitted by
    the renderer rather than printed as ``None``.
    """
    user_fields = user_fields if isinstance(user_fields, dict) else {}
    extracted = _extract_from_data(data)

    def pick(name: str) -> str:
        return _clean(user_fields.get(name)) or extracted.get(name, "")

    date_value = _clean(user_fields.get("date")) or datetime.now().strftime("%b %d, %Y").upper()

    return {
        "structure_name": pick("structure_name"),
        "project_name": pick("project_name"),
        "project_address": pick("project_address"),
        "company_address": _clean(user_fields.get("company_address")) or FIXED_US,
        "date": date_value,
        "rev": _clean(user_fields.get("rev") or user_fields.get("revision")) or "0",
        "description": _clean(user_fields.get("description")) or "Issue for Record",
        "originated_by": _clean(user_fields.get("originated_by")) or "PP",
        "checked_by": _clean(user_fields.get("checked_by")) or "SP",
        "approved_by": _clean(user_fields.get("approved_by")) or "SP",
    }


# --------------------------------------------------------------------------
# Asset lookup and text fitting
# --------------------------------------------------------------------------


def _find_asset(names: Iterable[str], label: str) -> str | None:
    """Return the first existing logo path, or ``None`` when it is missing.

    Missing assets are reported once per process, not once per PDF.
    """
    names = tuple(names)
    for name in names:
        for directory in (ASSETS_DIR, os.path.join(os.getcwd(), "assets")):
            path = os.path.abspath(os.path.join(directory, name))
            if os.path.isfile(path):
                return path
    if label not in _ASSET_WARNINGS:
        _ASSET_WARNINGS.add(label)
        print(
            f"[cover_page] WARNING: missing {label} logo; expected one of "
            f"{', '.join(names)} in {ASSETS_DIR!r} or ./assets. "
            "Blank space reserved.",
            flush=True,
        )
    return None


def _wrap(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Greedy word wrap using the same metrics the canvas will use."""
    text = _clean(text)
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and stringWidth(candidate, font, size) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _fit_size(
    text: str, font: str, preferred: float, max_width: float, minimum: float = 10.0
) -> float:
    """Shrink a single line just enough to stay inside the content width."""
    size = preferred
    while size > minimum and stringWidth(text, font, size) > max_width:
        size -= 0.5
    return size


def _draw_right_lines(
    canvas: pdfcanvas.Canvas,
    lines: Sequence[str],
    x_right: float,
    first_baseline: float,
    font: str,
    size: float,
    leading: float,
    colour: colors.Color,
) -> float:
    """Draw right-aligned lines and return the next baseline below the block."""
    canvas.setFillColor(colour)
    canvas.setFont(font, size)
    baseline = first_baseline
    for line in lines:
        canvas.drawRightString(x_right, baseline, line)
        baseline -= leading
    return baseline


def _draw_logo(
    canvas: pdfcanvas.Canvas,
    path: str | None,
    x_right: float,
    y_top: float,
    box: tuple[float, float],
) -> float:
    """Draw a logo inside ``box`` without stretching. Returns drawn height."""
    if not path:
        return 0.0
    try:
        reader = ImageReader(path)
        width_px, height_px = reader.getSize()
        if width_px <= 0 or height_px <= 0:
            return 0.0
        max_width, max_height = box
        scale = min(max_width / width_px, max_height / height_px)
        draw_width = width_px * scale
        draw_height = height_px * scale
        canvas.drawImage(
            reader,
            x_right - draw_width,
            y_top - draw_height,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        return draw_height
    except Exception as exc:  # a broken asset must not lose the whole report
        print(f"[cover_page] WARNING: failed to draw logo {path!r}: {exc}", flush=True)
        return 0.0


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def _fill_polygon(
    canvas: pdfcanvas.Canvas, points: Sequence[tuple[float, float]], colour: colors.Color
) -> None:
    path = canvas.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    canvas.setFillColor(colour)
    canvas.drawPath(path, fill=1, stroke=0)


def _draw_top_banner(canvas: pdfcanvas.Canvas) -> None:
    """Layered, angular blue/teal banner with thin white separations."""
    width = PAGE_WIDTH
    top = PAGE_HEIGHT
    height = BANNER_HEIGHT
    bottom = top - height

    # Medium teal base with an angled lower-left corner (not a plain rectangle).
    _fill_polygon(
        canvas,
        [
            (0, bottom),
            (width, bottom),
            (width, top),
            (0.10 * width, top),
            (0, top - 0.13 * height),
        ],
        BANNER_TEAL,
    )

    # Lighter blue band entering from the left and sweeping across the middle.
    _fill_polygon(
        canvas,
        [
            (0, bottom + 0.10 * height),
            (0, top - 0.34 * height),
            (0.30 * width, top - 0.12 * height),
            (0.61 * width, bottom + 0.30 * height),
            (0.43 * width, bottom + 0.16 * height),
        ],
        BANNER_LIGHT_BLUE,
    )

    # Broad cyan band overlapping the lower-right portion.
    _fill_polygon(
        canvas,
        [
            (0.34 * width, bottom),
            (0.74 * width, bottom),
            (0.90 * width, bottom + 0.30 * height),
            (0.56 * width, bottom + 0.26 * height),
        ],
        BANNER_CYAN,
    )

    # Dark navy triangular mass in the upper-right.
    _fill_polygon(
        canvas,
        [
            (0.60 * width, top),
            (width, top),
            (width, top - 0.54 * height),
            (0.86 * width, top - 0.20 * height),
        ],
        BANNER_NAVY,
    )

    # Medium blue diagonal band through the centre.
    _fill_polygon(
        canvas,
        [
            (0.31 * width, bottom),
            (0.425 * width, bottom),
            (0.575 * width, top),
            (0.455 * width, top),
        ],
        BANNER_BLUE,
    )

    # Light cyan wedge on the far left, below the light-blue band.
    _fill_polygon(
        canvas,
        [
            (0, bottom),
            (0.17 * width, bottom),
            (0.29 * width, bottom + 0.22 * height),
            (0, bottom + 0.40 * height),
        ],
        BANNER_CYAN,
    )

    # Thin white gaps along selected shape boundaries.
    canvas.saveState()
    canvas.setStrokeColor(WHITE)
    canvas.setLineWidth(2.3)
    canvas.setLineCap(1)
    canvas.line(0.31 * width, bottom, 0.455 * width, top)
    canvas.line(0.425 * width, bottom, 0.575 * width, top)
    canvas.setLineWidth(2.0)
    canvas.line(0.60 * width, top, 0.86 * width, top - 0.20 * height)
    canvas.line(0.86 * width, top - 0.20 * height, width, top - 0.54 * height)
    canvas.setLineWidth(1.6)
    canvas.line(0.34 * width, bottom, 0.56 * width, bottom + 0.26 * height)
    canvas.line(0.56 * width, bottom + 0.26 * height, 0.90 * width, bottom + 0.30 * height)
    canvas.restoreState()


def _draw_title(canvas: pdfcanvas.Canvas, fields: dict[str, str]) -> float:
    """Right-aligned three-line title. Returns the next baseline below it."""
    lines = ["STRUCTURAL CALCULATION", "HELICAL PILE DESIGN"]
    structure = _clean(fields.get("structure_name")).upper()
    if structure:
        lines.append(structure)

    baseline = TITLE_TOP
    for index, line in enumerate(lines):
        size = TITLE_SIZE
        if index == len(lines) - 1 and structure:
            # The structure name may be longer than the fixed title lines.
            size = _fit_size(line, "Helvetica-Bold", TITLE_SIZE, CONTENT_WIDTH, minimum=13.0)
        canvas.setFillColor(TITLE_BLUE)
        canvas.setFont("Helvetica-Bold", size)
        canvas.drawRightString(CONTENT_RIGHT, baseline, line)
        baseline -= TITLE_LEADING
    return baseline


def _draw_project_block(canvas: pdfcanvas.Canvas, fields: dict[str, str], baseline: float) -> float:
    """Project name and address in dark maroon. Returns the next baseline."""
    project = _clean(fields.get("project_name")).upper()
    if project:
        size = _fit_size(project, "Helvetica-Bold", PROJECT_SIZE, CONTENT_WIDTH, minimum=10.0)
        lines = _wrap(project, "Helvetica-Bold", size, CONTENT_WIDTH)
        baseline = _draw_right_lines(
            canvas,
            lines,
            CONTENT_RIGHT,
            baseline,
            "Helvetica-Bold",
            size,
            PROJECT_LEADING,
            PROJECT_MAROON,
        )

    address = _clean(fields.get("project_address"))
    if address:
        lines = _wrap(address, "Helvetica", ADDRESS_SIZE, CONTENT_WIDTH)
        baseline = _draw_right_lines(
            canvas,
            lines,
            CONTENT_RIGHT,
            baseline,
            "Helvetica",
            ADDRESS_SIZE,
            ADDRESS_LEADING,
            PROJECT_MAROON,
        )
    return baseline


def _draw_company_blocks(canvas: pdfcanvas.Canvas, fields: dict[str, str], top: float) -> float:
    """Stack the two right-aligned logo/address blocks. Returns the block bottom."""
    us_logo = _find_asset(_US_LOGO_NAMES, "U.S. Helicals")
    _draw_logo(canvas, us_logo, CONTENT_RIGHT, top, US_LOGO_BOX)
    us_box_height = US_LOGO_BOX[1]
    address_top = top - us_box_height - LOGO_ADDRESS_GAP
    company_address = _clean(fields.get("company_address")) or FIXED_US
    canvas.setFillColor(TEXT_DARK)
    canvas.setFont("Helvetica", PNA_LINE_SIZE)
    canvas.drawRightString(CONTENT_RIGHT, address_top, company_address)

    pna_top = address_top - PNA_LINE_LEADING - LOGO_BLOCK_GAP
    pna_logo = _find_asset(_PNA_LOGO_NAMES, "PNA Associates")
    _draw_logo(canvas, pna_logo, CONTENT_RIGHT, pna_top, PNA_LOGO_BOX)
    pna_box_height = PNA_LOGO_BOX[1]
    baseline = pna_top - pna_box_height - LOGO_ADDRESS_GAP
    canvas.setFillColor(TEXT_DARK)
    canvas.setFont("Helvetica", PNA_LINE_SIZE)
    for line in FIXED_PNA_LINES:
        canvas.drawRightString(CONTENT_RIGHT, baseline, line)
        baseline -= PNA_LINE_LEADING
    return baseline


def _revision_values(fields: dict[str, str]) -> list[str]:
    return [
        _clean(fields.get("rev") or fields.get("revision")) or "0",
        _clean(fields.get("date")),
        _clean(fields.get("description")) or "Issue for Record",
        _clean(fields.get("originated_by")) or "PP",
        _clean(fields.get("checked_by")) or "SP",
        _clean(fields.get("approved_by")) or "SP",
    ]


def _column_widths() -> list[float]:
    widths = [CONTENT_WIDTH * fraction for _label, fraction in REVISION_COLUMNS]
    widths[-1] += CONTENT_WIDTH - sum(widths)
    return widths


def _table_height() -> float:
    return TABLE_TITLE_HEIGHT + TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT


def _draw_revision_table(canvas: pdfcanvas.Canvas, fields: dict[str, str], top: float) -> float:
    """Draw the bordered revision table. Returns the table's bottom edge."""
    x = MARGIN
    width = CONTENT_WIDTH
    widths = _column_widths()
    y = top

    # Title row (merged).
    canvas.setFillColor(TABLE_HEADER)
    canvas.rect(x, y - TABLE_TITLE_HEIGHT, width, TABLE_TITLE_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawCentredString(x + width / 2, y - TABLE_TITLE_HEIGHT + 0.075 * inch, "REVISION RECORD")
    y -= TABLE_TITLE_HEIGHT

    # Column-header row.
    canvas.setFillColor(TABLE_HEADER)
    canvas.rect(x, y - TABLE_HEADER_HEIGHT, width, TABLE_HEADER_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7.3)
    cell_x = x
    for (label, _fraction), cell_width in zip(REVISION_COLUMNS, widths):
        size = _fit_size(label, "Helvetica-Bold", 7.3, cell_width - 6, minimum=5.6)
        canvas.setFont("Helvetica-Bold", size)
        canvas.drawCentredString(cell_x + cell_width / 2, y - TABLE_HEADER_HEIGHT + 0.070 * inch, label)
        cell_x += cell_width
    y -= TABLE_HEADER_HEIGHT

    # Data row.
    canvas.setFillColor(TABLE_ROW)
    canvas.rect(x, y - TABLE_ROW_HEIGHT, width, TABLE_ROW_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(TEXT_DARK)
    values = _revision_values(fields)
    cell_x = x
    for value, cell_width in zip(values, widths):
        size = _fit_size(value, "Helvetica", 7.3, cell_width - 6, minimum=5.6)
        canvas.setFont("Helvetica", size)
        canvas.drawCentredString(cell_x + cell_width / 2, y - TABLE_ROW_HEIGHT + 0.095 * inch, value)
        cell_x += cell_width
    y -= TABLE_ROW_HEIGHT

    # Cyan grid. The title row is deliberately one merged cell.
    canvas.saveState()
    canvas.setStrokeColor(TABLE_BORDER)
    canvas.setLineWidth(0.9)
    canvas.rect(x, y, width, _table_height(), fill=0, stroke=1)
    for horizontal in (y + TABLE_ROW_HEIGHT, y + TABLE_ROW_HEIGHT + TABLE_HEADER_HEIGHT):
        canvas.line(x, horizontal, x + width, horizontal)
    vertical = x
    for cell_width in widths[:-1]:
        vertical += cell_width
        canvas.line(vertical, y, vertical, y + TABLE_ROW_HEIGHT + TABLE_HEADER_HEIGHT)
    canvas.restoreState()
    return y


def _proprietary_lines() -> list[str]:
    return _wrap(PROPRIETARY, "Helvetica-Oblique", PROPRIETARY_SIZE, CONTENT_WIDTH - 2 * PROPRIETARY_PADDING_X)


def _proprietary_height() -> float:
    lines = _proprietary_lines()
    return max(
        0.52 * inch,
        len(lines) * PROPRIETARY_LEADING + 2 * PROPRIETARY_PADDING_Y,
    )


def _draw_proprietary_box(canvas: pdfcanvas.Canvas, bottom: float) -> None:
    x = MARGIN
    width = CONTENT_WIDTH
    height = _proprietary_height()
    canvas.setFillColor(PROPRIETARY_FILL)
    canvas.setStrokeColor(TABLE_BORDER)
    canvas.setLineWidth(0.9)
    canvas.rect(x, bottom, width, height, fill=1, stroke=1)

    lines = _proprietary_lines()
    text_height = len(lines) * PROPRIETARY_LEADING
    baseline = bottom + (height + text_height) / 2 - PROPRIETARY_LEADING + 0.02 * inch
    canvas.setFillColor(TEXT_DARK)
    canvas.setFont("Helvetica-Oblique", PROPRIETARY_SIZE)
    for line in lines:
        canvas.drawCentredString(x + width / 2, baseline, line)
        baseline -= PROPRIETARY_LEADING


def _draw_cover(canvas: pdfcanvas.Canvas, fields: dict[str, str]) -> None:
    # White page background, explicitly, so a coloured viewer background cannot
    # show through the vector art.
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)

    _draw_top_banner(canvas)

    baseline = _draw_title(canvas, fields)
    baseline -= TITLE_GAP_AFTER
    baseline = _draw_project_block(canvas, fields, baseline)
    baseline -= PROJECT_GAP_AFTER
    company_bottom = _draw_company_blocks(canvas, fields, baseline)

    # Anchor the table/proprietary box near the bottom, but never let them
    # collide with the company block for unusually long fields.
    proprietary_top = BOTTOM_MARGIN + _proprietary_height() + TABLE_TO_PROPRIETARY_GAP
    table_top = proprietary_top + _table_height()
    table_top = min(table_top, company_bottom - MIN_GAP_ABOVE_TABLE)

    table_bottom = _draw_revision_table(canvas, fields, table_top)
    proprietary_bottom = (
        table_bottom - TABLE_TO_PROPRIETARY_GAP - _proprietary_height()
    )
    _draw_proprietary_box(canvas, proprietary_bottom)


class _CoverFlowable(Flowable):
    """Compatibility flowable for callers that build a full-page document."""

    def __init__(self, fields: dict[str, str]):
        super().__init__()
        self.fields = fields

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        return PAGE_WIDTH, PAGE_HEIGHT

    def draw(self) -> None:
        _draw_cover(self.canv, self.fields)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_cover_page(fields: dict[str, Any] | None = None) -> list[Flowable]:
    """Return a one-page flowable list (for callers already using Platypus)."""
    return [_CoverFlowable(_normalise_fields(fields))]


def _normalise_fields(fields: dict[str, Any] | None) -> dict[str, str]:
    fields = fields if isinstance(fields, dict) else {}
    return {
        "structure_name": _clean(fields.get("structure_name")),
        "project_name": _clean(fields.get("project_name")),
        "project_address": _clean(fields.get("project_address")),
        "company_address": _clean(fields.get("company_address")) or FIXED_US,
        "date": _clean(fields.get("date")),
        "rev": _clean(fields.get("rev") or fields.get("revision")) or "0",
        "description": _clean(fields.get("description")) or "Issue for Record",
        "originated_by": _clean(fields.get("originated_by")) or "PP",
        "checked_by": _clean(fields.get("checked_by")) or "SP",
        "approved_by": _clean(fields.get("approved_by")) or "SP",
    }


def render_cover_page_pdf(fields: dict[str, Any] | None = None) -> bytes:
    """Render just the cover page and return PDF bytes."""
    buffer = io.BytesIO()
    canvas = pdfcanvas.Canvas(buffer, pagesize=A4, pageCompression=1)
    canvas.setTitle("Cover Page")
    canvas.setAuthor("PNA Technical Services")
    _draw_cover(canvas, _normalise_fields(fields))
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def build_cover_page_pdf(fields: dict[str, Any] | None = None) -> bytes:
    """Compatibility alias used by the web server."""
    return render_cover_page_pdf(fields)


def prepend_cover_page_pdf(pdf_bytes: bytes, cover_pdf: bytes) -> bytes:
    """Return ``cover_pdf`` followed by every page of ``pdf_bytes``."""
    import pymupdf

    base = pymupdf.open(stream=bytes(cover_pdf), filetype="pdf")
    try:
        rest = pymupdf.open(stream=bytes(pdf_bytes), filetype="pdf")
        try:
            base.insert_pdf(rest)
            return base.tobytes(garbage=4, deflate=True)
        finally:
            rest.close()
    finally:
        base.close()
