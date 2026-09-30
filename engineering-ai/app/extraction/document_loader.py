"""Stage 2: document ingestion.

Extracts raw text and tables from engineering project documents
(PDF, DOCX, XLSX, CSV, TXT) with full traceability -- every piece of
content carries a source location (file + page/paragraph/sheet/row).

Design rules (Hard Constraints):
- Conversion is mechanical and deterministic. No LLM involved.
- Nothing is invented: unreadable/blank content produces a warning,
  never a guessed value.
- Scanned PDFs have no text layer; they are detected and flagged as
  needing OCR (OCR itself is a later stage). They are never silently
  treated as empty documents.
- Units and numbers are left exactly as printed; nothing is converted.
"""  # noqa: D205

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

try:
    import pymupdf
except ImportError:  # older PyMuPDF releases expose the fitz module name
    import fitz as pymupdf  # type: ignore[no-redef]

try:
    import docx as docx_module
except ImportError:
    docx_module = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}

_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class UnsupportedFileTypeError(ValueError):
    """Raised when load_document receives a file type we cannot read."""


@dataclass
class TextChunk:
    """A piece of raw text and where it came from.

    location is a human-readable source, e.g. "page 3", "paragraph 12",
    "sheet: Loads, row 5" or "line 1..20".
    """

    location: str
    text: str


@dataclass
class Table:
    """A raw table: header row + data rows, with its source location."""

    location: str
    headers: list[str]
    rows: list[list[str]]

    def to_text(self) -> str:
        lines = [" | ".join(self.headers)]
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


@dataclass
class ExtractedDocument:
    """Everything extracted from a single source file."""

    path: Path
    file_type: str
    text_chunks: list[TextChunk] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def full_text(self) -> str:
        """Linear text combining all chunks and tables (for later LLM steps)."""
        parts: list[str] = []
        for chunk in self.text_chunks:
            parts.append(f"[{chunk.location}]\n{chunk.text}")
        for table in self.tables:
            parts.append(f"[{table.location} -- table]\n{table.to_text()}")
        return "\n\n".join(parts)

    def total_text_chars(self) -> int:
        return sum(len(c.text) for c in self.text_chunks)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _safe_str(value) -> str:
    """Normalise a cell to a clean string without ever inventing data."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # 324.0 -> "324" keeps parsing unambiguous
    return str(value).strip()


def _trim_row(row: list[str]) -> list[str]:
    while row and not row[-1]:
        row.pop()
    return row


def _trim_grid(grid: list[list[str]]) -> list[list[str]]:
    """Drop all-empty trailing rows and columns; keep the data that is there."""
    while grid and all(not c for c in grid[-1]):
        grid.pop()
    if not grid:
        return []
    max_cols = max(len(r) for r in grid)
    grid = [r + [""] * (max_cols - len(r)) for r in grid]
    while max_cols > 0 and all(r[max_cols - 1] == "" for r in grid):
        for r in grid:
            r.pop()
        max_cols -= 1
    return grid


def _decode_bytes(raw: bytes) -> str:
    """Decode text bytes, trying common encodings until one succeeds."""
    for encoding in _TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


# --------------------------------------------------------------------------
# Per-format extractors
# --------------------------------------------------------------------------


def _extract_pdf(path: Path) -> ExtractedDocument:
    doc = pymupdf.open(str(path))
    result = ExtractedDocument(path=path, file_type="pdf")

    for page_index, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        blocks = page.get_text("blocks") or []
        image_blocks = [b for b in blocks if len(b) > 6 and b[6] == 1]
        has_images = bool(page.get_images()) or bool(image_blocks)

        if not page_text and has_images:
            result.warnings.append(
                f"page {page_index}: no text layer but the page contains images -- "
                "likely scanned; OCR required (later stage). Values on this page "
                "cannot be read yet."
            )
        elif not page_text and not image_blocks:
            result.warnings.append(f"page {page_index}: empty page (no text, no images).")
        else:
            for block in blocks:
                if len(block) > 6 and block[6] == 0:
                    text = block[4].strip()
                    if text:
                        result.text_chunks.append(
                            TextChunk(f"page {page_index}", text)
                        )

        tables = page.find_tables()
        for table_index, table in enumerate(tables.tables, start=1):
            rows = _trim_grid([_trim_row([_safe_str(c) for c in row])
                               for row in table.extract()])
            if not rows:
                continue
            result.tables.append(
                Table(
                    location=f"page {page_index}, table {table_index}",
                    headers=rows[0],
                    rows=rows[1:],
                )
            )

    doc.close()
    return result


def _extract_docx(path: Path) -> ExtractedDocument:
    if docx_module is None:
        raise RuntimeError("python-docx is not installed; run: pip install python-docx")

    document = docx_module.Document(str(path))
    result = ExtractedDocument(path=path, file_type="docx")

    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            result.text_chunks.append(TextChunk(f"paragraph {index}", text))

    for index, table in enumerate(document.tables, start=1):
        headers, rows = _extract_docx_table(table)
        if headers:
            result.tables.append(
                Table(location=f"table {index}", headers=headers, rows=rows)
            )

    return result


def _extract_docx_table(table) -> tuple[list[str], list[list[str]]]:
    grid = [
        [_docx_cell_text(cell).strip() for cell in row.cells] for row in table.rows
    ]
    grid = _trim_grid(grid)
    if not grid:
        return [], []
    return grid[0], grid[1:]


def _docx_cell_text(cell) -> str:
    """Cell text, including any nested tables, so no content is dropped."""
    parts = [cell.text.strip()]
    for nested in cell.tables:
        headers, rows = _extract_docx_table(nested)
        if headers:
            parts.append(" | ".join(headers))
            parts.extend(" | ".join(r) for r in rows)
    return " ".join(p for p in parts if p)


def _extract_xlsx(path: Path) -> ExtractedDocument:
    if openpyxl is None:
        raise RuntimeError("openpyxl is not installed; run: pip install openpyxl")

    # data_only=True gives cached values (what Excel last saved).
    # data_only=False gives formulas so we never mistake a formula for empty.
    values_wb = openpyxl.load_workbook(path, data_only=True)
    formulas_wb = openpyxl.load_workbook(path, data_only=False)
    result = ExtractedDocument(path=path, file_type="xlsx")

    for sheet_index, sheet in enumerate(values_wb.worksheets, start=1):
        sheet_label = sheet.title or f"Sheet {sheet_index}"
        max_row = sheet.max_row or 0
        max_col = sheet.max_column or 0

        if max_row == 0:
            result.warnings.append(f"sheet: {sheet_label}: empty sheet.")
            continue

        values_grid: list[list[str]] = []
        for row in sheet.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
            values_grid.append([_safe_str(cell.value) for cell in row])
        values_grid = _trim_grid(values_grid)

        if not values_grid:
            result.warnings.append(f"sheet: {sheet_label}: sheet contains no read values.")
        else:
            result.tables.append(
                Table(
                    location=f"sheet: {sheet_label}",
                    headers=values_grid[0],
                    rows=values_grid[1:],
                )
            )

        # Row-by-row text chunks keep provenance for non-tabular sheets.
        for row_index, row in enumerate(values_grid, start=1):
            joined = " ".join(c for c in row if c)
            if joined:
                result.text_chunks.append(
                    TextChunk(f"sheet: {sheet_label}, row {row_index}", joined)
                )

        # Flag formula cells that have no cached value.
        formula_sheet = formulas_wb[sheet.title]
        for row in formula_sheet.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cached = sheet.cell(row=cell.row, column=cell.column).value
                    if cached is None:
                        result.warnings.append(
                            f"sheet: {sheet_label}, cell {cell.coordinate}: formula "
                            f"'{cell.value}' has no cached value (file was not "
                            "recalculated by Excel); value is unknown."
                        )

    values_wb.close()
    formulas_wb.close()
    return result


def _extract_csv(path: Path) -> ExtractedDocument:
    text = _decode_bytes(path.read_bytes())
    result = ExtractedDocument(path=path, file_type="csv")

    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect)
    grid = [_trim_row([_safe_str(c) for c in row]) for row in reader]
    grid = _trim_grid(grid)

    if not grid:
        result.warnings.append("file contains no rows.")
    else:
        result.tables.append(
            Table(
                location="whole file",
                headers=grid[0],
                rows=grid[1:],
            )
        )
        for row_index, row in enumerate(grid, start=1):
            joined = " ".join(c for c in row if c)
            if joined:
                result.text_chunks.append(TextChunk(f"row {row_index}", joined))

    return result


def _extract_txt(path: Path) -> ExtractedDocument:
    text = _decode_bytes(path.read_bytes()).strip()
    result = ExtractedDocument(path=path, file_type="txt")
    if not text:
        result.warnings.append("file is empty.")
        return result
    line_count = len(text.splitlines())
    result.text_chunks.append(TextChunk(f"line 1..{line_count}", text))
    return result


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
    ".csv": _extract_csv,
    ".txt": _extract_txt,
}


def load_document(path: str | Path) -> ExtractedDocument:
    """Extract text/tables from one supported file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(f"expected a file, got a directory: {file_path}")

    ext = file_path.suffix.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise UnsupportedFileTypeError(
            f"unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return extractor(file_path)


def load_directory(directory: str | Path) -> list[ExtractedDocument]:
    """Load every supported file in a directory (non-recursive)."""
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    results = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(load_document(file_path))
    return results


def load_many(paths: list[str | Path]) -> list[ExtractedDocument]:
    """Load a mix of files and directories."""
    results: list[ExtractedDocument] = []
    for entry in paths:
        entry_path = Path(entry)
        if entry_path.is_dir():
            results.extend(load_directory(entry_path))
        else:
            results.append(load_document(entry_path))
    return results


# --------------------------------------------------------------------------
# CLI: python -m app.extraction.document_loader <path-or-dir> [...]
# --------------------------------------------------------------------------


def _print_summary(document: ExtractedDocument) -> None:
    print(f"\n== {document.path} ({document.file_type}) ==")
    print(f"   text chunks : {len(document.text_chunks)} "
          f"({document.total_text_chars()} chars)")
    print(f"   tables      : {len(document.tables)}")
    for table in document.tables[:5]:
        print(f"     - {table.location}: {table.headers}")
    if len(document.tables) > 5:
        print(f"     ... and {len(document.tables) - 5} more")
    if document.warnings:
        print(f"   warnings    : {len(document.warnings)}")
        for warning in document.warnings[:10]:
            print(f"     - {warning}")
        if len(document.warnings) > 10:
            print(f"     ... and {len(document.warnings) - 10} more")


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m app.extraction.document_loader <path-or-dir> [...]")
        return 2

    errors = 0
    for entry in args:
        try:
            docs = load_many([entry])
            for doc in docs:
                _print_summary(doc)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError,
                UnsupportedFileTypeError) as exc:
            errors += 1
            print(f"\n!! {entry}: {exc}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())