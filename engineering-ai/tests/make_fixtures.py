"""Generate realistic test fixtures for the Stage 2 document loader.

Run from the repo root (engineering-ai/):

    python tests/make_fixtures.py

Produces tests/fixtures/ with one example per supported file type plus a
scanned-style PDF (no text layer) to exercise the OCR warning path.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document as DocxDocument
from docx.shared import Pt
from openpyxl import Workbook

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

PROJECT_INFO = """PROJECT: Oak Creek Wind Farm -- Substation Foundations
CLIENT: Ridgeview Power Co.
LOCATION: Bastrop County, Texas
SUBMISSION DATE: 2026-09-23

The substation equipment is supported on driven steel pipe piles.
Pile section: 324 mm OD nominal, wall thickness 12.7 mm.
Steel grade: Fy = 450 MPa (F355NH, per EN 10219).
Design pile length: 18.0 m embedded depth.
Required axial capacity (ULS): compression 980 kN, tension 610 kN.
Erection loads: shear 45 kN, moment 62 kN*m.
"""

PILE_HEADER = ["Pile", "Diameter (mm)", "Thickness (mm)", "Fy (MPa)", "Length (m)"]
PILE_ROWS = [
    ["HP1", "324", "12.7", "450", "18.0"],
    ["HP2", "406", "12.7", "450", "18.0"],
    ["HP3", "324", "9.5", "450", "15.0"],
]

LOAD_HEADER = ["Load Case", "Compression (kN)", "Tension (kN)", "Shear (kN)", "Moment (kN*m)"]
LOAD_ROWS = [
    ["ULS-1", "980", "610", "45", "62"],
    ["ULS-2", "720", "420", "38", "51"],
    ["SLS-1", "540", "300", "22", "34"],
]


FULL_REPORT = """PROJECT: Greenfield Substation & Control Building
PROJECT NO: G-2049
CLIENT: Meridian Utilities Co.
LOCATION: Comal County, Texas
STRUCTURE: Reinforced concrete foundations on helical piles
SUBMISSION DATE: 2026-08-15

EXECUTIVE SUMMARY
The foundation for the substation transformer is supported on eight (8) driven helical piles.
Pile diameter: 324 mm
Pile wall thickness: 12.7 mm
Yield strength (Fy): 345 MPa
Helix No: 3
Helix dia 1: 12 in
Helix dia 2: 10 in
Helix dia 3: 8 in

LOADS
ULS-1: compression 980 kN, tension 610 kN, shear 45 kN, moment 62 kN*m
ULS-2: compression 720 kN, tension 420 kN, shear 38 kN, moment 51 kN*m

RESULTS
ULS-1 capacity ratio: 0.78
ULS-2 capacity ratio: 0.65

REFERENCES
1. ASCE 7-22 Minimum Design Loads and Associated Criteria for Buildings and Other Structures
2. AISC 360-22 Specification for Structural Steel Buildings
3. Pile Design & Installation Specification, Project G-2049, Rev. B
"""


def make_txt():
    (FIXTURES / "project_info.txt").write_text(PROJECT_INFO, encoding="utf-8")
    (FIXTURES / "full_report.txt").write_text(FULL_REPORT, encoding="utf-8")


def make_csv_comma():
    lines = [",".join(PILE_HEADER)]
    lines += [",".join(r) for r in PILE_ROWS]
    lines += ["", f"Note,Generated fixture,{PILE_ROWS[0][2]},mm wall,{PILE_ROWS[0][1]} mm OD"]
    (FIXTURES / "pile_schedule.csv").write_text("\n".join(lines), encoding="utf-8")


def make_csv_semicolon():
    lines = [";".join(LOAD_HEADER)]
    lines += [";".join(r) for r in LOAD_ROWS]
    (FIXTURES / "loads_semicolon.csv").write_text("\n".join(lines), encoding="utf-8")


def make_xlsx():
    wb = Workbook()
    ws = wb.active
    ws.title = "Pile Schedule"
    ws.append(PILE_HEADER)
    for row in PILE_ROWS:
        ws.append(row)
    ws2 = wb.create_sheet("Loads")
    ws2.append(LOAD_HEADER)
    for row in LOAD_ROWS:
        ws2.append(row)
    # one formula cell with a cached value, to prove it is read as a value
    ws2["F1"] = "Resistance check"
    ws2["F2"] = "=C2+D2"
    wb.save(FIXTURES / "design_report.xlsx")


def make_docx():
    doc = DocxDocument()
    doc.add_heading("Helical Pile Design Report", level=0)
    doc.add_paragraph(f"Project: {PROJECT_INFO.splitlines()[0].split(':', 1)[1].strip()}")
    doc.add_paragraph(
        "The design is based on the geotechnical report and the pile schedule below."
    )

    table = doc.add_table(rows=1, cols=len(PILE_HEADER))
    table.style = "Table Grid"
    for i, header in enumerate(PILE_HEADER):
        table.rows[0].cells[i].text = header
    for row in PILE_ROWS:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value

    doc.save(FIXTURES / "design_report.docx")


def _draw_table(page, x0, y0, col_widths, row_h, header, rows):
    """Draw a ruled table that PyMuPDF's find_tables() can detect."""
    x = x0
    for width in col_widths:
        page.draw_line((x, y0), (x, y0 + row_h * (1 + len(rows))))
        x += width
    page.draw_line((x, y0), (x, y0 + row_h * (1 + len(rows))))
    for i in range(2 + len(rows)):
        y = y0 + i * row_h
        page.draw_line((x0, y), (x, y))
    y = y0
    for row_text in [header] + rows:
        x = x0 + 4
        for width, cell in zip(col_widths, row_text):
            page.insert_text((x, y + 15), cell, fontsize=9)
            x += width
        y += row_h


def make_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Helical Pile Design Report", fontsize=14)
    page.insert_text((72, 80), "Oak Creek Wind Farm -- Substation", fontsize=11)
    page.insert_text((72, 100),
                     "Piles are driven steel pipes, 324 mm OD, 12.7 mm wall, Fy 450 MPa.", fontsize=10)
    _draw_table(
        page,
        x0=72,
        y0=140,
        col_widths=[60, 110, 90, 70, 80],
        row_h=22,
        header=PILE_HEADER,
        rows=PILE_ROWS[:2],
    )
    doc.save(FIXTURES / "design_report.pdf")
    doc.close()


def make_scanned_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 400))
    pix.clear_with(200)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    doc.save(FIXTURES / "scanned_page.pdf")
    doc.close()


def main():
    make_txt()
    make_csv_comma()
    make_csv_semicolon()
    make_xlsx()
    make_docx()
    make_pdf()
    make_scanned_pdf()
    print(f"Fixtures written to {FIXTURES}")
    for path in sorted(FIXTURES.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()