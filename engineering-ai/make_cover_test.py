from app.report import cover_page_v2
import os

fields = {
    "structure_name": "",
    "project_name": "",
    "project_address": "",
    "date": "",
}
pdf = cover_page_v2.build_cover_page_pdf(fields)
out = os.path.join(os.path.dirname(__file__), "cover_test.pdf")
with open(out, "wb") as f:
    f.write(pdf)
print(f"Wrote {out}")
