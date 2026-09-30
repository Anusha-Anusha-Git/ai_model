from app.extraction.document_loader import load_document
import re
def norm(t):
    return re.sub(r"[\s_\-./()]+", " ", t.strip().lower()).strip()
doc = load_document("/mnt/c/Users/Anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf")
for t in doc.tables:
    print(f"{t.location}: {[norm(h) for h in t.headers]}")
