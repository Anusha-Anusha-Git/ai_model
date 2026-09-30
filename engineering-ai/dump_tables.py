from app.extraction.document_loader import load_document
doc = load_document("/mnt/c/Users/Anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf")
for t in doc.tables:
    if t.location in ("page 7, table 2", "page 7, table 3", "page 8, table 2", "page 8, table 3", "page 8, table 1", "page 12, table 1", "page 2, table 1", "page 3, table 1"):
        print("="*100)
        print(t.location, f"({len(t.rows)} data rows)")
        print("HDR:", t.headers)
        for r in t.rows:
            print("   ", r)
