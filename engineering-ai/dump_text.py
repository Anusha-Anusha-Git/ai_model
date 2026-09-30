from app.extraction.document_loader import load_document
doc = load_document("/mnt/c/Users/Anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf")
for c in doc.text_chunks:
    if c.location in ("page 1", "page 2", "page 4", "page 6", "page 7", "page 8"):
        print("=====", c.location)
        print(c.text)
