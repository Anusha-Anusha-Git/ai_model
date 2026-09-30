from app.extraction.field_extractor import extract_fields, _print_result
from app.extraction.document_loader import load_document
doc = load_document("/mnt/c/Users/Anusha/Downloads/PEMB (PERIMETER A)_Model raw.pdf")
result = extract_fields(doc)
_print_result(result)
