import sys
from pypdf import PdfReader

path = sys.argv[1] if len(sys.argv) > 1 else "RAGInitialLoadData/1.1 Propositional Logic p2.pdf"

reader = PdfReader(path)
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"--- Page {i+1} ---")
    print(text if text else "(empty)")
    print()
