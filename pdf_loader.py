import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pypdf is not installed. Run: pip install pypdf")


def load_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)

    if not path.exists():
        sys.exit(f"[Error] File not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        sys.exit(f"[Error] Expected a .pdf file, got: {path.suffix}")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        sys.exit(f"[Error] Could not open PDF '{pdf_path}': {exc}")

    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"[Warning] Could not extract text from page {page_num}: {exc}")
            text = ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)

    if not full_text.strip():
        sys.exit("[Error] No readable text found. The PDF may be scanned/image-only.")

    print(f"[Info] Loaded '{path.name}' ({len(reader.pages)} pages, {len(full_text):,} chars)")
    return full_text
