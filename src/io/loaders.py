import os
from pypdf import PdfReader
from docx import Document as DocxDocument

def read_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        texts = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts)

def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext in [".txt", ".md"]:
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
