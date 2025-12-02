# form_parser.py
from io import BytesIO
from typing import Tuple

from PyPDF2 import PdfReader
from docx import Document as DocxDocument


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_text_from_plain(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text(filename: str, file_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (detected_type, text).
    detected_type: 'pdf' | 'docx' | 'txt' | 'unknown'
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf", extract_text_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return "docx", extract_text_from_docx(file_bytes)
    if lower.endswith((".txt", ".csv")):
        return "txt", extract_text_from_plain(file_bytes)
    # Fallback – try plain decode
    return "unknown", extract_text_from_plain(file_bytes)
