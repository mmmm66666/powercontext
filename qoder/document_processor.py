"""Document text extraction and chunking."""

import re
import fitz  # PyMuPDF


def extract_text(file_path: str, filename: str) -> str:
    """Extract text content from PDF or TXT files."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return _extract_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def _extract_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """Split text into overlapping chunks with positional metadata.

    Strategy: split by double-newlines (paragraphs) first, then by sentences
    if a single paragraph exceeds chunk_size.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[dict] = []
    current = ""
    start_idx = 0

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph stays within limit, accumulate
        if len(current) + len(para) + 1 <= chunk_size:
            sep = "\n" if current else ""
            current += sep + para
            continue

        # Flush accumulated text
        if current:
            chunks.append({"text": current, "start_idx": start_idx})
            # Keep overlap from the tail of current
            overlap_text = current[-chunk_overlap:] if len(current) > chunk_overlap else ""
            current = overlap_text + "\n" + para
            start_idx = i
            continue

        # Single paragraph exceeds chunk_size -> split by sentences
        if len(para) > chunk_size:
            sentences = _split_sentences(para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    sep = "" if not current else " "
                    current += sep + sent
                else:
                    if current:
                        chunks.append({"text": current, "start_idx": i})
                    current = sent
            continue

        current = para
        start_idx = i

    if current.strip():
        chunks.append({"text": current.strip(), "start_idx": start_idx})

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (supports Chinese and English punctuation)."""
    parts = re.split(r"(?<=[.!?。！？；\n])\s*", text)
    return [s for s in parts if s.strip()]