from __future__ import annotations

import re
from typing import Iterable


def clean_text(text: str) -> str:
    """Normalize PDF extraction artifacts while keeping useful paragraph breaks."""
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def recursive_character_chunks(
    text: str, chunk_size: int = 1800, overlap: int = 250
) -> list[str]:
    """Split text near paragraph/sentence boundaries with controlled character overlap.

    Character-based sizing is intentionally used for a dependency-light baseline. It is
    not the same as token-based sizing; tune it against your document and evaluation set.
    """
    text = clean_text(text)
    if not text:
        return []
    if chunk_size < 100:
        raise ValueError("chunk_size should be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    separators = ("\n\n", "\n", ". ", "; ", " ")

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            min_boundary = start + int(chunk_size * 0.65)
            candidates = [text.rfind(sep, min_boundary, hard_end) for sep in separators]
            valid = [position + 1 for position in candidates if position >= min_boundary]
            if valid:
                end = max(valid)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Do not carry a leading partial whitespace sequence into the next chunk.
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start

    # Exact duplicates can occur with repeated headers in a very short document.
    return list(dict.fromkeys(chunk for chunk in chunks if chunk))


def table_rows_to_chunks(
    rows: Iterable[Iterable[object]], rows_per_chunk: int = 12
) -> list[str]:
    """Convert an extracted table into readable text chunks, repeating its header."""
    normalized: list[list[str]] = []
    for row in rows:
        cells = [clean_text(str(cell or "")).replace("\n", " ") for cell in row]
        if any(cells):
            normalized.append(cells)
    if len(normalized) < 2:
        return []

    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    data_rows = normalized[1:]
    if not data_rows:
        return []

    def row_to_text(row: list[str]) -> str:
        return " | ".join(cell.strip() for cell in row)

    result: list[str] = []
    for offset in range(0, len(data_rows), max(1, rows_per_chunk)):
        batch = data_rows[offset : offset + max(1, rows_per_chunk)]
        lines = ["TABLE HEADERS: " + row_to_text(header)]
        lines.extend(row_to_text(row) for row in batch)
        result.append("\n".join(lines))
    return result
