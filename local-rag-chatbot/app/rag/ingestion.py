from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from app.config import settings
from app.rag.chunker import clean_text, recursive_character_chunks, table_rows_to_chunks
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any]


def _stable_id(source_file: str, page: int, content_type: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    safe_name = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name}-p{page}-{content_type}-{index}-{digest}"


def _record(
    source_file: str,
    page_number: int,
    content_type: str,
    chunk_index: int,
    text: str,
    extra_metadata: dict[str, Any] | None = None,
) -> ChunkRecord:
    text = clean_text(text)
    metadata: dict[str, Any] = {
        "source_file": source_file,
        "page_number": int(page_number),
        "content_type": content_type,
        "chunk_index": int(chunk_index),
    }
    for key, value in (extra_metadata or {}).items():
        if value is not None and isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return ChunkRecord(
        id=_stable_id(source_file, page_number, content_type, chunk_index, text),
        text=text,
        metadata=metadata,
    )


def _extract_page_tables(pdf_page: Any) -> list[list[list[object]]]:
    try:
        return pdf_page.extract_tables() or []
    except Exception as exc:  # Table extraction varies with PDF layout.
        logger.warning("Table extraction failed on a page: %s", exc)
        return []


def parse_pdf(pdf_path: Path, vision_client: OllamaClient | None = None) -> list[ChunkRecord]:
    """Extract page text, table chunks and optional visual descriptions from a PDF.

    Pages are numbered from 1 using the physical page position in the PDF. Table rows
    retain their header in each table chunk. Visual descriptions are plain text created
    by an optional local vision model and are embedded alongside text chunks later.
    """
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.name}")
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    records: list[ChunkRecord] = []
    source_file = pdf_path.name
    vision_errors: list[str] = []

    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        for page_index, fitz_page in enumerate(fitz_doc):
            page_number = page_index + 1
            text = clean_text(fitz_page.get_text("text", sort=True))

            if text:
                for chunk_index, piece in enumerate(
                    recursive_character_chunks(
                        text,
                        chunk_size=settings.chunk_size_chars,
                        overlap=settings.chunk_overlap_chars,
                    )
                ):
                    records.append(
                        _record(source_file, page_number, "text", chunk_index, piece)
                    )

            # Tables are stored as separate structured text chunks. On designed reports,
            # rows are often individual positioned text blocks in two side-by-side columns.
            # Group blocks spatially under each TABLE heading to avoid mixing adjacent tables.
            blocks = []
            for block in fitz_page.get_text("blocks", sort=True):
                if len(block) >= 5:
                    block_text = clean_text(str(block[4]))
                    if block_text:
                        blocks.append((float(block[0]), float(block[1]), float(block[2]), block_text))
            heading_indexes = [
                idx for idx, block in enumerate(blocks)
                if re.search(r"(?i)\bTABLE\s+\d+\s*:", block[3])
            ]
            table_index = 0
            if heading_indexes:
                midpoint = float(fitz_page.rect.width) / 2.0
                for heading_pos, block_index in enumerate(heading_indexes):
                    x0, heading_y, _x1, heading_text = blocks[block_index]
                    side = 0 if x0 < midpoint else 1
                    next_y = float(fitz_page.rect.height) + 1
                    for next_index in heading_indexes[heading_pos + 1 :]:
                        nx, ny, _nx1, _nt = blocks[next_index]
                        if (0 if nx < midpoint else 1) == side and ny > heading_y:
                            next_y = min(next_y, ny)
                    section_blocks = [
                        item[3] for item in blocks
                        if (0 if item[0] < midpoint else 1) == side
                        and item[1] >= heading_y
                        and item[1] < next_y
                    ]
                    section = "\n".join(section_blocks).strip()
                    if len(section) < 80:
                        continue
                    table_heading_match = re.search(r"(?i)TABLE\s+\d+\s*:[^\n]*", section)
                    table_heading = table_heading_match.group(0).strip() if table_heading_match else heading_text.splitlines()[0]
                    for chunk_index, piece in enumerate(
                        recursive_character_chunks(
                            section,
                            chunk_size=settings.chunk_size_chars,
                            overlap=settings.chunk_overlap_chars,
                        )
                    ):
                        # Prefix every chunk with the heading so table identity survives splitting.
                        table_text = piece if table_heading.lower() in piece.lower() else f"{table_heading}\n{piece}"
                        records.append(
                            _record(
                                source_file, page_number, "table", chunk_index, table_text,
                                {"table_index": table_index},
                            )
                        )
                    table_index += 1
            else:
                # Fallback for unlabelled tables: use native extraction and repeat the header.
                pdf_page = plumber_doc.pages[page_index]
                for raw_table in _extract_page_tables(pdf_page):
                    table_chunks = table_rows_to_chunks(
                        raw_table, rows_per_chunk=settings.table_rows_per_chunk
                    )
                    for chunk_index, table_text in enumerate(table_chunks):
                        records.append(
                            _record(
                                source_file,
                                page_number,
                                "table",
                                chunk_index,
                                f"Table from PDF page {page_number}. Preserve the row labels and values.\n{table_text}",
                                {"table_index": table_index},
                            )
                        )
                    table_index += 1

            if settings.enable_vision:
                if vision_client is None:
                    vision_client = OllamaClient()
                # A moderate render size reduces memory and vision-token cost while keeping
                # infographics, charts and most small report labels inspectable.
                pix = fitz_page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
                png_bytes = pix.tobytes("png")
                try:
                    description = vision_client.describe_page_image(
                        png_bytes, page_number, has_extractable_text=len(text) >= 100
                    )
                    if description and "no meaningful visual" not in description.lower():
                        records.append(
                            _record(
                                source_file,
                                page_number,
                                "visual_description",
                                0,
                                f"Visual description for PDF page {page_number}:\n{description}",
                            )
                        )
                except Exception as exc:
                    vision_errors.append(f"page {page_number}: {exc}")
                    logger.exception("Vision processing failed for page %s", page_number)
                    # Fail visibly rather than silently claiming that the PDF was fully
                    # multimodally indexed. The exception is raised after extraction.
                    break

    if vision_errors:
        raise RuntimeError(
            "Visual indexing failed; text/table chunks were extracted but have not been "
            "persisted. Check that Ollama is running and the vision model is installed. "
            + " | ".join(vision_errors[:2])
        )

    if not records:
        raise ValueError(f"No extractable content found in {pdf_path.name}")
    return records
