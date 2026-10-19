from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import settings
from app.rag.ingestion import parse_pdf
from app.rag.store import collection_count, upsert_records
from app.services.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: Path) -> int:
    pdf_path = pdf_path.resolve()
    client = OllamaClient() if settings.enable_vision else None
    logger.info("Extracting %s (vision=%s)", pdf_path.name, settings.enable_vision)
    try:
        records = parse_pdf(pdf_path, vision_client=client)
        logger.info("Extracted %d chunks; embedding and storing locally", len(records))
        count = upsert_records(records, replace_source=pdf_path.name)
        logger.info("Indexed %d chunks. Total indexed chunks: %d", count, collection_count())
        return count
    finally:
        if settings.enable_vision and client is not None:
            client.unload_model(settings.vision_model)


def ingest_folder(folder: Path) -> int:
    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    total = 0
    for pdf_path in pdf_paths:
        total += ingest_pdf(pdf_path)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local PDF files into the local RAG store")
    parser.add_argument("--pdf", type=Path, help="Path to a single PDF file")
    parser.add_argument(
        "--folder", type=Path, default=settings.documents_dir,
        help="Folder containing PDFs (default: data/documents)",
    )
    args = parser.parse_args()
    if args.pdf:
        ingest_pdf(args.pdf)
    else:
        ingest_folder(args.folder)


if __name__ == "__main__":
    main()
