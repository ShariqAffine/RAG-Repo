from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    app_name: str = os.getenv("APP_NAME", "Local Multimodal RAG Chatbot")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    # Local Ollama models. No hosted API key is needed.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    vision_model: str = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
    enable_vision: bool = _bool("ENABLE_VISION", True)

    # Embedding model runs locally through Sentence Transformers.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # A cross-encoder reranker is optional because this laptop is CPU-based.
    enable_reranker: bool = _bool("ENABLE_RERANKER", False)
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))
    table_rows_per_chunk: int = int(os.getenv("TABLE_ROWS_PER_CHUNK", "12"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "12"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    rrf_constant: int = int(os.getenv("RRF_CONSTANT", "60"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

    documents_dir: Path = ROOT_DIR / "data" / "documents"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    evaluation_file: Path = ROOT_DIR / "data" / "evaluation" / "questions.json"


settings = Settings()

# Ensure local working directories exist without needing a separate setup step.
settings.documents_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
