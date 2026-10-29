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
    app_name: str = os.getenv("APP_NAME", "Multimodal RAG Chatbot (Hosted APIs)")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    # Hosted providers. Groq is used for chat by default; Gemini is fallback and handles vision + embeddings.
    chat_provider: str = os.getenv("CHAT_PROVIDER", "groq").strip().lower()
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    gemini_vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    enable_vision: bool = _bool("ENABLE_VISION", True)
    enable_gemini_fallback: bool = _bool("ENABLE_GEMINI_FALLBACK", True)

    # Download-free default: cross-encoder reranking is intentionally disabled/removed.
    enable_reranker: bool = _bool("ENABLE_RERANKER", False)
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))
    table_rows_per_chunk: int = int(os.getenv("TABLE_ROWS_PER_CHUNK", "12"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "12"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    rrf_constant: int = int(os.getenv("RRF_CONSTANT", "60"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))

    documents_dir: Path = ROOT_DIR / "data" / "documents"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    evaluation_file: Path = ROOT_DIR / "data" / "evaluation" / "questions.json"

settings = Settings()
settings.documents_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
