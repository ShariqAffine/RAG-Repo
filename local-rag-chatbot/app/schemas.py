from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranker: bool = False
    model: str | None = Field(default=None, max_length=120)


class SourceResult(BaseModel):
    source_id: str
    chunk_id: str
    source_file: str
    page_number: int | str
    content_type: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    model: str
    retrieved_chunks: int


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_model: str
    vision_enabled: bool
    indexed_chunks: int


class IngestResponse(BaseModel):
    source_file: str
    chunks_indexed: int
    indexed_total_chunks: int
    message: str
