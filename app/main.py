from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.ingest import ingest_pdf
from app.rag.engine import RAGEngine
from app.rag.store import collection_count
from app.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.services.llm_client import ProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Lightweight RAG chatbot using hosted Gemini/Groq APIs with local PDF parsing and ChromaDB storage.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(settings.root_dir / "static")), name="static")

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(settings.root_dir / "static" / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chat_provider=settings.chat_provider,
        chat_model=settings.groq_chat_model if settings.chat_provider == "groq" else settings.gemini_chat_model,
        gemini_key_configured=bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        groq_key_configured=bool(os.getenv("GROQ_API_KEY")),
        vision_enabled=settings.enable_vision,
        indexed_chunks=collection_count(),
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def upload_and_ingest(file: UploadFile = File(...)) -> IngestResponse:
    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    destination = settings.documents_dir / filename
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDFs must be 50 MB or smaller.")
        destination.write_bytes(content)
        count = await run_in_threadpool(ingest_pdf, destination)
        return IngestResponse(
            source_file=filename,
            chunks_indexed=count,
            indexed_total_chunks=collection_count(),
            message="PDF extracted locally; embeddings generated through Gemini API; vectors stored locally.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        history: list[dict[str, str]] = [item.model_dump() for item in request.history]
        result: dict[str, Any] = await run_in_threadpool(
            get_engine().answer,
            request.question.strip(),
            history,
            request.top_k,
            request.use_reranker,
            request.model,
        )
        return ChatResponse(**result)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models")
def models() -> dict[str, Any]:
    choices = []
    if os.getenv("GROQ_API_KEY"):
        choices.append(settings.groq_chat_model)
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        choices.append(settings.gemini_chat_model)
    default_model = settings.groq_chat_model if settings.chat_provider == "groq" else settings.gemini_chat_model
    return {"models": choices, "default_model": default_model, "available": bool(choices)}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    files = sorted(path.name for path in settings.documents_dir.glob("*.pdf"))
    return {"documents": files, "indexed_chunks": collection_count()}
