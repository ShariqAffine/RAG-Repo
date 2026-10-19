from __future__ import annotations

import logging
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
from app.services.ollama_client import OllamaError, OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Local-first RAG chatbot with text, tables and optional visual-page descriptions.",
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
    ollama = OllamaClient()
    return HealthResponse(
        status="ok",
        ollama_available=ollama.is_available(),
        ollama_model=settings.ollama_model,
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
            message="PDF extracted, embedded and indexed locally.",
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
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models")
def models() -> dict[str, Any]:
    ollama = OllamaClient()
    if not ollama.is_available():
        return {"models": [], "default_model": settings.ollama_model, "available": False}
    try:
        installed = ollama.list_models()
    except OllamaError:
        installed = []
    return {"models": installed, "default_model": settings.ollama_model, "available": True}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    files = sorted(path.name for path in settings.documents_dir.glob("*.pdf"))
    return {"documents": files, "indexed_chunks": collection_count()}
