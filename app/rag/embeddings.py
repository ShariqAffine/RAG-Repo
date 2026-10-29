from __future__ import annotations

import logging
import time
from typing import Iterable
import requests
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingAPIError(RuntimeError):
    pass

def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create hosted Gemini embeddings; no embedding model is downloaded locally."""
    if not texts:
        return []
    api_key = __import__('os').getenv("GEMINI_API_KEY") or __import__('os').getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EmbeddingAPIError("GEMINI_API_KEY is missing. Add it to your .env file before indexing or searching.")
    output: list[list[float]] = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_embedding_model}:embedContent"
    # Use one request per text for broad API compatibility; batch size limits request overhead.
    for batch in _batches(texts, 16):
        for text in batch:
            payload = {"model": f"models/{settings.gemini_embedding_model}", "content": {"parts": [{"text": text[:12000]}]}, "outputDimensionality": settings.embedding_dimensions}
            last_error = None
            for attempt in range(3):
                try:
                    response = requests.post(url, params={"key": api_key}, json=payload, timeout=settings.request_timeout_seconds)
                    if response.status_code == 429 and attempt < 2:
                        time.sleep(2 * (attempt + 1)); continue
                    response.raise_for_status()
                    vector = response.json().get("embedding", {}).get("values")
                    if not vector:
                        raise EmbeddingAPIError("Gemini returned an empty embedding.")
                    output.append(vector)
                    last_error = None
                    break
                except (requests.RequestException, ValueError, EmbeddingAPIError) as exc:
                    last_error = exc
                    if attempt < 2 and isinstance(exc, requests.RequestException) and getattr(exc.response, "status_code", None) in {429, 500, 502, 503}:
                        time.sleep(2 * (attempt + 1)); continue
                    break
            if last_error is not None:
                raise EmbeddingAPIError(f"Gemini embedding request failed: {last_error}") from last_error
    return output
