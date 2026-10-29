from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb

from app.config import settings
from app.rag.embeddings import embed_texts
from app.rag.ingestion import ChunkRecord


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name="local_rag_chunks",
        metadata={"hnsw:space": "cosine", "description": "Local PDF RAG chunks"},
    )


def collection_count() -> int:
    return int(get_collection().count())


def upsert_records(records: list[ChunkRecord], replace_source: str | None = None) -> int:
    collection = get_collection()
    if replace_source:
        try:
            collection.delete(where={"source_file": replace_source})
        except Exception:
            # A first ingestion may not have any matching documents.
            pass

    batch_size = 64
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embed_texts([item.text for item in batch])
        collection.upsert(
            ids=[item.id for item in batch],
            documents=[item.text for item in batch],
            metadatas=[item.metadata for item in batch],
            embeddings=vectors,
        )
    return len(records)


def get_all_records() -> dict[str, Any]:
    collection = get_collection()
    if collection.count() == 0:
        return {"ids": [], "documents": [], "metadatas": []}
    return collection.get(include=["documents", "metadatas"])


def vector_search(query_vector: list[float], n_results: int) -> dict[str, Any]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    return collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(1, n_results), count),
        include=["documents", "metadatas", "distances"],
    )
