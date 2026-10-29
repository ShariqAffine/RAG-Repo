# Architecture and decisions

## Goal
A lightweight multimodal PDF RAG app for Windows that avoids local model-weight downloads and uses hosted APIs.

## Architecture

`PDF -> PyMuPDF/pdfplumber extraction -> cleaning/chunking -> Gemini embeddings -> local ChromaDB`

`Question -> Gemini embedding -> vector retrieval + BM25 -> RRF fusion -> context assembly -> Groq chat (Gemini fallback) -> answer with source labels`

During ingestion, PDF pages can be rendered to PNG and sent to Gemini vision for chart/diagram/infographic descriptions. These descriptions become ordinary searchable text chunks.

## Decisions

- **PDF parsing:** PyMuPDF for page text/rendering; pdfplumber for table extraction fallback. Parsing remains local.
- **Embeddings:** Gemini hosted embedding API. No Sentence Transformers package or local embedding weights.
- **Generation:** Groq is the default chat provider; Gemini is configurable as the primary provider and can serve as fallback.
- **Vision:** Gemini multimodal API; `ENABLE_VISION=false` disables this step.
- **Retrieval:** ChromaDB persistent local vector index plus BM25 keyword ranking and Reciprocal Rank Fusion.
- **Reranking:** Local cross-encoder removed to avoid model downloads. Initial ranking uses hybrid retrieval.
- **Storage:** PDF files and ChromaDB index remain on disk locally. Do not commit `.env` or API keys.

## Tradeoffs and limitations

- This is not a fully local or privacy-preserving pipeline: text is sent to Gemini for embeddings, page images may be sent to Gemini for visual description, and retrieved context/questions go to the chat provider.
- API quotas, model names, and free-tier terms can change.
- Embedding model/dimension changes require deleting the old ChromaDB index and re-ingesting all documents.
- PDF layout/table extraction can be imperfect; inspect source pages for important values.
