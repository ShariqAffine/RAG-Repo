# Multimodal RAG Chatbot — Hosted APIs, Lightweight Local Setup

A Windows-friendly PDF RAG project that uses **hosted Gemini and Groq APIs** instead of downloading large local LLM, vision, embedding, or reranking models. Python packages are still installed locally; PDFs are parsed locally and ChromaDB stores the index on disk. Text/images sent to Gemini/Groq are processed by those cloud providers.

## What changed in this version

- **Groq** is the default chat provider, with **Gemini chat fallback** enabled by default.
- **Gemini API** creates semantic embeddings; no Sentence Transformers weights are downloaded.
- **Gemini multimodal API** describes charts, diagrams, infographics, and other visual page content during ingestion.
- **PyMuPDF + pdfplumber** extract PDF text and tables locally.
- **ChromaDB** persists vectors locally; **BM25 + Reciprocal Rank Fusion (RRF)** provide hybrid retrieval.
- Removed Ollama integration, local vision/embedding models, and local cross-encoder reranking.
- API keys are read from `.env`; do not commit or share that file.

## Requirements

- Windows 10/11, Python 3.10 or newer (64-bit recommended), internet access.
- Gemini API key: https://aistudio.google.com/app/apikey
- Groq API key: https://console.groq.com/keys

Provider free-tier limits, model availability, and pricing can change. Review each provider's current terms and quotas. This project makes API requests during ingestion and chat; large PDFs may generate multiple embedding and vision requests.

## Setup on Windows

1. Extract the ZIP to a writable folder, for example `C:\Users\you\Downloads\local-rag-chatbot`.
2. Run `setup_windows.bat`. It creates `.venv` and installs Python dependencies. **It does not install Ollama or download model weights.**
3. Open `.env` in a text editor and replace the placeholder values for `GEMINI_API_KEY` and `GROQ_API_KEY` with your keys. Keep the file private.
4. Start the app using `run_windows.bat`.
5. Open http://127.0.0.1:8000 in your browser.

If you prefer PowerShell, run from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use `setup_windows.bat` or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current PowerShell session, then activate again. Add API keys to `.env` before ingesting PDFs.

## Add and index PDFs

- Use the web UI to upload a PDF, or copy PDFs into `data/documents/`.
- To index all PDFs in that folder:

```powershell
.\.venv\Scripts\python.exe -m app.ingest
```

- To index one PDF:

```powershell
.\.venv\Scripts\python.exe -m app.ingest --pdf "data\documents\your-file.pdf"
```

Ingestion extracts text and tables locally, renders each page to a modest-size PNG for Gemini visual description (if `ENABLE_VISION=true`), requests Gemini embeddings, and stores chunks/vectors in local ChromaDB. Re-ingesting a filename replaces its prior indexed chunks.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required for embeddings and visual understanding; also used for Gemini fallback chat |
| `GROQ_API_KEY` | Required for default Groq chat |
| `CHAT_PROVIDER` | `groq` (default) or `gemini` |
| `GROQ_CHAT_MODEL` | Groq chat model ID |
| `GEMINI_CHAT_MODEL` | Gemini chat model ID |
| `GEMINI_VISION_MODEL` | Gemini model used to describe PDF page images |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model ID |
| `EMBEDDING_DIMENSIONS` | Embedding output size; default 768. Keep consistent for an index. |
| `ENABLE_GEMINI_FALLBACK` | If true, attempt Gemini chat when the selected non-Gemini provider fails |
| `ENABLE_VISION` | Set `false` to skip image/chart description and reduce API usage |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | Text chunk size and overlap (characters, not tokens) |
| `TABLE_ROWS_PER_CHUNK` | Rows per table chunk for fallback table extraction |
| `CANDIDATE_K` / `FINAL_TOP_K` | Hybrid retrieval candidate count and final context count |
| `MAX_CONTEXT_CHARS` | Maximum assembled source-context characters sent to chat API |

The default `GROQ_CHAT_MODEL` is `openai/gpt-oss-20b`; provider model IDs can change or be unavailable to a given account. Change the value in `.env` if needed. Gemini embedding vectors are stored in the ChromaDB collection. **If you change the embedding model or dimensionality, delete `data/chroma/` and re-ingest all PDFs** so old and new vectors are not mixed.

## RAG pipeline

1. Parse PDF text with PyMuPDF and tables with pdfplumber/layout heuristics.
2. Split text and table content into page-aware chunks; optionally create visual descriptions through Gemini.
3. Embed chunks using the Gemini embedding API.
4. Store text, metadata, and vectors in local ChromaDB.
5. At question time, embed the question, retrieve vector candidates, retrieve keyword candidates with BM25, and fuse rankings using RRF.
6. Send the question plus selected source context to Groq or Gemini; include source labels and return page/content-type citations.

## Privacy and cost notes

- PDF parsing and ChromaDB storage are local, but the content is **not fully local/private**: chunk text is sent to Gemini for embeddings, and page images are sent to Gemini if vision is enabled. Retrieved context and the question are sent to Groq/Gemini for answer generation.
- Use only documents you are authorized to send to those providers. For sensitive documents, set `ENABLE_VISION=false` or do not use this hosted-API configuration.
- Free quotas are not guaranteed and can change. A PDF with many pages can consume API quota through per-chunk embedding calls and per-page vision calls. Start with a short sample PDF and monitor provider usage.
- Never paste API keys into chat or commit `.env` to source control.

## Evaluation and tests

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the evaluation script (requires an indexed sample corpus and configured API keys):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
```

The evaluation code is a development aid; validate quality with your own representative questions and expected answers. API model output can vary between runs.

## Troubleshooting

- **Missing `GEMINI_API_KEY`:** add the key to `.env`; it is needed for embeddings even if Groq handles chat.
- **Missing `GROQ_API_KEY`:** add the key, or set `CHAT_PROVIDER=gemini` and configure Gemini.
- **401/403:** verify the key, project access, model availability, and provider quota.
- **429/rate limit:** wait, reduce PDF size or disable vision, and retry. Embedding requests include limited retry handling.
- **Vision failures:** set `ENABLE_VISION=false` to continue with text/table-only indexing.
- **Chroma dimension mismatch after config changes:** remove `data/chroma/` and re-index all PDFs.
- **PowerShell execution policy:** use the batch setup/run scripts or a session-scoped policy change as described above.

## Project layout

- `app/rag/ingestion.py` — PDF parsing, text/table/visual chunks
- `app/rag/chunker.py` — cleaning and chunking utilities
- `app/rag/embeddings.py` — hosted Gemini embeddings
- `app/rag/store.py` — persistent ChromaDB operations
- `app/rag/engine.py` — hybrid retrieval, RRF, grounded answer assembly
- `app/services/llm_client.py` — Groq/Gemini chat and Gemini vision API
- `app/main.py` — FastAPI endpoints
- `static/index.html` — browser interface
- `data/documents/` — sample/input PDFs
- `data/chroma/` — generated local vector index (not shipped as a prebuilt index)
