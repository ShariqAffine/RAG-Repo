# Local Multimodal RAG Chatbot

A local-first RAG application built around the supplied **World Bank Group Annual Report 2024 Summary** PDF. It runs a FastAPI backend and a small browser chat UI. Documents, embeddings, retrieval, and generation stay on your computer after the model files have been downloaded.

## Your laptop and recommended configuration

Your screenshot shows an 11th Gen Intel Core i5-1135G7 and 16 GB RAM on a 64-bit Windows machine. This project uses CPU embeddings, a compact local chat model, and optional vision/reranking stages. Expect initial model downloads and slower responses than a hosted GPU service.

**Recommended baseline**

- Chat LLM: Ollama `qwen2.5:3b` (small local instruction model).
- Visual page understanding: Ollama `qwen2.5vl:3b` (optional, but enabled in `.env.example` to index charts and infographics).
- Text embeddings: `sentence-transformers/all-MiniLM-L6-v2` running on CPU.
- Vector index: ChromaDB persisted to `data/chroma/`.
- Keyword retrieval: BM25.
- Hybrid ranking: Reciprocal Rank Fusion (RRF).
- Reranker: local `cross-encoder/ms-marco-MiniLM-L-6-v2`, disabled by default.
- API/UI: FastAPI + plain HTML/CSS/JavaScript.

No OpenAI, Azure, or other hosted LLM API key is required. The `.env` file contains configuration, not secrets. You can install several compatible Ollama models and choose among them in the UI; only the selected model runs for a given answer. Ollama and the embedding model need to be downloaded once; after that, model inference can run locally/offline.

## What the supplied PDF exercises

The included PDF is 15 pages and approximately 2.4 MB. It contains selectable text, financial tables, photographs, and infographic/chart-style pages. It is copied to:

`data/documents/WBAR 2024 English Executive Summary.pdf`

The ingestion process creates different records:

1. **Text chunks** — extracted per PDF page and split near paragraph/sentence boundaries.
2. **Table chunks** — tables extracted with `pdfplumber`, serialized as row-oriented text with the header repeated when large tables are split.
3. **Visual-description chunks** — if vision is enabled, each page is rendered to an image and a local vision model describes charts, diagrams, photographs and infographic relationships. Pages with little extractable text also get a transcription-focused prompt.

Every chunk stores the source filename, physical PDF page number, content type, and chunk index. The generated text descriptions are embedded alongside regular text/table chunks. This is **vision-assisted text retrieval**, not raw image-vector retrieval; see design trade-offs below.

## Architecture

```text
PDF(s)
  ├── PyMuPDF: page text + page rendering
  ├── pdfplumber: table extraction
  └── Ollama vision model (optional): page visual description
             ↓
    page-aware chunks + metadata
             ↓
    Sentence Transformers local embeddings
             ↓
    ChromaDB persistent vector index

User question
  ├── local query embedding → vector search
  ├── lexical tokens → BM25
  └── Reciprocal Rank Fusion (RRF)
             ↓
    optional local cross-encoder reranking
             ↓
    top-k evidence chunks + chat history
             ↓
    Ollama local LLM
             ↓
    answer with source IDs, filename, physical PDF page and excerpt
```

## 1. Install prerequisites on Windows

### A. Install Python

Install **Python 3.11 x64**. Python 3.11 is recommended for a smoother dependency installation with PyTorch/Sentence Transformers. During setup, enable the option to add Python to PATH if offered.

### B. Install Ollama

Install Ollama for Windows from <https://ollama.com/download/windows>, then open a new terminal. Ollama normally runs as a local service on `http://localhost:11434`.

Pull the chat and vision models:

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
ollama list
```

`qwen2.5vl:3b` is used to describe visual material in the PDF. It requires a recent Ollama version; if Ollama reports that this model needs a newer version, update Ollama and retry. The vision download is several GB, so allow disk space and time.

To test the chat model independently:

```powershell
ollama run qwen2.5:3b
```

Ask it a test question, then type `/bye` to exit.

### C. Set up the Python application

Extract the project ZIP, open PowerShell in the `local-rag-chatbot` folder, and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks script activation in the current terminal, use:

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

You can use `setup_windows.bat` for the venv and package installation if Python 3.11 is installed. Avoid running two setup methods at the same time.

The first call to Sentence Transformers downloads `all-MiniLM-L6-v2` from its public model repository. It does not require a hosted API key. Once downloaded, it is cached locally.

## 2. Configure `.env`

The default `.env.example` is appropriate as a starting point for this laptop. The main settings are:

| Variable | Purpose | Starting value |
|---|---|---|
| `OLLAMA_MODEL` | Model that generates chatbot answers | `qwen2.5:3b` |
| `ENABLE_VISION` | Generate visual summaries during ingestion | `true` |
| `VISION_MODEL` | Local model for page images | `qwen2.5vl:3b` |
| `EMBEDDING_MODEL` | Local semantic embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE_CHARS` | Target chunk length, in characters | `1800` |
| `CHUNK_OVERLAP_CHARS` | Repeated boundary context, in characters | `250` |
| `CANDIDATE_K` | Candidate pool used by retrieval | `12` |
| `FINAL_TOP_K` | Final evidence chunks | `5` |
| `ENABLE_RERANKER` | Enable local cross-encoder by default | `false` |
| `MAX_CONTEXT_CHARS` | Limit context sent to the LLM | `14000` |

Chunk sizes are **characters, not tokens**. This is deliberate for a dependency-light first implementation. If you need token-accurate chunks, use the tokenizer associated with your target embedding/model and evaluate the change.

To get a fast text-only first run without pulling a vision model, set `ENABLE_VISION=false`. Text and tables still work; charts and photos will not be described or independently searchable. After switching the setting, re-run ingestion to apply it.

## 3. Index the included sample PDF

Make sure Ollama is running and both models are installed when `ENABLE_VISION=true`. In the project root, with your virtual environment activated, run:

```powershell
python -m app.ingest
```

The command indexes all `*.pdf` files in `data/documents`. To index one PDF from another path:

```powershell
python -m app.ingest --pdf "C:\path\to\report.pdf"
```

The first ingestion can take several minutes on CPU because each page is rendered and described by the vision model. The embedding model is also downloaded on first use. Ingestion persists the index under `data/chroma/`; it does not need to re-embed the source PDF on every chat request.

If visual processing fails because a model was not found, pull the model named in `.env` and run ingestion again. If memory or speed is a problem, set `ENABLE_VISION=false` for text/table indexing. The app deliberately fails the indexing operation visibly when enabled visual processing fails rather than silently claiming that visual content was indexed.

## 4. Launch the application

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open <http://127.0.0.1:8000>. API docs are at <http://127.0.0.1:8000/docs> and the health endpoint is <http://127.0.0.1:8000/api/health>.

Alternatively, after setup, double-click `run_windows.bat` to start the API without `--reload`.

The UI lets you:

- Ask questions about indexed PDFs.
- Choose any model already installed in Ollama from the UI (install another model with `ollama pull <model-name>`, then refresh).
- See retrieved source excerpts, PDF page numbers and content types.
- Upload and index a new PDF.
- Change final top-k and optionally request reranking.

## 5. Test questions for the included PDF

Try:

- What is the World Bank Group's vision?
- How many people did the World Bank Group help feed according to the results infographic?
- What were total World Bank Group commitments in 2024, and how did they compare with 2023?
- What were IBRD commitments by region in 2024?
- What was IFC's total long-term investment commitment for 2024?
- What does the report say about providing electricity access in Sub-Saharan Africa by 2030?

Check the cited physical PDF page, particularly when asking questions about exact table values or chart labels.

## 6. Evaluate retrieval

After ingestion, run:

```powershell
python scripts/evaluate.py
```

The small hand-labeled dataset is in `data/evaluation/questions.json`. It reports Precision@5, Recall@5, reciprocal rank (RR) and Hit@5 per question and a macro average. Relevant pages are currently used as labels, so the metrics assess page-level retrieval rather than exact chunk-level relevance. This is a starter harness, not a statistically robust benchmark; add many representative and adversarial questions before tuning for production.

For a fair comparison, evaluate at least:

1. Vector-only retrieval.
2. BM25-only retrieval.
3. Hybrid retrieval with RRF (the current default).
4. Hybrid retrieval plus reranking.

Use identical questions and relevance judgments for each configuration.

## API endpoints

- `GET /` — browser chat interface.
- `GET /api/health` — Ollama reachability, configured model and indexed-chunk count.
- `GET /api/documents` — PDF files found in the document folder and the index count.
- `GET /api/models` — local Ollama models installed on this machine.
- `POST /api/ingest` — multipart form upload field named `file`; extracts, embeds and indexes the PDF.
- `POST /api/chat` — accepts JSON such as:

```json
{
  "question": "What were total World Bank Group commitments in 2024?",
  "history": [],
  "top_k": 5,
  "use_reranker": false,
  "model": "qwen2.5:3b"
}
```

## Design decisions: why this approach?

### Why Ollama instead of a hosted LLM API?

It avoids usage-based API charges and keeps documents, prompts and responses local. It also lets you swap among installed local models from the UI; `.env` defines the default model. The trade-off is CPU latency, memory usage, and generally lower answer quality than larger hosted models.

### Why a local Sentence Transformers embedding model?

Embedding is a separate task from answer generation. A compact general-purpose model is cheap to run on CPU, can be cached, and does not require an embedding API key. It is not domain-specific, so retrieval should be evaluated against your own questions before relying on it.

### Why ChromaDB?

It provides persistent local vector search with metadata and cosine-distance configuration, without operating a separate database service. For a production enterprise workload with access-control requirements, index size, concurrent users and governance, reassess whether managed search or a larger vector database is needed.

### Why hybrid retrieval and RRF?

Vector retrieval helps with paraphrases and semantic similarity; BM25 helps with exact terms, codes, acronyms and named entities. RRF combines rankings without assuming raw keyword and vector scores are calibrated to the same scale. The trade-off is that lexical ranking adds CPU work and duplicate candidates should be watched.

### Why optional reranking?

A cross-encoder can score the query and each candidate jointly and improve the final order. It adds a model download, RAM use and CPU latency; keep it disabled at first and enable it only if evaluation shows a benefit.

### Why not use raw image embeddings in this version?

This implementation uses the vision model to turn each page's visual information into a searchable text description. The text embedding model then indexes those descriptions together with text and table chunks. This simplifies one unified retrieval path and citation scheme. It is not ideal for fine-grained image similarity or exact chart-data extraction; a later version can add OCR/layout extraction, table-specific parsing, or a separate CLIP-style image embedding index.

### Why page-aware chunks?

This report is a designed annual summary with tables and infographics laid out on specific pages. Storing physical page number and content type makes citations interpretable. Page-based chunking can split long content across multiple chunks, while tables are separately formatted so column headers remain attached to row values.

## Limitations and safety notes

- Visual summaries are generated by a small local vision model and can misread charts or numbers. Verify exact numerical claims against the cited PDF page.
- Table extraction depends on the structure of the source PDF. Some tables may need OCR/layout-aware extraction or manual correction.
- The system uses short conversation history sent by the browser; it is not a durable user-memory store.
- This sample is intended for local experimentation. It has no user authentication or document-level authorization. Do not expose the server to a network or use it for confidential multi-user data without implementing authentication, access filtering and audit logging.
- Source documents are untrusted inputs. The generation prompt instructs the LLM not to follow instructions embedded in documents, but prompt injection protection is not guaranteed.
- Model licenses vary. Check the terms of each model before using it commercially or distributing it.

## Common issues

**`Ollama is not responding`**: confirm the Ollama app/service is running and `http://localhost:11434/api/tags` opens locally.

**`model was not found`**: run `ollama pull qwen2.5:3b` or `ollama pull qwen2.5vl:3b`, matching the value in `.env`.

**Very slow visual indexing**: `ENABLE_VISION=true` sends each page image to a local model. For a quick prototype, set it to `false` and re-run `python -m app.ingest`; text and extracted tables remain available.

**Embedding download failed**: verify internet access for the initial download. Once cached, the embedding model can be used locally.

**Change chunking/model settings**: re-run ingestion after changing chunk settings, the embedding model, table strategy, or vision setting. If you switch embedding models, use a fresh index (delete `data/chroma/` after stopping the app) because old vectors are not compatible with the new embedding space.
