# Architecture and Trade-offs

## Data model

Each PDF produces a collection of `ChunkRecord` objects with:

- `id`: deterministic ID for source file, page, content type, chunk number, and text hash.
- `text`: searchable text extracted from the page, table, or vision model.
- `metadata`: source filename, physical PDF page, `content_type`, `chunk_index`, and optional table index.

Physical PDF page numbering is used in citations. Printed page numbers inside the report can be offset from physical page numbers (the cover is physical page 1), so always confirm the page label in the PDF viewer.

## Ingestion path

1. PyMuPDF extracts page text and renders pages.
2. `pdfplumber` extracts native tables where possible. Each table is serialized into readable row text with its header preserved.
3. If enabled, Ollama's vision model receives a rendered page and describes visually important content. For pages with very little selectable text it is also asked to transcribe legible page content.
4. Recursive character splitting creates page-aware text chunks.
5. Sentence Transformers creates normalized vectors.
6. ChromaDB stores vectors, text and metadata on disk.

## Retrieval path

- The query is embedded using the same Sentence Transformers model as the indexed chunks.
- ChromaDB returns the nearest vector candidates using cosine distance.
- BM25 ranks the corpus using lexical token overlap.
- Reciprocal Rank Fusion combines the result rankings.
- Optional cross-encoder reranking reorders the candidate list.
- The selected chunks are assembled as evidence with source labels and supplied to the local Ollama chat model.

## Why these choices for a 16 GB CPU laptop?

| Choice | Benefit | Trade-off / alternative |
|---|---|---|
| Ollama local model | Local inference, no hosted token-based billing | CPU latency and lower capability than large hosted models |
| 3B chat model | Less memory than 7B+ models | May need careful prompting and citation verification |
| Local MiniLM embeddings | Fast enough for small local corpora, no hosted embedding bill | Not necessarily best for domain-specific retrieval |
| ChromaDB | Easy persistent local vector index | A managed search engine may be better for production, scale, or fine-grained authorization |
| BM25 + vector + RRF | Combines exact terms and semantic similarity without score calibration | More compute than vector-only retrieval |
| Character-based recursive chunks | Simple, readable, respects common boundaries | Token length is approximate; evaluate against token-aware splitting |
| Separate table chunks | Row/column labels remain close to values | Can duplicate table text already found in page text |
| Vision-generated text descriptions | Makes visuals discoverable in the same text index | Descriptions may omit details or misread numbers; does not support raw image similarity |
| Optional cross-encoder | Can improve ordering of top candidates | Additional model download, latency and CPU/RAM use |

## Known limitations and useful next improvements

1. Build an evaluation set with chunk-level relevance judgements and compare vector-only, BM25-only, hybrid and reranked retrieval.
2. Track extraction quality on tables, columns, charts, and scanned pages separately.
3. For chart-specific numerical questions, use a layout-aware parser or a vision model in an evidence-verification step; do not blindly trust a generated caption.
4. Add document-level access filters before serving non-public data. This sample is a single-user local prototype and does not implement authentication.
5. Add prompt-injection and poisoned-document tests.
6. For larger corpora, use batched ingestion, incremental updates, retries, observability, and a versioned evaluation/embedding pipeline.
