from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `python scripts/evaluate.py` work from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rag.engine import RAGEngine  # noqa: E402
from app.rag.store import collection_count  # noqa: E402


from app.rag.metrics import retrieval_metrics  # noqa: E402


def main() -> None:
    if collection_count() == 0:
        raise SystemExit("No indexed chunks. Run `python -m app.ingest` first.")
    dataset = json.loads(settings.evaluation_file.read_text(encoding="utf-8"))
    engine = RAGEngine()
    reports = []
    for item in dataset:
        question = item["question"]
        relevant_pages = set(int(page) for page in item["relevant_pdf_pages"])
        results = engine.retrieve(question, top_k=10, use_reranker=False)
        # Measure at page level and count a page once even if several chunks came from it.
        pages = list(dict.fromkeys(int(result.metadata["page_number"]) for result in results))
        metrics = retrieval_metrics(pages, relevant_pages, k=5)
        reports.append({"question": question, "expected_pages": sorted(relevant_pages), "retrieved_pages": pages[:5], **metrics})

    if not reports:
        print("Evaluation dataset is empty.")
        return
    average = {
        key: sum(float(row[key]) for row in reports) / len(reports)
        for key in ("precision_at_k", "recall_at_k", "reciprocal_rank", "hit_at_k")
    }
    print(json.dumps({"per_question": reports, "macro_average_at_5": average}, indent=2))


if __name__ == "__main__":
    main()
