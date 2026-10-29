
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make `python scripts/evaluate.py` work from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rag.engine import RAGEngine  # noqa: E402
from app.rag.metrics import retrieval_metrics  # noqa: E402
from app.rag.answer_metrics import  answer_metrics
from app.rag.store import collection_count  # noqa: E402
from app.services.llm_client import ProviderError  # noqa: E402



def main() -> None:
    if collection_count() == 0:
        raise SystemExit(
            "No indexed chunks. Run `python -m app.ingest` first."
        )

    dataset = json.loads(
        settings.evaluation_file.read_text(encoding="utf-8")
    )
    engine = RAGEngine()
    reports = []

    for item in dataset:
        question = item["question"]
        relevant_pages = {
            int(page) for page in item["relevant_pdf_pages"]
        }

        # Retrieval evaluation.
        results = engine.retrieve(
            question, top_k=10, use_reranker=False
        )

        pages = list(
            dict.fromkeys(
                int(result.metadata["page_number"])
                for result in results
            )
        )
        metrics = retrieval_metrics(pages, relevant_pages, k=5)

        # Answer generation and latency measurement.
        start_time = time.perf_counter()

        try:
            answer_result = engine.answer(
                question, top_k=5, use_reranker=False
            )
            generated_answer = answer_result["answer"]
            model_used = answer_result["model"]
            api_error = None

        except (ProviderError, TimeoutError) as exc:
            generated_answer = None
            model_used = None
            api_error = str(exc)

        answer_latency = round(
            time.perf_counter() - start_time, 3
        )

        print("QUESTION:", question)
        print("REFERENCE:", item.get("reference_answer"))
        print("GENERATED:", generated_answer)
        print("MODEL:", model_used)

        if api_error:
            print("API ERROR:", api_error)

        answer_scores = {
                "answer_precision": None,
                "answer_recall": None,
                "answer_f1": None,
                "numeric_fact_recall": None,
            }

        if generated_answer is not None and item.get("reference_answer"):
            answer_scores = answer_metrics(
                generated_answer,
                item["reference_answer"],
            )
        reports.append({
            "question": question,
            "reference_answer": item.get("reference_answer"),
            "generated_answer": generated_answer,
            **answer_scores,
            #"answer_correctness": correctness,
            "model": model_used,
            "api_error": api_error,
            "expected_pages": sorted(relevant_pages),
            "retrieved_pages": pages[:5],
            **metrics,
            "answer_latency_seconds": answer_latency,
            
        })

    if not reports:
        print("Evaluation dataset is empty.")
        return

    # Existing retrieval metrics.
    average = {
        key: sum(float(row[key]) for row in reports) / len(reports)
        for key in (
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "hit_at_k",
        )
    }

    # Answer-generation metrics.
    answer_success_count = sum(
        1 for row in reports
        if row["generated_answer"] is not None
    )

    latencies = [
        row["answer_latency_seconds"]
        for row in reports
    ]

    

    summary = {
        **average,
        "answer_generation_success_rate": (
            answer_success_count / len(reports)
        ),
        "average_answer_latency_seconds": round(
            sum(latencies) / len(latencies), 3
        ),
        "api_failure_count": sum(
            1 for row in reports
            if row["api_error"] is not None
        ),
    }

    score_keys = (
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "numeric_fact_recall",
        )
    
    for key in score_keys:
        scores = [
            row[key]
            for row in reports
            if row.get(key) is not None
        ]

        summary[f"average_{key}"] = (
            round(sum(scores) / len(scores), 3)
            if scores
            else None
        )

        summary[f"{key}_evaluated_count"] = len(scores)

    print(json.dumps({
        "per_question": reports,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
