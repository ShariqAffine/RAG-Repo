from __future__ import annotations


def retrieval_metrics(
    retrieved_pages: list[int], relevant_pages: set[int], k: int
) -> dict[str, float]:
    """Compute basic page-level Precision@k, Recall@k, RR and Hit@k."""
    top_pages = retrieved_pages[:k]
    relevant_found = sum(1 for page in top_pages if page in relevant_pages)
    precision = relevant_found / len(top_pages) if top_pages else 0.0
    recall = relevant_found / len(relevant_pages) if relevant_pages else 0.0
    reciprocal_rank = 0.0
    for rank, page in enumerate(top_pages, start=1):
        if page in relevant_pages:
            reciprocal_rank = 1.0 / rank
            break
    hit = 1.0 if relevant_found else 0.0
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "hit_at_k": hit,
    }
