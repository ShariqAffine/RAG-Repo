from app.rag.metrics import retrieval_metrics


def test_retrieval_metrics():
    result = retrieval_metrics([2, 4, 7, 8], {4, 8}, k=3)
    assert result["precision_at_k"] == 1 / 3
    assert result["recall_at_k"] == 0.5
    assert result["reciprocal_rank"] == 0.5
    assert result["hit_at_k"] == 1.0
