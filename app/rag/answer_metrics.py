
import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize an answer before comparing its tokens."""
    text = text.lower()
    text = re.sub(r"\[\s*s\d+\s*\]", " ", text)
    text = re.sub(r"【[^】]*】", " ", text)
    text = re.sub(r"\*\*|__|[*_`]", "", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\u2011", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text.strip()


def answer_metrics(
    generated_answer: str,
    reference_answer: str,
) -> dict:
    """Calculate token precision, recall, F1, and numeric recall."""
    generated_tokens = normalize_answer(generated_answer).split()
    reference_tokens = normalize_answer(reference_answer).split()

    if not generated_tokens or not reference_tokens:
        precision = recall = f1 = 0.0
    else:
        generated_counts = Counter(generated_tokens)
        reference_counts = Counter(reference_tokens)

        common = sum(
            (generated_counts & reference_counts).values()
        )

        precision = common / len(generated_tokens)
        recall = common / len(reference_tokens)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    # Compare numeric values mentioned in the reference.
    def extract_numbers(text: str) -> set:
        values = re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?", text)
        return {value.replace(",", "") for value in values}

    reference_numbers = extract_numbers(reference_answer)
    generated_numbers = extract_numbers(generated_answer)

    numeric_recall = (
        len(reference_numbers & generated_numbers)
        / len(reference_numbers)
        if reference_numbers
        else None
    )

    return {
        "answer_precision": round(precision, 3),
        "answer_recall": round(recall, 3),
        "answer_f1": round(f1, 3),
        "numeric_fact_recall": (
            round(numeric_recall, 3)
            if numeric_recall is not None
            else None
        ),
    }
