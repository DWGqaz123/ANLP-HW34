from rouge_score import rouge_scorer


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def retrieval_recall(retrieved_docs, relevant_docs):
    """Fraction of relevant docs found in retrieved set."""
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs)
    hits = sum(1 for d in relevant_docs if d in retrieved_set)
    return hits / len(relevant_docs)


def rouge_l(prediction, reference):
    score = _scorer.score(reference, prediction)
    return score["rougeL"].fmeasure


def aggregate(results):
    """results: list of dicts with 'recall' and 'rouge_l' keys."""
    n = len(results)
    avg_recall = sum(r["recall"] for r in results) / n
    avg_rouge = sum(r["rouge_l"] for r in results) / n
    return {"recall@k": avg_recall, "rouge_l": avg_rouge, "n": n}
