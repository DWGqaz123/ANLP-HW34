from datasets import load_dataset
from config import DATASET_NAME, DATASET_SUBSET


def load_ragbench(split="test"):
    ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split=split)

    # Build corpus: all unique documents across the split
    corpus = []
    seen = set()
    for row in ds:
        for doc in row["documents"]:
            if doc not in seen:
                seen.add(doc)
                corpus.append(doc)

    queries = [row["question"] for row in ds]
    answers = [row["response"] for row in ds]
    # relevant_docs[i] = set of doc strings relevant to query i
    relevant_docs = [set(row["documents"]) for row in ds]

    return corpus, queries, answers, relevant_docs
