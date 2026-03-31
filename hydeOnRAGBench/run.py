import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import argparse
from tqdm import tqdm

from data import load_ragbench
from index import load_embed_model, build_index, search
from hyde import generate_hypothetical_doc, generate_answer
from evaluate import retrieval_recall, rouge_l, aggregate
from config import TOP_K, DATASET_SUBSET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=DATASET_SUBSET)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default="results.json")
    p.add_argument("--no_hyde", action="store_true",
                   help="baseline: embed raw query instead of hypothetical doc")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading RAGBench [{args.subset}] ...")
    corpus, queries, answers, relevant_docs = load_ragbench(args.split)

    print(f"Building FAISS index over {len(corpus)} documents ...")
    embed_model = load_embed_model()
    index, _ = build_index(embed_model, corpus)

    results = []
    for i, query in enumerate(tqdm(queries, desc="Processing queries")):
        # Step 1: generate hypothetical document (HyDE) or use raw query
        if args.no_hyde:
            search_text = query
        else:
            search_text = generate_hypothetical_doc(query)

        # Step 2+3: embed and retrieve
        _, idxs = search(index, embed_model, [search_text], args.top_k)
        retrieved = [corpus[j] for j in idxs[0]]

        # Step 4: generate answer
        pred_answer = generate_answer(query, retrieved)

        recall = retrieval_recall(retrieved, relevant_docs[i])
        rl = rouge_l(pred_answer, answers[i])
        results.append({
            "query": query,
            "hypothetical_doc": search_text if not args.no_hyde else None,
            "retrieved_docs": retrieved,
            "pred_answer": pred_answer,
            "gold_answer": answers[i],
            "recall": recall,
            "rouge_l": rl,
        })

    metrics = aggregate(results)
    print(f"\n=== Results ({args.subset}, top_k={args.top_k}) ===")
    print(f"Retrieval Recall@{args.top_k}: {metrics['recall@k']:.4f}")
    print(f"ROUGE-L:                      {metrics['rouge_l']:.4f}")

    with open(args.output, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
