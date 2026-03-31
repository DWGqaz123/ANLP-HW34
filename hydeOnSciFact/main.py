from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

from config import (
    DATASET_SPLIT,
    DEFAULT_OUTPUT_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    MAX_QUERIES,
    PREFER_BEIR,
    TOP_KS,
)
from embed import Embedder
from evaluate import evaluate_run
from generate_hyde import HyDEGenerator
from load_data import load_scifact_data
from retrieve import build_faiss_index, retrieve_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyDE vs baseline retrieval on SciFact")
    parser.add_argument("--mode", choices=["baseline", "hyde", "both"], default="both")
    parser.add_argument("--split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--prefer_beir", action="store_true", default=PREFER_BEIR)
    parser.add_argument("--no_prefer_beir", action="store_true")
    parser.add_argument("--max_queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--llm_model", type=str, default=LLM_MODEL)
    parser.add_argument("--top_ks", type=str, default=",".join(map(str, TOP_KS)))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def parse_top_ks(top_ks_str: str) -> List[int]:
    ks = sorted({int(x.strip()) for x in top_ks_str.split(",") if x.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("top_ks must contain positive integers.")
    return ks


def print_summary_table(baseline_metrics: Dict[str, float] | None, hyde_metrics: Dict[str, float] | None) -> None:
    keys = ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    print("\n=== SciFact Retrieval Summary ===")
    print(f"{'Metric':<12} {'Baseline':>12} {'HyDE':>12}")
    print("-" * 38)
    for key in keys:
        b = baseline_metrics.get(key) if baseline_metrics else None
        h = hyde_metrics.get(key) if hyde_metrics else None
        btxt = f"{b:.4f}" if isinstance(b, float) else "-"
        htxt = f"{h:.4f}" if isinstance(h, float) else "-"
        print(f"{key:<12} {btxt:>12} {htxt:>12}")


def run_single_mode(
    mode: str,
    query_ids: List[str],
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    corpus_doc_ids: List[str],
    index,
    embedder: Embedder,
    max_k: int,
    hyde_generator: HyDEGenerator | None,
    top_ks: List[int],
) -> tuple[Dict[str, float], List[dict]]:
    # Build retrieval text inputs.
    retrieval_texts: List[str] = []
    hyde_docs: Dict[str, str | None] = {}

    for qid in tqdm(query_ids, desc=f"Prepare-{mode}"):
        query = queries[qid]
        if mode == "baseline":
            retrieval_texts.append(query)
            hyde_docs[qid] = None
        else:
            if hyde_generator is None:
                raise ValueError("HyDE mode requires hyde_generator.")
            hyde_doc = hyde_generator.generate(query)
            retrieval_texts.append(hyde_doc)
            hyde_docs[qid] = hyde_doc

    query_vectors = embedder.encode(retrieval_texts)
    _, all_indices = retrieve_top_k(index, query_vectors, max_k)

    per_query_retrieved: Dict[str, List[str]] = {}
    per_query_rows: List[dict] = []

    for i, qid in enumerate(query_ids):
        retrieved_doc_ids = [corpus_doc_ids[j] for j in all_indices[i].tolist()]
        per_query_retrieved[qid] = retrieved_doc_ids

        row = {
            "mode": mode,
            "query_id": qid,
            "query": queries[qid],
            "hyde_document": hyde_docs[qid],
            "gold_doc_ids": sorted(qrels[qid]),
            "retrieved_doc_ids": retrieved_doc_ids,
            "hit@1": int(any(d in qrels[qid] for d in retrieved_doc_ids[:1])),
            "hit@5": int(any(d in qrels[qid] for d in retrieved_doc_ids[:5])),
            "hit@10": int(any(d in qrels[qid] for d in retrieved_doc_ids[:10])),
        }
        per_query_rows.append(row)

    metrics = evaluate_run(per_query_retrieved, qrels, top_ks)
    return metrics, per_query_rows


def main() -> None:
    args = parse_args()
    prefer_beir = args.prefer_beir and not args.no_prefer_beir
    top_ks = parse_top_ks(args.top_ks)
    max_k = max(top_ks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels, data_source = load_scifact_data(
        split=args.split,
        prefer_beir=prefer_beir,
        max_queries=args.max_queries,
    )
    query_ids = list(queries.keys())
    corpus_doc_ids = list(corpus.keys())
    corpus_texts = [corpus[did] for did in corpus_doc_ids]

    print(f"Loaded dataset source: {data_source}")
    print(f"Corpus size: {len(corpus_texts)} | Queries: {len(query_ids)}")

    embedder = Embedder(model_name=args.embed_model)
    corpus_vectors = embedder.encode(corpus_texts)
    index = build_faiss_index(corpus_vectors)

    baseline_metrics = None
    hyde_metrics = None
    all_rows: List[dict] = []

    if args.mode in {"baseline", "both"}:
        baseline_metrics, baseline_rows = run_single_mode(
            mode="baseline",
            query_ids=query_ids,
            queries=queries,
            qrels=qrels,
            corpus_doc_ids=corpus_doc_ids,
            index=index,
            embedder=embedder,
            max_k=max_k,
            hyde_generator=None,
            top_ks=top_ks,
        )
        all_rows.extend(baseline_rows)

    if args.mode in {"hyde", "both"}:
        hyde_generator = HyDEGenerator(model_name=args.llm_model)
        hyde_metrics, hyde_rows = run_single_mode(
            mode="hyde",
            query_ids=query_ids,
            queries=queries,
            qrels=qrels,
            corpus_doc_ids=corpus_doc_ids,
            index=index,
            embedder=embedder,
            max_k=max_k,
            hyde_generator=hyde_generator,
            top_ks=top_ks,
        )
        all_rows.extend(hyde_rows)

    print_summary_table(baseline_metrics, hyde_metrics)

    metrics_payload = {
        "config": {
            "split": args.split,
            "prefer_beir": prefer_beir,
            "data_source": data_source,
            "embed_model": args.embed_model,
            "llm_model": args.llm_model,
            "top_ks": top_ks,
            "max_queries": args.max_queries,
            "mode": args.mode,
        },
        "baseline": baseline_metrics,
        "hyde": hyde_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    per_query_path = output_dir / "per_query_results.jsonl"
    with per_query_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved per-query results: {per_query_path}")


if __name__ == "__main__":
    main()
