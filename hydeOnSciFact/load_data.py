from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

from datasets import load_dataset


Corpus = Dict[str, str]
Queries = Dict[str, str]
Qrels = Dict[str, Set[str]]


def _build_doc_text(title: str, abstract) -> str:
    if isinstance(abstract, list):
        abstract_text = " ".join([x for x in abstract if isinstance(x, str)])
    else:
        abstract_text = abstract if isinstance(abstract, str) else ""
    title = title if isinstance(title, str) else ""
    return f"Title: {title}\nAbstract: {abstract_text}".strip()


def _try_load_beir_scifact(split: str) -> Tuple[Corpus, Queries, Qrels]:
    """
    Try several common BEIR-on-HF layouts.
    Returns corpus, queries, qrels if successful; raises on failure.
    """
    last_error = None

    # Layout A: one dataset with configs corpus/queries/qrels
    try:
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("BeIR/scifact", "queries", split=split)
        qrels_ds = load_dataset("BeIR/scifact", "qrels", split=split)
        return _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds)
    except Exception as e:  # pragma: no cover
        last_error = e

    # Layout B: qrels in dedicated dataset
    try:
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("BeIR/scifact", "queries", split=split)
        qrels_ds = load_dataset("BeIR/scifact-qrels", split=split)
        return _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds)
    except Exception as e:  # pragma: no cover
        last_error = e

    # Layout C: mteb style dataset naming
    try:
        corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("mteb/scifact", "queries", split=split)
        qrels_ds = load_dataset("mteb/scifact", "qrels", split=split)
        return _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds)
    except Exception as e:  # pragma: no cover
        last_error = e

    raise RuntimeError(f"Failed to load BEIR-formatted SciFact. Last error: {last_error}")


def _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds) -> Tuple[Corpus, Queries, Qrels]:
    corpus: Corpus = {}
    for row in corpus_ds:
        doc_id = str(row.get("_id", row.get("doc_id", row.get("id"))))
        if doc_id == "None":
            continue
        text = row.get("text", "")
        title = row.get("title", "")
        if title and text:
            corpus[doc_id] = f"Title: {title}\nPassage: {text}"
        elif text:
            corpus[doc_id] = text
        else:
            corpus[doc_id] = _build_doc_text(title, row.get("abstract", ""))

    queries: Queries = {}
    for row in queries_ds:
        qid = str(row.get("_id", row.get("query_id", row.get("id"))))
        query_text = row.get("text", row.get("query", row.get("claim", "")))
        if qid != "None" and isinstance(query_text, str) and query_text.strip():
            queries[qid] = query_text

    qrels: Qrels = defaultdict(set)
    for row in qrels_ds:
        qid = str(row.get("query-id", row.get("query_id", row.get("qid", row.get("id")))))
        did = str(row.get("corpus-id", row.get("doc_id", row.get("did"))))
        score = row.get("score", row.get("label", 1))
        if qid == "None" or did == "None":
            continue
        if isinstance(score, (int, float)) and score <= 0:
            continue
        qrels[qid].add(did)

    # Keep only queries that have labels and existing docs.
    filtered_queries: Queries = {}
    filtered_qrels: Qrels = {}
    for qid, query in queries.items():
        gold = {did for did in qrels.get(qid, set()) if did in corpus}
        if gold:
            filtered_queries[qid] = query
            filtered_qrels[qid] = gold

    return corpus, filtered_queries, filtered_qrels


def _load_raw_scifact(split: str) -> Tuple[Corpus, Queries, Qrels]:
    # Raw SciFact layout from HF datasets.
    corpus_ds = load_dataset("scifact", "corpus", split="train")

    # test split in raw scifact often has no evidence labels; fallback later if needed.
    claims_ds = load_dataset("scifact", "claims", split=split)

    corpus: Corpus = {}
    for row in corpus_ds:
        doc_id = str(row.get("doc_id", row.get("id")))
        if doc_id == "None":
            continue
        corpus[doc_id] = _build_doc_text(row.get("title", ""), row.get("abstract", ""))

    queries: Queries = {}
    qrels: Qrels = {}

    for row in claims_ds:
        qid = str(row.get("id"))
        claim = row.get("claim", "")
        evidence = row.get("evidence", {})
        if not isinstance(claim, str) or not claim.strip():
            continue

        gold_doc_ids: Set[str] = set()
        if isinstance(evidence, dict):
            for doc_id, ev_list in evidence.items():
                doc_id_str = str(doc_id)
                if doc_id_str not in corpus:
                    continue

                include = False
                if isinstance(ev_list, list) and ev_list:
                    for ev in ev_list:
                        if not isinstance(ev, dict):
                            continue
                        label = str(ev.get("label", "")).upper()
                        if label in {"SUPPORT", "SUPPORTS", "CONTRADICT", "REFUTES"}:
                            include = True
                            break
                    # Some dataset variants may omit label in evidence entries.
                    if not include:
                        include = True

                if include:
                    gold_doc_ids.add(doc_id_str)

        if gold_doc_ids:
            queries[qid] = claim
            qrels[qid] = gold_doc_ids

    return corpus, queries, qrels


def load_scifact_data(
    split: str = "test",
    prefer_beir: bool = True,
    max_queries: int | None = None,
) -> Tuple[Corpus, Queries, Qrels, str]:
    """
    Returns: corpus, queries, qrels, source_name
    """
    if prefer_beir:
        try:
            corpus, queries, qrels = _try_load_beir_scifact(split=split)
            source = "beir_scifact"
        except Exception:
            corpus, queries, qrels = _load_raw_scifact(split=split)
            source = "raw_scifact"
    else:
        corpus, queries, qrels = _load_raw_scifact(split=split)
        source = "raw_scifact"

    # If chosen split has no labels in raw dataset, fallback to validation.
    if not queries and source == "raw_scifact" and split != "validation":
        corpus, queries, qrels = _load_raw_scifact(split="validation")
        source = "raw_scifact_validation_fallback"

    if max_queries is not None:
        keep_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in keep_ids}
        qrels = {qid: qrels[qid] for qid in keep_ids}

    if not corpus or not queries or not qrels:
        raise RuntimeError("Loaded empty corpus/queries/qrels. Please check dataset access and split.")

    return corpus, queries, qrels, source
