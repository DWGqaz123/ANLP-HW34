# ANLP-HW34

This is the repo for Assignment 3&4 of 11711-A Advanced Natural Language Processing-Spring 2026

Assignment instruction: https://cmu-l3.github.io/anlp-spring2026/assignments/assignment3&4

#### Team members:
Wenguang Dong
Carnegie Mellon University
wenguand@andrew.cmu.edu

Yichen Ji
Carnegie Mellon University
yichenj@andrew.cmu.edu

Juan Pablo Naranjo
Carnegie Mellon University
jpnaranj@andrew.cmu.edu


### Clarification
hyde is the reproduction of original paper with generator switch to gpt-4o-mini

hydeOnSciFact applys the hyde on SciFact dataset with EMBED_MODEL = "BAAI/bge-base-en-v1.5" and LLM_MODEL = "gpt-4o-mini"

### Result Error Analysis
`result_error_analysis.ipynb` provides query-level error analysis for DL19 runs in `all_results/dl19`.

Current DL19 analysis workflow:
1. Load official qrels (relevance judgments).
2. Compare top-k rankings across run files (`bm25`, `contriever`, `hyde`).
3. Identify failure queries where HyDE underperforms the baseline on `nDCG@10` or `Recall@10`.

The notebook also exports:
- `all_results/dl19/error_analysis/comparison_per_query.jsonl`
- `all_results/dl19/error_analysis/failure_cases.jsonl`
