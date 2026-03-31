# HyDE on SciFact (Retrieval-only)

## Install

```bash
cd hydeOnSciFact
pip install -r requirements.txt
```

Set Hugging Face token:

```bash
export HF_TOKEN=your_token
```

## Run (Notebooks)

Open and run these notebooks from top to bottom:

- `run_baseline.ipynb`: baseline dense retrieval on SciFact
- `run_hyde.ipynb`: HyDE dense retrieval on SciFact

Outputs are saved to:

- `results/baseline/metrics.json`
- `results/baseline/per_query_results.jsonl`
- `results/hyde/metrics.json`
- `results/hyde/per_query_results.jsonl`
