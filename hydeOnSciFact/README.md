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

## Run

Run both baseline + HyDE comparison:

```bash
python main.py --mode both --split test --prefer_beir --top_ks 1,3,5,10
```

Run baseline only:

```bash
python main.py --mode baseline --split test --prefer_beir --top_ks 1,3,5,10
```

Run HyDE only:

```bash
python main.py --mode hyde --split test --prefer_beir --top_ks 1,3,5,10
```

Outputs are saved to `results/`:
- `metrics.json`
- `per_query_results.jsonl`
