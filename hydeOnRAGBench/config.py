import os

HF_TOKEN = os.getenv("HF_TOKEN")

LLM_MODEL = "Qwen/Qwen3.5-27B"
EMBED_MODEL = "BAAI/bge-m3"

DATASET_NAME = "rungalileo/ragbench"
DATASET_SUBSET = "covidqa"  # change to any RAGBench subset

TOP_K = 5
EMBED_BATCH_SIZE = 32
MAX_NEW_TOKENS = 512


# prompt to generate hypothetical passage for HyDE
HYDE_PROMPT = (
    "Please write a passage that could answer the following question.\n"
    "Question: {query}\n"
    "Passage:"
)

# prompt to generate final answer based on retrieved passages
ANSWER_PROMPT = (
    "Answer the question based on the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)
