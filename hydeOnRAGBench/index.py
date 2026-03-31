import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
from config import EMBED_MODEL, EMBED_BATCH_SIZE


def load_embed_model():
    return BGEM3FlagModel(EMBED_MODEL, use_fp16=True)


def encode(model, texts):
    out = model.encode(texts, batch_size=EMBED_BATCH_SIZE, max_length=8192)
    vecs = out["dense_vecs"].astype(np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def build_index(model, corpus):
    vecs = encode(model, corpus)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, vecs


def search(index, model, query_texts, top_k):
    q_vecs = encode(model, query_texts)
    scores, indices = index.search(q_vecs, top_k)
    return scores, indices
