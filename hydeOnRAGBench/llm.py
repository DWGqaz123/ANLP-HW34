from huggingface_hub import InferenceClient
from config import HF_TOKEN, LLM_MODEL, MAX_NEW_TOKENS


_client = None

def get_client():
    global _client
    if _client is None:
        _client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
    return _client


def generate(prompt):
    client = get_client()
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_NEW_TOKENS,
    )
    return resp.choices[0].message.content.strip()
