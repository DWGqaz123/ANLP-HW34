from __future__ import annotations

from huggingface_hub import InferenceClient

from config import (
    HF_TOKEN,
    HYDE_DO_SAMPLE,
    HYDE_MAX_NEW_TOKENS,
    HYDE_PROMPT_TEMPLATE,
    HYDE_TEMPERATURE,
    LLM_MODEL,
)


class HyDEGenerator:
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        hf_token: str | None = HF_TOKEN,
        max_new_tokens: int = HYDE_MAX_NEW_TOKENS,
        temperature: float = HYDE_TEMPERATURE,
        do_sample: bool = HYDE_DO_SAMPLE,
    ) -> None:
        self.client = InferenceClient(model=model_name, token=hf_token, timeout=120)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def generate(self, query: str) -> str:
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)

        # Preferred path: chat-completions API.
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            pass

        # Fallback: text generation API.
        generated = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            return_full_text=False,
        )
        return str(generated).strip()
