"""OpenAI-compatible LLM client. Defaults to Qwen3-Max via DashScope."""
from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional


def _client():
    from openai import OpenAI

    base_url = os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_complete(prompt: str, user_content: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    model = model or os.environ.get("LLM_MODEL", "qwen3-max")
    client = _client()
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed: {last_error}")


def embed_texts(texts: Iterable[str], model: Optional[str] = None, batch_size: int = 16) -> List[List[float]]:
    """Embed texts with an OpenAI-compatible embedding endpoint (Qwen3-Embedding)."""
    model = model or os.environ.get("EMBED_MODEL", "text-embedding-v4")
    client = _client()
    vectors = []
    batch = []
    texts = list(texts)
    for i, text in enumerate(texts, 1):
        batch.append(text if text else " ")
        if len(batch) >= batch_size or i == len(texts):
            last_error = None
            for attempt in range(5):
                try:
                    response = client.embeddings.create(model=model, input=batch)
                    vectors.extend([row.embedding for row in response.data])
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    time.sleep(2 ** attempt)
            if last_error is not None:
                raise RuntimeError(f"Embedding call failed: {last_error}")
            batch = []
    return vectors
