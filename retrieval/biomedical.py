"""Biomedical retrieval + answering.

Logic from the original notebook: dense top-k (Chroma, cosine) and sparse top-k
(BM25) are interleaved without duplicates, the first 3 contexts go to the LLM
(Groq by default) with a medical-assistant system prompt.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config
import llm
import shared
from retrieval.base import Context, RagResult

DOMAIN = "biomedical"
COLLECTION = "biomedical"
SYSTEM = (
    "You are a precise medical assistant. Synthesize the provided medical contexts to "
    "answer the question accurately. If the answer cannot be found in the contexts, "
    "state that clearly."
)


def is_ready() -> bool:
    return shared.collection_ready(COLLECTION)


@lru_cache(maxsize=1)
def _index():
    collection = shared.get_collection(COLLECTION, DOMAIN)
    _, docs, _ = shared.fetch_all(collection)
    bm25 = shared.build_bm25([d.lower().split(" ") for d in docs])
    return collection, docs, bm25


def retrieve_context(query: str, final_k: int = 3) -> list[str]:
    collection, docs, bm25 = _index()

    dense_texts = shared.dense_query(collection, query, final_k)[1]

    scores = bm25.get_scores(query.lower().split(" "))
    top = np.argsort(scores)[::-1][:final_k]
    sparse_texts = [docs[i] for i in top]

    combined: list[str] = []
    for d, s in zip(dense_texts, sparse_texts):
        if d not in combined:
            combined.append(d)
        if s not in combined:
            combined.append(s)
    return combined[:final_k]


def answer(query: str) -> RagResult:
    contexts = retrieve_context(query)
    if not contexts:
        return RagResult("Unable to retrieve context for this question.", [], DOMAIN)

    block = "\n\n".join(f"[Context {i + 1}]: {t}" for i, t in enumerate(contexts))
    prompt = f"Contexts:\n{block}\n\nQuestion: {query}\n\nAnswer:"
    provider, model = config.DOMAIN_LLM[DOMAIN]
    text = llm.generate(prompt, provider, model, system=SYSTEM, temperature=0.0, max_tokens=300)
    return RagResult(text, [Context(t, f"Context {i + 1}") for i, t in enumerate(contexts)], DOMAIN)
