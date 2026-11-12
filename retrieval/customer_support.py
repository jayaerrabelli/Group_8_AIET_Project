"""Customer-support retrieval + answering.

Logic from the original notebook: dense top-50 (Chroma) + BM25 top-50 fused with
RRF (constant 60) -> top 20 candidates -> cross-encoder rerank -> top 5 go to
Gemini with a support-assistant prompt.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

import config
import llm
import shared
from retrieval.base import Context, RagResult

DOMAIN = "customer_support"
COLLECTION = "customer_support"


def is_ready() -> bool:
    return shared.collection_ready(COLLECTION)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*", str(text).lower())


@lru_cache(maxsize=1)
def _index():
    collection = shared.get_collection(COLLECTION, DOMAIN)
    ids, docs, metas = shared.fetch_all(collection)
    bm25 = shared.build_bm25([tokenize_text(d) for d in docs])
    position = {cid: i for i, cid in enumerate(ids)}
    return collection, ids, docs, metas, bm25, position


def hybrid_retrieve(query, dense_k=50, sparse_k=50, top_k=10, rrf_constant=60):
    collection, ids, docs, metas, bm25, position = _index()

    dense_ids = shared.dense_query(collection, query, dense_k)[0]
    sparse_scores = bm25.get_scores(tokenize_text(query))
    sparse_idx = np.argsort(sparse_scores)[::-1][: min(sparse_k, len(ids))]

    rrf = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_constant + rank)
    for rank, idx in enumerate(sparse_idx, start=1):
        cid = ids[int(idx)]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_constant + rank)

    results = []
    for cid, score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        pos = position[cid]
        results.append({"id": cid, "text": docs[pos], "metadata": metas[pos], "rrf_score": float(score)})
    return results


def rerank(query, candidates, top_n=5):
    scores = shared.get_reranker().predict([[query, c["text"]] for c in candidates])
    ranked = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    return sorted(ranked, key=lambda c: c["rerank_score"], reverse=True)[:top_n]


def answer(query: str) -> RagResult:
    candidates = hybrid_retrieve(query, dense_k=50, sparse_k=50, top_k=20)
    top = rerank(query, candidates, top_n=5)

    final_context = "\n\n---\n\n".join(
        f"[Source {rank} | {item['metadata']['source_dataset']} | {item['metadata']['source_id']}]\n{item['text']}"
        for rank, item in enumerate(top, start=1)
    )
    prompt = f"""You are an expert customer support assistant.

Answer the customer's question using only the retrieved support documentation below.
Do not invent product behavior, settings, commands or troubleshooting steps.
If the documentation does not contain enough information, clearly say what is missing
and ask for the product/model, software version or exact error message when useful.
Give concise, numbered troubleshooting steps when the context supports them.
Mention important warnings found in the documentation.

Retrieved support documentation:
{final_context}

Customer question:
{query}

Answer:"""
    provider, model = config.DOMAIN_LLM[DOMAIN]
    contexts = [
        Context(
            item["text"],
            source=f"{item['metadata']['source_dataset']} | {item['metadata']['source_id']}",
            score=item["rerank_score"],
        )
        for item in top
    ]
    return RagResult(llm.generate(prompt, provider, model), contexts, DOMAIN)
