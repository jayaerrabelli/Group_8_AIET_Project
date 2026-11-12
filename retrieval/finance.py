"""Finance retrieval + answering.

Logic from the original notebook: dense top-50 (Chroma) fused with BM25 ranks
using Reciprocal Rank Fusion (constant 10); the top 10 fused chunks go to Gemini.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config
import llm
import shared
from retrieval.base import Context, RagResult

DOMAIN = "finance"
COLLECTION = "finance"
RRF_CONSTANT = 10
DENSE_CANDIDATES = 50
TOP_CONTEXTS = 10


def is_ready() -> bool:
    return shared.collection_ready(COLLECTION)


@lru_cache(maxsize=1)
def _index():
    collection = shared.get_collection(COLLECTION, DOMAIN)
    ids, docs, _ = shared.fetch_all(collection)
    bm25 = shared.build_bm25([d.lower().split(" ") for d in docs])
    return collection, ids, docs, bm25


def hybrid_search_rrf(query: str):
    """Return [(chunk_id, rrf_score), ...] sorted best-first."""
    collection, ids, _, bm25 = _index()

    dense_ids = shared.dense_query(collection, query, DENSE_CANDIDATES)[0]
    dense_ranks = {cid: rank for rank, cid in enumerate(dense_ids)}

    sparse_scores = bm25.get_scores(query.lower().split(" "))
    sparse_order = np.argsort(sparse_scores)[::-1]
    sparse_ranks = {ids[idx]: rank for rank, idx in enumerate(sparse_order)}

    rrf_scores = {}
    for cid in ids:
        dense_rank = dense_ranks.get(cid, 100)  # penalty if outside the dense top 50
        sparse_rank = sparse_ranks.get(cid, 100)
        rrf_scores[cid] = 1.0 / (RRF_CONSTANT + dense_rank) + 1.0 / (RRF_CONSTANT + sparse_rank)

    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)


def answer(query: str) -> RagResult:
    _, ids, docs, _ = _index()
    id_to_doc = dict(zip(ids, docs))
    ranked = hybrid_search_rrf(query)[:TOP_CONTEXTS]

    contexts = [Context(id_to_doc[cid], source=cid, score=score) for cid, score in ranked]
    context_str = "\n\n===\n\n".join(
        f"[Source: {c.source} | RRF Score: {c.score:.4f}]\n{c.text}" for c in contexts
    )
    prompt = f"""You are an expert financial assistant. Use the following hybrid search context fragments to answer the question at the end.
If you don't know the answer, say that you don't know.

Context:
{context_str}

Question: {query}

Answer:"""
    provider, model = config.DOMAIN_LLM[DOMAIN]
    return RagResult(llm.generate(prompt, provider, model), contexts, DOMAIN)
