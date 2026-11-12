"""General-knowledge retrieval + answering.

Logic from the original notebook: Qdrant hybrid query (dense + sparse BM25 fused
with RRF, top 100) -> cross-encoder rerank -> top 5 go to Gemini.
"""
from __future__ import annotations

import config
import llm
import shared
from retrieval.base import Context, RagResult

DOMAIN = "general_knowledge"
COLLECTION = "ragbench_hybrid"
FIRST_STAGE_K = 100
FINAL_K = 5


def is_ready() -> bool:
    return shared.qdrant_ready(COLLECTION)


def hybrid_retrieve_context(query: str, top_k: int = FIRST_STAGE_K):
    from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

    client = shared.get_qdrant()
    if not (client.collection_exists(COLLECTION) and client.count(COLLECTION).count):
        raise shared.not_ingested(DOMAIN)

    dense_query = shared.embed_query(query)
    sparse_query = next(iter(shared.get_sparse_model().embed([query])))

    response = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_query, using="dense", limit=top_k),
            Prefetch(
                query=SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                ),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    return response.points if hasattr(response, "points") else response


def answer(query: str) -> RagResult:
    initial = hybrid_retrieve_context(query, FIRST_STAGE_K)
    if not initial:
        return RagResult("No relevant context found in the database.", [], DOMAIN)

    scores = shared.get_reranker().predict([[query, r.payload["text"]] for r in initial])
    order = sorted(range(len(initial)), key=lambda i: scores[i], reverse=True)[:FINAL_K]
    final = [(initial[i], float(scores[i])) for i in order]

    formatted = "\n\n".join(f"Document [{i + 1}]: {r.payload['text']}" for i, (r, _) in enumerate(final))
    prompt = f"""You are an expert assistant for a highly specialized RAG system.
Answer the user's question accurately using the provided context chunks.
- Synthesize information across the chunks if the answer is split.
- Ignore minor typos in the question (e.g., spelling of names/words) when mapping them to the context.
- Only if the context is completely irrelevant to the question, reply with "I don't have enough information to answer that based on the provided context."

Context:
{formatted}

Question: {query}

Answer:"""
    provider, model = config.DOMAIN_LLM[DOMAIN]
    contexts = [
        Context(r.payload["text"], source=f"{r.payload.get('domain', '?')} | doc {r.payload['doc_id']}", score=s)
        for r, s in final
    ]
    return RagResult(llm.generate(prompt, provider, model), contexts, DOMAIN)
