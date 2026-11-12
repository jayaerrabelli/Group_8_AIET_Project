"""Helpers shared by the ingestion and retrieval modules.

Heavy libraries are imported lazily, so importing this module is cheap and
each model / client is created once per process.
"""
from __future__ import annotations

import json
from functools import lru_cache

import config


class NotIngestedError(RuntimeError):
    """Raised when a domain's index has not been built yet."""


def not_ingested(domain: str) -> NotIngestedError:
    return NotIngestedError(
        f"The '{domain}' index has not been built yet. "
        f"Run:  python -m ingestion.ingest_{domain}"
    )


# ---- Models -------------------------------------------------------------------
def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL, device=_device())


def embed_texts(texts, batch_size: int = 64) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []
    return get_embedder().encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    return get_embedder().encode(query).tolist()


def embedding_dim() -> int:
    return len(embed_query("dimension probe"))


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL, device=_device())


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


# ---- ChromaDB -----------------------------------------------------------------
@lru_cache(maxsize=1)
def get_chroma():
    import chromadb
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH))


def recreate_collection(name: str, metadata: dict):
    """Drop and re-create a collection. Embeddings are always supplied by us."""
    client = get_chroma()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, embedding_function=None, metadata=metadata)


def get_collection(name: str, domain: str):
    try:
        collection = get_chroma().get_collection(name=name, embedding_function=None)
    except Exception as exc:
        raise not_ingested(domain) from exc
    if collection.count() == 0:
        raise not_ingested(domain)
    return collection


def collection_ready(name: str) -> bool:
    try:
        return get_chroma().get_collection(name=name, embedding_function=None).count() > 0
    except Exception:
        return False


def add_in_batches(collection, ids, documents, metadatas, batch_size: int = 500):
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embed_texts(documents[start:end]),
        )
        print(f"  stored {end:,} / {len(ids):,} chunks")


def fetch_all(collection, page: int = 2000):
    """Return (ids, documents, metadatas) for every chunk in a collection."""
    ids, docs, metas = [], [], []
    offset = 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
        if not batch["ids"]:
            break
        ids += batch["ids"]
        docs += batch["documents"]
        metas += batch["metadatas"]
        offset += len(batch["ids"])
    return ids, docs, metas


def dense_query(collection, query: str, n_results: int):
    """Return (ids, documents, metadatas, distances) of the nearest chunks."""
    n = min(n_results, collection.count())
    out = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return out["ids"][0], out["documents"][0], out["metadatas"][0], out["distances"][0]


def build_bm25(tokenized_corpus):
    from rank_bm25 import BM25Okapi
    return BM25Okapi(tokenized_corpus)


# ---- Qdrant (general knowledge) -----------------------------------------------
@lru_cache(maxsize=1)
def get_qdrant():
    from qdrant_client import QdrantClient
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def close_qdrant():
    """Release Qdrant's local file lock (only one process may hold it)."""
    if get_qdrant.cache_info().currsize:
        get_qdrant().close()
        get_qdrant.cache_clear()


def qdrant_ready(name: str) -> bool:
    try:
        client = get_qdrant()
        return bool(client.collection_exists(name)) and client.count(name).count > 0
    except Exception:
        return False


# ---- Data ---------------------------------------------------------------------
def resolve_limit(domain: str, cli_value):
    return config.MAX_RECORDS[domain] if cli_value is None else cli_value


def load_ragbench(config_name: str, limit: int | None = None):
    """Load the train split of one RAGBench subset (limit 0/None = everything)."""
    from datasets import load_dataset
    ds = load_dataset(config.RAGBENCH_REPO, config_name, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def normalize_documents(value) -> list[str]:
    """Turn a RAGBench `documents` field into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [value]
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item).strip() for item in value if item is not None and str(item).strip()]
