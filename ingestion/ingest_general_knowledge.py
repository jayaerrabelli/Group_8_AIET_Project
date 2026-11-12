"""General-knowledge ingestion: RAGBench hotpotqa + expertqa -> Qdrant (dense + sparse).

Adapted from original/capstone_rag_project_general_knowledge_domain.py.
This is the only domain that uses Qdrant instead of ChromaDB: it stores a dense
vector (MiniLM) and a sparse BM25 vector (fastembed) per chunk for hybrid search.

Qdrant's local mode allows ONE process at a time, so stop the Streamlit app
before running this.

Run:  python -m ingestion.ingest_general_knowledge [--max-records N]
"""
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

import shared

DOMAIN = "general_knowledge"
COLLECTION = "ragbench_hybrid"
DATASETS = ["hotpotqa", "expertqa"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 256


def main(max_records=None):
    from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

    limit = shared.resolve_limit(DOMAIN, max_records)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    records = []
    for name in DATASETS:
        print(f"Loading {name} ...")
        for i, row in enumerate(shared.load_ragbench(name, limit)):
            doc_id = row.get("id", str(i))
            for doc in row.get("documents", []):
                for chunk in splitter.split_text(doc):
                    records.append({"doc_id": doc_id, "domain": name, "text": chunk})
    print(f"Created {len(records):,} chunks")

    sparse_model = shared.get_sparse_model()
    client = shared.get_qdrant()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=shared.embedding_dim(), distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        dense = shared.embed_texts(texts)
        sparse = list(sparse_model.embed(texts))
        points = [
            PointStruct(
                id=start + j,
                vector={
                    "dense": dense[j],
                    "sparse": {
                        "indices": sparse[j].indices.tolist(),
                        "values": sparse[j].values.tolist(),
                    },
                },
                payload=rec,
            )
            for j, rec in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  uploaded {min(start + BATCH_SIZE, len(records)):,} / {len(records):,}")

    shared.close_qdrant()
    print(f"Done: {len(records):,} chunks in Qdrant collection '{COLLECTION}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-records", type=int, default=None,
                        help="records per source dataset (0 = all; the notebook used all)")
    main(parser.parse_args().max_records)
