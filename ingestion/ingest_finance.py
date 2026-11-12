"""Finance ingestion: RAGBench finqa -> ChromaDB collection 'finance'.

Adapted from original/capstone_rag_project_finance_domain.py
(rows are merged into one text, then chunked 500/150, as in the notebook).

Run:  python -m ingestion.ingest_finance [--max-records N]
"""
import argparse
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

import shared

DOMAIN = "finance"
COLLECTION = "finance"
DATASET = "finqa"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150


def main(max_records=None):
    limit = shared.resolve_limit(DOMAIN, max_records)

    print(f"Loading {DATASET} ...")
    all_texts = []
    for row in shared.load_ragbench(DATASET, limit):
        docs = row["documents"]
        if isinstance(docs, str):
            try:
                docs = json.loads(docs)
            except Exception:
                pass
        if isinstance(docs, list):
            all_texts.append("\n".join(str(doc) for doc in docs))
        else:
            all_texts.append(str(docs))

    merged_text = "\n\n".join(all_texts)
    print(f"Merged text: {len(merged_text):,} characters")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(merged_text)
    print(f"Created {len(chunks):,} chunks")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection = shared.recreate_collection(COLLECTION, {"hnsw:space": "cosine"})
    shared.add_in_batches(collection, ids, chunks, metadatas)
    print(f"Done: {collection.count():,} chunks in collection '{COLLECTION}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-records", type=int, default=None,
                        help="records to ingest (0 = all)")
    main(parser.parse_args().max_records)
