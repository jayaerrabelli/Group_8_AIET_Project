"""Customer-support ingestion: RAGBench techqa + emanual -> ChromaDB collection 'customer_support'.

Adapted from original/capstone_rag_project_customer_support_domain.py
(per-document chunking with dataset / source-id metadata).

Run:  python -m ingestion.ingest_customer_support [--max-records N]
"""
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

import shared

DOMAIN = "customer_support"
COLLECTION = "customer_support"
DATASETS = ["techqa", "emanual"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150


def main(max_records=None):
    limit = shared.resolve_limit(DOMAIN, max_records)

    # NOTE: the notebook export contained literal "\\n" strings here; real newlines
    # are what the splitter needs in order to split on paragraph / line boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids, chunks, metadatas = [], [], []
    for name in DATASETS:
        print(f"Loading {name} ...")
        for row in shared.load_ragbench(name, limit):
            source_id = str(row["id"])
            for doc_idx, text in enumerate(shared.normalize_documents(row["documents"])):
                for chunk_idx, chunk in enumerate(splitter.split_text(text)):
                    if not chunk.strip():
                        continue
                    ids.append(f"{name}_{source_id}_doc{doc_idx}_chunk{chunk_idx}")
                    chunks.append(chunk)
                    metadatas.append({
                        "source_dataset": name,
                        "source_id": source_id,
                        "document_index": int(doc_idx),
                        "chunk_index": int(chunk_idx),
                    })

    assert chunks, "No chunks were created. Check the source documents."
    print(f"Created {len(chunks):,} chunks")

    collection = shared.recreate_collection(COLLECTION, {"hnsw:space": "cosine"})
    shared.add_in_batches(collection, ids, chunks, metadatas)
    print(f"Done: {collection.count():,} chunks in collection '{COLLECTION}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-records", type=int, default=None,
                        help="records per source dataset (0 = all)")
    main(parser.parse_args().max_records)
