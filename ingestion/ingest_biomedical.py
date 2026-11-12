"""Biomedical ingestion: RAGBench covidqa + pubmedqa -> ChromaDB collection 'biomedical'.

Adapted from original/capstone_rag_project_biomedical_research_domain.py
(same chunking, HNSW settings and unified-knowledge-base approach; the index is
now persisted to disk instead of living in memory).

Run:  python -m ingestion.ingest_biomedical [--max-records N]
"""
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

import shared

DOMAIN = "biomedical"
COLLECTION = "biomedical"
DATASETS = ["covidqa", "pubmedqa"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
HNSW = {"hnsw:space": "cosine", "hnsw:construction_ef": 100, "hnsw:M": 16}


def main(max_records=None):
    limit = shared.resolve_limit(DOMAIN, max_records)

    all_documents = []
    for name in DATASETS:
        print(f"Loading {name} ...")
        for item in shared.load_ragbench(name, limit):
            docs = item["documents"]
            if isinstance(docs, list):
                all_documents.extend(docs)
            else:
                all_documents.append(docs)

    knowledge_base = "\n".join(all_documents)
    print(f"Unified knowledge base: {len(knowledge_base):,} characters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = [d.page_content for d in splitter.create_documents([knowledge_base])]
    print(f"Created {len(chunks):,} chunks")

    ids = [f"bio_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection = shared.recreate_collection(COLLECTION, HNSW)
    shared.add_in_batches(collection, ids, chunks, metadatas)
    print(f"Done: {collection.count():,} chunks in collection '{COLLECTION}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-records", type=int, default=None,
                        help="records per source dataset (0 = all)")
    main(parser.parse_args().max_records)
