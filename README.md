# AIET Project: Multi-Domain RAG Assistant

**Group 8.** A Streamlit app that takes a question, uses an LLM to classify which domain it belongs to, and then runs that domain's own retrieval pipeline and answer generation.

| Domain | RAGBench subsets | Vector store | Retrieval logic | Answer LLM (default) |
|---|---|---|---|---|
| Biomedical | covidqa, pubmedqa | ChromaDB | dense + BM25, interleaved | Groq |
| Finance | finqa | ChromaDB | dense + BM25, Reciprocal Rank Fusion | Gemini |
| Customer support | techqa, emanual | ChromaDB | dense + BM25, RRF, cross-encoder rerank | Gemini |
| General knowledge | hotpotqa, expertqa | Qdrant (local) | dense + sparse BM25, RRF, cross-encoder rerank | Gemini |

## How it works

```
Question -> router.py (LLM picks a domain) -> retrieval/<domain>.py -> answer + sources
                                              (that domain's own retrieval logic)

ingestion/ingest_<domain>.py  ->  data/chroma_db  or  data/qdrant_db   (built once)
```

## Project structure

```
original/        the four team notebooks exported as .py, untouched (record keeping)
ingestion/       one ingestion script per domain (+ run_all.py)
retrieval/       one retrieval + answering module per domain
router.py        LLM query -> domain classifier
registry.py      the list of domains (descriptions feed the router prompt)
llm.py           Gemini / Groq / Ollama wrapper
shared.py        embeddings, Chroma / Qdrant helpers
config.py        paths, models, record limits (all overridable via .env)
app.py           Streamlit UI
```

## Setup

```bash
git clone https://github.com/jayaerrabelli/Group_8_AIET_Project.git
cd Group_8_AIET_Project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your API keys
```

Free API keys: [Google AI Studio](https://aistudio.google.com) (Gemini) and [Groq](https://console.groq.com).

## Build the indexes (once)

```bash
python -m ingestion.run_all                      # all four domains
python -m ingestion.ingest_finance               # or one at a time
python -m ingestion.ingest_general_knowledge --max-records 200
```

The first run downloads the datasets and the embedding model. `--max-records` sets the number of training records per source dataset (`0` means all).

Qdrant's local mode allows only one process at a time, so stop the app before re-running `ingest_general_knowledge`. Restart the app after re-ingesting any domain.

## Run the app

```bash
streamlit run app.py
```

The sidebar shows which domains are indexed, lets you pick a domain manually (skipping the router), and accepts API keys if they aren't in `.env`.

## Changing models or running fully local

Edit `.env`, for example `FINANCE_MODEL=...`, or switch a domain to a local model with `FINANCE_PROVIDER=ollama` and `FINANCE_MODEL=llama3.2:3b` (install [Ollama](https://ollama.com) and `pip install ollama`). The router can be switched the same way with `ROUTER_PROVIDER` and `ROUTER_MODEL`.

## Adding a domain

1. Add `ingestion/ingest_<key>.py` and `retrieval/<key>.py` (expose `answer(query) -> RagResult` and `is_ready()`).
2. Add an entry to `registry.py` with a specific description.
3. Add its LLM defaults to `config.py`.

## Notes on the notebook-to-project conversion

- Colab-only code (`!pip`, `google.colab.userdata`) was replaced by `.env` keys.
- Evaluation code (token metrics, Excel export, validation splits) was not carried over and stays in `original/`.
- Biomedical now persists to disk (the notebook kept Chroma in memory) and uses ChromaDB directly instead of through LangChain; chunking, HNSW settings and retrieval logic are unchanged.
- The customer-support export had literal `\\n` strings as splitter separators and prompt joins; these are now real newlines.
- General knowledge ingests 500 records per dataset by default (the notebook used the full train split on a GPU).
