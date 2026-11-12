"""Central configuration.

Every value can be overridden with an environment variable or a `.env` file
in the project root (see .env.example).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(ROOT / ".env")

# ---- Storage ----------------------------------------------------------------
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
CHROMA_PATH = DATA_DIR / "chroma_db"   # biomedical, finance, customer_support
QDRANT_PATH = DATA_DIR / "qdrant_db"   # general_knowledge

# ---- Data and models --------------------------------------------------------
RAGBENCH_REPO = "rungalileo/ragbench"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Training records ingested PER SOURCE DATASET (0 = use all).
# The notebooks used 100 for biomedical / finance / customer support and the
# full train split for general knowledge (too heavy for a CPU laptop).
MAX_RECORDS = {
    "biomedical": int(os.getenv("BIOMEDICAL_MAX_RECORDS", 100)),
    "finance": int(os.getenv("FINANCE_MAX_RECORDS", 100)),
    "customer_support": int(os.getenv("CUSTOMER_SUPPORT_MAX_RECORDS", 100)),
    "general_knowledge": int(os.getenv("GENERAL_KNOWLEDGE_MAX_RECORDS", 500)),
}

# ---- LLMs (provider, model) -------------------------------------------------
# Defaults are the models used in the original notebooks. Override with e.g.
#   FINANCE_MODEL=gemini-3.5-flash-lite   or   FINANCE_PROVIDER=ollama
# Providers: gemini | groq | ollama
_DEFAULT_LLMS = {
    "biomedical": ("groq", "qwen/qwen3.8-27b"),
    "finance": ("gemini", "gemini-3.5-flash-lite"),
    "customer_support": ("gemini", "gemini-3.5-flash-lite"),
    "general_knowledge": ("gemini", "gemini-3.8-flash"),
}
DOMAIN_LLM = {
    key: (
        os.getenv(f"{key.upper()}_PROVIDER", provider),
        os.getenv(f"{key.upper()}_MODEL", model),
    )
    for key, (provider, model) in _DEFAULT_LLMS.items()
}

# The router (query -> domain classifier)
ROUTER_PROVIDER = os.getenv("ROUTER_PROVIDER", "gemini")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-2.5-flash-lite")
