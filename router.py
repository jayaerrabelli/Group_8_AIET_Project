"""LLM-based query router: decides which domain pipeline should answer."""
from __future__ import annotations

import re

import config
import llm
from registry import DOMAINS


def build_prompt(query: str) -> str:
    options = "\n".join(f"- {d.key}: {d.description}" for d in DOMAINS.values())
    return (
        "You are a query router for a RAG system. Classify the user query into "
        "exactly one of these domains:\n"
        f"{options}\n\n"
        "If the query does not clearly fit a specialised domain, choose general_knowledge.\n"
        f"Query: {query}\n\n"
        "Reply with only the domain key and nothing else."
    )


def parse_label(text: str):
    text = (text or "").strip().lower()
    for key in DOMAINS:
        if key in text:
            return key
    normalized = re.sub(r"[\s\-]+", "_", text)
    for key in DOMAINS:
        if key in normalized:
            return key
    return None


def classify(query: str):
    """Return (domain_key or None, raw model reply)."""
    raw = llm.generate(
        build_prompt(query),
        provider=config.ROUTER_PROVIDER,
        model=config.ROUTER_MODEL,
        temperature=0.0,
    )
    return parse_label(raw), raw
