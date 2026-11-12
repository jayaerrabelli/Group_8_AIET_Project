"""The list of domains. To add a domain: add an entry here plus one
ingestion module and one retrieval module (see README)."""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    description: str  # shown to the router LLM, so be specific
    module: str       # retrieval module exposing answer(query) and is_ready()


DOMAINS: dict[str, Domain] = {
    d.key: d
    for d in [
        Domain(
            "biomedical",
            "Biomedical research",
            "Medical and biomedical research questions: diseases, symptoms, treatments, "
            "clinical studies, drugs, COVID-19, and findings from PubMed or scientific papers.",
            "retrieval.biomedical",
        ),
        Domain(
            "finance",
            "Finance",
            "Finance and corporate financial reports: earnings, revenue, expenses, lending, "
            "assets, balance sheets, year-over-year changes, and numerical questions about "
            "company filings.",
            "retrieval.finance",
        ),
        Domain(
            "customer_support",
            "Customer support",
            "Technical and product customer support: troubleshooting, error messages, "
            "software or hardware problems, how-to steps, device settings, and "
            "electronics or appliance manuals.",
            "retrieval.customer_support",
        ),
        Domain(
            "general_knowledge",
            "General knowledge",
            "General encyclopedic knowledge: people, places, history, science, culture, "
            "organisations and other factual questions that do not clearly belong to the "
            "specialised domains above.",
            "retrieval.general_knowledge",
        ),
    ]
}


def _module(key: str):
    return importlib.import_module(DOMAINS[key].module)


def answer(key: str, query: str):
    """Run the domain-specific retrieval + generation pipeline."""
    return _module(key).answer(query)


def is_ready(key: str) -> bool:
    try:
        return bool(_module(key).is_ready())
    except Exception:
        return False
