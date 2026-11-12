"""Streamlit UI: classify the question with an LLM, then run the matching domain pipeline.

Run:  streamlit run app.py
"""
import os

import streamlit as st

import llm
import registry
import router
import shared
from registry import DOMAINS

st.set_page_config(page_title="Group 8 RAG Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Group 8 RAG Assistant")
st.caption("Your question is classified by an LLM, then answered by the matching domain pipeline.")

AUTO = "Auto-detect (LLM router)"
label_to_key = {d.label: d.key for d in DOMAINS.values()}


def safe(text: str) -> str:
    """Stop Streamlit from turning $...$ (common in finance answers) into LaTeX."""
    return text.replace("$", "\\$")


# ---- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("API keys")
    st.caption("Optional if they are already in your .env file.")
    google_key = st.text_input("Google (Gemini) API key", type="password")
    groq_key = st.text_input("Groq API key", type="password")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.header("Routing")
    choice = st.selectbox("Domain", [AUTO] + list(label_to_key), help="Pick a domain to skip the router.")

    st.header("Index status")
    for d in DOMAINS.values():
        st.write(("✅ " if registry.is_ready(d.key) else "⚠️ not ingested · ") + d.label)

    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()


# ---- Chat ---------------------------------------------------------------------
def render_assistant(entry: dict):
    if entry.get("domain"):
        how = "selected manually" if entry.get("manual") else "auto-detected"
        st.caption(f"Domain: **{DOMAINS[entry['domain']].label}** ({how})")
    st.markdown(safe(entry["answer"]))
    if entry.get("contexts"):
        with st.expander(f"Sources ({len(entry['contexts'])})"):
            for i, c in enumerate(entry["contexts"], start=1):
                score = f" · score {c['score']:.4f}" if c.get("score") is not None else ""
                st.markdown(f"**{i}. {c['source']}**{score}")
                st.text(c["text"])


if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(safe(message["content"]))
        else:
            render_assistant(message)

query = st.chat_input("Ask a question...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(safe(query))

    with st.chat_message("assistant"):
        entry = None
        try:
            manual = choice != AUTO
            if manual:
                domain = label_to_key[choice]
            else:
                with st.spinner("Routing your question..."):
                    domain, _raw = router.classify(query)

            if domain is None:
                entry = {
                    "role": "assistant",
                    "answer": "I couldn't decide which domain this belongs to. "
                              "Try rephrasing, or pick a domain in the sidebar.",
                }
            else:
                with st.spinner(f"Searching {DOMAINS[domain].label} ..."):
                    result = registry.answer(domain, query)
                entry = {
                    "role": "assistant",
                    "domain": domain,
                    "manual": manual,
                    "answer": result.answer,
                    "contexts": [vars(c) for c in result.contexts],
                }
        except llm.MissingAPIKey as exc:
            st.error(f"🔑 {exc}")
        except shared.NotIngestedError as exc:
            st.error(f"📦 {exc}")
        except Exception as exc:  # show anything else instead of crashing the page
            st.error(f"Something went wrong: {exc}")

        if entry:
            render_assistant(entry)
            st.session_state.history.append(entry)
