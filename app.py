# Student: Kofi Assan | Index: 1002300129 | CS4241-Introduction to Artificial Intelligence
"""
Streamlit UI: query, retrieved chunks, scores, final prompt, answer.
Run from project root: streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.pipeline import (  # noqa: E402
    PipelineLog,
    apply_feedback_boost,
    call_llm,
    run_llm_only,
)
from rag.prompts import build_context_block, build_rag_prompt, select_context  # noqa: E402
from rag.retrieval import (  # noqa: E402
    hybrid_retrieve,
    pure_vector_topk,
    retrieve_with_optional_expansion,
)
from rag.store import FaissStore  # noqa: E402

logging.basicConfig(level=logging.INFO)
INDEX_DIR = ROOT / "data" / "index"

st.set_page_config(page_title="ACity RAG (CS4241)", layout="wide")
st.title("Academic City RAG Assistant")
st.caption("Kofi Assan · 1002300129 · CS4241 — manual RAG (no LangChain/LlamaIndex)")

if not (INDEX_DIR / "index.faiss").is_file():
    st.error(
        "Index not found. In a terminal at the project root, run:\n\n"
        "`python scripts/download_data.py` then `python scripts/build_index.py`"
    )
    st.stop()

@st.cache_resource
def load_store():
    return FaissStore.load(INDEX_DIR)


store = load_store()

with st.sidebar:
    st.subheader("Retrieval")
    use_hybrid = st.toggle("Hybrid (vector + BM25)", value=True)
    use_expansion = st.toggle("Query expansion (Part B)", value=False, disabled=not use_hybrid)
    top_k = st.slider("Top-k", 3, 20, 8)
    prompt_style = st.selectbox("Prompt style", ["strict", "concise"])
    st.subheader("Part G — feedback boost")
    st.caption("Prefer answers grounded in:")
    boost_election = st.checkbox("Election CSV", value=False)
    boost_budget = st.checkbox("Budget PDF", value=True)
    compare_llm_only = st.checkbox("Also run LLM-only (Part E)", value=False)

query = st.text_input("Your question", placeholder="Ask about elections or the 2025 budget…")

if st.button("Run RAG", type="primary") and query.strip():
    boost_sources = set()
    if boost_election:
        boost_sources.add("ghana_elections")
    if boost_budget:
        boost_sources.add("budget_2025")

    plog = PipelineLog()
    if use_hybrid:
        if use_expansion:
            hits, expanded = retrieve_with_optional_expansion(
                store, query, k=top_k, use_expansion=True
            )
        else:
            hits = hybrid_retrieve(store, query, k=top_k)
            expanded = query
    else:
        hits = pure_vector_topk(store, query, k=top_k)
        expanded = query
    if boost_sources:
        hits = apply_feedback_boost(hits, boost_sources)

    plog.add("query", {"text": query})
    plog.add(
        "retrieval",
        {
            "mode": "hybrid" if use_hybrid else "vector_only",
            "query_expansion": bool(use_expansion) if use_hybrid else False,
            "expanded_query": expanded,
            "feedback_boost": sorted(boost_sources),
            "hits": [
                {
                    "source": h.chunk.source,
                    "vector_score": round(h.vector_score, 4),
                    "bm25_score": round(h.bm25_score, 4),
                    "fused_score": round(h.fused_score, 4),
                    "text_preview": h.chunk.text[:300]
                    + ("…" if len(h.chunk.text) > 300 else ""),
                }
                for h in hits
            ],
        },
    )

    selected = select_context(hits, max_chars=6000)
    context_block = build_context_block(selected)
    final_prompt = build_rag_prompt(expanded, context_block, style=prompt_style)
    plog.add(
        "context_selection",
        {
            "num_chunks": len(selected),
            "sources": list({h.chunk.source for h in selected}),
        },
    )
    plog.add(
        "prompt",
        {"style": prompt_style, "final_prompt": final_prompt, "prompt_chars": len(final_prompt)},
    )

    with st.spinner("Calling LLM…"):
        answer = call_llm(final_prompt)
    plog.add("generation", {"answer_preview": answer[:500]})

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved chunks & scores"):
        for h in hits:
            st.markdown(
                f"**{h.chunk.source}** · vec={h.vector_score:.4f} · bm25={h.bm25_score:.4f} · fused={h.fused_score:.4f}"
            )
            st.text(h.chunk.text[:1200] + ("…" if len(h.chunk.text) > 1200 else ""))

    with st.expander("Final prompt sent to LLM"):
        st.code(final_prompt, language="text")

    with st.expander("Pipeline log (JSON-like)"):
        st.json(plog.stages)

    if compare_llm_only:
        with st.spinner("LLM-only baseline…"):
            base_ans, base_log = run_llm_only(query)
        st.subheader("Baseline: LLM without retrieval")
        st.write(base_ans)
        with st.expander("LLM-only prompt"):
            st.code(
                next(
                    (s["final_prompt"] for s in base_log.stages if s["stage"] == "prompt"),
                    "",
                ),
                language="text",
            )

elif query.strip():
    st.info("Click **Run RAG** to query the index.")

st.markdown(
    "---\n**Submission checklist:** GitHub repo `ai_1002300129`, deploy URL, "
    "invite **GodwinDansoAcity**, email lecturer with subject line format from the QP."
)
