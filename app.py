# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""
Streamlit UI: query, retrieved chunks, scores, final prompt, answer.
Run from project root: streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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

# SVG: grid scroll; draggable wave peaks (JS); sea motion via translate only (no ball-synced pulse)
_LIME = "#DFFF00"
_BALL_DUR_A = 3.2  # seconds — primary ball along path
_BALL_DUR_B = 4.65  # seconds — second ball (slower lap)
_BALL_B_BEGIN = 0.85  # seconds — stagger second ball so both are visible
# Background (ghost / distant) waves — smaller, dimmer balls
_BG_BALL_DUR_1 = 2.95
_BG_BALL_DUR_2 = 3.45
_BG_BALL_DUR_3 = 4.1
_BG_BALL_B2 = 0.5
_BG_BALL_B3 = 1.05

_WAVE_PEAK_DRAG_JS = """
<script>
(function () {
  var svg = document.getElementById("waveSvg");
  var grp = document.getElementById("wavePeakGroup");
  if (!svg || !grp) return;
  var KEY = "acity_wave_peak_sy";
  var MIN = 0.5;
  var MAX = 1.95;
  var SENS = 0.0045;
  var CX = 600;
  var CY = 100;

  function applyScale(sy) {
    sy = Math.max(MIN, Math.min(MAX, sy));
    grp.setAttribute(
      "transform",
      "translate(" + CX + "," + CY + ") scale(1," + sy + ") translate(" + -CX + "," + -CY + ")"
    );
    try { sessionStorage.setItem(KEY, String(sy)); } catch (e) {}
  }

  var currentSy = 1;
  try {
    var sv = sessionStorage.getItem(KEY);
    if (sv) {
      var n = parseFloat(sv);
      if (!isNaN(n)) currentSy = n;
    }
  } catch (e) {}
  applyScale(currentSy);

  var drag = false;
  var startY = 0;
  var startSy = 1;

  function down(e) {
    if (e.button != null && e.button !== 0) return;
    drag = true;
    startY = e.clientY;
    startSy = currentSy;
    try { svg.setPointerCapture(e.pointerId); } catch (x) {}
    e.preventDefault();
    svg.style.cursor = "grabbing";
  }
  function move(e) {
    if (!drag) return;
    var dy = startY - e.clientY;
    currentSy = startSy + dy * SENS;
    applyScale(currentSy);
  }
  function up(e) {
    if (!drag) return;
    drag = false;
    svg.style.cursor = "ns-resize";
    try { svg.releasePointerCapture(e.pointerId); } catch (x) {}
  }

  svg.style.cursor = "ns-resize";
  svg.addEventListener("pointerdown", down);
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
})();
</script>
"""

_WAVE_BANNER_HTML = (
    f"""
<style>html,body{{margin:0;padding:0;background:#000;height:100%;}}</style>
<div id="waveBannerShell" style="font-family:'Segoe UI',system-ui,sans-serif;width:100%;
  margin:0;padding:0;box-sizing:border-box;background:#000000;">
  <svg id="waveSvg" viewBox="0 0 1200 200" xmlns="http://www.w3.org/2000/svg"
       preserveAspectRatio="xMidYMid meet" role="img"
       aria-label="Sea wave banner. Drag vertically on the waves to raise or lower peaks."
       style="width:100%;height:268px;display:block;background:#000000;touch-action:none;">
    <defs>
      <pattern id="gridScroll" width="48" height="48" patternUnits="userSpaceOnUse">
        <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#1c1c1c" stroke-width="0.55"/>
      </pattern>
      <path id="ghostWaveA" d="M 0,58 C 160,18 320,118 480,58 S 800,18 960,58 S 1120,118 1200,58"/>
      <path id="ghostWaveB" d="M 0,82 C 180,128 360,32 540,82 S 900,128 1080,82 S 1140,38 1200,82"/>
      <path id="ghostWaveD" d="M 0,22 C 140,92 280,-8 420,68 S 700,-12 840,68 S 1020,22 1200,68"/>
    </defs>
    <rect width="1200" height="200" fill="#000000"/>
    <rect width="1200" height="200" fill="url(#gridScroll)" opacity="0.78">
      <animateTransform attributeName="transform" type="translate" additive="replace"
        values="0 0; 48 0" dur="18s" repeatCount="indefinite" calcMode="linear"/>
    </rect>
    <g id="wavePeakGroup" transform="translate(600,100) scale(1,1) translate(-600,-100)">
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -10; 0 0; 0 9; 0 0"
          keyTimes="0;0.25;0.5;0.75;1" dur="5.5s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
        <use href="#ghostWaveA" fill="none" stroke="#3d4a2a" stroke-width="1.25" opacity="0.45"
             stroke-linecap="round"/>
        <circle r="3.6" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.85">
          <animateMotion dur="{_BG_BALL_DUR_1}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveA"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.40;0.40;0" keyTimes="0;0.06;0.90;1"
            dur="{_BG_BALL_DUR_1}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 5; 0 -12; 0 4; 0 11; 0 5"
          keyTimes="0;0.25;0.5;0.75;1" dur="4.2s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1"/>
        <use href="#ghostWaveB" fill="none" stroke="#2a3320" stroke-width="1" opacity="0.35"
             stroke-linecap="round"/>
        <circle r="3.2" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.75">
          <animateMotion dur="{_BG_BALL_DUR_2}s" begin="{_BG_BALL_B2}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveB"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.34;0.34;0" keyTimes="0;0.08;0.91;1"
            dur="{_BG_BALL_DUR_2}s" begin="{_BG_BALL_B2}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -7; 0 6; 0 0"
          keyTimes="0;0.33;0.66;1" dur="4.8s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1"/>
        <use href="#ghostWaveD" fill="none" stroke="#2f3824" stroke-width="1.2" opacity="0.4"
             stroke-linecap="round"/>
        <circle r="2.8" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.65">
          <animateMotion dur="{_BG_BALL_DUR_3}s" begin="{_BG_BALL_B3}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveD"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.28;0.28;0" keyTimes="0;0.07;0.90;1"
            dur="{_BG_BALL_DUR_3}s" begin="{_BG_BALL_B3}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -14; 0 0; 0 12; 0 0"
          keyTimes="0;0.25;0.5;0.75;1" dur="4s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.4 0 0.6 1;0.4 0 0.6 1;0.4 0 0.6 1;0.4 0 0.6 1"/>
        <path id="ragWavePath"
              d="M 4,70 C 140,8 280,132 420,70 S 700,8 840,70 S 1020,132 1196,70"
              fill="none" stroke="{_LIME}" stroke-width="2.25" stroke-linecap="round"/>
        <circle r="7" fill="{_LIME}" stroke="#000000" stroke-width="1.5">
          <animateMotion dur="{_BALL_DUR_A}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ragWavePath"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;1;1;0" keyTimes="0;0.055;0.90;1"
            dur="{_BALL_DUR_A}s" repeatCount="indefinite"/>
        </circle>
        <circle r="5.5" fill="{_LIME}" stroke="#000000" stroke-width="1.2">
          <animateMotion dur="{_BALL_DUR_B}s" begin="{_BALL_B_BEGIN}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ragWavePath"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;1;1;0" keyTimes="0;0.16;0.91;1"
            dur="{_BALL_DUR_B}s" begin="{_BALL_B_BEGIN}s" repeatCount="indefinite"/>
        </circle>
      </g>
    </g>
    <text x="28" y="168" fill="#ffffff" font-size="26" font-weight="700" letter-spacing="0.06em"
          font-family="Segoe UI,system-ui,sans-serif">ACADEMIC</text>
    <text x="28" y="194" fill="{_LIME}" font-size="26" font-weight="700" letter-spacing="0.06em"
          font-family="Segoe UI,system-ui,sans-serif">RAG</text>
  </svg>
</div>
"""
    + _WAVE_PEAK_DRAG_JS
)

st.set_page_config(page_title="ACity RAG (IT3241)", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #000000 !important; }
      section[data-testid="stSidebar"] { background-color: #0a0a0a !important; }
      div[data-testid="stToolbar"] { background-color: transparent !important; }
      /* High-visibility primary — Run RAG */
      .stApp button[kind="primary"],
      .stApp button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #f4ff66 0%, #DFFF00 42%, #b8cf00 100%) !important;
        color: #0a0a0a !important;
        border: 2px solid #f8ff99 !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.7rem 2rem !important;
        min-height: 3.1rem !important;
        box-shadow:
          0 0 32px rgba(223, 255, 0, 0.65),
          0 0 60px rgba(223, 255, 0, 0.25),
          0 8px 24px rgba(0, 0, 0, 0.55) !important;
        text-shadow: none !important;
      }
      .stApp button[kind="primary"]:hover,
      .stApp button[data-testid="baseButton-primary"]:hover {
        filter: brightness(1.12) saturate(1.05) !important;
        box-shadow:
          0 0 40px rgba(223, 255, 0, 0.85),
          0 0 80px rgba(223, 255, 0, 0.35),
          0 10px 28px rgba(0, 0, 0, 0.55) !important;
      }
      .stApp button[kind="primary"]:focus-visible,
      .stApp button[data-testid="baseButton-primary"]:focus-visible {
        outline: 3px solid #f8ff99 !important;
        outline-offset: 3px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Academic City RAG Assistant")
st.caption("Kofi Assan · 10022300129 · IT3241 — manual RAG (no LangChain/LlamaIndex)")
components.html(_WAVE_BANNER_HTML, height=288, scrolling=False)

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
    "---\n**Submission checklist:** GitHub repo `ai_10022300129`, deploy URL, "
    "invite **GodwinDansoAcity**, email lecturer with subject line format from the QP."
)
