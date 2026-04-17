# CS4241 — Introduction to Artificial Intelligence (2026)

**Student:** Kofi Assan  
**Index number:** 1002300129  
**Repository name (submit as):** `ai_1002300129`

Manual RAG for Academic City: Ghana election results (CSV) + 2025 budget PDF.  
**No LangChain, LlamaIndex, or pre-built RAG frameworks** — chunking, embeddings, FAISS, retrieval, and prompts are implemented in this repo.

## Architecture (Part F)

Full diagram + component description: `docs/ARCHITECTURE.md`

```text
User query
    → Embedding (sentence-transformers)
    → FAISS top-k (cosine via normalized vectors + inner product)
    → Hybrid: BM25 scores on candidate pool, fused with vector scores
    → Context selection (truncate by char budget, ranked by fused score)
    → Prompt template (anti-hallucination rules + injected context)
    → LLM (OpenAI API if `OPENAI_API_KEY` is set)
    → Response + stage logs
```

**Why this fits the domain:** Election data is structured per row; budget text is long-form policy prose. Row-level CSV chunks preserve column semantics; sliding windows with overlap capture budget sections that span chunk boundaries. Hybrid retrieval recovers exact tokens (candidate names, figures) that pure dense retrieval sometimes misses.

## Innovation (Part G)

**Session feedback boost:** In the Streamlit sidebar, preferred sources (election CSV vs budget PDF) add a small weight to fused retrieval scores so follow-up queries favor user-trusted corpora.

## Setup

1. **Python**: This repo deploys with `runtime.txt` (**3.11.9**).  
   - If you use **Python 3.13**, local `sentence-transformers` may not install (PyTorch support).  
   - The project therefore supports **OpenAI embeddings** as a fallback for building the index on any Python version.

2. Create a virtual environment and install dependencies:

```bash
cd "c:\Users\HP\Downloads\KOFI_ASSAN - INTRO_TO_AI EXAM - 1002300129"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI key for real answers (optional for UI testing).
   - To force embeddings backend, set: `EMBEDDINGS_BACKEND=openai` (or leave default `auto`).
   - Chat provider options:
     - OpenAI: `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...`
     - Groq: `LLM_PROVIDER=groq` + `GROQ_API_KEY=...`
     - Auto-select (default): `LLM_PROVIDER=auto` prefers Groq key when present.

4. Download data and build the index:

```bash
python scripts/download_data.py
python scripts/build_index.py
```

To compare chunking configs for **Part A**, rebuild with explicit settings:

```bash
# Example Config A
python scripts/build_index.py 600 80

# Example Config B
python scripts/build_index.py 900 120
```

Each build writes `data/index/build_config.json` capturing the chunking + embedding settings used.

5. Run the app:

```bash
streamlit run app.py
```

## Deploy on Render

1. Push this repo to GitHub (see submission repo name above).
2. In [Render](https://render.com): **New** → **Blueprint** → connect the repository. Render reads `render.yaml` and creates the web service.
3. In the service **Environment** tab, set **OPENAI_API_KEY** (required for real answers). Optionally set **HF_TOKEN** if Hugging Face throttles the embedding model download during build.
4. After deploy, copy the **public URL** for your exam email.

The **build** step installs dependencies, downloads the CSV/PDF, and runs `scripts/build_index.py`. If the build exceeds Render’s time limit, build the index on your machine, then force-add the artifacts (they are gitignored by default):

```bash
git add -f data/index data/raw
git commit -m "Add prebuilt index for Render"
git push
```

Redeploy so the build can skip long work (you may shorten `buildCommand` in `render.yaml` to `pip install -r requirements.txt` only if both `data/raw` and `data/index` are committed).

## Chunking (Part A)

- **CSV:** One chunk per cleaned row; all columns concatenated with labels so retrieval sees full context for that record.  
- **PDF:** ~900 characters per window, **120-character overlap** (~13%). Justification: large enough for policy sentences; overlap limits information split across boundaries. Compare smaller/larger windows in `experiment_logs/` and note retrieval quality.

## Retrieval failure cases (Part B)

Document your own runs in `experiment_logs/`. Typical patterns to test:

1. **Exact name or year:** Pure vector retrieval may rank a semantically similar but wrong constituency or section; **hybrid BM25** tends to fix this by matching rare tokens.  
2. **Ambiguous query:** Short queries can pull generic budget chunks; mitigation: query expansion toggle in `rag/retrieval.py` or stricter `select_context` / user clarification in the UI.

## Experiment evidence runner (Parts B–E)

To generate **evidence JSON** (retrieval hits, prompts, answers) you can attach to your manual writeups:

```bash
# One query
python scripts/run_experiments.py --query "What is the budget deficit target?" --hybrid --llm-only

# Multiple queries (one per line, # comments allowed)
python scripts/run_experiments.py --queries-file experiment_logs/queries.txt --hybrid --query-expansion --llm-only
```

Outputs are written to `experiment_logs/auto_runs/`. You should still write your own manual reflections using the templates in `experiment_logs/`.

## Submission (from question paper)

- Push to GitHub: repo **`ai_1002300129`** (replace if your index differs).  
- Deploy (this repo includes **Render** via `render.yaml`; or use Streamlit Cloud / Railway) and record the public URL.  
- Add **GodwinDansoAcity** / `godwin.danso@acity.edu.gh` as collaborator.  
- Email the lecturer with subject: `CS4241-Introduction to Artificial Intelligence-2026:[your index and name]`.  
- Include: repo link, deployed URL, **video walkthrough (≤2 min)**, **manual** experiment logs, and this documentation.

## Files

| Path | Role |
|------|------|
| `rag/chunking.py` | Cleaning + chunk strategies |
| `rag/embeddings.py` | Sentence-transformers embeddings |
| `rag/store.py` | FAISS persist/load |
| `rag/retrieval.py` | Top-k, BM25, hybrid fusion |
| `rag/prompts.py` | Context window + templates |
| `rag/pipeline.py` | End-to-end logging + LLM call |
| `app.py` | Streamlit UI |
| `scripts/download_data.py` | Fetch exam datasets |
| `scripts/build_index.py` | Build `data/index/` |
| `render.yaml` | Render Blueprint (build + start commands) |
| `runtime.txt` | Python version for Render/native Python builds |

Student name and index appear in the README and in each source file header as required.
