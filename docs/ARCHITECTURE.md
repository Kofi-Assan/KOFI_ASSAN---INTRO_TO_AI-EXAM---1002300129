# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence

# Part F — Architecture & System Design

## High-level architecture

```mermaid
flowchart LR
  U[User (Streamlit UI)] -->|query| Q[Query handler]

  subgraph IndexBuild[Offline / Build-time]
    D1[CSV: Ghana_Election_Result.csv] --> C1[Clean + chunk CSV rows]
    D2[PDF: 2025 Budget Statement] --> C2[Extract text + sliding window chunking]
    C1 --> E[Embedding pipeline]
    C2 --> E
    E --> V[(FAISS index + chunks.json)]
  end

  subgraph Runtime[Online / Query-time]
    Q --> R1[Dense retrieval (FAISS top-k)]
    Q --> R2[Keyword scoring (BM25)]
    R1 --> F[Fuse scores + rank]
    R2 --> F
    F --> S[Context selection (char budget)]
    S --> P[Prompt template + hallucination controls]
    P --> L[LLM (OpenAI Chat API)]
    L --> A[Answer]
    F --> Lg[Stage logs: retrieval + scores + final prompt]
    S --> Lg
    P --> Lg
    L --> Lg
  end

  V --> R1
  A --> U
  Lg --> U
```

## Components and interactions (what to explain in your writeup)
- **Chunking**: CSV row-level chunks preserve table semantics; PDF sliding windows preserve long-form policy context with overlap to avoid boundary loss.
- **Embeddings**: `sentence-transformers` by default; optional OpenAI embeddings fallback.
- **Vector store**: FAISS inner-product index over normalized vectors (cosine-like).
- **Hybrid retrieval**: BM25 keyword scoring + dense retrieval fused to reduce exact-token failures (names/years/figures).
- **Context selection**: selects top ranked chunks within a character budget; trims last chunk if needed.
- **Prompting**: context injection + explicit “don’t invent” constraints; strict vs concise prompt variants.
- **Logging (Part D)**: pipeline stages store retrieval hits + scores, selected context metadata, and the final prompt sent to the LLM.

