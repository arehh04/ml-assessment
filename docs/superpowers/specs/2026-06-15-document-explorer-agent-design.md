# Document Explorer Agent — Design Spec

**Date:** 2026-06-15  
**Status:** Approved  
**Replaces:** Chat-style `app.py` (single Q&A box)

---

## Overview

Replace the chat-style Streamlit UI with a Document Explorer Agent that lets the user select any indexed document, auto-generates three template questions about it, runs them in parallel, and streams Gemini answers — with a BM25 fast-path that appears instantly while the full dense+MMR retrieval upgrades the result in the background.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Left panel                │  Right panel            │
│  ─────────────             │  ──────────────         │
│  Document dropdown         │  Card 1 · Q1            │
│  (unique sources from      │  Card 2 · Q2            │
│   ChromaDB metadata)       │  Card 3 · Q3            │
│                            │                         │
│  Doc info                  │  Each card:             │
│  (filename, chunk count)   │  - Question text        │
│                            │  - ⚡ BM25 preview      │
│  [Explore →] button        │  - ✨ Full answer       │
│                            │     (streaming)         │
└─────────────────────────────────────────────────────┘
```

**Session state keys:**
- `selected_doc` — currently chosen document filename
- `exploring` — bool, True while retrieval is running
- `results` — list of 3 result dicts (one per question)

---

## Question Templates

Fixed strings — no LLM call, zero tokens spent on question generation:

```python
QUESTION_TEMPLATES = [
    "What is the main policy objective of this document?",
    "What organizations or agencies are referenced in this document?",
    "What are the key requirements or recommendations in this document?",
]
```

---

## Two-Phase Retrieval Per Question

Each of the three question cards runs two phases sequentially, but all three cards run in parallel via `ThreadPoolExecutor(max_workers=3)`.

### Phase 1 — BM25 fast path (~0.5–1 s)

1. Filter `all_docs` to only chunks whose `source` metadata matches the selected document filename.
2. Run `sparse_retrieve` (BM25) on those document-specific chunks, `k=3`.
3. Call `answer_question(question, bm25_chunks, llm)` — no streaming, returns immediately.
4. Write result to `st.session_state.results[i]["preview"]`.
5. UI renders with badge **⚡ BM25 preview**.

### Phase 2 — Full pipeline (~3–5 s, background)

1. Run `dense_retrieve`, `sparse_retrieve` (full corpus), `mmr_retrieve` in parallel threads.
2. `reciprocal_rank_fusion` → `apply_mmr(k=8)`.
3. Stream the answer token-by-token via `llm.stream()`.
4. Replace Phase 1 answer in the card; update badge to **✨ Full retrieval**.

---

## Parallelism

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def explore_document(selected_doc, all_docs, store, embedding_model, llm):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_question, q, i, selected_doc, ...): i
            for i, q in enumerate(QUESTION_TEMPLATES)
        }
        for future in as_completed(futures):
            idx = futures[future]
            st.session_state.results[idx] = future.result()
```

Each `run_question` call performs both Phase 1 and Phase 2 sequentially, updating session state between phases so Streamlit can re-render incrementally.

---

## Streaming

```python
def stream_answer(question, docs, llm):
    from src.prompt import build_prompt, format_context
    prompt = build_prompt()
    context = format_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    for chunk in llm.stream(messages):
        yield chunk.content
```

In the UI, the card's full-answer placeholder uses `st.write_stream(stream_answer(...))`.

---

## UI State Machine Per Card

```
IDLE → BM25_DONE → STREAMING → DONE
```

| State       | Badge              | Content                    |
|-------------|-------------------|----------------------------|
| IDLE        | (spinner)          | "Retrieving…"              |
| BM25_DONE   | ⚡ BM25 preview    | Non-streaming answer       |
| STREAMING   | ✨ Full retrieval  | Token-by-token stream      |
| DONE        | ✨ Full retrieval  | Complete answer + sources  |

---

## Confidence Badge

Reuse the existing FALLBACK_PHRASE check from the current `app.py`:

```python
FALLBACK_PHRASE = "cannot find sufficient evidence"
is_fallback = FALLBACK_PHRASE in answer.lower()
```

- Green badge if grounded, red badge if fallback — same logic as today.

---

## Cached Resources (unchanged)

`load_embedding_model`, `load_store`, `load_llm` remain `@st.cache_resource` functions. `get_all_documents(store)` is called once at startup to populate `all_docs` and the document dropdown.

---

## Document Dropdown Population

```python
sources = sorted({
    os.path.basename(doc.metadata.get("source", "unknown"))
    for doc in all_docs
})
selected_doc = st.selectbox("Select document", sources)
```

Chunk count for the selected document:

```python
chunk_count = sum(
    1 for doc in all_docs
    if os.path.basename(doc.metadata.get("source", "")) == selected_doc
)
st.caption(f"{chunk_count} chunks indexed")
```

---

## Sidebar

Keep existing sidebar items (retrieval pipeline info, embedding model, LLM label). Remove the "Clear chat" button — not applicable in this mode. Add a small "About" note explaining the two-phase retrieval strategy.

---

## Error Handling

- If `llm` is None (no API key): `st.error(...)` + `st.stop()` — same as today.
- If ChromaDB is empty: `st.warning(...)` + `st.stop()` — same as today.
- If the selected document has zero matching chunks (edge case): show an inline warning in the card instead of running retrieval.
- Streaming errors: wrap `llm.stream()` in a try/except; fall back to `answer_question()` non-streaming.

---

## Files Changed

| File | Action |
|------|--------|
| `app.py` | Full rewrite — replaces chat UI with Document Explorer |
| No other files | All src modules (`retriever.py`, `fusion.py`, `qa.py`, etc.) are unchanged |

---

## Out of Scope

- Saving or exporting answers
- Multi-document comparison
- Custom question input (the three templates cover the document's essential dimensions)
- Caching BM25 index across sessions (acceptable at this scale)
