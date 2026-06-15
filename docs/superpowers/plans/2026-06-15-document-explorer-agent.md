# Document Explorer Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the chat-style Streamlit UI with a Document Explorer Agent that selects a document, auto-generates three fixed questions, and answers them with a BM25 instant preview followed by a full hybrid-retrieval streaming answer.

**Architecture:** Helper logic (`bm25_fast_path`, `full_pipeline_retrieve`, `stream_answer`) lives in `src/explorer.py` so it can be unit-tested without Streamlit. `app.py` is a full rewrite that imports these helpers and orchestrates a three-phase flow: (1) parallel BM25 previews, (2) parallel full retrieval, (3) sequential streaming per card. All three questions run in parallel for phases 1 and 2 via `ThreadPoolExecutor(max_workers=3)`.

**Tech Stack:** Streamlit, LangChain BM25Retriever, ChromaDB, HuggingFaceEmbeddings, Gemini 2.5 Flash (`langchain-google-genai`), `concurrent.futures.ThreadPoolExecutor`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/explorer.py` | **Create** | `bm25_fast_path`, `full_pipeline_retrieve`, `stream_answer`, `FALLBACK_PHRASE` |
| `tests/test_explorer.py` | **Create** | Unit tests for `src/explorer.py` helpers |
| `app.py` | **Full rewrite** | Streamlit Document Explorer Agent UI |

Existing `src/` modules (`retriever.py`, `fusion.py`, `qa.py`, `prompt.py`, etc.) are **not modified**.

---

### Task 1: Write failing tests for the explorer helpers

**Files:**
- Create: `tests/test_explorer.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_explorer.py` with this content:

```python
import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Policy objective is to ensure AI safety.",
            metadata={"source": "data/documents/100.txt"},
        ),
        Document(
            page_content="The agency responsible is NIST.",
            metadata={"source": "data/documents/100.txt"},
        ),
        Document(
            page_content="Requirements include regular audits.",
            metadata={"source": "data/documents/100.txt"},
        ),
        Document(
            page_content="Unrelated content from another document.",
            metadata={"source": "data/documents/200.txt"},
        ),
    ]


# ── bm25_fast_path ────────────────────────────────────────────────────────────

def test_bm25_fast_path_filters_to_selected_document(sample_docs):
    from src.explorer import bm25_fast_path
    with patch("src.explorer.answer_question") as mock_answer:
        mock_answer.return_value = {"answer": "AI safety.", "sources": []}
        bm25_fast_path("What is the objective?", "100.txt", sample_docs, MagicMock())
        docs_passed = mock_answer.call_args[0][1]
        assert all(
            os.path.basename(d.metadata["source"]) == "100.txt"
            for d in docs_passed
        ), "Must pass only chunks from the selected document to answer_question"


def test_bm25_fast_path_returns_answer_and_sources(sample_docs):
    from src.explorer import bm25_fast_path
    with patch("src.explorer.answer_question") as mock_answer:
        mock_answer.return_value = {
            "answer": "AI safety.",
            "sources": [{"source": "100.txt", "page": None, "content": "x"}],
        }
        result = bm25_fast_path("What is the objective?", "100.txt", sample_docs, MagicMock())
        assert result["answer"] == "AI safety."
        assert result["sources"] == [{"source": "100.txt", "page": None, "content": "x"}]


def test_bm25_fast_path_no_matching_chunks_returns_fallback():
    from src.explorer import bm25_fast_path, FALLBACK_PHRASE
    docs = [
        Document(page_content="Content.", metadata={"source": "data/documents/999.txt"})
    ]
    result = bm25_fast_path("What is the objective?", "000.txt", docs, MagicMock())
    assert FALLBACK_PHRASE in result["answer"]
    assert result["sources"] == []


# ── full_pipeline_retrieve ────────────────────────────────────────────────────

def test_full_pipeline_retrieve_calls_all_retrievers(sample_docs):
    from src.explorer import full_pipeline_retrieve
    mock_store = MagicMock()
    mock_embedding = MagicMock()

    with patch("src.explorer.dense_retrieve", return_value=sample_docs[:2]) as mock_dense, \
         patch("src.explorer.sparse_retrieve", return_value=sample_docs[1:3]) as mock_sparse, \
         patch("src.explorer.mmr_retrieve", return_value=sample_docs[:3]) as mock_mmr, \
         patch("src.explorer.reciprocal_rank_fusion", return_value=sample_docs[:3]), \
         patch("src.explorer.apply_mmr", return_value=sample_docs[:2]):
        full_pipeline_retrieve("What is the objective?", mock_store, sample_docs, mock_embedding, k=8)
        mock_dense.assert_called_once_with(mock_store, "What is the objective?", k=20)
        mock_sparse.assert_called_once_with(sample_docs, "What is the objective?", k=20)
        mock_mmr.assert_called_once_with(mock_store, "What is the objective?", k=20, fetch_k=60)


def test_full_pipeline_retrieve_passes_k_to_apply_mmr(sample_docs):
    from src.explorer import full_pipeline_retrieve
    mock_store = MagicMock()
    mock_embedding = MagicMock()

    with patch("src.explorer.dense_retrieve", return_value=sample_docs[:2]), \
         patch("src.explorer.sparse_retrieve", return_value=sample_docs[1:3]), \
         patch("src.explorer.mmr_retrieve", return_value=sample_docs[:3]), \
         patch("src.explorer.reciprocal_rank_fusion", return_value=sample_docs[:3]), \
         patch("src.explorer.apply_mmr", return_value=sample_docs[:2]) as mock_apply:
        full_pipeline_retrieve("Q", mock_store, sample_docs, mock_embedding, k=5)
        mock_apply.assert_called_once_with("Q", sample_docs[:3], mock_embedding, k=5)


def test_full_pipeline_retrieve_returns_list(sample_docs):
    from src.explorer import full_pipeline_retrieve
    with patch("src.explorer.dense_retrieve", return_value=sample_docs[:2]), \
         patch("src.explorer.sparse_retrieve", return_value=sample_docs[1:3]), \
         patch("src.explorer.mmr_retrieve", return_value=sample_docs[:3]), \
         patch("src.explorer.reciprocal_rank_fusion", return_value=sample_docs[:3]), \
         patch("src.explorer.apply_mmr", return_value=sample_docs[:2]):
        result = full_pipeline_retrieve("Q", MagicMock(), sample_docs, MagicMock(), k=8)
        assert isinstance(result, list)
        assert len(result) == 2


# ── stream_answer ─────────────────────────────────────────────────────────────

def test_stream_answer_yields_content_from_llm_chunks(sample_docs):
    from src.explorer import stream_answer
    chunk1 = MagicMock()
    chunk1.content = "The "
    chunk2 = MagicMock()
    chunk2.content = "answer."
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([chunk1, chunk2])

    mock_template = MagicMock()
    mock_template.format_messages.return_value = [MagicMock()]

    with patch("src.explorer.build_prompt", return_value=mock_template), \
         patch("src.explorer.format_context", return_value="Some context."):
        result = list(stream_answer("What?", sample_docs, mock_llm))

    assert result == ["The ", "answer."]


def test_stream_answer_passes_question_to_prompt(sample_docs):
    from src.explorer import stream_answer
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([])
    mock_template = MagicMock()
    mock_template.format_messages.return_value = []

    with patch("src.explorer.build_prompt", return_value=mock_template), \
         patch("src.explorer.format_context", return_value="ctx"):
        list(stream_answer("My question?", sample_docs, mock_llm))

    mock_template.format_messages.assert_called_once_with(
        context="ctx", question="My question?"
    )
```

- [ ] **Step 2: Run tests to confirm they fail with import error**

```
pytest tests/test_explorer.py -v
```

Expected output:
```
ModuleNotFoundError: No module named 'src.explorer'
```

---

### Task 2: Implement src/explorer.py

**Files:**
- Create: `src/explorer.py`

- [ ] **Step 1: Create the module**

Create `src/explorer.py` with this content:

```python
import os
from langchain_core.documents import Document

from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
from src.fusion import reciprocal_rank_fusion, apply_mmr
from src.qa import answer_question
from src.prompt import build_prompt, format_context

FALLBACK_PHRASE = "cannot find sufficient evidence"


def bm25_fast_path(question: str, doc_filename: str, all_docs: list, llm) -> dict:
    doc_chunks = [
        d for d in all_docs
        if os.path.basename(d.metadata.get("source", "")) == doc_filename
    ]
    if not doc_chunks:
        return {"answer": f"{FALLBACK_PHRASE} in this document.", "sources": []}
    retrieved = sparse_retrieve(doc_chunks, question, k=3)
    result = answer_question(question, retrieved, llm)
    return {"answer": result["answer"], "sources": result["sources"]}


def full_pipeline_retrieve(
    question: str,
    store,
    all_docs: list,
    embedding_model,
    k: int = 8,
) -> list:
    dense_results  = dense_retrieve(store, question, k=20)
    sparse_results = sparse_retrieve(all_docs, question, k=20)
    mmr_results    = mmr_retrieve(store, question, k=20, fetch_k=60)
    fused          = reciprocal_rank_fusion([dense_results, sparse_results, mmr_results])
    return apply_mmr(question, fused, embedding_model, k=k)


def stream_answer(question: str, docs: list, llm):
    prompt = build_prompt()
    context = format_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    for chunk in llm.stream(messages):
        yield chunk.content
```

- [ ] **Step 2: Run the new tests**

```
pytest tests/test_explorer.py -v
```

Expected: 8 passed

- [ ] **Step 3: Run the full test suite to check for regressions**

```
pytest tests/ --ignore=tests/test_integration.py -v
```

Expected: All 58 existing tests + 8 new = 66 passed

- [ ] **Step 4: Commit**

```
git add src/explorer.py tests/test_explorer.py
git commit -m "feat: add explorer helpers (bm25_fast_path, full_pipeline_retrieve, stream_answer)"
```

---

### Task 3: Rewrite app.py — document picker and card skeleton

**Files:**
- Rewrite: `app.py`

- [ ] **Step 1: Replace app.py with the new file**

Overwrite `app.py` completely with this content:

```python
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.embedding import get_embedding_model
from src.vector_store import get_or_create_store, get_all_documents
from src.qa import format_sources
from src.explorer import FALLBACK_PHRASE, bm25_fast_path, full_pipeline_retrieve, stream_answer

QUESTION_TEMPLATES = [
    "What is the main policy objective of this document?",
    "What organizations or agencies are referenced in this document?",
    "What are the key requirements or recommendations in this document?",
]

st.set_page_config(
    page_title="Document Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Context chunks", min_value=1, max_value=16, value=8)
    st.divider()
    st.markdown("**Two-phase retrieval**")
    st.markdown(
        "- ⚡ Phase 1: BM25 only (document-filtered, instant)\n"
        "- ✨ Phase 2: Dense + Sparse + MMR → RRF → MMR diversification"
    )
    st.divider()
    st.markdown("**Embedding:** `all-MiniLM-L6-v2`")
    st.markdown("**LLM:** Gemini 2.5 Flash · temp=0")


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_store(_embedding_model):
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return get_or_create_store(chroma_dir, _embedding_model)


@st.cache_resource(show_spinner="Loading LLM...")
def load_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )


# ── Load resources ────────────────────────────────────────────────────────────
embedding_model = load_embedding_model()
store = load_store(embedding_model)
llm = load_llm()

if llm is None:
    st.error("⚠️ GOOGLE_API_KEY not set in .env — LLM unavailable.")
    st.stop()

all_docs = get_all_documents(store)
if not all_docs:
    st.warning("⚠️ ChromaDB is empty. Run `python src/main.py --ingest` first.")
    st.stop()

st.success(f"✅ {len(all_docs):,} chunks indexed.")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Document Explorer Agent")
st.caption(
    "Select a document · 3 questions auto-generated · "
    "BM25 instant preview → full hybrid retrieval streams in background"
)

# ── Document picker ───────────────────────────────────────────────────────────
sources = sorted({
    os.path.basename(d.metadata.get("source", "unknown"))
    for d in all_docs
})

left_col, right_col = st.columns([1, 2])

with left_col:
    selected_doc = st.selectbox("Select document", sources)
    chunk_count = sum(
        1 for d in all_docs
        if os.path.basename(d.metadata.get("source", "")) == selected_doc
    )
    st.caption(f"{chunk_count} chunks indexed for this document")
    st.divider()
    explore = st.button("Explore →", use_container_width=True, type="primary")

with right_col:
    if not explore:
        st.info("Select a document on the left and click **Explore →** to begin.")
```

- [ ] **Step 2: Confirm app renders without crashing**

Run:
```
streamlit run app.py
```

Open `http://localhost:8501`. Expected: Two-column layout appears. Left panel shows a document dropdown with 646 entries. Right panel shows the "Select a document" info message. No errors in the terminal.

Stop the server with Ctrl+C.

---

### Task 4: Phase 1 — parallel BM25 previews

**Files:**
- Modify: `app.py` — append to the `with right_col:` block

- [ ] **Step 1: Add the card layout and Phase 1 execution inside the `with right_col:` block**

Replace the `with right_col:` block (from `with right_col:` to end of file) with this expanded version:

```python
with right_col:
    if not explore:
        st.info("Select a document on the left and click **Explore →** to begin.")
    else:
        # Build card layout — 3 cards with updateable slots
        badge_slots = []
        preview_slots = []
        full_slots = []
        source_slots = []

        for i, question in enumerate(QUESTION_TEMPLATES):
            with st.container(border=True):
                st.markdown(f"**Q{i+1}:** {question}")
                badge_slots.append(st.empty())
                preview_slots.append(st.empty())
                full_slots.append(st.empty())
                source_slots.append(st.empty())

        # ── Phase 1: parallel BM25 fast path ─────────────────────────────────
        for slot in badge_slots:
            slot.markdown("⏳ Retrieving BM25 preview…")

        with ThreadPoolExecutor(max_workers=3) as ex:
            bm25_futures = {
                ex.submit(bm25_fast_path, q, selected_doc, all_docs, llm): i
                for i, q in enumerate(QUESTION_TEMPLATES)
            }
            bm25_results = [None, None, None]
            for future in as_completed(bm25_futures):
                idx = bm25_futures[future]
                bm25_results[idx] = future.result()

        for i, res in enumerate(bm25_results):
            badge_slots[i].markdown("⚡ **BM25 preview**")
            preview_slots[i].markdown(res["answer"])
```

- [ ] **Step 2: Test Phase 1 in the browser**

Run:
```
streamlit run app.py
```

Select any document (e.g. `100.txt`) and click **Explore →**. Expected:
- 3 cards appear immediately
- Each shows "⏳ Retrieving BM25 preview…" then switches to "⚡ **BM25 preview**" with a non-streaming answer within ~1–2 seconds.
- Cards for all 3 questions appear at roughly the same time (parallel).

Stop the server with Ctrl+C.

---

### Task 5: Phase 2 — parallel full retrieval and streaming answers

**Files:**
- Modify: `app.py` — append Phase 2 and Phase 3 inside the `else:` block

- [ ] **Step 1: Add Phase 2 and Phase 3 after the Phase 1 block**

Append the following code inside the `else:` block, directly after the Phase 1 `for i, res in enumerate(bm25_results):` loop:

```python
        # ── Phase 2: parallel full retrieval ─────────────────────────────────
        for slot in badge_slots:
            slot.markdown("⚡ **BM25 preview** · upgrading with full retrieval…")

        with ThreadPoolExecutor(max_workers=3) as ex:
            full_futures = {
                ex.submit(
                    full_pipeline_retrieve, q, store, all_docs, embedding_model, top_k
                ): i
                for i, q in enumerate(QUESTION_TEMPLATES)
            }
            full_docs_list = [None, None, None]
            for future in as_completed(full_futures):
                idx = full_futures[future]
                full_docs_list[idx] = future.result()

        # ── Phase 3: stream answers sequentially ─────────────────────────────
        for i, (question, docs) in enumerate(zip(QUESTION_TEMPLATES, full_docs_list)):
            badge_slots[i].markdown("✨ **Full retrieval** — streaming…")
            preview_slots[i].empty()

            full_answer = full_slots[i].write_stream(stream_answer(question, docs, llm))

            is_fallback = FALLBACK_PHRASE in full_answer.lower()
            badge_slots[i].markdown(
                "🔴 No evidence found" if is_fallback else "🟢 ✨ **Full retrieval**"
            )

            srcs = format_sources(docs)
            if srcs:
                src_lines = "\n".join(
                    f"- `{s['source']}`"
                    + (f" · Page {s['page']}" if s.get("page") else "")
                    for s in srcs
                )
                source_slots[i].markdown(f"**Sources:**\n{src_lines}")
```

- [ ] **Step 2: Run the full flow in the browser**

Run:
```
streamlit run app.py
```

Select a document and click **Explore →**. Expected sequence:
1. 3 cards appear, each showing "⏳ Retrieving BM25 preview…"
2. Within ~1–2 s: all 3 cards update to "⚡ **BM25 preview**" with answers
3. Badges update to "⚡ **BM25 preview** · upgrading with full retrieval…"
4. Card 1 clears and begins streaming the full answer token-by-token
5. Card 1 finishes: green badge "🟢 ✨ **Full retrieval**" + sources list
6. Card 2 begins streaming, then finishes
7. Card 3 begins streaming, then finishes
8. No terminal errors

- [ ] **Step 3: Test fallback behaviour**

Select a document whose content is unlikely to answer the questions (any document works since BM25 filters to 3 chunks — a very short document with generic content may trigger the fallback). If the answer contains "cannot find sufficient evidence", the badge should show "🔴 No evidence found".

- [ ] **Step 4: Commit**

```
git add app.py
git commit -m "feat: rewrite app.py as Document Explorer Agent with two-phase retrieval"
```

---

### Task 6: Run full test suite and confirm no regressions

**Files:** None

- [ ] **Step 1: Run all unit tests**

```
pytest tests/ --ignore=tests/test_integration.py -v
```

Expected: 66 passed, 0 failed, 0 errors.

If any test fails, read the error, trace it back to the relevant file, fix, and re-run before committing.

- [ ] **Step 2: Final smoke test — explore two different documents**

Run:
```
streamlit run app.py
```

Pick two different documents from the dropdown. For each, click **Explore →** and verify:
- BM25 preview appears for all 3 cards simultaneously
- Full streaming answer replaces each preview card by card
- Sources list appears under each answer
- No Python exceptions in the terminal

- [ ] **Step 3: Commit test run confirmation**

```
git add .
git commit -m "test: verify 66 unit tests pass after Document Explorer rewrite"
```

Only commit if there were any unstaged changes. If the test run produced no file changes, skip this step.
