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
    st.markdown("**LLM:** Gemini 2.5 Flash Lite · temp=0")


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
        model="gemini-2.5-flash-lite-preview-06-17",
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
                try:
                    bm25_results[idx] = future.result()
                except Exception as exc:
                    bm25_results[idx] = {"answer": f"{FALLBACK_PHRASE} (retrieval error: {exc})", "sources": []}

        for i, res in enumerate(bm25_results):
            badge_slots[i].markdown("⚡ **BM25 preview**")
            preview_slots[i].markdown(res["answer"] if res else f"⚠️ {FALLBACK_PHRASE}")

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
                try:
                    full_docs_list[idx] = future.result()
                except Exception as exc:
                    full_docs_list[idx] = []

        # ── Phase 3: stream answers sequentially ─────────────────────────────
        for i, (question, docs) in enumerate(zip(QUESTION_TEMPLATES, full_docs_list)):
            badge_slots[i].markdown("✨ **Full retrieval** — streaming…")
            preview_slots[i].empty()

            full_answer = full_slots[i].write_stream(stream_answer(question, docs or [], llm))

            is_fallback = FALLBACK_PHRASE in full_answer.lower()
            badge_slots[i].markdown(
                "🔴 No evidence found" if is_fallback else "🟢 ✨ **Full retrieval**"
            )

            srcs = format_sources(docs or [])
            if srcs:
                src_lines = "\n".join(
                    f"- `{s['source']}`"
                    + (f" · Page {s['page']}" if s.get("page") else "")
                    for s in srcs
                )
                source_slots[i].markdown(f"**Sources:**\n{src_lines}")
