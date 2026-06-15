import os
from langchain_core.documents import Document

from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
from src.fusion import reciprocal_rank_fusion, apply_mmr
from src.qa import answer_question
from src.prompt import build_prompt, format_context

FALLBACK_PHRASE = "cannot find sufficient evidence"


def bm25_fast_path(question: str, doc_filename: str, all_docs: list[Document], llm) -> dict:
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
    all_docs: list[Document],
    embedding_model,
    k: int = 8,
) -> list[Document]:
    dense_results  = dense_retrieve(store, question, k=20)
    sparse_results = sparse_retrieve(all_docs, question, k=20)
    mmr_results    = mmr_retrieve(store, question, k=20, fetch_k=60)
    fused          = reciprocal_rank_fusion([dense_results, sparse_results, mmr_results])
    return apply_mmr(question, fused, embedding_model, k=k)


def stream_answer(question: str, docs: list[Document], llm):
    prompt = build_prompt()
    context = format_context(docs)
    messages = prompt.format_messages(context=context, question=question)
    for chunk in llm.stream(messages):
        yield chunk.content
