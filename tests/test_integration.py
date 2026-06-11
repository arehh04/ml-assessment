"""
Integration tests — require:
  1. Documents in data/documents/ (run: python scripts/download_data.py)
  2. GOOGLE_API_KEY in .env (for the LLM test only)

Skip tests with: pytest tests/test_integration.py -m "not integration"
Run all with:    pytest tests/test_integration.py -v -m integration
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
def test_full_pipeline_retrieval(tmp_path):
    """Test retrieval pipeline without LLM call."""
    from src.embedding import get_embedding_model
    from src.ingest import load_documents, build_vector_store
    from src.vector_store import get_all_documents
    from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
    from src.fusion import reciprocal_rank_fusion, apply_mmr

    docs_dir = "data/documents"
    if not os.path.isdir(docs_dir) or not os.listdir(docs_dir):
        pytest.skip("data/documents is empty — run scripts/download_data.py first")

    embedding_model = get_embedding_model()
    chroma_dir = str(tmp_path / "chroma")

    documents = load_documents(docs_dir)
    assert len(documents) > 0, "No documents loaded"

    store = build_vector_store(documents, chroma_dir, embedding_model)
    all_docs = get_all_documents(store)
    assert len(all_docs) > 0

    query = "What are the key principles of AI governance?"

    dense_results = dense_retrieve(store, query, k=10)
    sparse_results = sparse_retrieve(all_docs, query, k=10)
    mmr_results = mmr_retrieve(store, query, k=10, fetch_k=30)

    assert len(dense_results) > 0
    assert len(sparse_results) > 0
    assert len(mmr_results) > 0

    fused = reciprocal_rank_fusion([dense_results, sparse_results, mmr_results])
    assert len(fused) > 0

    final = apply_mmr(query, fused, embedding_model, k=5)
    assert 1 <= len(final) <= 5

    # No duplicate content in final results
    contents = [d.page_content for d in final]
    assert len(contents) == len(set(contents))


@pytest.mark.integration
def test_full_pipeline_with_llm(tmp_path):
    """End-to-end test including LLM call. Requires GOOGLE_API_KEY."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")

    docs_dir = "data/documents"
    if not os.path.isdir(docs_dir) or not os.listdir(docs_dir):
        pytest.skip("data/documents is empty — run scripts/download_data.py first")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.embedding import get_embedding_model
    from src.ingest import load_documents, build_vector_store
    from src.vector_store import get_all_documents
    from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
    from src.fusion import reciprocal_rank_fusion, apply_mmr
    from src.qa import answer_question

    embedding_model = get_embedding_model()
    chroma_dir = str(tmp_path / "chroma")
    documents = load_documents(docs_dir)
    store = build_vector_store(documents, chroma_dir, embedding_model)
    all_docs = get_all_documents(store)

    query = "What is AI governance and why does it matter?"

    dense_results = dense_retrieve(store, query, k=20)
    sparse_results = sparse_retrieve(all_docs, query, k=20)
    mmr_results = mmr_retrieve(store, query, k=20, fetch_k=60)
    fused = reciprocal_rank_fusion([dense_results, sparse_results, mmr_results])
    final = apply_mmr(query, fused, embedding_model, k=8)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    result = answer_question(query, final, llm)

    assert "answer" in result
    assert len(result["answer"]) > 50
    assert "sources" in result
    assert len(result["sources"]) > 0
    assert "cannot find sufficient evidence" not in result["answer"].lower()
