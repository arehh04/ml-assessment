import pytest
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
from src.embedding import get_embedding_model


@pytest.fixture
def embedding_model():
    return get_embedding_model()


@pytest.fixture
def populated_store(sample_documents, tmp_chroma_dir, embedding_model):
    return Chroma.from_documents(
        documents=sample_documents,
        embedding=embedding_model,
        persist_directory=tmp_chroma_dir,
    )


def test_dense_retrieve_returns_documents(populated_store):
    results = dense_retrieve(populated_store, "AI governance", k=3)
    assert len(results) == 3
    assert all(isinstance(d, Document) for d in results)


def test_dense_retrieve_relevance(populated_store):
    results = dense_retrieve(populated_store, "transparency in AI systems", k=3)
    contents = " ".join(d.page_content for d in results).lower()
    assert "transparent" in contents or "ai" in contents


def test_dense_retrieve_k_capped_by_corpus_size(populated_store, sample_documents):
    results = dense_retrieve(populated_store, "AI", k=100)
    assert len(results) <= len(sample_documents)


def test_sparse_retrieve_returns_documents(sample_documents):
    results = sparse_retrieve(sample_documents, "GDPR privacy regulations", k=3)
    assert len(results) <= 3
    assert all(isinstance(d, Document) for d in results)


def test_sparse_retrieve_keyword_match(sample_documents):
    results = sparse_retrieve(sample_documents, "GDPR", k=5)
    assert len(results) >= 1
    assert any("GDPR" in d.page_content for d in results)


def test_sparse_retrieve_empty_corpus():
    results = sparse_retrieve([], "query", k=5)
    assert results == []


def test_mmr_retrieve_returns_documents(populated_store):
    results = mmr_retrieve(populated_store, "AI governance oversight", k=3, fetch_k=6)
    assert 1 <= len(results) <= 3
    assert all(isinstance(d, Document) for d in results)


def test_mmr_retrieve_diversity(populated_store):
    results = mmr_retrieve(populated_store, "AI governance", k=4, fetch_k=7)
    contents = [d.page_content for d in results]
    # No exact duplicates
    assert len(contents) == len(set(contents))
