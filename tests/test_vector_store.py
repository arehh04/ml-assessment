import pytest
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.vector_store import get_or_create_store, get_all_documents
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


def test_get_or_create_store_creates_new(tmp_chroma_dir, embedding_model):
    store = get_or_create_store(tmp_chroma_dir, embedding_model)
    assert store is not None


def test_get_or_create_store_loads_existing(tmp_chroma_dir, sample_documents, embedding_model):
    # First: create and populate
    Chroma.from_documents(
        documents=sample_documents,
        embedding=embedding_model,
        persist_directory=tmp_chroma_dir,
    )
    # Second: load without re-adding documents
    store = get_or_create_store(tmp_chroma_dir, embedding_model)
    result = store.similarity_search("AI governance", k=2)
    assert len(result) >= 1


def test_get_all_documents_returns_all(populated_store, sample_documents):
    docs = get_all_documents(populated_store)
    assert len(docs) == len(sample_documents)
    assert all(isinstance(d, Document) for d in docs)


def test_get_all_documents_preserves_content(populated_store, sample_documents):
    docs = get_all_documents(populated_store)
    contents = {d.page_content for d in docs}
    expected_contents = {d.page_content for d in sample_documents}
    assert contents == expected_contents


def test_get_all_documents_preserves_metadata(populated_store):
    docs = get_all_documents(populated_store)
    assert all("source" in d.metadata for d in docs)
