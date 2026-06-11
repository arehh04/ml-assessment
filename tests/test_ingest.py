from langchain_core.documents import Document
from src.ingest import load_documents, build_vector_store


def test_load_documents_from_txt(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is a test document about AI governance frameworks.")
    docs = load_documents(str(tmp_path))
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


def test_load_documents_empty_dir(tmp_path):
    docs = load_documents(str(tmp_path))
    assert docs == []


def test_load_documents_nonexistent_dir():
    docs = load_documents("/nonexistent/path/that/does/not/exist")
    assert docs == []


def test_load_documents_returns_metadata(tmp_path):
    txt_file = tmp_path / "policy.txt"
    txt_file.write_text("AI policy document content here.")
    docs = load_documents(str(tmp_path))
    assert len(docs) >= 1
    assert "source" in docs[0].metadata


def test_build_vector_store_creates_collection(tmp_path, sample_documents):
    from src.embedding import get_embedding_model
    embedding_model = get_embedding_model()
    chroma_dir = str(tmp_path / "chroma")
    store = build_vector_store(sample_documents, chroma_dir, embedding_model)
    assert store is not None
    result = store.similarity_search("AI governance", k=2)
    assert len(result) >= 1


def test_build_vector_store_raises_on_empty_documents(tmp_path):
    from src.embedding import get_embedding_model
    import pytest
    embedding_model = get_embedding_model()
    chroma_dir = str(tmp_path / "chroma")
    with pytest.raises(ValueError, match="No chunks to index"):
        build_vector_store([], chroma_dir, embedding_model)
