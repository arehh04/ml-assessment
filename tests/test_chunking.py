from langchain_core.documents import Document
from src.chunking import split_documents


def test_split_returns_documents():
    docs = [Document(page_content="word " * 400, metadata={"source": "a.pdf", "page": 1})]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


def test_split_respects_chunk_size():
    docs = [Document(page_content="word " * 400, metadata={"source": "a.pdf", "page": 1})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # some tolerance for splitter boundaries


def test_split_preserves_metadata():
    docs = [Document(page_content="word " * 400, metadata={"source": "test.pdf", "page": 5})]
    chunks = split_documents(docs)
    assert all(c.metadata["source"] == "test.pdf" for c in chunks)


def test_split_empty_input():
    assert split_documents([]) == []


def test_split_short_document_returns_one_chunk():
    docs = [Document(page_content="Short text.", metadata={"source": "a.pdf", "page": 1})]
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."
