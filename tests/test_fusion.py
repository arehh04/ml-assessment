import pytest
from langchain_core.documents import Document
from src.fusion import reciprocal_rank_fusion


@pytest.fixture
def doc_a():
    return Document(page_content="AI governance frameworks are essential.", metadata={"source": "a.pdf"})


@pytest.fixture
def doc_b():
    return Document(page_content="Machine learning requires oversight.", metadata={"source": "b.pdf"})


@pytest.fixture
def doc_c():
    return Document(page_content="Transparency builds public trust.", metadata={"source": "c.pdf"})


@pytest.fixture
def doc_d():
    return Document(page_content="GDPR governs personal data privacy.", metadata={"source": "d.pdf"})


def test_rrf_returns_documents(doc_a, doc_b, doc_c):
    result = reciprocal_rank_fusion([[doc_a, doc_b], [doc_b, doc_c]])
    assert len(result) >= 1
    assert all(isinstance(d, Document) for d in result)


def test_rrf_deduplicates(doc_a, doc_b):
    result = reciprocal_rank_fusion([[doc_a, doc_b], [doc_a, doc_b]])
    contents = [d.page_content for d in result]
    assert len(contents) == len(set(contents))


def test_rrf_promotes_consistent_top_rank(doc_a, doc_b, doc_c, doc_d):
    # doc_a is rank 1 in both lists -> should score highest
    result = reciprocal_rank_fusion([[doc_a, doc_b, doc_c], [doc_a, doc_d, doc_b]])
    assert result[0].page_content == doc_a.page_content


def test_rrf_single_list(doc_a, doc_b, doc_c):
    result = reciprocal_rank_fusion([[doc_a, doc_b, doc_c]])
    assert result[0].page_content == doc_a.page_content


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_rrf_empty_input():
    result = reciprocal_rank_fusion([])
    assert result == []


def test_rrf_preserves_all_unique_docs(doc_a, doc_b, doc_c, doc_d):
    result = reciprocal_rank_fusion([[doc_a, doc_b], [doc_c, doc_d]])
    assert len(result) == 4
