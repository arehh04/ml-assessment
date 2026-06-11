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


from src.fusion import apply_mmr
from src.embedding import get_embedding_model


@pytest.fixture
def embedding_model():
    return get_embedding_model()


@pytest.fixture
def diverse_documents():
    return [
        Document(page_content="AI governance frameworks establish accountability.", metadata={"source": "a.pdf"}),
        Document(page_content="AI governance policies define oversight roles.", metadata={"source": "b.pdf"}),
        Document(page_content="GDPR requires data minimization and user consent.", metadata={"source": "c.pdf"}),
        Document(page_content="Machine learning bias can harm marginalized groups.", metadata={"source": "d.pdf"}),
        Document(page_content="AI governance accountability is central to ethics.", metadata={"source": "e.pdf"}),
        Document(page_content="Quantum computing will reshape cryptography.", metadata={"source": "f.pdf"}),
    ]


def test_apply_mmr_returns_k_documents(diverse_documents, embedding_model):
    result = apply_mmr("AI governance", diverse_documents, embedding_model, k=3)
    assert len(result) == 3
    assert all(isinstance(d, Document) for d in result)


def test_apply_mmr_no_duplicates(diverse_documents, embedding_model):
    result = apply_mmr("AI governance", diverse_documents, embedding_model, k=4)
    contents = [d.page_content for d in result]
    assert len(contents) == len(set(contents))


def test_apply_mmr_k_larger_than_candidates(diverse_documents, embedding_model):
    result = apply_mmr("AI governance", diverse_documents[:2], embedding_model, k=10)
    assert len(result) == 2  # capped at available candidates


def test_apply_mmr_empty_candidates(embedding_model):
    result = apply_mmr("query", [], embedding_model, k=5)
    assert result == []


def test_apply_mmr_selects_diverse_results(diverse_documents, embedding_model):
    # docs[0] and docs[4] are semantically near-identical (both "AI governance ... accountab...")
    # MMR with lambda_mult=0.3 should select at most 1 of these near-identical docs
    result = apply_mmr("AI governance accountability", diverse_documents, embedding_model, k=3, lambda_mult=0.3)
    selected_contents = [d.page_content for d in result]
    near_identical_count = sum(
        1 for c in selected_contents
        if "AI governance" in c and "accountab" in c.lower()
    )
    assert near_identical_count <= 1
