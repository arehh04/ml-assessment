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
