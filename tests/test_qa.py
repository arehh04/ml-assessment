import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.qa import answer_question, format_sources


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate_content.return_value = MagicMock(text="AI governance refers to frameworks and policies.")
    return llm


@pytest.fixture
def retrieved_docs():
    return [
        Document(
            page_content="AI governance refers to the frameworks and policies that guide AI development.",
            metadata={"source": "governance.pdf", "page": 1},
        ),
        Document(
            page_content="Transparency requirements are central to AI governance frameworks.",
            metadata={"source": "transparency.pdf", "page": 5},
        ),
    ]


def test_answer_question_returns_dict(mock_llm, retrieved_docs):
    result = answer_question(
        query="What is AI governance?",
        retrieved_docs=retrieved_docs,
        llm=mock_llm,
    )
    assert isinstance(result, dict)


def test_answer_question_has_answer_key(mock_llm, retrieved_docs):
    result = answer_question("What is AI governance?", retrieved_docs, mock_llm)
    assert "answer" in result
    assert isinstance(result["answer"], str)


def test_answer_question_has_sources_key(mock_llm, retrieved_docs):
    result = answer_question("What is AI governance?", retrieved_docs, mock_llm)
    assert "sources" in result
    assert isinstance(result["sources"], list)


def test_answer_question_has_query_key(mock_llm, retrieved_docs):
    result = answer_question("What is AI governance?", retrieved_docs, mock_llm)
    assert result["query"] == "What is AI governance?"


def test_answer_question_sources_contain_metadata(mock_llm, retrieved_docs):
    result = answer_question("What is AI governance?", retrieved_docs, mock_llm)
    for source in result["sources"]:
        assert "source" in source
        assert "content" in source


def test_answer_question_sources_count_matches_docs(mock_llm, retrieved_docs):
    result = answer_question("What is AI governance?", retrieved_docs, mock_llm)
    assert len(result["sources"]) == len(retrieved_docs)


def test_format_sources():
    docs = [
        Document(page_content="Content.", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="More content.", metadata={"source": "b.pdf", "page": 7}),
    ]
    sources = format_sources(docs)
    assert len(sources) == 2
    assert sources[0]["source"] == "a.pdf"
    assert sources[0]["page"] == 3
    assert sources[0]["content"] == "Content."
    assert sources[1]["source"] == "b.pdf"


def test_answer_question_calls_llm(mock_llm, retrieved_docs):
    answer_question("test query", retrieved_docs, mock_llm)
    assert mock_llm.generate_content.called


def test_format_sources_missing_metadata():
    docs = [Document(page_content="Content with no metadata.", metadata={})]
    sources = format_sources(docs)
    assert len(sources) == 1
    assert sources[0]["source"] == "Unknown"
    assert sources[0]["page"] is None
    assert sources[0]["content"] == "Content with no metadata."
