from langchain_core.documents import Document
from src.prompt import build_prompt, format_context


def test_build_prompt_returns_template():
    from langchain_core.prompts import ChatPromptTemplate
    prompt = build_prompt()
    assert isinstance(prompt, ChatPromptTemplate)


def test_prompt_has_context_variable():
    prompt = build_prompt()
    assert "context" in prompt.input_variables


def test_prompt_has_question_variable():
    prompt = build_prompt()
    assert "question" in prompt.input_variables


def test_format_context_includes_source():
    docs = [
        Document(page_content="AI governance is important.", metadata={"source": "policy.pdf", "page": 3}),
    ]
    context = format_context(docs)
    assert "AI governance is important." in context
    assert "policy.pdf" in context


def test_format_context_multiple_docs():
    docs = [
        Document(page_content="First doc content.", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Second doc content.", metadata={"source": "b.pdf", "page": 2}),
    ]
    context = format_context(docs)
    assert "First doc content." in context
    assert "Second doc content." in context
    assert "a.pdf" in context
    assert "b.pdf" in context


def test_format_context_missing_metadata():
    docs = [Document(page_content="Content only.", metadata={})]
    context = format_context(docs)
    assert "Content only." in context


def test_prompt_renders_with_values():
    prompt = build_prompt()
    messages = prompt.format_messages(context="Some context.", question="What is AI?")
    full_text = " ".join(m.content for m in messages)
    assert "Some context." in full_text
    assert "What is AI?" in full_text
