from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from src.prompt import build_prompt, format_context


def format_sources(documents: list[Document]) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", None),
            "content": doc.page_content,
        }
        for doc in documents
    ]


def answer_question(
    query: str,
    retrieved_docs: list[Document],
    llm,
) -> dict:
    """Generate a grounded answer from retrieved documents using the LLM."""
    prompt = build_prompt()
    context = format_context(retrieved_docs)
    messages = prompt.format_messages(context=context, question=query)
    response: BaseMessage = llm.invoke(messages)
    return {
        "query": query,
        "answer": response.content,
        "sources": format_sources(retrieved_docs),
    }
