from langchain_core.documents import Document
from src.prompt import format_context, format_prompt


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
    context = format_context(retrieved_docs)
    prompt_text = format_prompt(context, query)
    response = llm.generate_content(prompt_text)
    return {
        "query": query,
        "answer": response.text,
        "sources": format_sources(retrieved_docs),
    }
