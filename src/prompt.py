from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM_TEMPLATE = """You are an AI assistant answering questions using only the provided context from AI governance documents.

Instructions:
1. Use ONLY the retrieved context below. Do not use outside knowledge.
2. If the answer is not found in the context, respond with exactly: "I cannot find sufficient evidence in the provided documents to answer this question."
3. Cite the source document and page for each claim you make.
4. Be concise and factual.

Context:
{context}"""

_HUMAN_TEMPLATE = "Question: {question}"


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", _HUMAN_TEMPLATE),
        ]
    )


def format_prompt(context: str, question: str) -> str:
    return _SYSTEM_TEMPLATE.format(context=context) + "\n\n" + _HUMAN_TEMPLATE.format(question=question)


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a numbered context block with source citations."""
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        citation = f"[{i}] Source: {source}" + (f", Page {page}" if page else "")
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n".join(parts)
