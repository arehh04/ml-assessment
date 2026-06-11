from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def dense_retrieve(vector_store: Chroma, query: str, k: int = 20) -> list[Document]:
    """Semantic similarity search using dense embeddings."""
    return vector_store.similarity_search(query, k=k)


def sparse_retrieve(
    documents: list[Document],
    query: str,
    k: int = 20,
) -> list[Document]:
    """Keyword-based BM25 retrieval over a document list."""
    if not documents:
        return []
    retriever = BM25Retriever.from_documents(documents, k=k)
    return retriever.invoke(query)


def mmr_retrieve(
    vector_store: Chroma,
    query: str,
    k: int = 20,
    fetch_k: int = 60,
) -> list[Document]:
    """Maximum Marginal Relevance search for diverse dense candidates."""
    return vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
    )
