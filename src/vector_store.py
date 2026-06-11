from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def get_or_create_store(persist_dir: str, embedding_model: Embeddings) -> Chroma:
    """Load existing ChromaDB store or create an empty one."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )


def get_all_documents(vector_store: Chroma) -> list[Document]:
    """Fetch every stored chunk — used for BM25 indexing."""
    result = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
