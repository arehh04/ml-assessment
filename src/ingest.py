import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_chroma import Chroma
from src.chunking import split_documents


def load_documents(docs_dir: str) -> list[Document]:
    """Load all PDF and TXT files from a directory."""
    if not os.path.isdir(docs_dir):
        return []

    documents: list[Document] = []

    pdf_loader = PyPDFDirectoryLoader(docs_dir, silent_errors=True)
    try:
        documents.extend(pdf_loader.load())
    except Exception:
        pass

    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )
    try:
        documents.extend(txt_loader.load())
    except Exception:
        pass

    return documents


def build_vector_store(
    documents: list[Document],
    persist_dir: str,
    embedding_model,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chroma:
    """Chunk documents, embed, and persist to ChromaDB."""
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return store
