"""
Usage:
    python src/main.py --query "What is AI governance?"
    python src/main.py --query "What does OECD say about AI?" --top-k 5
    python src/main.py --ingest  # Re-index documents from data/documents/
"""
import argparse
import os
import sys

# Ensure the project root is on sys.path so `from src.X import ...` works
# when this file is invoked directly as `python src/main.py`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv

load_dotenv()

from src.embedding import get_embedding_model
from src.ingest import load_documents, build_vector_store
from src.vector_store import get_or_create_store, get_all_documents
from src.retriever import dense_retrieve, sparse_retrieve, mmr_retrieve
from src.fusion import reciprocal_rank_fusion, apply_mmr
from src.qa import answer_question


def _get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Copy .env.example to .env and add your key.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )


def ingest(docs_dir: str, chroma_dir: str) -> None:
    print(f"Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    if not documents:
        print(f"No documents found in {docs_dir}. Run scripts/download_data.py first.")
        return
    print(f"Loaded {len(documents)} pages. Chunking and indexing...")
    embedding_model = get_embedding_model()
    store = build_vector_store(documents, chroma_dir, embedding_model)
    all_docs = get_all_documents(store)
    print(f"Indexed {len(all_docs)} chunks into ChromaDB at {chroma_dir}")


def query(question: str, chroma_dir: str, top_k: int = 8) -> None:
    embedding_model = get_embedding_model()
    store = get_or_create_store(chroma_dir, embedding_model)
    all_docs = get_all_documents(store)

    if not all_docs:
        print("ChromaDB is empty. Run: python src/main.py --ingest")
        return

    print(f"Retrieving context for: {question!r}")

    dense_results = dense_retrieve(store, question, k=20)
    sparse_results = sparse_retrieve(all_docs, question, k=20)
    mmr_results = mmr_retrieve(store, question, k=20, fetch_k=60)

    fused = reciprocal_rank_fusion([dense_results, sparse_results, mmr_results])
    final_docs = apply_mmr(question, fused, embedding_model, k=top_k)

    print(f"Using {len(final_docs)} context chunks.\n")

    llm = _get_llm()
    result = answer_question(question, final_docs, llm)

    print("=" * 60)
    print(f"Answer:\n{result['answer']}")
    print("\nSources:")
    for src in result["sources"]:
        page_info = f", Page {src['page']}" if src.get("page") else ""
        print(f"  - {src['source']}{page_info}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG system for AI governance documents")
    parser.add_argument("--query", "-q", type=str, help="Question to answer")
    parser.add_argument("--ingest", action="store_true", help="Re-ingest documents")
    parser.add_argument("--top-k", type=int, default=8, help="Number of context chunks (default: 8)")
    args = parser.parse_args()

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    docs_dir = os.getenv("DOCS_DIR", "./data/documents")

    if args.ingest:
        ingest(docs_dir, chroma_dir)
    elif args.query:
        query(args.query, chroma_dir, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
