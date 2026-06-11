# src/embedding.py — minimal stub for Task 3 tests, will be fully implemented in Task 4
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
