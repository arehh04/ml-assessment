import hashlib
from langchain_core.documents import Document


def _doc_id(doc: Document) -> str:
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.
    k=60 is the standard constant from the original RRF paper.
    Higher k -> more uniform weighting across ranks.
    """
    if not result_lists:
        return []

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = _doc_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]
