import hashlib
import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity


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


def apply_mmr(
    query: str,
    candidates: list[Document],
    embedding_model,
    k: int = 8,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """
    Maximum Marginal Relevance diversification over a candidate list.
    lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if not candidates:
        return []

    k = min(k, len(candidates))

    query_emb = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    doc_embs = np.array(embedding_model.embed_documents([d.page_content for d in candidates]))

    relevance_scores = cosine_similarity(query_emb, doc_embs)[0]

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < k and remaining:
        if not selected_indices:
            best = max(remaining, key=lambda i: relevance_scores[i])
        else:
            selected_embs = doc_embs[selected_indices]
            mmr_scores: dict[int, float] = {}
            for i in remaining:
                rel = relevance_scores[i]
                redundancy = float(
                    cosine_similarity(doc_embs[i].reshape(1, -1), selected_embs).max()
                )
                mmr_scores[i] = lambda_mult * rel - (1 - lambda_mult) * redundancy
            best = max(remaining, key=lambda i: mmr_scores[i])

        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]
