from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]
    indices: list[int]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_scores(chunk_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    # normalize chunks row-wise
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed_chunks = chunk_embeddings / norms

    # normalize query
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        return np.zeros(len(chunk_embeddings), dtype=np.float32)
    normed_query = query_embedding / q_norm

    return normed_chunks @ normed_query


def retrieve_top_k(chunks, chunk_embeddings, query_embedding, top_k=3) -> RetrievalResult:
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("chunks and chunk_embeddings length mismatch")

    top_k = min(top_k, len(chunks))
    scores = _compute_scores(chunk_embeddings, query_embedding)

    # sort descending
    ranked = np.argsort(scores)[::-1]
    top_idx = ranked[:top_k].tolist()

    return RetrievalResult(
        chunks=[chunks[i] for i in top_idx],
        scores=[float(scores[i]) for i in top_idx],
        indices=top_idx,
    )
