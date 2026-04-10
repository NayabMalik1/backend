import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from app.settings import SUPPORT_EMBEDDINGS_PATH


def load_support_embeddings() -> Dict[str, List[dict]]:
    if not os.path.isfile(SUPPORT_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Support embeddings file not found: {SUPPORT_EMBEDDINGS_PATH}"
        )

    with open(SUPPORT_EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).flatten()
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def build_prototype(embeddings: List[np.ndarray]) -> np.ndarray:
    arr = np.stack([l2_normalize(e) for e in embeddings], axis=0)
    proto = np.mean(arr, axis=0)
    return l2_normalize(proto)


def get_weighted_family_items(items: List[dict]) -> Tuple[List[np.ndarray], List[float]]:
    """
    support images ko زیادہ importance
    train images ko کم importance
    """
    embeddings = []
    weights = []

    for item in items:
        emb = np.array(item["embedding"], dtype=np.float32).flatten()
        emb = l2_normalize(emb)

        source = item.get("source", "support")

        if source == "support":
            weight = 1.0
        elif source == "train":
            weight = 0.65
        else:
            weight = 0.8

        embeddings.append(emb)
        weights.append(weight)

    return embeddings, weights


def weighted_topk_score(
    query_embedding: np.ndarray,
    embeddings: List[np.ndarray],
    weights: List[float],
    top_k: int = 5
) -> float:
    if not embeddings:
        return 0.0

    sims = []
    for emb, w in zip(embeddings, weights):
        sim = cosine_similarity(query_embedding, emb)
        sims.append(sim * w)

    sims = sorted(sims, reverse=True)
    k = min(top_k, len(sims))
    return float(np.mean(sims[:k]))


def weighted_best_score(
    query_embedding: np.ndarray,
    embeddings: List[np.ndarray],
    weights: List[float]
) -> float:
    if not embeddings:
        return 0.0

    best = -1.0
    for emb, w in zip(embeddings, weights):
        sim = cosine_similarity(query_embedding, emb) * w
        if sim > best:
            best = sim
    return float(best)


def compare_with_support(query_embedding: np.ndarray) -> Dict[str, float]:
    db = load_support_embeddings()
    family_scores = {}

    query_embedding = l2_normalize(query_embedding)

    for family, items in db.items():
        if not items:
            continue

        support_embeddings, source_weights = get_weighted_family_items(items)

        if not support_embeddings:
            continue

        # ----- prototype -----
        prototype = build_prototype(support_embeddings)
        prototype_score = cosine_similarity(query_embedding, prototype)

        # ----- local nearest evidence -----
        best_score = weighted_best_score(
            query_embedding=query_embedding,
            embeddings=support_embeddings,
            weights=source_weights,
        )

        top3_score = weighted_topk_score(
            query_embedding=query_embedding,
            embeddings=support_embeddings,
            weights=source_weights,
            top_k=3,
        )

        top5_score = weighted_topk_score(
            query_embedding=query_embedding,
            embeddings=support_embeddings,
            weights=source_weights,
            top_k=5,
        )

        # ----- fusion -----
        # prototype = class center
        # best/topk = local evidence
        final_score = (
            0.40 * prototype_score +
            0.25 * best_score +
            0.20 * top3_score +
            0.15 * top5_score
        )

        family_scores[family] = float(final_score)

    return family_scores