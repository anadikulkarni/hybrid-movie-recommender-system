"""
evaluator.py
------------
Evaluation metrics for recommender systems:
  - RMSE and MAE (rating prediction quality)
  - Precision@K, Recall@K, F1@K (ranking quality)
  - NDCG@K (ranking quality with position discount)
  - Coverage (catalog coverage)
  - Novelty / Popularity Bias
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
import logging

logger = logging.getLogger(__name__)


# ─── Rating Prediction Metrics ───────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ─── Ranking Metrics ─────────────────────────────────────────────────────────

def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of relevant items retrieved in top-K."""
    top_k = recommended[:k]
    if not relevant:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Binary relevance: 1 if item is relevant, 0 otherwise.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: List[int], relevant: set, k: int) -> float:
    """Average Precision: mean P@i for each hit in top-K."""
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    if not relevant:
        return 0.0
    return score / min(len(relevant), k)


# ─── Evaluator Class ─────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates a recommender model on a held-out test set.

    Parameters
    ----------
    ratings_train : pd.DataFrame   Training ratings (userId, movieId, rating).
    ratings_test : pd.DataFrame    Test ratings (userId, movieId, rating).
    relevance_threshold : float    Minimum rating to consider an item relevant.
    """

    def __init__(
        self,
        ratings_train: pd.DataFrame,
        ratings_test: pd.DataFrame,
        relevance_threshold: float = 4.0,
    ):
        self.train = ratings_train
        self.test = ratings_test
        self.threshold = relevance_threshold

    def evaluate_rating_prediction(
        self,
        predict_fn: Callable[[int, int], float],
        user_id_map: dict,
        item_id_map: dict,
        sample_size: int = 5000,
    ) -> Dict[str, float]:
        """
        Compute RMSE and MAE on sampled test pairs.

        Parameters
        ----------
        predict_fn : callable  Takes (user_idx, item_idx) → predicted rating.
        user_id_map : dict     Maps raw userId → internal index.
        item_id_map : dict     Maps raw movieId → internal index.
        sample_size : int      Number of test ratings to evaluate.
        """
        sample = self.test.sample(min(sample_size, len(self.test)), random_state=42)
        y_true, y_pred = [], []

        for _, row in sample.iterrows():
            uid = user_id_map.get(row["userId"])
            iid = item_id_map.get(row["movieId"])
            if uid is None or iid is None:
                continue
            try:
                pred = predict_fn(uid, iid)
                y_true.append(row["rating"])
                y_pred.append(pred)
            except Exception:
                continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return {
            "RMSE": round(rmse(y_true, y_pred), 4),
            "MAE": round(mae(y_true, y_pred), 4),
            "n_evaluated": len(y_true),
        }

    def evaluate_ranking(
        self,
        recommend_fn: Callable[[int, int], list],
        user_id_map: dict,
        k: int = 10,
        n_users: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate ranking quality on a sample of test users.

        Parameters
        ----------
        recommend_fn : callable   Takes (userId, n) → list of movieIds.
        user_id_map : dict        Maps raw userId → internal index.
        k : int                   Cutoff rank.
        n_users : int             Number of users to evaluate.

        Returns
        -------
        Dict with mean Precision@K, Recall@K, F1@K, NDCG@K, MAP@K.
        """
        test_users = self.test["userId"].unique()
        sampled = np.random.choice(
            test_users, size=min(n_users, len(test_users)), replace=False
        )

        metrics = {"precision": [], "recall": [], "f1": [], "ndcg": [], "ap": []}

        for user_id in sampled:
            if user_id not in user_id_map:
                continue

            relevant = set(
                self.test[
                    (self.test["userId"] == user_id) &
                    (self.test["rating"] >= self.threshold)
                ]["movieId"].values
            )
            if not relevant:
                continue

            try:
                recs = recommend_fn(user_id, k * 3)  # over-fetch then cut
            except Exception:
                continue

            rec_ids = [r if isinstance(r, int) else r[0] for r in recs]

            metrics["precision"].append(precision_at_k(rec_ids, relevant, k))
            metrics["recall"].append(recall_at_k(rec_ids, relevant, k))
            metrics["f1"].append(f1_at_k(rec_ids, relevant, k))
            metrics["ndcg"].append(ndcg_at_k(rec_ids, relevant, k))
            metrics["ap"].append(average_precision(rec_ids, relevant, k))

        result = {f"Precision@{k}": round(np.mean(metrics["precision"]), 4),
                  f"Recall@{k}": round(np.mean(metrics["recall"]), 4),
                  f"F1@{k}": round(np.mean(metrics["f1"]), 4),
                  f"NDCG@{k}": round(np.mean(metrics["ndcg"]), 4),
                  f"MAP@{k}": round(np.mean(metrics["ap"]), 4),
                  "n_users_evaluated": len(metrics["precision"])}
        return result

    def coverage(self, all_recs: List[List[int]], catalog_size: int) -> float:
        """Catalog coverage: fraction of items ever recommended."""
        unique_recs = set(item for recs in all_recs for item in recs)
        return round(len(unique_recs) / catalog_size, 4)
