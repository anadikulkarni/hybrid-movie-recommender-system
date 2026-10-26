"""
baselines.py
------------
Simple reference models that every recommender here has to beat:

  - BiasBaseline: predicts global mean + user bias + item bias (rating prediction)
  - PopularityRecommender: recommends the most-rated unseen items (ranking)
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class BiasBaseline:
    """
    Baseline predictor: r̂(u, i) = μ + b_u + b_i.

    Biases are damped means of the residuals, so a user or item with only a
    few ratings is pulled towards zero instead of trusting a noisy average.

    Parameters
    ----------
    damping : float
        Pseudo-count added to each mean's denominator.
    """

    def __init__(self, damping: float = 10.0):
        self.damping = damping
        self.global_mean: float = 0.0
        self.user_bias: np.ndarray = None
        self.item_bias: np.ndarray = None

    def fit(self, user_item_matrix: csr_matrix) -> "BiasBaseline":
        """Learn the global mean, then user biases, then item biases."""
        matrix = user_item_matrix.toarray().astype(float)
        rated = matrix != 0
        self.global_mean = float(matrix[rated].mean())

        residual = np.where(rated, matrix - self.global_mean, 0.0)
        self.user_bias = residual.sum(axis=1) / (rated.sum(axis=1) + self.damping)

        residual = np.where(rated, residual - self.user_bias[:, None], 0.0)
        self.item_bias = residual.sum(axis=0) / (rated.sum(axis=0) + self.damping)

        logger.info(f"Bias baseline fitted (global mean {self.global_mean:.3f})")
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict the rating for a (user_idx, item_idx) pair."""
        score = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        return float(np.clip(score, 1.0, 5.0))


class PopularityRecommender:
    """Recommends the items with the most ratings in training, minus those already seen."""

    def __init__(self):
        self.item_counts: np.ndarray = None

    def fit(self, user_item_matrix: csr_matrix) -> "PopularityRecommender":
        self.item_counts = np.asarray((user_item_matrix != 0).sum(axis=0)).ravel().astype(float)
        return self

    def recommend(
        self, user_idx: int, user_item_matrix: csr_matrix, n: int = 10
    ) -> List[Tuple[int, float]]:
        """Return the n most-rated items the user hasn't rated, as (item_idx, count) tuples."""
        scores = self.item_counts.copy()
        scores[user_item_matrix.getrow(user_idx).nonzero()[1]] = -np.inf
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]
