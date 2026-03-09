"""
collaborative_filter.py
------------------------
User-based and Item-based Collaborative Filtering using cosine similarity
on the sparse user-item ratings matrix.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """
    Memory-based Collaborative Filtering.

    Supports both user-based and item-based CF with optional mean-centering
    to reduce user bias.

    Parameters
    ----------
    mode : str
        'user' for user-based CF, 'item' for item-based CF.
    k : int
        Number of nearest neighbors to consider.
    mean_center : bool
        Whether to subtract per-user mean rating before computing similarity.
    """

    def __init__(self, mode: str = "item", k: int = 20, mean_center: bool = True):
        assert mode in ("user", "item"), "mode must be 'user' or 'item'"
        self.mode = mode
        self.k = k
        self.mean_center = mean_center
        self.similarity_matrix: np.ndarray = None
        self.user_item_matrix: csr_matrix = None
        self.user_means: np.ndarray = None

    def fit(self, user_item_matrix: csr_matrix) -> "CollaborativeFilter":
        """
        Compute similarity matrix from the user-item ratings matrix.

        Parameters
        ----------
        user_item_matrix : csr_matrix
            Shape (n_users, n_items) with explicit zeros for unrated items.
        """
        self.user_item_matrix = user_item_matrix
        matrix = user_item_matrix.toarray().astype(float)

        if self.mean_center:
            # Replace 0 (unrated) with NaN for proper mean calculation
            rated = matrix.copy()
            rated[rated == 0] = np.nan
            self.user_means = np.nanmean(rated, axis=1, keepdims=True)
            self.user_means = np.nan_to_num(self.user_means)
            # Center only rated entries
            mask = matrix != 0
            matrix[mask] -= np.broadcast_to(self.user_means, matrix.shape)[mask]

        if self.mode == "user":
            logger.info("Computing user-user cosine similarity...")
            self.similarity_matrix = cosine_similarity(matrix)
        else:
            logger.info("Computing item-item cosine similarity...")
            self.similarity_matrix = cosine_similarity(matrix.T)

        logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        return self

    def predict_user_item(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for a (user_idx, item_idx) pair."""
        matrix = self.user_item_matrix.toarray().astype(float)

        if self.mode == "item":
            item_sims = self.similarity_matrix[item_idx]
            # Find k most similar items that this user has rated
            user_ratings = matrix[user_idx]
            rated_mask = user_ratings != 0
            rated_mask[item_idx] = False

            if rated_mask.sum() == 0:
                return self.user_means[user_idx, 0] if self.mean_center else 3.0

            sims = item_sims[rated_mask]
            ratings = user_ratings[rated_mask]

            # Top-k neighbors
            top_k = min(self.k, len(sims))
            top_idx = np.argsort(np.abs(sims))[-top_k:]
            sims_k = sims[top_idx]
            ratings_k = ratings[top_idx]

            denom = np.sum(np.abs(sims_k))
            if denom < 1e-9:
                return self.user_means[user_idx, 0] if self.mean_center else 3.0

            pred = np.dot(sims_k, ratings_k) / denom
            if self.mean_center:
                pred += self.user_means[user_idx, 0]
            return float(np.clip(pred, 1.0, 5.0))

        else:  # user-based
            user_sims = self.similarity_matrix[user_idx]
            item_ratings = matrix[:, item_idx]
            rated_mask = item_ratings != 0
            rated_mask[user_idx] = False

            if rated_mask.sum() == 0:
                return self.user_means[user_idx, 0] if self.mean_center else 3.0

            sims = user_sims[rated_mask]
            ratings = item_ratings[rated_mask]

            top_k = min(self.k, len(sims))
            top_idx = np.argsort(np.abs(sims))[-top_k:]
            sims_k = sims[top_idx]
            ratings_k = ratings[top_idx]

            if self.mean_center:
                # Subtract neighbor means
                neighbor_means = self.user_means[np.where(rated_mask)[0][top_idx], 0]
                ratings_k = ratings_k - neighbor_means

            denom = np.sum(np.abs(sims_k))
            if denom < 1e-9:
                return self.user_means[user_idx, 0] if self.mean_center else 3.0

            pred = np.dot(sims_k, ratings_k) / denom
            if self.mean_center:
                pred += self.user_means[user_idx, 0]
            return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self, user_idx: int, n: int = 10, exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-N items for a user.

        Parameters
        ----------
        user_idx : int
            Internal (mapped) user index.
        n : int
            Number of recommendations.
        exclude_seen : bool
            If True, do not recommend already-rated items.

        Returns
        -------
        List of (item_idx, predicted_score) tuples sorted descending.
        """
        matrix = self.user_item_matrix.toarray().astype(float)
        user_ratings = matrix[user_idx]
        n_items = matrix.shape[1]

        candidates = []
        for item_idx in range(n_items):
            if exclude_seen and user_ratings[item_idx] != 0:
                continue
            score = self.predict_user_item(user_idx, item_idx)
            candidates.append((item_idx, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def get_similar_items(self, item_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """Return n most similar items to the given item."""
        assert self.mode == "item", "get_similar_items requires mode='item'"
        sims = self.similarity_matrix[item_idx]
        top_idx = np.argsort(sims)[::-1][1:n + 1]
        return [(int(i), float(sims[i])) for i in top_idx]
