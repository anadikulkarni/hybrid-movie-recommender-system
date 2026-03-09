"""
matrix_factorization.py
------------------------
Latent-factor model via Truncated SVD (similar to Simon Funk SVD).
Decomposes the user-item matrix into user and item latent factor matrices
and uses them for rating prediction and recommendation.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from typing import List, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MatrixFactorization:
    """
    SVD-based Matrix Factorization recommender.

    Uses scipy's truncated SVD on the mean-centered user-item matrix
    to learn dense latent factor representations for users and items.

    Parameters
    ----------
    n_factors : int
        Number of latent factors (dimensionality of embeddings).
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.user_factors: np.ndarray = None   # shape (n_users, n_factors)
        self.item_factors: np.ndarray = None   # shape (n_items, n_factors)
        self.user_means: np.ndarray = None
        self.global_mean: float = 0.0
        self._fitted = False

    def fit(self, user_item_matrix: csr_matrix) -> "MatrixFactorization":
        """
        Decompose the user-item matrix using truncated SVD.

        Parameters
        ----------
        user_item_matrix : csr_matrix
            Shape (n_users, n_items).
        """
        logger.info(f"Fitting SVD with {self.n_factors} factors...")
        matrix = user_item_matrix.toarray().astype(float)

        # Mean-center per user to reduce rating scale bias
        rated_mask = matrix != 0
        self.global_mean = matrix[rated_mask].mean()

        self.user_means = np.zeros(matrix.shape[0])
        for u in range(matrix.shape[0]):
            rated = matrix[u][rated_mask[u]]
            self.user_means[u] = rated.mean() if len(rated) > 0 else self.global_mean

        centered = matrix.copy()
        for u in range(matrix.shape[0]):
            centered[u][rated_mask[u]] -= self.user_means[u]

        # Truncated SVD: matrix ≈ U * diag(S) * Vt
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, S, Vt = svds(centered, k=k)

        # Absorb singular values into user factors
        self.user_factors = U * S[np.newaxis, :]   # (n_users, k)
        self.item_factors = Vt.T                    # (n_items, k)

        self._fitted = True
        logger.info(
            f"SVD complete. User factors: {self.user_factors.shape}, "
            f"Item factors: {self.item_factors.shape}"
        )
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict the rating for a (user_idx, item_idx) pair."""
        assert self._fitted, "Model must be fit before prediction."
        score = (
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            + self.user_means[user_idx]
        )
        return float(np.clip(score, 1.0, 5.0))

    def predict_batch(
        self, user_indices: np.ndarray, item_indices: np.ndarray
    ) -> np.ndarray:
        """Vectorized batch prediction for arrays of (user, item) pairs."""
        assert self._fitted
        scores = (
            np.einsum("ij,ij->i",
                      self.user_factors[user_indices],
                      self.item_factors[item_indices])
            + self.user_means[user_indices]
        )
        return np.clip(scores, 1.0, 5.0)

    def recommend(
        self,
        user_idx: int,
        user_item_matrix: csr_matrix,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-N items for a user using dot-product scoring.

        Parameters
        ----------
        user_idx : int
            Internal user index.
        user_item_matrix : csr_matrix
            Original matrix (used to identify already-seen items).
        n : int
            Number of recommendations.
        exclude_seen : bool
            Exclude items the user has already rated.

        Returns
        -------
        List of (item_idx, predicted_score) tuples, sorted descending.
        """
        assert self._fitted
        user_vec = self.user_factors[user_idx]  # (k,)
        scores = self.item_factors @ user_vec + self.user_means[user_idx]  # (n_items,)
        scores = np.clip(scores, 1.0, 5.0)

        if exclude_seen:
            seen = user_item_matrix.getrow(user_idx).nonzero()[1]
            scores[seen] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]

    def get_similar_items(self, item_idx: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find n most similar items using cosine similarity in latent space.
        """
        assert self._fitted
        item_vec = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        item_norm = np.linalg.norm(item_vec)
        if item_norm < 1e-9 or np.any(norms < 1e-9):
            return []
        sims = (self.item_factors @ item_vec) / (norms * item_norm + 1e-9)
        top_idx = np.argsort(sims)[::-1][1:n + 1]
        return [(int(i), float(sims[i])) for i in top_idx]

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Return the latent factor vector for a user."""
        return self.user_factors[user_idx]

    def save(self, path: str):
        """Persist model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MatrixFactorization":
        """Load a persisted model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
