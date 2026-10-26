"""
hybrid_recommender.py
---------------------
Ensemble hybrid recommender that blends Collaborative Filtering,
Matrix Factorization, and Content-Based Filtering via weighted score fusion.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Optional
import logging

from .collaborative_filter import CollaborativeFilter
from .matrix_factorization import MatrixFactorization, DEFAULT_N_FACTORS
from .content_based import ContentBasedFilter

logger = logging.getLogger(__name__)

# Blend weights (CF, MF, CB) chosen by train.py on a validation split
DEFAULT_WEIGHTS = (0.6, 0.4, 0.0)


class HybridRecommender:
    """
    Weighted hybrid recommender combining three recommendation strategies.

    The final score for each candidate item is:
        score = w_cf * cf_score + w_mf * mf_score + w_cb * cb_score

    where cf_score is the item-KNN neighbourhood score, mf_score the SVD
    predicted rating and cb_score the genre similarity to the user's taste,
    each min-max scaled to [0, 1]. Weights are normalised to sum to 1.0 internally.

    Parameters
    ----------
    w_cf : float   Weight for Collaborative Filtering (item-based).
    w_mf : float   Weight for Matrix Factorization (SVD).
    w_cb : float   Weight for Content-Based Filtering.
    n_factors : int  Number of SVD latent factors.
    cf_k : int       Number of CF neighbors.
    """

    def __init__(
        self,
        w_cf: float = DEFAULT_WEIGHTS[0],
        w_mf: float = DEFAULT_WEIGHTS[1],
        w_cb: float = DEFAULT_WEIGHTS[2],
        n_factors: int = DEFAULT_N_FACTORS,
        cf_k: int = 20,
    ):
        self.set_weights(w_cf, w_mf, w_cb)

        # Plain cosine on raw ratings: for ranking, "watched by the same people"
        # beats mean-centred rating agreement
        self.cf = CollaborativeFilter(mode="item", k=cf_k, mean_center=False)
        self.mf = MatrixFactorization(n_factors=n_factors)
        self.cb = ContentBasedFilter()

        self.user_item_matrix: csr_matrix = None
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self.item_id_map: dict = {}
        self.reverse_item_map: dict = {}
        self.user_id_map: dict = {}
        self._fitted = False

    def set_weights(self, w_cf: float, w_mf: float, w_cb: float) -> "HybridRecommender":
        """Set the blend weights, normalised to sum to 1.0."""
        total = w_cf + w_mf + w_cb
        self.w_cf = w_cf / total
        self.w_mf = w_mf / total
        self.w_cb = w_cb / total
        return self

    def fit(
        self,
        user_item_matrix: csr_matrix,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        user_id_map: dict,
        item_id_map: dict,
    ) -> "HybridRecommender":
        """
        Train all sub-models.

        Parameters
        ----------
        user_item_matrix : csr_matrix  Shape (n_users, n_items).
        movies : pd.DataFrame          Must have 'movieId' and 'genres'.
        ratings : pd.DataFrame         Full ratings df with userId/movieId/rating.
        user_id_map : dict             Maps raw userId → internal index.
        item_id_map : dict             Maps raw movieId → internal index.
        """
        logger.info("=== Training Hybrid Recommender ===")
        self.user_item_matrix = user_item_matrix
        self.movies = movies
        self.ratings = ratings
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.reverse_item_map = {v: k for k, v in item_id_map.items()}

        logger.info("[1/3] Fitting Collaborative Filter...")
        self.cf.fit(user_item_matrix)

        logger.info("[2/3] Fitting Matrix Factorization...")
        self.mf.fit(user_item_matrix)

        logger.info("[3/3] Fitting Content-Based Filter...")
        self.cb.fit(movies)
        # Internal item index of each movie in the CB profiles; -1 for movies
        # dropped in preprocessing, which have no index
        self._cb_item_index = np.array([item_id_map.get(mid, -1) for mid in self.cb.movie_ids])

        self._fitted = True
        logger.info("=== Training complete ===")
        return self

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]; constant scores all become 0.5."""
        if scores.size == 0:
            return scores
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return np.full(scores.shape, 0.5)
        return (scores - mn) / (mx - mn)

    def _cb_item_scores(self, user_id: int) -> np.ndarray:
        """Content-based score for every item, in internal item-index order."""
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        by_movie = self.cb.score_movies(user_ratings)
        known = self._cb_item_index >= 0
        scores = np.zeros(self.user_item_matrix.shape[1])
        scores[self._cb_item_index[known]] = by_movie[known]
        return scores

    def component_scores(
        self, user_id: int, exclude_seen: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Score every candidate item with each sub-model, min-max scaled to [0, 1].

        Parameters
        ----------
        user_id : int        Raw (original) userId.
        exclude_seen : bool  Leave out items the user has already rated.

        Returns
        -------
        (candidates, cf, mf, cb): candidate item indices and three aligned score arrays.
        """
        assert self._fitted, "Call .fit() before scoring"

        user_idx = self.user_id_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"userId {user_id} not found in training data.")

        n_items = self.user_item_matrix.shape[1]
        keep = np.ones(n_items, dtype=bool)
        if exclude_seen:
            keep[self.user_item_matrix.getrow(user_idx).nonzero()[1]] = False
        candidates = np.flatnonzero(keep)

        cf_scores = self.cf.neighborhood_scores(user_idx)

        user_vec = self.mf.user_factors[user_idx]
        mf_scores = np.clip(
            self.mf.item_factors @ user_vec + self.mf.user_means[user_idx], 1.0, 5.0
        )

        cb_scores = self._cb_item_scores(user_id)

        return (
            candidates,
            self._normalize_scores(cf_scores[candidates]),
            self._normalize_scores(mf_scores[candidates]),
            self._normalize_scores(cb_scores[candidates]),
        )

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Parameters
        ----------
        user_id : int   Raw (original) userId.
        n : int         Number of items to return.
        exclude_seen : bool  Exclude already-rated items.

        Returns
        -------
        pd.DataFrame with columns: rank, movieId, title, genres, predicted_score.
        """
        candidates, cf_norm, mf_norm, cb_norm = self.component_scores(user_id, exclude_seen)

        # --- Weighted fusion ---
        fused = self.w_cf * cf_norm + self.w_mf * mf_norm + self.w_cb * cb_norm
        order = np.argsort(-fused, kind="stable")[:n]
        top = [(int(candidates[j]), float(fused[j])) for j in order]

        # Map back to movieIds and enrich with metadata
        rows = []
        for rank, (item_idx, score) in enumerate(top, 1):
            movie_id = self.reverse_item_map[item_idx]
            movie_row = self.movies[self.movies["movieId"] == movie_id]
            title = movie_row.iloc[0]["title"] if not movie_row.empty else f"Movie {movie_id}"
            genres = movie_row.iloc[0]["genres"] if not movie_row.empty else "Unknown"
            rows.append({
                "rank": rank,
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "predicted_score": round(score, 4),
            })

        return pd.DataFrame(rows)

    def explain(self, user_id: int, movie_id: int) -> Dict:
        """
        Return per-model score breakdown for a (user, item) pair.

        Each score is the sub-model's raw score before scaling: the CF
        neighbourhood score, the MF predicted rating and the CB genre similarity.
        Useful for interpretability and debugging.
        """
        assert self._fitted
        user_idx = self.user_id_map.get(user_id)
        item_idx = self.item_id_map.get(movie_id)
        if user_idx is None or item_idx is None:
            return {"error": "user_id or movie_id not found"}

        cf_score = float(self.cf.neighborhood_scores(user_idx)[item_idx])
        mf_score = self.mf.predict(user_idx, item_idx)
        cb_score = float(self._cb_item_scores(user_id)[item_idx])

        return {
            "userId": user_id,
            "movieId": movie_id,
            "cf_score": round(cf_score, 3),
            "mf_score": round(mf_score, 3),
            "cb_score": round(cb_score, 3),
            "weights": {"cf": self.w_cf, "mf": self.w_mf, "cb": self.w_cb},
        }
