"""
hybrid_recommender.py
---------------------
Ensemble hybrid recommender that blends Collaborative Filtering,
Matrix Factorization, and Content-Based Filtering via weighted score fusion.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional
import logging

from .collaborative_filter import CollaborativeFilter
from .matrix_factorization import MatrixFactorization
from .content_based import ContentBasedFilter

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Weighted hybrid recommender combining three recommendation strategies.

    The final score for each candidate item is:
        score = w_cf * cf_score + w_mf * mf_score + w_cb * cb_score

    Weights are normalised to sum to 1.0 internally.

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
        w_cf: float = 0.3,
        w_mf: float = 0.5,
        w_cb: float = 0.2,
        n_factors: int = 50,
        cf_k: int = 20,
    ):
        total = w_cf + w_mf + w_cb
        self.w_cf = w_cf / total
        self.w_mf = w_mf / total
        self.w_cb = w_cb / total

        self.cf = CollaborativeFilter(mode="item", k=cf_k, mean_center=True)
        self.mf = MatrixFactorization(n_factors=n_factors)
        self.cb = ContentBasedFilter()

        self.user_item_matrix: csr_matrix = None
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self.item_id_map: dict = {}
        self.reverse_item_map: dict = {}
        self.user_id_map: dict = {}
        self._fitted = False

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

        self._fitted = True
        logger.info("=== Training complete ===")
        return self

    def _normalize_scores(self, items_scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """Min-max normalize a list of (item_idx, score) to [0, 1]."""
        if not items_scores:
            return {}
        scores = np.array([s for _, s in items_scores])
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-9:
            return {i: 0.5 for i, _ in items_scores}
        return {i: float((s - mn) / (mx - mn)) for i, s in items_scores}

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
        assert self._fitted, "Call .fit() before .recommend()"

        user_idx = self.user_id_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"userId {user_id} not found in training data.")

        n_items = self.user_item_matrix.shape[1]
        seen = set(self.user_item_matrix.getrow(user_idx).nonzero()[1])
        candidates = [i for i in range(n_items) if not (exclude_seen and i in seen)]

        # --- CF scores (item-based) ---
        cf_raw = [(i, self.cf.predict_user_item(user_idx, i)) for i in candidates]
        cf_norm = self._normalize_scores(cf_raw)

        # --- MF scores ---
        user_vec = self.mf.user_factors[user_idx]
        mf_scores = self.mf.item_factors @ user_vec + self.mf.user_means[user_idx]
        mf_scores = np.clip(mf_scores, 1.0, 5.0)
        mf_raw = [(i, float(mf_scores[i])) for i in candidates]
        mf_norm = self._normalize_scores(mf_raw)

        # --- CB scores ---
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        cb_recs = self.cb.recommend(user_ratings, n=len(candidates) + 50, exclude_seen=False)
        # CB returns (movieId, score) — convert to item_idx space
        cb_by_movie = {self.item_id_map.get(mid): sc for mid, sc in cb_recs
                       if self.item_id_map.get(mid) is not None}
        cb_raw = [(i, cb_by_movie.get(i, 0.0)) for i in candidates]
        cb_norm = self._normalize_scores(cb_raw)

        # --- Weighted fusion ---
        fused = []
        for i in candidates:
            score = (
                self.w_cf * cf_norm.get(i, 0.0) +
                self.w_mf * mf_norm.get(i, 0.0) +
                self.w_cb * cb_norm.get(i, 0.0)
            )
            fused.append((i, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:n]

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

        Useful for interpretability and debugging.
        """
        assert self._fitted
        user_idx = self.user_id_map.get(user_id)
        item_idx = self.item_id_map.get(movie_id)
        if user_idx is None or item_idx is None:
            return {"error": "user_id or movie_id not found"}

        cf_score = self.cf.predict_user_item(user_idx, item_idx)
        mf_score = self.mf.predict(user_idx, item_idx)
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        cb_recs = dict(self.cb.recommend(user_ratings, n=500, exclude_seen=False))
        cb_score = cb_recs.get(movie_id, 0.0)

        return {
            "userId": user_id,
            "movieId": movie_id,
            "cf_score": round(cf_score, 3),
            "mf_score": round(mf_score, 3),
            "cb_score": round(cb_score, 3),
            "weights": {"cf": self.w_cf, "mf": self.w_mf, "cb": self.w_cb},
        }
