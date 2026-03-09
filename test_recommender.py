"""
tests/test_recommender.py
--------------------------
Unit and integration tests for CineMatch.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.collaborative_filter import CollaborativeFilter
from src.matrix_factorization import MatrixFactorization
from src.content_based import ContentBasedFilter
from src.hybrid_recommender import HybridRecommender
from src.evaluator import (
    rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, Evaluator
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_matrix():
    """5x6 user-item sparse matrix."""
    data = np.array([
        [5, 4, 0, 0, 1, 0],
        [0, 0, 4, 0, 4, 3],
        [3, 0, 0, 2, 0, 4],
        [0, 3, 4, 0, 0, 2],
        [1, 0, 0, 5, 0, 0],
    ], dtype=float)
    return csr_matrix(data)


@pytest.fixture
def small_ratings():
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        "movieId": [10, 20, 30, 10, 40, 20, 50, 30, 60, 10],
        "rating": [5.0, 4.0, 3.0, 2.0, 4.0, 5.0, 3.0, 4.0, 2.0, 1.0],
        "timestamp": [1000] * 10,
    })


@pytest.fixture
def small_movies():
    return pd.DataFrame({
        "movieId": [10, 20, 30, 40, 50, 60],
        "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F"],
        "genres": ["Action|Drama", "Comedy", "Romance|Drama", "Action", "Sci-Fi", "Comedy|Romance"],
    })


# ─── Metric Tests ─────────────────────────────────────────────────────────────

def test_rmse_perfect():
    y = np.array([4.0, 3.0, 5.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([4.0, 3.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_known():
    y_true = np.array([4.0, 3.0])
    y_pred = np.array([3.0, 4.0])
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_precision_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6}
    assert precision_at_k(recs, relevant, k=5) == pytest.approx(2 / 5)


def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    relevant = {1, 3, 6}
    assert recall_at_k(recs, relevant, k=5) == pytest.approx(2 / 3)


def test_ndcg_at_k_perfect():
    relevant = {1, 2, 3}
    recs = [1, 2, 3, 4, 5]
    # Perfect: all relevant items at top 3
    score = ndcg_at_k(recs, relevant, k=3)
    assert score == pytest.approx(1.0)


def test_ndcg_at_k_empty():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


# ─── Collaborative Filter Tests ───────────────────────────────────────────────

class TestCollaborativeFilter:

    def test_fit_user_mode(self, small_matrix):
        cf = CollaborativeFilter(mode="user", k=2)
        cf.fit(small_matrix)
        assert cf.similarity_matrix.shape == (5, 5)

    def test_fit_item_mode(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2)
        cf.fit(small_matrix)
        assert cf.similarity_matrix.shape == (6, 6)

    def test_similarity_diagonal_is_one(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2, mean_center=False)
        cf.fit(small_matrix)
        diag = np.diag(cf.similarity_matrix)
        assert np.allclose(diag, 1.0, atol=0.01)

    def test_predict_returns_valid_rating(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2)
        cf.fit(small_matrix)
        pred = cf.predict_user_item(0, 2)
        assert 1.0 <= pred <= 5.0

    def test_recommend_returns_n_items(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2)
        cf.fit(small_matrix)
        recs = cf.recommend(0, n=3, exclude_seen=True)
        assert len(recs) <= 3

    def test_recommend_excludes_seen(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2)
        cf.fit(small_matrix)
        seen = set(small_matrix.getrow(0).nonzero()[1])
        recs = cf.recommend(0, n=10, exclude_seen=True)
        rec_ids = {i for i, _ in recs}
        assert seen.isdisjoint(rec_ids)


# ─── Matrix Factorization Tests ───────────────────────────────────────────────

class TestMatrixFactorization:

    def test_fit_shapes(self, small_matrix):
        mf = MatrixFactorization(n_factors=3)
        mf.fit(small_matrix)
        assert mf.user_factors.shape == (5, 3)
        assert mf.item_factors.shape == (6, 3)

    def test_predict_in_range(self, small_matrix):
        mf = MatrixFactorization(n_factors=3)
        mf.fit(small_matrix)
        for u in range(5):
            for i in range(6):
                pred = mf.predict(u, i)
                assert 1.0 <= pred <= 5.0, f"Out-of-range prediction: {pred}"

    def test_recommend_length(self, small_matrix):
        mf = MatrixFactorization(n_factors=3)
        mf.fit(small_matrix)
        recs = mf.recommend(0, small_matrix, n=5)
        assert len(recs) == 5

    def test_recommend_descending_scores(self, small_matrix):
        mf = MatrixFactorization(n_factors=3)
        mf.fit(small_matrix)
        recs = mf.recommend(0, small_matrix, n=5)
        scores = [s for _, s in recs]
        assert scores == sorted(scores, reverse=True)

    def test_similar_items(self, small_matrix):
        mf = MatrixFactorization(n_factors=3)
        mf.fit(small_matrix)
        sims = mf.get_similar_items(0, n=3)
        assert len(sims) == 3
        assert all(1 <= i < 6 for i, _ in sims)


# ─── Content-Based Filter Tests ───────────────────────────────────────────────

class TestContentBasedFilter:

    def test_fit_profile_shape(self, small_movies):
        cb = ContentBasedFilter()
        cb.fit(small_movies)
        assert cb.item_profiles.shape[0] == len(small_movies)

    def test_recommend_returns_list(self, small_movies, small_ratings):
        cb = ContentBasedFilter()
        cb.fit(small_movies)
        user_ratings = small_ratings[small_ratings["userId"] == 1]
        recs = cb.recommend(user_ratings, n=5)
        assert isinstance(recs, list)

    def test_similar_movies_excludes_self(self, small_movies):
        cb = ContentBasedFilter()
        cb.fit(small_movies)
        sims = cb.get_similar_movies(10, n=3)
        movie_ids = [mid for mid, _ in sims]
        assert 10 not in movie_ids


# ─── Hybrid Recommender Tests ─────────────────────────────────────────────────

class TestHybridRecommender:

    def test_fit_and_recommend(self, small_matrix, small_movies, small_ratings):
        user_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        item_id_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}

        h = HybridRecommender(w_cf=0.3, w_mf=0.5, w_cb=0.2, n_factors=2, cf_k=2)
        h.fit(small_matrix, small_movies, small_ratings, user_id_map, item_id_map)

        recs = h.recommend(1, n=3)
        assert isinstance(recs, pd.DataFrame)
        assert "title" in recs.columns
        assert "predicted_score" in recs.columns
        assert len(recs) <= 3

    def test_explain_returns_dict(self, small_matrix, small_movies, small_ratings):
        user_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        item_id_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}

        h = HybridRecommender(w_cf=0.3, w_mf=0.5, w_cb=0.2, n_factors=2, cf_k=2)
        h.fit(small_matrix, small_movies, small_ratings, user_id_map, item_id_map)
        explain = h.explain(1, 20)
        assert "cf_score" in explain
        assert "mf_score" in explain
        assert "cb_score" in explain


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluator:

    def test_coverage(self):
        ev = Evaluator(pd.DataFrame(), pd.DataFrame())
        all_recs = [[1, 2, 3], [2, 4, 5], [6, 7]]
        cov = ev.coverage(all_recs, catalog_size=10)
        assert cov == pytest.approx(0.7)
