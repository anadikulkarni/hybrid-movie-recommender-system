"""
tests/test_recommender.py
--------------------------
Unit and integration tests for CineMatch.
Run with: pytest tests/ -v
"""

import json
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.baselines import BiasBaseline, PopularityRecommender
from src.collaborative_filter import CollaborativeFilter
from src.matrix_factorization import MatrixFactorization
from src.content_based import ContentBasedFilter
from src.hybrid_recommender import HybridRecommender
from src.evaluator import (
    rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, Evaluator
)
from generate_data import generate, STAR_SHARES
from train import weight_grid


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

    def test_item_prediction_keeps_user_scale(self):
        # A user who gives everything 2 stars should be predicted 2, not 2 + their mean
        data = np.array([
            [2, 2, 2, 0, 0],
            [5, 4, 0, 1, 3],
            [1, 0, 2, 5, 4],
        ], dtype=float)
        cf = CollaborativeFilter(mode="item", k=2)
        cf.fit(csr_matrix(data))
        assert cf.predict_user_item(0, 3) == pytest.approx(2.0)
        assert cf.predict_user_item(0, 4) == pytest.approx(2.0)

    def test_neighborhood_scores_sum_similarities(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2, mean_center=False)
        cf.fit(small_matrix)
        rated = small_matrix.getrow(0).nonzero()[1]
        scores = cf.neighborhood_scores(0)
        unseen = 2
        assert scores[unseen] == pytest.approx(cf.similarity_matrix[unseen, rated].sum())
        # A rated item's own similarity of 1 is not counted
        own = rated[0]
        others = [j for j in rated if j != own]
        assert scores[own] == pytest.approx(cf.similarity_matrix[own, others].sum())

    def test_item_recommend_ranks_by_neighborhood(self, small_matrix):
        cf = CollaborativeFilter(mode="item", k=2, mean_center=False)
        cf.fit(small_matrix)
        scores = cf.neighborhood_scores(0)
        unseen = [i for i in range(6) if small_matrix[0, i] == 0]
        expected = sorted(unseen, key=lambda i: -scores[i])[:2]
        assert [i for i, _ in cf.recommend(0, n=2)] == expected


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

    def test_weights_are_normalised(self):
        h = HybridRecommender(w_cf=2, w_mf=1, w_cb=1)
        assert (h.w_cf, h.w_mf, h.w_cb) == pytest.approx((0.5, 0.25, 0.25))

    def test_component_scores_scaled_and_unseen(self, small_matrix, small_movies, small_ratings):
        user_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        item_id_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}

        h = HybridRecommender(n_factors=2, cf_k=2)
        h.fit(small_matrix, small_movies, small_ratings, user_id_map, item_id_map)
        candidates, cf, mf, cb = h.component_scores(1)
        seen = set(small_matrix.getrow(0).nonzero()[1])
        assert seen.isdisjoint(candidates.tolist())
        for scores in (cf, mf, cb):
            assert len(scores) == len(candidates)
            assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_mf_only_weights_follow_mf_ranking(self, small_matrix, small_movies, small_ratings):
        user_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        item_id_map = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}

        h = HybridRecommender(w_cf=0.0, w_mf=1.0, w_cb=0.0, n_factors=2, cf_k=2)
        h.fit(small_matrix, small_movies, small_ratings, user_id_map, item_id_map)
        hybrid_order = h.recommend(1, n=3)["movieId"].tolist()
        reverse = {v: k for k, v in item_id_map.items()}
        mf_order = [reverse[i] for i, _ in h.mf.recommend(0, small_matrix, n=3)]
        assert hybrid_order == mf_order


# ─── Baseline Tests ───────────────────────────────────────────────────────────

class TestBaselines:

    def test_bias_baseline_damped_means(self):
        data = np.array([
            [5, 3],
            [3, 0],
        ], dtype=float)
        b = BiasBaseline(damping=1.0).fit(csr_matrix(data))
        assert b.global_mean == pytest.approx(11 / 3)
        # user 0's residuals (+4/3, -2/3) sum to 2/3, damped over 2 + 1 ratings
        assert b.user_bias[0] == pytest.approx((2 / 3) / 3)
        assert b.predict(1, 1) == pytest.approx(b.global_mean + b.user_bias[1] + b.item_bias[1])

    def test_bias_baseline_clips_to_scale(self, small_matrix):
        b = BiasBaseline().fit(small_matrix)
        b.user_bias[0] = 3.0
        assert b.predict(0, 0) == pytest.approx(5.0)

    def test_popularity_excludes_seen_and_orders_by_count(self, small_matrix):
        pop = PopularityRecommender().fit(small_matrix)
        recs = pop.recommend(0, small_matrix, n=3)
        seen = set(small_matrix.getrow(0).nonzero()[1])
        assert seen.isdisjoint(i for i, _ in recs)
        counts = [c for _, c in recs]
        assert counts == sorted(counts, reverse=True)


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluator:

    def test_coverage(self):
        ev = Evaluator(pd.DataFrame(), pd.DataFrame())
        all_recs = [[1, 2, 3], [2, 4, 5], [6, 7]]
        cov = ev.coverage(all_recs, catalog_size=10)
        assert cov == pytest.approx(0.7)

    def test_ranking_is_repeatable_and_json_safe(self):
        test = pd.DataFrame({
            "userId": np.repeat(np.arange(1, 21), 2),
            "movieId": np.tile([1, 2], 20),
            "rating": np.tile([5.0, 2.0], 20),
        })
        ev = Evaluator(pd.DataFrame(), test)
        user_id_map = {u: u - 1 for u in range(1, 21)}
        recommend_fn = lambda user_id, n: [1, 3] if user_id % 2 else [3, 1]
        first = ev.evaluate_ranking(recommend_fn, user_id_map, k=2, n_users=10)
        second = ev.evaluate_ranking(recommend_fn, user_id_map, k=2, n_users=10)
        assert first == second
        json.dumps(first)

    def test_ranking_all_users(self):
        test = pd.DataFrame({"userId": [1, 2, 3], "movieId": [1, 1, 1], "rating": [5.0, 5.0, 5.0]})
        ev = Evaluator(pd.DataFrame(), test)
        result = ev.evaluate_ranking(lambda u, n: [1], {1: 0, 2: 1, 3: 2}, k=1, n_users=None)
        assert result["n_users_evaluated"] == 3
        assert result["Precision@1"] == pytest.approx(1.0)


# ─── Pipeline Tests ───────────────────────────────────────────────────────────

def test_weight_grid_sums_to_one():
    grid = weight_grid(0.1)
    assert len(grid) == 66
    assert all(sum(w) == pytest.approx(1.0) for w in grid)
    assert (1.0, 0.0, 0.0) in grid and (0.0, 1.0, 0.0) in grid and (0.0, 0.0, 1.0) in grid


# ─── Synthetic Data Tests ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    return generate(output_dir=None, seed=7, n_users=200, n_movies=300, n_ratings=8000)


class TestGenerateData:

    def test_shapes_and_unique_pairs(self, small_dataset):
        movies, users, ratings = small_dataset
        assert len(movies) == 300 and len(users) == 200 and len(ratings) == 8000
        assert not ratings.duplicated(["userId", "movieId"]).any()
        assert (movies["genres"].str.len() > 0).all()

    def test_star_shares_match_movielens(self, small_dataset):
        _, _, ratings = small_dataset
        shares = ratings["rating"].value_counts(normalize=True).sort_index().values
        assert shares == pytest.approx(STAR_SHARES, abs=0.01)

    def test_ratings_carry_learnable_signal(self, small_dataset):
        # The bias baseline must beat the global mean on held-out ratings
        _, _, ratings = small_dataset
        test = ratings.sample(frac=0.2, random_state=0)
        train = ratings.drop(test.index)
        users = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()))}
        items = {m: i for i, m in enumerate(sorted(ratings["movieId"].unique()))}
        matrix = csr_matrix(
            (train["rating"], (train["userId"].map(users), train["movieId"].map(items))),
            shape=(len(users), len(items)),
        )
        bias = BiasBaseline().fit(matrix)
        preds = [bias.predict(users[u], items[m]) for u, m in zip(test["userId"], test["movieId"])]
        y = test["rating"].values
        assert rmse(y, np.array(preds)) < rmse(y, np.full(len(y), train["rating"].mean())) - 0.05
