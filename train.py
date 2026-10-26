"""
train.py
--------
End-to-end training and evaluation pipeline for CineMatch.

  1. Split the ratings 80/20 into train and test, stratified by user.
  2. Carve a validation split out of train and use it to choose the number of
     SVD factors and the hybrid's blend weights. The test set is never used to tune.
  3. Refit on the full training set and evaluate every model on the test set,
     next to baselines: the global mean and a bias model for ratings,
     popularity for rankings.

Usage:
    python train.py                       # Tune, train all models, evaluate, save results
    python train.py --model mf            # Train only Matrix Factorization
    python train.py --n_factors 100       # Fix the SVD factors instead of tuning them
    python train.py --weights 0.3 0.5 0.2 # Fix the hybrid weights (CF, MF, CB)
    python train.py --data_dir data       # Specify data directory
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.baselines import BiasBaseline, PopularityRecommender
from src.collaborative_filter import CollaborativeFilter
from src.matrix_factorization import MatrixFactorization, DEFAULT_N_FACTORS
from src.content_based import ContentBasedFilter
from src.hybrid_recommender import HybridRecommender, DEFAULT_WEIGHTS
from src.evaluator import Evaluator, ndcg_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 4.0          # a test rating of 4+ counts as a good recommendation
FACTOR_OPTIONS = (5, 10, 20, 30, 50)
WEIGHT_STEP = 0.1                  # grid step for the hybrid weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train CineMatch recommender models.")
    parser.add_argument("--data_dir", default="data", help="Directory with CSV data files")
    parser.add_argument("--model", default="all", choices=["all", "cf", "mf", "cb", "hybrid"],
                        help="Which model(s) to train")
    parser.add_argument("--n_factors", type=int, default=None,
                        help="SVD latent factors (default: chosen on the validation split)")
    parser.add_argument("--weights", type=float, nargs=3, default=None, metavar=("CF", "MF", "CB"),
                        help="Hybrid blend weights (default: chosen on the validation split)")
    parser.add_argument("--cf_k", type=int, default=20, help="CF neighbor count")
    parser.add_argument("--test_size", type=float, default=0.2, help="Train/test split ratio")
    parser.add_argument("--val_size", type=float, default=0.125,
                        help="Share of the training ratings held out for tuning")
    parser.add_argument("--top_k", type=int, default=10, help="Recommendation cutoff")
    parser.add_argument("--output_dir", default="outputs", help="Where to save results")
    parser.add_argument("--save_model", action="store_true", help="Persist MF model to disk")
    return parser.parse_args()


def to_matrix(ratings: pd.DataFrame, loader: DataLoader, shape) -> csr_matrix:
    """Sparse user-item matrix of the given ratings, in the loader's index space."""
    row = ratings["userId"].map(loader.user_id_map)
    col = ratings["movieId"].map(loader.item_id_map)
    return csr_matrix((ratings["rating"].values, (row, col)), shape=shape)


def relevant_items(ratings: pd.DataFrame) -> dict:
    """movieIds each user rated at or above the relevance threshold."""
    good = ratings[ratings["rating"] >= RELEVANCE_THRESHOLD]
    return good.groupby("userId")["movieId"].apply(set).to_dict()


def choose_n_factors(fit_matrix, val_ratings, loader, k):
    """Pick the SVD factor count with the best validation NDCG@k."""
    relevant = relevant_items(val_ratings)
    scores = {}
    for n_factors in FACTOR_OPTIONS:
        mf = MatrixFactorization(n_factors=n_factors).fit(fit_matrix)
        ndcgs = []
        for user_id, rel in relevant.items():
            recs = mf.recommend(loader.user_id_map[user_id], fit_matrix, n=k)
            ndcgs.append(ndcg_at_k([loader.reverse_item_map[i] for i, _ in recs], rel, k))
        scores[n_factors] = round(float(np.mean(ndcgs)), 4)
        logger.info(f"  {n_factors:>3} factors: validation NDCG@{k} = {scores[n_factors]}")
    return max(scores, key=scores.get), scores


def weight_grid(step: float = WEIGHT_STEP):
    """Every (CF, MF, CB) weight triple on a grid that sums to 1, single models included."""
    n = round(1 / step)
    return [
        (round(a * step, 2), round(b * step, 2), round((n - a - b) * step, 2))
        for a in range(n + 1)
        for b in range(n + 1 - a)
    ]


def choose_weights(hybrid: HybridRecommender, val_ratings, loader, k):
    """Pick the blend weights with the best validation NDCG@k."""
    relevant = relevant_items(val_ratings)
    item_to_movie = np.array([loader.reverse_item_map[i] for i in range(len(loader.item_id_map))])

    # Each sub-model's scores depend only on the user, so compute them once
    cached = []
    for user_id, rel in relevant.items():
        candidates, cf, mf, cb = hybrid.component_scores(user_id)
        cached.append((item_to_movie[candidates], cf, mf, cb, rel))

    scores = {}
    for w_cf, w_mf, w_cb in weight_grid():
        ndcgs = []
        for movie_ids, cf, mf, cb, rel in cached:
            fused = w_cf * cf + w_mf * mf + w_cb * cb
            top = movie_ids[np.argsort(-fused, kind="stable")[:k]]
            ndcgs.append(ndcg_at_k(top.tolist(), rel, k))
        scores[(w_cf, w_mf, w_cb)] = float(np.mean(ndcgs))
    best = max(scores, key=scores.get)
    logger.info(
        f"  best weights CF/MF/CB = {best}: validation NDCG@{k} = {scores[best]:.4f} "
        f"(MF alone: {scores[(0.0, 1.0, 0.0)]:.4f})"
    )
    return best, scores


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 1. Load & preprocess data ─────────────────────────────────────────────
    loader = DataLoader(data_dir=args.data_dir)
    loader.load().preprocess(min_user_ratings=5, min_movie_ratings=5)
    summary = loader.summary()
    logger.info(f"Dataset summary: {json.dumps(summary, indent=2)}")

    user_item_matrix = loader.build_user_item_matrix()
    train_ratings, test_ratings = loader.train_test_split(test_size=args.test_size)

    # ── 2. Build train-only matrix (to avoid data leakage) ───────────────────
    train_matrix = to_matrix(train_ratings, loader, user_item_matrix.shape)

    evaluator = Evaluator(train_ratings, test_ratings, relevance_threshold=RELEVANCE_THRESHOLD)
    results = {}

    # ── 3. Tune on a validation split carved out of train ────────────────────
    n_factors = args.n_factors
    weights = tuple(args.weights) if args.weights else None
    tune_factors = n_factors is None and args.model in ("all", "mf", "hybrid")
    tune_weights = weights is None and args.model in ("all", "hybrid")

    if tune_factors or tune_weights:
        fit_ratings, val_ratings = train_test_split(
            train_ratings, test_size=args.val_size, random_state=42,
            stratify=train_ratings["userId"],
        )
        fit_matrix = to_matrix(fit_ratings, loader, user_item_matrix.shape)
        logger.info(f"Tuning on {len(val_ratings):,} validation ratings held out of train")
        tuning = {}

        if tune_factors:
            n_factors, factor_scores = choose_n_factors(fit_matrix, val_ratings, loader, args.top_k)
            tuning["n_factors"] = n_factors
            tuning[f"validation_NDCG@{args.top_k}_by_factors"] = {str(f): s for f, s in factor_scores.items()}

        if tune_weights:
            hybrid_val = HybridRecommender(n_factors=n_factors, cf_k=args.cf_k)
            hybrid_val.fit(fit_matrix, loader.movies, fit_ratings, loader.user_id_map, loader.item_id_map)
            weights, weight_scores = choose_weights(hybrid_val, val_ratings, loader, args.top_k)
            tuning["weights"] = {"cf": weights[0], "mf": weights[1], "cb": weights[2]}
            tuning[f"validation_NDCG@{args.top_k}"] = round(weight_scores[weights], 4)

        results["Tuning"] = tuning
    n_factors = n_factors or DEFAULT_N_FACTORS
    weights = weights or DEFAULT_WEIGHTS

    def ranking(recommend_fn):
        return evaluator.evaluate_ranking(
            recommend_fn=recommend_fn, user_id_map=loader.user_id_map, k=args.top_k, n_users=None,
        )

    def rating(predict_fn):
        return evaluator.evaluate_rating_prediction(
            predict_fn=predict_fn, user_id_map=loader.user_id_map, item_id_map=loader.item_id_map,
        )

    def to_movie_ids(recs):
        return [int(loader.reverse_item_map[i]) for i, _ in recs]

    # ── 4. Baselines ──────────────────────────────────────────────────────────
    global_mean = float(train_ratings["rating"].mean())
    results["GlobalMean"] = rating(lambda u, i: global_mean)
    logger.info(f"Global mean rating metrics: {results['GlobalMean']}")

    bias = BiasBaseline().fit(train_matrix)
    results["BiasBaseline"] = rating(bias.predict)
    logger.info(f"Bias baseline rating metrics: {results['BiasBaseline']}")

    popularity = PopularityRecommender().fit(train_matrix)
    results["Popularity"] = ranking(
        lambda user_id, n: to_movie_ids(
            popularity.recommend(loader.user_id_map[user_id], train_matrix, n=n)
        )
    )
    logger.info(f"Popularity ranking metrics: {results['Popularity']}")

    # ── 5. Train & evaluate models ────────────────────────────────────────────

    if args.model in ("all", "mf"):
        t0 = time.time()
        mf = MatrixFactorization(n_factors=n_factors)
        mf.fit(train_matrix)
        logger.info(f"MF training time: {time.time()-t0:.1f}s")

        mf_rating_metrics = rating(mf.predict)
        logger.info(f"MF Rating metrics: {mf_rating_metrics}")

        def mf_recommend_fn(user_id, n):
            uidx = loader.user_id_map.get(user_id)
            if uidx is None:
                return []
            return to_movie_ids(mf.recommend(uidx, train_matrix, n=n))

        mf_rank_metrics = ranking(mf_recommend_fn)
        logger.info(f"MF Ranking metrics: {mf_rank_metrics}")
        results["MatrixFactorization"] = {"n_factors": n_factors, **mf_rating_metrics, **mf_rank_metrics}

        if args.save_model:
            mf.save(str(output_dir / "mf_model.pkl"))

    if args.model in ("all", "cf"):
        # Rating prediction uses mean-centred (adjusted cosine) similarity;
        # ranking uses plain cosine neighbourhood scores, as in the hybrid
        t0 = time.time()
        cf = CollaborativeFilter(mode="item", k=args.cf_k)
        cf.fit(train_matrix)
        cf_rank = CollaborativeFilter(mode="item", k=args.cf_k, mean_center=False)
        cf_rank.fit(train_matrix)
        logger.info(f"CF training time: {time.time()-t0:.1f}s")

        cf_rating_metrics = rating(cf.predict_user_item)
        logger.info(f"CF Rating metrics: {cf_rating_metrics}")

        cf_rank_metrics = ranking(
            lambda user_id, n: to_movie_ids(cf_rank.recommend(loader.user_id_map[user_id], n=n))
        )
        logger.info(f"CF Ranking metrics: {cf_rank_metrics}")
        results["CollaborativeFilter"] = {**cf_rating_metrics, **cf_rank_metrics}

    if args.model in ("all", "cb"):
        t0 = time.time()
        cb = ContentBasedFilter()
        cb.fit(loader.movies)
        logger.info(f"CB training time: {time.time()-t0:.1f}s")

        # Content-based doesn't predict ratings, so it's scored on rankings only
        ratings_by_user = dict(tuple(train_ratings.groupby("userId")))
        cb_rank_metrics = ranking(
            lambda user_id, n: [mid for mid, _ in cb.recommend(ratings_by_user[user_id], n=n)]
        )
        logger.info(f"CB Ranking metrics: {cb_rank_metrics}")

        sample_user_id = int(train_ratings["userId"].iloc[0])
        cb_recs = cb.recommend(ratings_by_user[sample_user_id], n=10)
        rec_movies = [loader.get_movie_title(mid) for mid, _ in cb_recs[:5]]
        logger.info(f"CB sample recs for user {sample_user_id}: {rec_movies}")
        results["ContentBased"] = {**cb_rank_metrics, "sample_user": sample_user_id, "top5_recs": rec_movies}

    if args.model in ("all", "hybrid"):
        t0 = time.time()
        hybrid = HybridRecommender(
            w_cf=weights[0], w_mf=weights[1], w_cb=weights[2],
            n_factors=n_factors,
            cf_k=args.cf_k,
        )
        hybrid.fit(
            train_matrix,
            loader.movies,
            train_ratings,
            loader.user_id_map,
            loader.item_id_map,
        )
        logger.info(f"Hybrid training time: {time.time()-t0:.1f}s")

        def hybrid_recommend_fn(user_id, n):
            df = hybrid.recommend(user_id, n=n)
            return df["movieId"].tolist()

        hybrid_rank_metrics = ranking(hybrid_recommend_fn)
        logger.info(f"Hybrid Ranking metrics: {hybrid_rank_metrics}")
        results["Hybrid"] = {
            "weights": {"cf": weights[0], "mf": weights[1], "cb": weights[2]},
            **hybrid_rank_metrics,
        }

        # Sample recommendation output
        sample_uid = int(train_ratings["userId"].iloc[42])
        recs_df = hybrid.recommend(sample_uid, n=10)
        logger.info(f"\nSample recommendations for user {sample_uid}:\n{recs_df.to_string(index=False)}")
        recs_df.to_csv(output_dir / "sample_recommendations.csv", index=False)

    # ── 6. Save results ───────────────────────────────────────────────────────
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Pretty print summary table
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    main()
