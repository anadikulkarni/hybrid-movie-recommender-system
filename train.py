"""
train.py
--------
End-to-end training and evaluation pipeline for CineMatch.

Usage:
    python train.py                    # Train all models, evaluate, save results
    python train.py --model mf         # Train only Matrix Factorization
    python train.py --n_factors 100    # Use 100 SVD latent factors
    python train.py --data_dir data    # Specify data directory
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.collaborative_filter import CollaborativeFilter
from src.matrix_factorization import MatrixFactorization
from src.content_based import ContentBasedFilter
from src.hybrid_recommender import HybridRecommender
from src.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CineMatch recommender models.")
    parser.add_argument("--data_dir", default="data", help="Directory with CSV data files")
    parser.add_argument("--model", default="all", choices=["all", "cf", "mf", "cb", "hybrid"],
                        help="Which model(s) to train")
    parser.add_argument("--n_factors", type=int, default=50, help="SVD latent factors")
    parser.add_argument("--cf_k", type=int, default=20, help="CF neighbor count")
    parser.add_argument("--test_size", type=float, default=0.2, help="Train/test split ratio")
    parser.add_argument("--top_k", type=int, default=10, help="Recommendation cutoff")
    parser.add_argument("--output_dir", default="outputs", help="Where to save results")
    parser.add_argument("--save_model", action="store_true", help="Persist MF model to disk")
    return parser.parse_args()


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
    row = train_ratings["userId"].map(loader.user_id_map)
    col = train_ratings["movieId"].map(loader.item_id_map)
    from scipy.sparse import csr_matrix
    train_matrix = csr_matrix(
        (train_ratings["rating"].values, (row, col)),
        shape=user_item_matrix.shape,
    )

    evaluator = Evaluator(train_ratings, test_ratings, relevance_threshold=4.0)
    results = {}

    # ── 3. Train & evaluate models ────────────────────────────────────────────

    if args.model in ("all", "mf"):
        t0 = time.time()
        mf = MatrixFactorization(n_factors=args.n_factors)
        mf.fit(train_matrix)
        logger.info(f"MF training time: {time.time()-t0:.1f}s")

        mf_rating_metrics = evaluator.evaluate_rating_prediction(
            predict_fn=mf.predict,
            user_id_map=loader.user_id_map,
            item_id_map=loader.item_id_map,
        )
        logger.info(f"MF Rating metrics: {mf_rating_metrics}")

        def mf_recommend_fn(user_id, n):
            uidx = loader.user_id_map.get(user_id)
            if uidx is None:
                return []
            recs = mf.recommend(uidx, train_matrix, n=n)
            return [int(loader.reverse_item_map[i]) for i, _ in recs if i in loader.reverse_item_map]

        mf_rank_metrics = evaluator.evaluate_ranking(
            recommend_fn=mf_recommend_fn,
            user_id_map=loader.user_id_map,
            k=args.top_k,
        )
        logger.info(f"MF Ranking metrics: {mf_rank_metrics}")
        results["MatrixFactorization"] = {**mf_rating_metrics, **mf_rank_metrics}

        if args.save_model:
            mf.save(str(output_dir / "mf_model.pkl"))

    if args.model in ("all", "cf"):
        t0 = time.time()
        cf = CollaborativeFilter(mode="item", k=args.cf_k)
        cf.fit(train_matrix)
        logger.info(f"CF training time: {time.time()-t0:.1f}s")

        cf_rating_metrics = evaluator.evaluate_rating_prediction(
            predict_fn=cf.predict_user_item,
            user_id_map=loader.user_id_map,
            item_id_map=loader.item_id_map,
            sample_size=2000,
        )
        logger.info(f"CF Rating metrics: {cf_rating_metrics}")
        results["CollaborativeFilter"] = cf_rating_metrics

    if args.model in ("all", "cb"):
        t0 = time.time()
        cb = ContentBasedFilter()
        cb.fit(loader.movies)
        logger.info(f"CB training time: {time.time()-t0:.1f}s")

        # Content-based doesn't predict ratings — show sample recs instead
        sample_user_id = train_ratings["userId"].iloc[0]
        sample_ratings = train_ratings[train_ratings["userId"] == sample_user_id]
        cb_recs = cb.recommend(sample_ratings, n=10)
        rec_movies = [loader.get_movie_title(mid) for mid, _ in cb_recs[:5]]
        logger.info(f"CB sample recs for user {sample_user_id}: {rec_movies}")
        results["ContentBased"] = {"sample_user": sample_user_id, "top5_recs": rec_movies}

    if args.model in ("all", "hybrid"):
        t0 = time.time()
        hybrid = HybridRecommender(
            w_cf=0.3, w_mf=0.5, w_cb=0.2,
            n_factors=args.n_factors,
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

        hybrid_rank_metrics = evaluator.evaluate_ranking(
            recommend_fn=hybrid_recommend_fn,
            user_id_map=loader.user_id_map,
            k=args.top_k,
            n_users=50,
        )
        logger.info(f"Hybrid Ranking metrics: {hybrid_rank_metrics}")
        results["Hybrid"] = hybrid_rank_metrics

        # Sample recommendation output
        sample_uid = train_ratings["userId"].iloc[42]
        recs_df = hybrid.recommend(sample_uid, n=10)
        logger.info(f"\nSample recommendations for user {sample_uid}:\n{recs_df.to_string(index=False)}")
        recs_df.to_csv(output_dir / "sample_recommendations.csv", index=False)

    # ── 4. Save results ───────────────────────────────────────────────────────
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
