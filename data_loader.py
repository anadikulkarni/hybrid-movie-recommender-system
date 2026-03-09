"""
data_loader.py
--------------
Handles loading, validating, and preprocessing the MovieLens-style dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess ratings, movies, and users data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.ratings: pd.DataFrame = None
        self.movies: pd.DataFrame = None
        self.users: pd.DataFrame = None
        self.user_item_matrix: csr_matrix = None
        self.user_id_map: dict = {}
        self.item_id_map: dict = {}
        self.reverse_user_map: dict = {}
        self.reverse_item_map: dict = {}

    def load(self) -> "DataLoader":
        """Load all datasets from disk."""
        logger.info("Loading datasets...")
        self.ratings = pd.read_csv(self.data_dir / "ratings.csv")
        self.movies = pd.read_csv(self.data_dir / "movies.csv")
        self.users = pd.read_csv(self.data_dir / "users.csv")
        self._validate()
        logger.info(
            f"Loaded {len(self.ratings):,} ratings | "
            f"{len(self.movies):,} movies | "
            f"{len(self.users):,} users"
        )
        return self

    def _validate(self):
        """Basic sanity checks on loaded data."""
        assert "userId" in self.ratings.columns, "ratings must have userId"
        assert "movieId" in self.ratings.columns, "ratings must have movieId"
        assert "rating" in self.ratings.columns, "ratings must have rating"
        assert self.ratings["rating"].between(0.5, 5.0).all(), "Ratings out of expected range"
        logger.info("Data validation passed.")

    def preprocess(self, min_user_ratings: int = 5, min_movie_ratings: int = 5) -> "DataLoader":
        """
        Filter cold-start users/items and build contiguous index maps.

        Parameters
        ----------
        min_user_ratings : int
            Drop users with fewer than this many ratings.
        min_movie_ratings : int
            Drop movies with fewer than this many ratings.
        """
        logger.info("Preprocessing: filtering sparse users and movies...")
        user_counts = self.ratings["userId"].value_counts()
        movie_counts = self.ratings["movieId"].value_counts()

        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index

        self.ratings = self.ratings[
            self.ratings["userId"].isin(valid_users) &
            self.ratings["movieId"].isin(valid_movies)
        ].reset_index(drop=True)

        # Build contiguous integer maps for matrix construction
        users = sorted(self.ratings["userId"].unique())
        items = sorted(self.ratings["movieId"].unique())
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.item_id_map = {m: i for i, m in enumerate(items)}
        self.reverse_user_map = {i: u for u, i in self.user_id_map.items()}
        self.reverse_item_map = {i: m for m, i in self.item_id_map.items()}

        logger.info(
            f"After filtering: {len(self.ratings):,} ratings | "
            f"{len(users)} users | {len(items)} movies"
        )
        return self

    def build_user_item_matrix(self) -> csr_matrix:
        """Construct a sparse user-item ratings matrix."""
        row = self.ratings["userId"].map(self.user_id_map)
        col = self.ratings["movieId"].map(self.item_id_map)
        data = self.ratings["rating"].values

        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)

        self.user_item_matrix = csr_matrix(
            (data, (row, col)), shape=(n_users, n_items)
        )
        logger.info(
            f"Built user-item matrix: {n_users} x {n_items} | "
            f"Density: {self.user_item_matrix.nnz / (n_users * n_items):.4%}"
        )
        return self.user_item_matrix

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split ratings into train/test sets stratified by user."""
        train, test = train_test_split(
            self.ratings, test_size=test_size, random_state=random_state,
            stratify=self.ratings["userId"]
        )
        logger.info(f"Train: {len(train):,} | Test: {len(test):,}")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def get_movie_title(self, movie_id: int) -> str:
        """Return movie title for a given movieId."""
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            return f"Movie {movie_id}"
        return row.iloc[0]["title"]

    def summary(self) -> dict:
        """Return a dict of key dataset statistics."""
        return {
            "n_ratings": len(self.ratings),
            "n_users": self.ratings["userId"].nunique(),
            "n_movies": self.ratings["movieId"].nunique(),
            "avg_rating": round(self.ratings["rating"].mean(), 3),
            "rating_std": round(self.ratings["rating"].std(), 3),
            "sparsity": 1 - (len(self.ratings) /
                             (self.ratings["userId"].nunique() * self.ratings["movieId"].nunique())),
        }
