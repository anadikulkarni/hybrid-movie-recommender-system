"""
content_based.py
----------------
Content-Based Filtering using TF-IDF on movie genre tags.
Builds a movie profile matrix and recommends based on cosine similarity
to a user's taste profile derived from their rating history.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """
    TF-IDF + Cosine Similarity content-based recommender.

    Builds a genre-based item profile for each movie, then computes
    a weighted user profile from their historical ratings. New recommendations
    are scored by cosine similarity to the user's profile vector.

    Parameters
    ----------
    min_df : int
        Minimum genre frequency for TF-IDF vocabulary.
    """

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split("|"),
            token_pattern=None,
            min_df=min_df,
        )
        self.item_profiles: np.ndarray = None   # (n_movies, n_features)
        self.movie_ids: np.ndarray = None
        self._movie_id_to_idx: dict = {}

    def fit(self, movies: pd.DataFrame) -> "ContentBasedFilter":
        """
        Build TF-IDF item profiles from genre strings.

        Parameters
        ----------
        movies : pd.DataFrame
            Must contain 'movieId' and 'genres' columns.
            genres should be pipe-separated, e.g. 'Action|Drama'.
        """
        logger.info("Building content profiles with TF-IDF on genres...")
        genres = movies["genres"].fillna("Unknown")
        self.item_profiles = self.vectorizer.fit_transform(genres).toarray()
        self.movie_ids = movies["movieId"].values
        self._movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        logger.info(
            f"Item profiles: {self.item_profiles.shape} "
            f"| Vocab: {len(self.vectorizer.vocabulary_)} genres"
        )
        return self

    def build_user_profile(
        self, user_ratings: pd.DataFrame, rating_threshold: float = 3.5
    ) -> np.ndarray:
        """
        Compute a user taste profile as a weighted average of item vectors.

        Items rated >= threshold are included; weights are (rating - threshold).

        Parameters
        ----------
        user_ratings : pd.DataFrame
            Subset of ratings df for one user. Must have 'movieId' and 'rating'.
        rating_threshold : float
            Minimum rating to include an item in the profile.

        Returns
        -------
        np.ndarray of shape (n_features,) normalized user profile, or None.
        """
        liked = user_ratings[user_ratings["rating"] >= rating_threshold]
        if liked.empty:
            return None

        weights = (liked["rating"] - rating_threshold).values
        vectors = []
        for mid in liked["movieId"]:
            idx = self._movie_id_to_idx.get(mid)
            if idx is not None:
                vectors.append(self.item_profiles[idx])

        if not vectors:
            return None

        profile = np.average(np.array(vectors), axis=0, weights=weights[:len(vectors)])
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 1e-9 else profile

    def recommend(
        self,
        user_ratings: pd.DataFrame,
        n: int = 10,
        exclude_seen: bool = True,
        rating_threshold: float = 3.5,
    ) -> List[Tuple[int, float]]:
        """
        Recommend top-N items for a user.

        Parameters
        ----------
        user_ratings : pd.DataFrame
            Must contain 'movieId' and 'rating' for the target user.
        n : int
            Number of items to return.
        exclude_seen : bool
            Exclude items the user has already rated.
        rating_threshold : float
            Only items rated at or above this form the user profile.

        Returns
        -------
        List of (movieId, score) tuples sorted by score descending.
        """
        user_profile = self.build_user_profile(user_ratings, rating_threshold)
        if user_profile is None:
            logger.warning("Could not build user profile — insufficient ratings.")
            return []

        scores = cosine_similarity([user_profile], self.item_profiles)[0]  # (n_movies,)

        if exclude_seen:
            seen_ids = set(user_ratings["movieId"].values)
            for mid in seen_ids:
                idx = self._movie_id_to_idx.get(mid)
                if idx is not None:
                    scores[idx] = -1.0

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(self.movie_ids[i]), float(scores[i])) for i in top_idx]

    def get_similar_movies(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find n movies with highest genre-similarity to the given movie.

        Parameters
        ----------
        movie_id : int
            Source movieId.
        n : int
            Number of similar movies to return.

        Returns
        -------
        List of (movieId, similarity_score) tuples.
        """
        idx = self._movie_id_to_idx.get(movie_id)
        if idx is None:
            logger.error(f"movieId {movie_id} not found in item profiles.")
            return []

        sim_scores = cosine_similarity(
            [self.item_profiles[idx]], self.item_profiles
        )[0]
        sim_scores[idx] = -1.0  # exclude self
        top_idx = np.argsort(sim_scores)[::-1][:n]
        return [(int(self.movie_ids[i]), float(sim_scores[i])) for i in top_idx]

    @property
    def feature_names(self) -> list:
        """Return the list of genre features."""
        return self.vectorizer.get_feature_names_out().tolist()
