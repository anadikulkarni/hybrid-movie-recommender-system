"""
generate_data.py
----------------
Generates a synthetic MovieLens-style dataset for CineMatch.

The ratings have structure a recommender can learn, planted the way real
rating data behaves:
  - Biases: some users rate generously, and some movies are better liked
  - Hidden tastes: every user and movie gets an 8-dimensional latent vector,
    and ratings rise with their dot product
  - Genres that matter: part of each movie's latent vector comes from its
    genres, so movies sharing genres attract the same audiences
  - Self-selection: who rates what depends on user activity, movie popularity
    and how much the user is likely to enjoy the movie (missing not at random)

Sizes and the rating distribution follow the real MovieLens 100K:
  - 943 users, 1,682 movies, 100,000 ratings
  - 1-5 stars in the shares 6% / 11% / 27% / 34% / 21% (mean ~3.5, std ~1.1)
  - 1-3 genres per movie from 18 categories, weighted by real genre frequency

Because the structure is known, the models' results can be checked against it:
anything that learns should beat predicting the global mean.

Usage:
    python generate_data.py
    python generate_data.py --output_dir my_data --seed 123
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
# How common each genre is in MovieLens 100K (share of movies tagged with it)
GENRE_FREQUENCY = np.array([
    0.15, 0.08, 0.025, 0.07, 0.30, 0.06, 0.03, 0.43, 0.013, 0.014,
    0.05, 0.03, 0.04, 0.15, 0.06, 0.15, 0.04, 0.016,
])
STAR_SHARES = np.array([0.061, 0.114, 0.271, 0.342, 0.212])  # 1 to 5 stars

N_FACTORS = 8          # dimensions of the hidden taste vectors
GENRE_SHARE = 0.6      # share of a movie's taste vector explained by its genres
USER_BIAS_SD = 0.35
ITEM_BIAS_SD = 0.45
NOISE_SD = 0.7         # rating noise no model can explain
SELECTION = 0.6        # how strongly expected enjoyment drives what gets rated

OCCUPATIONS = [
    "technician", "other", "writer", "executive", "administrator",
    "student", "lawyer", "educator", "scientist", "programmer",
]

BASE_TITLES = [
    "Toy Story", "GoldenEye", "Four Rooms", "Get Shorty", "Copycat",
    "Twelve Monkeys", "Babe", "Dead Man Walking", "Seven", "Usual Suspects",
    "Braveheart", "Batman Forever", "Forrest Gump", "Shawshank Redemption",
    "Pulp Fiction", "Silence of Lambs", "Star Wars", "Schindlers List", "Fargo",
    "English Patient", "Titanic", "Good Will Hunting", "LA Confidential",
    "Jerry Maguire", "As Good as It Gets", "Full Monty", "Boogie Nights",
    "Donnie Brasco", "Trainspotting", "Heat", "Casino", "Apollo 13",
    "Outbreak", "Speed", "Die Hard with Vengeance", "Waterworld",
    "Dangerous Minds", "Clockers", "Leaving Las Vegas", "Restoration",
    "Mortal Kombat", "Ace Ventura", "Casper", "Free Willy 2", "Mad Love",
    "Mr Hollands Opus", "Muppet Treasure Island", "Rob Roy", "City of Angels",
    "Jerry Springer: Ringmaster",
]


def generate(
    output_dir: str = "data",
    seed: int = 42,
    n_users: int = 943,
    n_movies: int = 1682,
    n_ratings: int = 100_000,
):
    """
    Build the movies, users and ratings tables.

    Writes movies.csv, users.csv and ratings.csv to output_dir unless it is None.

    Returns
    -------
    (movies, users, ratings) DataFrames.
    """
    rng = np.random.default_rng(seed)

    # Movies: 1-3 genres each, with common genres (Drama, Comedy) picked more often
    genre_weights = GENRE_FREQUENCY / GENRE_FREQUENCY.sum()
    genre_matrix = np.zeros((n_movies, len(GENRES)))
    for i, count in enumerate(rng.choice([1, 2, 3], n_movies, p=[0.5, 0.35, 0.15])):
        genre_matrix[i, rng.choice(len(GENRES), count, replace=False, p=genre_weights)] = 1

    titles = BASE_TITLES + [f"Film {i}" for i in range(len(BASE_TITLES) + 1, n_movies + 1)]
    movies = pd.DataFrame({
        "movieId": range(1, n_movies + 1),
        "title": titles[:n_movies],
        "year": rng.integers(1990, 2000, n_movies),
        "genres": ["|".join(GENRES[j] for j in np.flatnonzero(row)) for row in genre_matrix],
    })

    users = pd.DataFrame({
        "userId": range(1, n_users + 1),
        "age": rng.integers(18, 65, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(OCCUPATIONS, n_users),
    })

    # Hidden tastes: a movie's vector is part genre, part its own
    genre_vectors = rng.normal(0, 1, (len(GENRES), N_FACTORS))
    from_genres = genre_matrix @ genre_vectors / np.sqrt(genre_matrix.sum(axis=1, keepdims=True))
    item_factors = (
        np.sqrt(GENRE_SHARE) * from_genres
        + np.sqrt(1 - GENRE_SHARE) * rng.normal(0, 1, (n_movies, N_FACTORS))
    )
    user_factors = rng.normal(0, 1, (n_users, N_FACTORS))
    user_bias = rng.normal(0, USER_BIAS_SD, n_users)
    item_bias = rng.normal(0, ITEM_BIAS_SD, n_movies)
    affinity = user_factors @ item_factors.T / np.sqrt(N_FACTORS)   # (n_users, n_movies)

    # Who rates what: heavy-tailed activity and popularity (well-liked movies
    # are watched more), plus a pull towards movies the user will enjoy
    activity = rng.lognormal(0, 1.0, n_users)
    popularity = rng.lognormal(0, 1.2, n_movies) * np.exp(0.8 * item_bias / ITEM_BIAS_SD)
    enjoyment = (affinity - affinity.mean()) / affinity.std()
    log_weight = (
        np.log(activity)[:, None] + np.log(popularity)[None, :] + SELECTION * enjoyment
    )

    # Weighted sampling without replacement (Efraimidis-Spirakis): each pair
    # draws key = log(U) / weight, and the n_ratings largest keys are kept
    keys = np.log(rng.random(log_weight.shape)) / np.exp(log_weight - log_weight.max())
    chosen = np.argpartition(keys.ravel(), -n_ratings)[-n_ratings:]
    u, m = np.unravel_index(chosen, log_weight.shape)

    # Ratings: true score plus noise, cut into stars at MovieLens's shares
    latent = user_bias[u] + item_bias[m] + affinity[u, m] + rng.normal(0, NOISE_SD, n_ratings)
    cut_points = np.quantile(latent, np.cumsum(STAR_SHARES)[:-1])
    stars = 1 + np.searchsorted(cut_points, latent)

    ratings = (
        pd.DataFrame({
            "userId": u + 1,
            "movieId": m + 1,
            "rating": stars.astype(float),
            "timestamp": rng.integers(874_724_710, 893_286_638, n_ratings),
        })
        .sort_values(["userId", "movieId"])
        .reset_index(drop=True)
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        movies.to_csv(output_path / "movies.csv", index=False)
        users.to_csv(output_path / "users.csv", index=False)
        ratings.to_csv(output_path / "ratings.csv", index=False)

        print(f"Dataset generated in '{output_dir}/'")
        print(f"  ratings : {len(ratings):,}")
        print(f"  movies  : {len(movies):,}")
        print(f"  users   : {len(users):,}")

    return movies, users, ratings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MovieLens-style data.")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
