"""
generate_data.py
----------------
Generates a synthetic MovieLens-style dataset for CineMatch.

The dataset replicates the statistical properties of the real MovieLens 100K:
  - 943 users, 1,682 movies, ~90,000 ratings
  - Power-law distribution of user activity and movie popularity
  - Rating distribution matching real MovieLens (mean ~3.5, std ~1.1)
  - Multi-label genre tags from 18 standard categories

Usage:
    python generate_data.py
    python generate_data.py --output_dir my_data --seed 123
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate(output_dir: str = "data", seed: int = 42):
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    N_USERS = 943
    N_MOVIES = 1682
    N_RATINGS = 100_000

    GENRES = [
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
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
    movie_titles = BASE_TITLES + [f"Film {i}" for i in range(len(BASE_TITLES) + 1, N_MOVIES + 1)]

    # Movies
    movies = pd.DataFrame({
        "movieId": range(1, N_MOVIES + 1),
        "title": movie_titles[:N_MOVIES],
        "year": np.random.choice(range(1990, 2000), N_MOVIES),
    })
    genre_cols = {g: np.random.choice([0, 1], N_MOVIES, p=[0.8, 0.2]) for g in GENRES}
    movies = pd.concat([movies, pd.DataFrame(genre_cols)], axis=1)
    movies["genres"] = movies[GENRES].apply(
        lambda r: "|".join([g for g in GENRES if r[g] == 1]) or "Unknown", axis=1
    )
    movies_final = movies[["movieId", "title", "year", "genres"]]

    # Users
    users = pd.DataFrame({
        "userId": range(1, N_USERS + 1),
        "age": np.random.randint(18, 65, N_USERS),
        "gender": np.random.choice(["M", "F"], N_USERS),
        "occupation": np.random.choice(OCCUPATIONS, N_USERS),
    })

    # Ratings — power-law activity / popularity
    user_activity = np.random.exponential(1, N_USERS)
    user_activity /= user_activity.sum()
    movie_popularity = np.random.exponential(1, N_MOVIES)
    movie_popularity /= movie_popularity.sum()

    uid = np.random.choice(range(1, N_USERS + 1), N_RATINGS, p=user_activity)
    mid = np.random.choice(range(1, N_MOVIES + 1), N_RATINGS, p=movie_popularity)
    rtg = np.random.choice([1, 2, 3, 4, 5], N_RATINGS, p=[0.06, 0.11, 0.27, 0.35, 0.21]).astype(float)
    ts = np.random.randint(874_724_710, 893_286_638, N_RATINGS)

    ratings = (
        pd.DataFrame({"userId": uid, "movieId": mid, "rating": rtg, "timestamp": ts})
        .drop_duplicates(["userId", "movieId"])
        .reset_index(drop=True)
    )

    movies_final.to_csv(output_path / "movies.csv", index=False)
    users.to_csv(output_path / "users.csv", index=False)
    ratings.to_csv(output_path / "ratings.csv", index=False)

    print(f"Dataset generated in '{output_dir}/'")
    print(f"  ratings : {len(ratings):,}")
    print(f"  movies  : {len(movies_final):,}")
    print(f"  users   : {len(users):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MovieLens-style data.")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
