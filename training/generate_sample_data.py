"""
Generates a small synthetic MovieLens-shaped dataset (movies.csv,
ratings.csv) for local development and testing, so the full pipeline
can be run end-to-end without downloading the real ~250MB MovieLens
25M dataset first.

Ratings are generated from a latent-factor model (each user and movie
gets a small random taste/style vector; rating = their dot product,
clipped and noised) - the same underlying structure SVD assumes when
it factorizes a rating matrix. This gives the sample dataset genuine
collaborative-filterable signal beyond what a genre/popularity-only
model would produce, so a local evaluation run is actually meaningful
rather than trivially matching a popularity baseline.

This is NOT a substitute for the real dataset - it exists purely so
`train.py` and `evaluate.py` have something to run against locally.
For real numbers worth putting on a resume, download the actual
MovieLens 25M dataset from https://grouplens.org/datasets/movielens/
and point these scripts at ml-25m/movies.csv and ml-25m/ratings.csv
instead.

Run: python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

RNG_SEED = 42
N_MOVIES = 200
N_USERS = 600
N_LATENT_FACTORS = 6
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Animation"]

rng = np.random.default_rng(RNG_SEED)


def generate_movies():
    rows = []
    for movie_id in range(1, N_MOVIES + 1):
        n_genres = rng.integers(1, 3)
        genres = "|".join(rng.choice(GENRES, n_genres, replace=False))
        rows.append({"movieId": movie_id, "title": f"Sample Movie {movie_id}", "genres": genres})
    return pd.DataFrame(rows)


def generate_ratings(movies_df):
    """Generates ratings from a latent-factor model: each user gets a
    random taste vector, each movie a random style vector, and the
    rating is (roughly) their dot product plus noise - genuinely
    collaborative-filterable structure, the same assumption SVD makes.
    Rating counts per user and per movie are both skewed (Zipfian) so
    cold-start users and a popularity long-tail both show up, matching
    real-world rating distributions.
    """
    user_vecs = rng.normal(0, 1, size=(N_USERS + 1, N_LATENT_FACTORS))
    movie_vecs = rng.normal(0, 1, size=(N_MOVIES + 1, N_LATENT_FACTORS))
    movie_bias = rng.normal(0, 0.4, size=N_MOVIES + 1)  # some movies are just generally better/worse

    movie_popularity = rng.zipf(a=1.6, size=N_MOVIES) + 1
    movie_popularity = movie_popularity / movie_popularity.sum()

    rows = []
    for user_id in range(1, N_USERS + 1):
        n_ratings = int(rng.choice([3, 5, 10, 20, 40, 70], p=[0.15, 0.2, 0.25, 0.2, 0.13, 0.07]))
        rated_movies = rng.choice(
            movies_df["movieId"].values, size=min(n_ratings, N_MOVIES), replace=False, p=movie_popularity
        )

        for movie_id in rated_movies:
            raw_score = np.dot(user_vecs[user_id], movie_vecs[movie_id]) + movie_bias[movie_id]
            rating = 3.0 + raw_score  # center around a 3.0 baseline
            rating = np.clip(rating + rng.normal(0, 0.3), 0.5, 5.0)
            rating = round(rating * 2) / 2  # snap to nearest 0.5, like real MovieLens ratings
            rows.append({"userId": user_id, "movieId": int(movie_id), "rating": rating, "timestamp": 0})

    return pd.DataFrame(rows)


def main():
    os.makedirs("data", exist_ok=True)

    movies_df = generate_movies()
    ratings_df = generate_ratings(movies_df)

    movies_df.to_csv("data/movies.csv", index=False)
    ratings_df.to_csv("data/ratings.csv", index=False)

    print(f"Generated {len(movies_df)} movies and {len(ratings_df)} ratings ({ratings_df.userId.nunique()} users).")
    print("Wrote data/movies.csv and data/ratings.csv")


if __name__ == "__main__":
    main()
