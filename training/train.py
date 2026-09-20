"""
RecoServe training pipeline.

Loads MovieLens-format ratings/movies, loads them into Postgres, trains
a collaborative-filtering model (SVD), generates top-N recommendations
per user, computes a popularity-based cold-start fallback list, and
writes both to Postgres for the serving API to read.

Run on a schedule (cron / GitHub Actions) - never called from the
request path. The serving API only ever reads what this script writes.

For local development/testing without the real dataset:
    python generate_sample_data.py
    python train.py

For production, download the real dataset from
https://grouplens.org/datasets/movielens/ (the 25M version) and place
movies.csv / ratings.csv in data/ before running this script.
"""

import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
TOP_N = int(os.environ.get("TOP_N", 10))
# How many ratings a movie needs before it's eligible for the
# popularity fallback list - avoids a movie with one 5-star rating
# outranking genuinely well-established titles. Lower this for small
# local/sample datasets via the env var; the real MovieLens dataset
# comfortably clears the default of 50.
POPULARITY_MIN_RATINGS = int(os.environ.get("POPULARITY_MIN_RATINGS", 50))

engine = create_engine(DATABASE_URL)


def load_data():
    movies_df = pd.read_csv("data/movies.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    return movies_df, ratings_df


def load_reference_data_into_db(movies_df, ratings_df):
    """Loads movies and ratings into Postgres. Movies are upserted
    (idempotent across repeated runs); ratings are only inserted if the
    table is currently empty, since in production ratings would arrive
    continuously from the application rather than being reloaded from
    a CSV on every training run."""
    with engine.begin() as conn:
        for _, row in movies_df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO movies (id, title, genres) VALUES (:id, :title, :genres)
                    ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, genres = EXCLUDED.genres
                    """
                ),
                {"id": int(row.movieId), "title": row.title, "genres": row.genres},
            )

        existing = conn.execute(text("SELECT COUNT(*) FROM ratings")).scalar()
        if existing == 0:
            conn.execute(
                text(
                    "INSERT INTO ratings (user_id, movie_id, rating) VALUES (:user_id, :movie_id, :rating)"
                ),
                [
                    {"user_id": int(r.userId), "movie_id": int(r.movieId), "rating": float(r.rating)}
                    for r in ratings_df.itertuples()
                ],
            )


def train_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    return model, trainset


def generate_top_n(model, ratings_df, n=TOP_N):
    """For every user, predict a score for every movie they haven't
    rated yet, and keep the top N."""
    all_movie_ids = ratings_df["movieId"].unique()
    recommendations = {}

    for uid in ratings_df["userId"].unique():
        rated = set(ratings_df.loc[ratings_df.userId == uid, "movieId"])
        unrated = [m for m in all_movie_ids if m not in rated]

        predictions = [(m, model.predict(uid, m).est) for m in unrated]
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations[uid] = predictions[:n]

    return recommendations


def compute_popularity_fallback(ratings_df, n=TOP_N):
    """Cold-start list for users with little/no history: highest
    average rating among movies with a reasonable number of ratings."""
    stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
    eligible = stats[stats["count"] >= POPULARITY_MIN_RATINGS]

    if eligible.empty:
        # Sample/small datasets may not have any movie clearing the
        # threshold - fall back to whatever has the most ratings at all
        # rather than returning an empty list.
        eligible = stats.sort_values("count", ascending=False).head(max(n * 2, 20))

    top = eligible.sort_values("mean", ascending=False).head(n)
    return list(top.index)


def write_recommendations(recommendations, popularity_fallback):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM recommendations"))
        for user_id, recs in recommendations.items():
            for rank, (movie_id, score) in enumerate(recs):
                conn.execute(
                    text(
                        """
                        INSERT INTO recommendations (user_id, movie_id, rank, score)
                        VALUES (:user_id, :movie_id, :rank, :score)
                        """
                    ),
                    {"user_id": int(user_id), "movie_id": int(movie_id), "rank": rank, "score": float(score)},
                )

        conn.execute(text("DELETE FROM popularity_fallback"))
        for rank, movie_id in enumerate(popularity_fallback):
            conn.execute(
                text("INSERT INTO popularity_fallback (movie_id, rank) VALUES (:movie_id, :rank)"),
                {"movie_id": int(movie_id), "rank": rank},
            )


def main():
    print("Loading movies and ratings...")
    movies_df, ratings_df = load_data()

    print("Loading reference data into Postgres...")
    load_reference_data_into_db(movies_df, ratings_df)

    print("Training SVD model...")
    model, _trainset = train_model(ratings_df)

    print("Generating top-N recommendations per user...")
    recommendations = generate_top_n(model, ratings_df)

    print("Computing popularity fallback for cold-start users...")
    popularity_fallback = compute_popularity_fallback(ratings_df)

    print("Writing results to Postgres...")
    write_recommendations(recommendations, popularity_fallback)

    print(f"Done. Wrote recommendations for {len(recommendations)} users.")


if __name__ == "__main__":
    main()
