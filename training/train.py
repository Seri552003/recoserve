"""
RecoServe training pipeline.

Loads MovieLens ratings, trains a collaborative-filtering model (SVD),
generates top-N recommendations per user, computes a popularity-based
cold-start fallback list, and writes both to Postgres for the serving
API to read.

Run on a schedule (cron / GitHub Actions) - never called from the
request path. The serving API only ever reads what this script writes.
"""

import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
TOP_N = 10

engine = create_engine(DATABASE_URL)


def load_ratings():
    # Expects ratings.csv (userId, movieId, rating, timestamp) from
    # https://grouplens.org/datasets/movielens/
    return pd.read_csv("data/ratings.csv")


def train_model(ratings_df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    return model, trainset


def generate_top_n(model, trainset, ratings_df, n=TOP_N):
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
    stats = stats[stats["count"] >= 50]  # avoid movies with 1-2 five-star ratings
    top = stats.sort_values("mean", ascending=False).head(n)
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
    print("Loading ratings...")
    ratings_df = load_ratings()

    print("Training SVD model...")
    model, trainset = train_model(ratings_df)

    print("Generating top-N recommendations per user...")
    recommendations = generate_top_n(model, trainset, ratings_df)

    print("Computing popularity fallback for cold-start users...")
    popularity_fallback = compute_popularity_fallback(ratings_df)

    print("Writing results to Postgres...")
    write_recommendations(recommendations, popularity_fallback)

    print(f"Done. Wrote recommendations for {len(recommendations)} users.")


if __name__ == "__main__":
    main()
