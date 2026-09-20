"""
Evaluates RecoServe's model against a held-out test split, and against
a naive popularity baseline, using Precision@K and Recall@K.

This is what generates the numbers that belong in the README - don't
hand-write those numbers, run this and paste the real output.

Run: python evaluate.py
"""

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

K = 10
RELEVANCE_THRESHOLD = 4.0  # a rating >= this counts as "relevant"


def precision_recall_at_k(predictions, k=K, threshold=RELEVANCE_THRESHOLD):
    user_est_true = defaultdict(list)
    for uid, _iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]

        n_relevant = sum(true_r >= threshold for (_, true_r) in ratings)
        n_recommended_relevant = sum(true_r >= threshold for (_, true_r) in top_k)

        precisions[uid] = n_recommended_relevant / k if k else 0
        recalls[uid] = n_recommended_relevant / n_relevant if n_relevant else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall


def popularity_baseline_precision_recall(train_ratings_df, testset, k=K, threshold=RELEVANCE_THRESHOLD):
    """Same metric, but 'predicting' with a fixed popularity ranking
    (average rating) instead of the trained model - this is the
    baseline the SVD model needs to beat to justify its complexity."""
    pop_rank = train_ratings_df.groupby("movieId")["rating"].mean().to_dict()
    global_mean = train_ratings_df["rating"].mean()

    user_est_true = defaultdict(list)
    for uid, iid, true_r in testset:
        est = pop_rank.get(iid, global_mean)
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        n_relevant = sum(true_r >= threshold for (_, true_r) in ratings)
        n_recommended_relevant = sum(true_r >= threshold for (_, true_r) in top_k)
        precisions[uid] = n_recommended_relevant / k if k else 0
        recalls[uid] = n_recommended_relevant / n_relevant if n_relevant else 0

    return sum(precisions.values()) / len(precisions), sum(recalls.values()) / len(recalls)


def main():
    ratings_df = pd.read_csv("data/ratings.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    svd_precision, svd_recall = precision_recall_at_k(predictions)
    print(f"SVD model         -> Precision@{K}: {svd_precision:.3f}  Recall@{K}: {svd_recall:.3f}")

    # Rebuild the train-split ratings as a DataFrame to compute the
    # popularity baseline on the same split the model was trained on.
    train_rows = [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), r)
                  for (u, i, r) in trainset.all_ratings()]
    train_df = pd.DataFrame(train_rows, columns=["userId", "movieId", "rating"])

    pop_precision, pop_recall = popularity_baseline_precision_recall(train_df, testset)
    print(f"Popularity baseline -> Precision@{K}: {pop_precision:.3f}  Recall@{K}: {pop_recall:.3f}")

    lift = (svd_precision / pop_precision) if pop_precision > 0 else float("inf")
    print(f"\nSVD precision is {lift:.2f}x the popularity baseline.")


if __name__ == "__main__":
    main()
