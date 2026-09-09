"""
Evaluates RecoServe's model against a held-out test split, and against
a naive popularity baseline, using Precision@K and Recall@K.

This is what generates the numbers that belong in the README - don't
hand-write those numbers, run this and paste the real output.
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


def main():
    ratings_df = pd.read_csv("data/ratings.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=50, n_epochs=20, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    precision, recall = precision_recall_at_k(predictions)
    print(f"SVD model  -> Precision@{K}: {precision:.3f}  Recall@{K}: {recall:.3f}")

    # TODO: repeat the same evaluation using the popularity_fallback
    # ranking (from train.py) instead of model predictions, to produce
    # the baseline comparison row in the README table.


if __name__ == "__main__":
    main()
