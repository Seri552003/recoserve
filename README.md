# RecoServe — Movie Recommendation Service

A production-shaped recommendation service built on the MovieLens dataset: an offline training pipeline generates personalized recommendations, a cached serving API delivers them in milliseconds, and a cold-start fallback handles new users. The goal was to build the *system* around a recommender, not just the model — the kind of architecture real recommendation services (Netflix, Amazon) actually use, just at a scale you can run on free-tier infra.

**Live demo:** [recoserve.vercel.app](#) *(replace with your deployed link)*

---

## Problem & Approach

Most "recommendation engine" resume projects stop at a Jupyter notebook that prints `model.predict()` output. That's the least interesting 25% of the problem. The real engineering questions are: how do you serve recommendations fast without running inference on every request? How do you keep the model fresh without retraining live? What do you show a user with no history at all? RecoServe is built to answer all three.

**Design principle: offline training, online serving.** The model is trained periodically as a batch job — never inside the request path. The serving API only ever reads precomputed, cached results. This is the same split used by real production recommender systems, and it's why the API stays fast regardless of how expensive the training step is.

## Architecture

```
 MovieLens dataset (ratings.csv, movies.csv)
              │
              ▼
   ┌───────────────────────┐   (scheduled every 6h)
   │   Training Pipeline     │
   │   (Python, surprise)    │
   │   • Loads ratings        │
   │   • Trains SVD model     │
   │   • Generates top-N      │
   │     recs per user        │
   │   • Computes popularity  │
   │     fallback list        │
   └───────────┬─────────────┘
               │ writes
               ▼
   ┌────────────────────┐        ┌─────────────────────┐
   │    PostgreSQL         │◄──────►│       Redis            │
   │  (movies, ratings,     │  warm  │ (precomputed top-N,     │
   │   precomputed recs)    │  cache │  TTL matches retrain     │
   │                        │        │  interval)               │
   └────────────────────┘        └───────────┬─────────────┘
                                                │ cache-first read
                                                ▼
                                   ┌──────────────────────┐
                                   │   Serving API (Node)    │
                                   │   GET /recommendations/  │
                                   │        :userId            │
                                   └───────────┬──────────────┘
                                                │
                                                ▼
                                   ┌──────────────────────┐
                                   │    React Frontend        │
                                   └──────────────────────┘
```

## Why This Split (the design decisions worth defending)

- **Python for training, Node for serving.** Training needs pandas/scikit-learn/surprise — Python's ecosystem is the right tool there. Serving is just reads from Redis/Postgres — no ML libraries needed on that path, so a lightweight Node API keeps the serving layer simple and fast. Two services, each doing the thing it's good at, communicating only through the shared database — not one monolith awkwardly mixing both.
- **Redis cache-first, Postgres fallback.** Precomputed recommendations are written to both. Redis is checked first (sub-5ms); a cache miss (e.g., after a Redis restart) falls back to Postgres rather than failing, then re-warms the cache.
- **Cold start via popularity fallback.** A brand-new user has no rating history, so collaborative filtering has nothing to work with. The training pipeline also computes a global popularity-ranked list; the serving API returns that when a user has fewer than N ratings, rather than an empty or broken response.

## Evaluation

Model quality measured against a held-out test split (80/20), evaluated at K=10:

| Metric | RecoServe (SVD) | Popularity baseline |
|---|---|---|
| Precision@10 | 0.31 | 0.14 |
| Recall@10 | 0.18 | 0.09 |
| RMSE (held-out ratings) | 0.87 | — |

*(Placeholder numbers — replace with your actual evaluation run before this goes on your resume or GitHub. Run `python training/evaluate.py` to generate real ones.)*

## Features

- Offline batch training on the MovieLens dataset (SVD-based collaborative filtering via the `surprise` library)
- Cache-first serving API (Redis, Postgres fallback) — precomputed recommendations, not live inference per request
- Cold-start handling via popularity fallback for new/low-history users
- Scheduled retraining (cron) keeps recommendations fresh without touching the request path
- Evaluation harness reporting precision@K / recall@K against a popularity baseline
- Fully containerized (Docker Compose) and deployed live

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Training pipeline | Python, pandas, `scikit-surprise` (SVD) | Standard, well-documented collaborative filtering tooling |
| Serving API | Node.js, Express | Lightweight, fast reads — no ML dependencies needed here |
| Cache | Redis (Upstash) | Sub-5ms precomputed recommendation lookups |
| Database | PostgreSQL (Supabase/Neon) | Movies, ratings, precomputed recs, fallback list |
| Frontend | React | Minimal UI to browse recommendations by user |
| Scheduling | cron / GitHub Actions scheduled workflow | Triggers retraining every N hours |
| Deployment | Docker Compose (local), Render (API + training), Vercel (frontend) | Free-tier friendly, all pieces independently deployable |

## Dataset

[MovieLens 25M](https://grouplens.org/datasets/movielens/) — 25 million ratings across ~62,000 movies from ~162,000 users. Publicly available, no scraping or API keys required. `movies.csv` and `ratings.csv` are loaded directly by the training pipeline.

## Project Structure

```
recoserve/
├── training/
│   ├── train.py            # Loads data, trains SVD, generates + writes top-N recs
│   ├── evaluate.py         # Precision@K / Recall@K against held-out split
│   └── requirements.txt
├── serving-api/
│   ├── src/
│   │   ├── server.js
│   │   ├── routes/
│   │   │   └── recommendations.js
│   │   └── services/
│   │       └── recommendationService.js   # cache-first read + cold-start logic
│   ├── package.json
│   └── .env.example
├── frontend/                # React app (browse recommendations)
├── docker-compose.yml
└── README.md
```

## Running Locally

```bash
# 1. Start Postgres + Redis
docker-compose up -d

# 2. Run the training pipeline once (populates the DB with recommendations)
cd training
pip install -r requirements.txt
python train.py

# 3. Start the serving API
cd ../serving-api
cp .env.example .env
npm install
npm run dev

# 4. (optional) Run evaluation
cd ../training
python evaluate.py
```

## What I'd Improve With More Time

- Swap batch SVD for an incremental/online update so new ratings affect recommendations before the next full retrain
- A/B test the learned model against the popularity baseline on real (simulated) user sessions, not just offline precision/recall
- Add implicit feedback signals (watch time, clicks) alongside explicit ratings

## License

MIT
