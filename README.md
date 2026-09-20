# RecoServe — Movie Recommendation Service

A production-shaped recommendation service: an offline training pipeline generates personalized recommendations, a cached serving API delivers them in milliseconds, and a cold-start fallback handles new users. The goal was to build the *system* around a recommender, not just the model.

---

## Problem & Approach

Most "recommendation engine" resume projects stop at a notebook that prints `model.predict()` output. That's the least interesting part of the problem. The real engineering questions are: how do you serve recommendations fast without running inference on every request? How do you keep the model fresh without retraining live? What do you show a user with no history at all? RecoServe answers all three.

**Design principle: offline training, online serving.** The model is trained periodically as a batch job — never inside the request path. The serving API only ever reads precomputed, cached results.

## Architecture

```
 MovieLens-format dataset (movies.csv, ratings.csv)
              │
              ▼
   ┌───────────────────────┐   (scheduled every 6h)
   │   Training Pipeline     │
   │   (Python, surprise)    │
   │   • Loads movies/ratings │
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

## Why This Split

- **Python for training, Node for serving.** Training needs pandas/scikit-learn/surprise. Serving is just reads from Redis/Postgres — no ML libraries needed on that path, so the serving layer stays simple and fast.
- **Redis cache-first, Postgres fallback.** A cache miss (e.g. after a Redis restart) falls back to Postgres rather than failing, then re-warms the cache on the way out.
- **Cold start via popularity fallback.** A brand-new user has no rating history, so collaborative filtering has nothing to work with. The training pipeline also computes a global popularity-ranked list; the serving API returns that when a user has fewer than 5 ratings.

## What's Actually Been Tested — Real Results

Everything below was run for real against a local Postgres + Redis instance during development, not estimated.

### End-to-end pipeline

```bash
python generate_sample_data.py   # synthetic dataset for local dev, see note below
python train.py                  # loads data into Postgres, trains SVD, writes recommendations
```

**Real output from a run in this repo:** 200 movies, 10,795 ratings, 600 users, recommendations written for all 600. Verified directly in Postgres afterward (`SELECT COUNT(*)` on each table).

### Cache behavior (real, measured)

Hitting `/recommendations/1` repeatedly against the local serving API:

| Call | Latency |
|---|---|
| 1st call (cache miss → Postgres) | 24ms |
| 2nd call onward (cache hit) | 5–6ms |

*(These numbers are from Postgres and Redis both running locally on the same machine as the API, with a small sample dataset — the cache-miss path here is already fast because there's no network hop to the database. In a real deployment, Postgres and the API are on different hosts, so the cache-miss latency (and therefore the cache's benefit) will be meaningfully larger — a network round trip to Supabase/Neon typically runs 50–150ms depending on region, which is the more realistic number to quote once deployed. Re-run this same test against your deployed instance and use that number, not the localhost one above, on a resume.)*

### Cold-start fallback (real, verified)

Requesting recommendations for a user ID with zero rating history returns the precomputed popularity list, not an error or empty response — confirmed directly against the running API.

### Model evaluation (real, on the local sample dataset)

```bash
python evaluate.py
```

**Actual output:**
```
SVD model            -> Precision@10: 0.160  Recall@10: 0.695
Popularity baseline  -> Precision@10: 0.154  Recall@10: 0.676
SVD precision is 1.04x the popularity baseline.
```

**Being transparent about this number:** the sample dataset generator (`generate_sample_data.py`) creates ratings from a latent-factor model, the same structure SVD assumes — genuinely collaborative-filterable, not just genre/popularity noise. Even so, on ~11K synthetic ratings across 600 users, SVD only edges out the popularity baseline by about 4%, not the larger margin (commonly cited around 2x on real benchmarks) you'd see on the actual MovieLens dataset with millions of real, idiosyncratic human ratings. This is expected: a small, synthetic sample simply doesn't carry as much learnable signal as a large real one. **Don't quote "2x baseline" on a resume until you've run `evaluate.py` against the real MovieLens 25M dataset and it actually shows that** — quote the real number your own run produces.

## Features

- Offline batch training (SVD-based collaborative filtering via the `surprise` library)
- Cache-first serving API (Redis, Postgres fallback) — precomputed recommendations, not live inference per request
- Cold-start handling via popularity fallback, verified working
- Evaluation harness reporting Precision@K / Recall@K against a popularity baseline, with a real comparison built in
- Minimal React frontend to look up a user's recommendations and see live response latency
- Fully containerized (Docker Compose)

## Tech Stack

| Layer | Choice |
|---|---|
| Training pipeline | Python, pandas, `scikit-surprise` (SVD) |
| Serving API | Node.js, Express |
| Frontend | React, Vite |
| Cache | Redis |
| Database | PostgreSQL |
| Scheduling | cron / GitHub Actions scheduled workflow |

## Dataset

For local development, `generate_sample_data.py` creates a synthetic MovieLens-shaped dataset (`data/movies.csv`, `data/ratings.csv`) with genuine latent-factor structure, so the whole pipeline can be run without a large download.

**For production / resume-worthy numbers**, download the real [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/) (25 million ratings, ~62,000 movies, ~162,000 users) and place `movies.csv` / `ratings.csv` in `training/data/` instead — the pipeline code is identical either way, only the CSVs change.

## Project Structure

```
recoserve/
├── training/
│   ├── generate_sample_data.py   # synthetic dataset for local dev/testing
│   ├── schema.sql                 # Postgres schema — run this first
│   ├── train.py                   # loads data, trains SVD, writes recommendations
│   ├── evaluate.py                # Precision@K / Recall@K vs popularity baseline
│   └── requirements.txt
├── serving-api/
│   ├── src/
│   │   ├── server.js
│   │   ├── routes/recommendations.js
│   │   └── services/recommendationService.js   # cache-first read + cold-start logic
│   ├── package.json
│   └── .env.example
├── frontend/                      # React app - look up a user's recommendations
├── docker-compose.yml
└── README.md
```

## Running Locally

```bash
# 1. Start Postgres + Redis
docker-compose up -d

# 2. Set up the training environment
cd training
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL to match docker-compose

# 3. Generate sample data and run the pipeline
python generate_sample_data.py
psql "$DATABASE_URL" -f schema.sql
python train.py

# 4. Check the model quality
python evaluate.py

# 5. Start the serving API
cd ../serving-api
cp .env.example .env
npm install
npm run dev              # http://localhost:4000

# 6. Start the frontend
cd ../frontend
cp .env.example .env
npm install
npm run dev               # http://localhost:5174
```

Open the frontend, enter a user ID from 1–600 (or whatever range your data generated), and see their recommendations plus live response latency. Try the same ID twice to see the cache-hit speed-up, and try a very high ID (e.g. 99999) to see the cold-start fallback.

## Deployment

- **Training pipeline**: run on a schedule via GitHub Actions (a scheduled workflow) or a cron job on any host — it's a one-shot script, not a long-running service.
- **Serving API**: Render or Railway, with `DATABASE_URL` pointing at a managed Postgres (Supabase/Neon) and `REDIS_URL` at Upstash.
- **Frontend**: Vercel or Netlify, `VITE_API_URL` pointing at the deployed API.

## What I'd Improve With More Time

- Test against the real MovieLens 25M dataset and update the evaluation numbers above with the real result
- Measure cache-vs-no-cache latency against a genuinely remote Postgres instance, not localhost, for a realistic number
- Swap batch SVD for an incremental/online update so new ratings affect recommendations before the next full retrain
- Add implicit feedback signals (watch time, clicks) alongside explicit ratings

## License

MIT
