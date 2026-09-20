-- RecoServe schema. Run once against a fresh database before the first
-- training run: psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS movies (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  genres TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  movie_id INTEGER NOT NULL REFERENCES movies(id),
  rating NUMERIC(2,1) NOT NULL,
  rated_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);

-- Precomputed top-N recommendations, written by the training pipeline.
-- The serving API only ever reads this table - it never computes a
-- recommendation itself.
CREATE TABLE IF NOT EXISTS recommendations (
  user_id INTEGER NOT NULL,
  movie_id INTEGER NOT NULL REFERENCES movies(id),
  rank INTEGER NOT NULL,
  score NUMERIC(6,4) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id, rank);

-- Cold-start fallback for users with too little rating history for a
-- personalized prediction to be meaningful.
CREATE TABLE IF NOT EXISTS popularity_fallback (
  movie_id INTEGER NOT NULL REFERENCES movies(id),
  rank INTEGER NOT NULL
);
