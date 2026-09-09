const Redis = require('ioredis');
const { Pool } = require('pg');

const redis = new Redis(process.env.REDIS_URL);
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

const CACHE_TTL_SECONDS = 6 * 60 * 60; // matches the training pipeline's retrain interval
const COLD_START_MIN_RATINGS = 5;

function cacheKey(userId) {
  return `recs:${userId}`;
}

/**
 * Returns top-N recommendations for a user. Cache-first (Redis), with
 * a Postgres fallback on a cache miss (e.g. after a Redis restart),
 * which also re-warms the cache. Users below the cold-start threshold
 * get the precomputed popularity fallback list instead of a
 * personalized (and likely poor-quality) prediction.
 */
async function getRecommendationsForUser(userId) {
  const cached = await redis.get(cacheKey(userId));
  if (cached) {
    return JSON.parse(cached);
  }

  const ratingCount = await getUserRatingCount(userId);

  const recs =
    ratingCount < COLD_START_MIN_RATINGS
      ? await getPopularityFallback()
      : await getPersonalizedRecommendations(userId);

  await redis.set(cacheKey(userId), JSON.stringify(recs), 'EX', CACHE_TTL_SECONDS);
  return recs;
}

async function getUserRatingCount(userId) {
  const { rows } = await pool.query('SELECT COUNT(*) FROM ratings WHERE user_id = $1', [userId]);
  return parseInt(rows[0].count, 10);
}

async function getPersonalizedRecommendations(userId) {
  const { rows } = await pool.query(
    `SELECT r.movie_id, r.score, m.title
     FROM recommendations r
     JOIN movies m ON m.id = r.movie_id
     WHERE r.user_id = $1
     ORDER BY r.rank ASC`,
    [userId]
  );
  return rows;
}

async function getPopularityFallback() {
  const { rows } = await pool.query(
    `SELECT p.movie_id, m.title
     FROM popularity_fallback p
     JOIN movies m ON m.id = p.movie_id
     ORDER BY p.rank ASC`
  );
  return rows;
}

module.exports = { getRecommendationsForUser };
