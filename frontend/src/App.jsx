import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:4000';

export default function App() {
  const [userId, setUserId] = useState('1');
  const [recs, setRecs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [latencyMs, setLatencyMs] = useState(null);

  async function fetchRecommendations(e) {
    e.preventDefault();
    setLoading(true);
    setError('');
    const start = performance.now();
    try {
      const { data } = await axios.get(`${API_URL}/recommendations/${userId}`);
      setRecs(data.recommendations);
      setLatencyMs(Math.round(performance.now() - start));
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch recommendations');
      setRecs(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>RecoServe</h1>
      <p className="subtitle">Enter a user ID to see their precomputed, cached recommendations.</p>

      <form onSubmit={fetchRecommendations} className="lookup-form">
        <input
          type="number"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="User ID"
          min="1"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {latencyMs !== null && (
        <p className="latency">Response time: {latencyMs}ms (first call warms the cache; try again to see the cached speed)</p>
      )}

      {recs && (
        <div className="recs-grid">
          {recs.map((r) => (
            <div className="rec-card" key={r.movie_id}>
              <h3>{r.title}</h3>
              {r.score !== undefined && r.score !== null && (
                <p className="score">predicted rating: {Number(r.score).toFixed(2)}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
