const express = require('express');
const { getRecommendationsForUser } = require('../services/recommendationService');

const router = express.Router();

router.get('/:userId', async (req, res) => {
  try {
    const recs = await getRecommendationsForUser(req.params.userId);
    res.json({ userId: req.params.userId, recommendations: recs });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to fetch recommendations' });
  }
});

module.exports = router;
