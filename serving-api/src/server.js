require('dotenv').config();
const express = require('express');
const cors = require('cors');
const recommendationsRouter = require('./routes/recommendations');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => res.json({ status: 'ok' }));
app.use('/recommendations', recommendationsRouter);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`RecoServe API running on :${PORT}`));
