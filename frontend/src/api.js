import axios from 'axios';

const api = axios.create({
  baseURL: '/',
  timeout: 120000,
});

export async function fetchMarketAnalysis(days) {
  const params = days ? { days } : {};
  const response = await api.get('/api/market-analysis', { params });
  return response.data;
}

export async function fetchPricePrediction({ horizon, lookback, frequency }) {
  const params = {
    horizon,
    lookback,
    frequency,
  };
  const response = await api.get('/api/price-prediction', { params });
  return response.data;
}

export async function fetchDirectionPrediction({ lookback, threshold, horizon }) {
  const params = {
    lookback,
    threshold,
    horizon,
  };
  const response = await api.get('/api/direction-prediction', { params });
  return response.data;
}

export async function analyzeSentimentText(text) {
  const response = await api.post('/api/sentiment-analysis', { text });
  return response.data;
}

export async function fetchRiskClustering({ days, seed }) {
  const params = {
    days,
    seed,
  };
  const response = await api.get('/api/risk-clustering', { params });
  return response.data;
}

export async function askChatbot({ message, history }) {
  const response = await api.post('/api/chatbot', {
    message,
    history,
  });
  return response.data;
}

export async function analyzeImageNews({ image, question }) {
  const formData = new FormData();
  formData.append('image', image);
  if (question) {
    formData.append('question', question);
  }

  const response = await api.post('/api/image-analysis', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}
