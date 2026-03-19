import { useState } from 'react';
import ChatbotWidget from './components/ChatbotWidget';
import Sidebar from './components/Sidebar';
import PlaceholderModule from './components/PlaceholderModule';
import DirectionPredictionPage from './pages/DirectionPredictionPage';
import ImageAnalysisPage from './pages/ImageAnalysisPage';
import MarketAnalysisPage from './pages/MarketAnalysisPage';
import PricePredictionPage from './pages/PricePredictionPage';
import RiskClusteringPage from './pages/RiskClusteringPage';
import SentimentAnalysisPage from './pages/SentimentAnalysisPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('market-analysis');

  return (
    <div className="min-h-screen bg-canvas p-4 md:p-6">
      <div className="mx-auto grid w-full max-w-[1500px] gap-6 lg:grid-cols-[290px_minmax(0,1fr)]">
        <Sidebar activeTab={activeTab} onChangeTab={setActiveTab} />

        <main className="space-y-6">
          {activeTab === 'market-analysis' && <MarketAnalysisPage />}
          {activeTab === 'price-prediction' && <PricePredictionPage />}
          {activeTab === 'direction-prediction' && <DirectionPredictionPage />}
          {activeTab === 'sentiment-analysis' && <SentimentAnalysisPage />}
          {activeTab === 'risk-clustering' && <RiskClusteringPage />}
          {activeTab === 'image-analysis' && <ImageAnalysisPage />}
        </main>
      </div>

      <ChatbotWidget />
    </div>
  );
}
