import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import StrategyLab from './pages/StrategyLab';
import AgentMonitor from './pages/AgentMonitor';
import './App.css';

function App() {
  const [marketData, setMarketData] = useState<any>(null);
  const [agentDecisions, setAgentDecisions] = useState<any[]>([]);
  const [pricePredictions, setPricePredictions] = useState<any[]>([]);
  const [sentimentData, setSentimentData] = useState<any[]>([]);

  // Mock data initialization
  useEffect(() => {
    // Initialize with mock data
    setMarketData({
      symbol: 'AAPL',
      price: 175.23,
      change: 1.25,
      changePercent: 0.72,
      volume: 12500000
    });

    // Mock agent decisions
    setAgentDecisions([
      {
        id: 'rl-001',
        type: 'RL',
        action: 'buy',
        confidence: 0.85,
        reason: 'Strong momentum detected',
        timestamp: Date.now()
      },
      {
        id: 'lstm-001',
        type: 'LSTM',
        action: 'hold',
        confidence: 0.72,
        reason: 'Price stability expected',
        timestamp: Date.now()
      }
    ]);

    // Mock predictions
    setPricePredictions([
      {
        id: 'lstm-001',
        predictedPrice: 176.50,
        confidence: 0.75,
        direction: 'up',
        timestamp: Date.now()
      }
    ]);

    // Mock sentiment
    setSentimentData([
      {
        id: 'news-001',
        sentiment: 'positive',
        confidence: 0.85,
        topics: ['earnings', 'market'],
        timestamp: Date.now()
      }
    ]);
  }, []);

  return (
    <Router>
      <div className="App">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard 
              marketData={marketData}
              agentDecisions={agentDecisions}
              pricePredictions={pricePredictions}
              sentimentData={sentimentData}
            />} />
            <Route path="/strategy-lab" element={<StrategyLab />} />
            <Route path="/agent-monitor" element={<AgentMonitor />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;