import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip as ChartTooltip, Legend as ChartLegend } from 'chart.js';
import './Dashboard.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend
);

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
}

interface AgentDecision {
  id: string;
  type: string;
  action: string;
  confidence: number;
  reason: string;
  timestamp: number;
}

interface PricePrediction {
  id: string;
  predictedPrice: number;
  confidence: number;
  direction: string;
  timestamp: number;
}

interface SentimentData {
  id: string;
  sentiment: string;
  confidence: number;
  topics: string[];
  timestamp: number;
}

interface DashboardProps {
  marketData: MarketData | null;
  agentDecisions: AgentDecision[];
  pricePredictions: PricePrediction[];
  sentimentData: SentimentData[];
}

const Dashboard: React.FC<DashboardProps> = ({ 
  marketData, 
  agentDecisions, 
  pricePredictions, 
  sentimentData 
}) => {
  const [livePriceData, setLivePriceData] = useState<number[]>([170, 172, 171, 173, 175, 174, 176]);
  const [timeLabels, setTimeLabels] = useState<string[]>(['09:30', '09:45', '10:00', '10:15', '10:30', '10:45', '11:00']);

  // Generate mock live price data
  useEffect(() => {
    const interval = setInterval(() => {
      setLivePriceData(prev => {
        const newData = [...prev.slice(1), prev[prev.length - 1] + (Math.random() - 0.5)];
        return newData;
      });
      
      setTimeLabels(prev => {
        const now = new Date();
        const minutes = now.getMinutes();
        const hours = now.getHours();
        const newLabel = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
        return [...prev.slice(1), newLabel];
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Chart data for live prices
  const priceChartData = {
    labels: timeLabels,
    datasets: [
      {
        label: 'AAPL Price',
        data: livePriceData,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Live Price Chart',
      },
    },
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Market Overview</h2>
      </div>

      {/* Market Summary Card */}
      {marketData && (
        <div className="market-summary">
          <div className="market-card">
            <h3>{marketData.symbol}</h3>
            <div className="price-info">
              <span className="current-price">${marketData.price.toFixed(2)}</span>
              <span className={`change ${marketData.change >= 0 ? 'positive' : 'negative'}`}>
                {marketData.change >= 0 ? '+' : ''}{marketData.change.toFixed(2)} ({marketData.changePercent.toFixed(2)}%)
              </span>
            </div>
            <div className="volume">Volume: {marketData.volume.toLocaleString()}</div>
          </div>
        </div>
      )}

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={priceChartData} options={chartOptions}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#8884d8" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Agent Decisions Section */}
      <div className="agents-section">
        <h3>Agent Decisions</h3>
        <div className="decisions-grid">
          {agentDecisions.map((decision) => (
            <div key={decision.id} className="decision-card">
              <h4>{decision.type} Agent</h4>
              <p>Action: <strong>{decision.action}</strong></p>
              <p>Confidence: {(decision.confidence * 100).toFixed(0)}%</p>
              <p>Reason: {decision.reason}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Predictions Section */}
      <div className="predictions-section">
        <h3>Price Predictions</h3>
        <div className="predictions-grid">
          {pricePredictions.map((prediction) => (
            <div key={prediction.id} className="prediction-card">
              <h4>LSTM Prediction</h4>
              <p>Predicted Price: <strong>${prediction.predictedPrice.toFixed(2)}</strong></p>
              <p>Confidence: {(prediction.confidence * 100).toFixed(0)}%</p>
              <p>Direction: <span className={prediction.direction}>{prediction.direction}</span></p>
            </div>
          ))}
        </div>
      </div>

      {/* Sentiment Section */}
      <div className="sentiment-section">
        <h3>Sentiment Analysis</h3>
        <div className="sentiment-grid">
          {sentimentData.map((sentiment) => (
            <div key={sentiment.id} className="sentiment-card">
              <h4>News/NLP Agent</h4>
              <p>Sentiment: <span className={sentiment.sentiment}>{sentiment.sentiment}</span></p>
              <p>Confidence: {(sentiment.confidence * 100).toFixed(0)}%</p>
              <p>Topics: {sentiment.topics.join(', ')}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;