import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import StrategyLab from './pages/StrategyLab';
import AgentMonitor from './pages/AgentMonitor';
import GARLTraining from './pages/GARLTraining';
import './App.css';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
}

interface HealthStatus {
  status: string;
  timestamp: string;
  services: {
    backend: boolean;
    database: boolean;
    redis: boolean;
  };
}

// Dashboard Component
const Dashboard: React.FC<{
  marketData: MarketData[];
  healthStatus: HealthStatus | null;
  error: string | null;
}> = ({ marketData, healthStatus, error }) => {
  return (
    <main className="App-main">
      {error && (
        <div className="error-banner">
          <p>⚠️ {error}</p>
        </div>
      )}

      <section className="services-overview">
        <h2>System Services</h2>
        <div className="services-grid">
          <div className="service-card">
            <h3>🔧 C++ Backend</h3>
            <span className={`service-status ${healthStatus?.services?.backend ? 'online' : 'offline'}`}>
              {healthStatus?.services?.backend ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="service-card">
            <h3>🗄️ Database</h3>
            <span className={`service-status ${healthStatus?.services?.database ? 'online' : 'offline'}`}>
              {healthStatus?.services?.database ? 'Online' : 'Offline'}
            </span>
          </div>
          <div className="service-card">
            <h3>⚡ Redis Cache</h3>
            <span className={`service-status ${healthStatus?.services?.redis ? 'online' : 'offline'}`}>
              {healthStatus?.services?.redis ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </section>

      <section className="market-data">
        <h2>Market Data</h2>
        {marketData.length > 0 ? (
          <div className="market-grid">
            {marketData.map((item) => (
              <div key={item.symbol} className="market-card">
                <h3>{item.symbol}</h3>
                <div className="price">${item.price.toFixed(2)}</div>
                <div className={`change ${item.change >= 0 ? 'positive' : 'negative'}`}>
                  {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)} 
                  ({item.changePercent >= 0 ? '+' : ''}{item.changePercent.toFixed(2)}%)
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-data">
            <p>📊 No market data available. Backend services may be starting up...</p>
          </div>
        )}
      </section>

      <section className="architecture">
        <h2>System Architecture</h2>
        <div className="architecture-diagram">
          <div className="arch-component">
            <h4>Frontend (React)</h4>
            <p>This dashboard</p>
          </div>
          <div className="arch-arrow">→</div>
          <div className="arch-component">
            <h4>C++ Backend</h4>
            <p>High-performance trading engine</p>
          </div>
          <div className="arch-arrow">→</div>
          <div className="arch-component">
            <h4>Python Agents</h4>
            <p>ML & Analytics</p>
          </div>
          <div className="arch-arrow">→</div>
          <div className="arch-component">
            <h4>Rust GPU</h4>
            <p>GPU acceleration</p>
          </div>
        </div>
      </section>
    </main>
  );
};

function App() {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealthStatus = async () => {
      try {
        const response = await fetch('http://192.168.1.129:5000/health');
        if (response.ok) {
          const health = await response.json();
          setHealthStatus(health);
        }
      } catch (err) {
        console.warn('Health check failed:', err);
      }
    };

    const fetchMarketData = async () => {
      try {
        const response = await fetch('http://192.168.1.129:5000/api/market/data');
        if (response.ok) {
          const data = await response.json();
          setMarketData(data);
        }
      } catch (err) {
        setError('Failed to fetch market data');
        console.error('Market data fetch failed:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthStatus();
    fetchMarketData();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchHealthStatus();
      fetchMarketData();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="App">
        <div className="loading">
          <h2>Loading Stock Market Dashboard...</h2>
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="App">
        <Header />

        <div className="status-bar">
          <span className={`status-indicator ${healthStatus?.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
            {healthStatus?.status === 'healthy' ? '🟢' : '🔴'} 
            {healthStatus?.status?.toUpperCase() || 'UNKNOWN'}
          </span>
          {healthStatus?.timestamp && (
            <span className="timestamp">
              Last updated: {new Date(healthStatus.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>

        <Routes>
          <Route 
            path="/" 
            element={<Dashboard marketData={marketData} healthStatus={healthStatus} error={error} />} 
          />
          <Route path="/strategy-lab" element={<StrategyLab />} />
          <Route path="/agent-monitor" element={<AgentMonitor />} />
          <Route path="/ga-rl-training" element={<GARLTraining />} />
        </Routes>

        <footer className="App-footer">
          <p>Stock Market Lab - Multi-language Architecture Demo</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;