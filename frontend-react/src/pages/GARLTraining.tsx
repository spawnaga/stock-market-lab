import React, { useState, useEffect, useCallback } from 'react';
import apiService, {
  GARLStatus,
  GARLHistory,
  GARLChromosome,
  GARLSignal
} from '../services/api';
import './GARLTraining.css';

interface TrainingConfig {
  symbol: string;
  population_size: number;
  num_generations: number;
  start_date: string;
  end_date: string;
}

const GARLTraining: React.FC = () => {
  const [status, setStatus] = useState<GARLStatus | null>(null);
  const [history, setHistory] = useState<GARLHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [config, setConfig] = useState<TrainingConfig>({
    symbol: 'AAPL',
    population_size: 20,
    num_generations: 50,
    start_date: '',
    end_date: ''
  });

  const [testSignal, setTestSignal] = useState<GARLSignal | null>(null);

  // Fetch status
  const fetchStatus = useCallback(async () => {
    try {
      const data = await apiService.getGARLStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, []);

  // Fetch history
  const fetchHistory = useCallback(async () => {
    try {
      const data = await apiService.getGARLHistory();
      setHistory(data);
    } catch {
      // History may not be available if not initialized
    }
  }, []);

  // Initialize system
  const handleInitialize = async () => {
    try {
      setError(null);
      await apiService.initializeGARL({
        symbol: config.symbol,
        population_size: config.population_size,
        num_generations: config.num_generations
      });
      setSuccessMessage('GA+RL system initialized successfully!');
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize');
    }
  };

  // Start training
  const handleStartTraining = async () => {
    try {
      setError(null);
      await apiService.startGARLTraining({
        symbol: config.symbol,
        start_date: config.start_date || undefined,
        end_date: config.end_date || undefined
      });
      setSuccessMessage('Training started! Monitor progress below.');
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training');
    }
  };

  // Stop training
  const handleStopTraining = async () => {
    try {
      setError(null);
      await apiService.stopGARLTraining();
      setSuccessMessage('Training stop signal sent.');
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop training');
    }
  };

  // Test signal
  const handleTestSignal = async () => {
    try {
      setError(null);
      const signal = await apiService.getGARLSignal({
        price_change_1d: Math.random() * 0.04 - 0.02,
        price_change_5d: Math.random() * 0.1 - 0.05,
        price_change_20d: Math.random() * 0.2 - 0.1,
        volume_ratio: 0.8 + Math.random() * 0.4,
        rsi: 30 + Math.random() * 40,
        macd: Math.random() * 2 - 1,
        macd_signal: Math.random() * 2 - 1,
        bb_position: Math.random(),
        position: 0,
        portfolio_value_change: 0,
        time_in_position: 0
      });
      setTestSignal(signal);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get signal');
    }
  };

  // Initial load
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await fetchStatus();
      await fetchHistory();
      setLoading(false);
    };
    load();

    // Poll for updates when training
    const interval = setInterval(() => {
      fetchStatus();
      if (status?.is_training) {
        fetchHistory();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchHistory, status?.is_training]);

  // Clear messages after timeout
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => setSuccessMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  if (loading) {
    return (
      <div className="garl-training">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading GA+RL Training System...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="garl-training">
      <h1>GA+RL Training System</h1>
      <p className="description">
        Train AI trading agents using Genetic Algorithm optimized Deep Q-Networks.
        The GA evolves neural network hyperparameters while DQN learns trading decisions.
      </p>

      {error && <div className="error-message">{error}</div>}
      {successMessage && <div className="success-message">{successMessage}</div>}

      {/* System Status */}
      <section className="status-section">
        <h2>System Status</h2>
        <div className="status-grid">
          <div className="status-item">
            <label>Module Available:</label>
            <span className={status?.available ? 'status-ok' : 'status-error'}>
              {status?.available ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="status-item">
            <label>Initialized:</label>
            <span className={status?.initialized ? 'status-ok' : 'status-warning'}>
              {status?.initialized ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="status-item">
            <label>Training:</label>
            <span className={status?.is_training ? 'status-active' : 'status-idle'}>
              {status?.is_training ? 'In Progress' : 'Idle'}
            </span>
          </div>
          <div className="status-item">
            <label>Trained Agent:</label>
            <span className={status?.has_trained_agent ? 'status-ok' : 'status-warning'}>
              {status?.has_trained_agent ? 'Available' : 'Not Available'}
            </span>
          </div>
        </div>
      </section>

      {/* Configuration */}
      <section className="config-section">
        <h2>Training Configuration</h2>
        <div className="config-form">
          <div className="form-group">
            <label>Symbol:</label>
            <input
              type="text"
              value={config.symbol}
              onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
              placeholder="AAPL"
              disabled={status?.is_training}
            />
          </div>
          <div className="form-group">
            <label>Population Size:</label>
            <input
              type="number"
              value={config.population_size}
              onChange={(e) => setConfig({ ...config, population_size: parseInt(e.target.value) || 20 })}
              min={5}
              max={100}
              disabled={status?.is_training}
            />
          </div>
          <div className="form-group">
            <label>Generations:</label>
            <input
              type="number"
              value={config.num_generations}
              onChange={(e) => setConfig({ ...config, num_generations: parseInt(e.target.value) || 50 })}
              min={5}
              max={200}
              disabled={status?.is_training}
            />
          </div>
          <div className="form-group">
            <label>Start Date (optional):</label>
            <input
              type="date"
              value={config.start_date}
              onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
              disabled={status?.is_training}
            />
          </div>
          <div className="form-group">
            <label>End Date (optional):</label>
            <input
              type="date"
              value={config.end_date}
              onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
              disabled={status?.is_training}
            />
          </div>
        </div>

        <div className="action-buttons">
          {!status?.initialized && (
            <button className="btn-primary" onClick={handleInitialize}>
              Initialize System
            </button>
          )}
          {status?.initialized && !status?.is_training && (
            <button className="btn-primary" onClick={handleStartTraining}>
              Start Training
            </button>
          )}
          {status?.is_training && (
            <button className="btn-danger" onClick={handleStopTraining}>
              Stop Training
            </button>
          )}
        </div>
      </section>

      {/* Training Progress */}
      {status?.training_progress && (
        <section className="progress-section">
          <h2>Training Progress</h2>
          <div className="progress-info">
            <div className="progress-item">
              <label>Generation:</label>
              <span>{status.training_progress.generation} / {status.training_progress.total_generations}</span>
            </div>
            <div className="progress-item">
              <label>Chromosome:</label>
              <span>{status.training_progress.chromosome} / {status.training_progress.total_chromosomes}</span>
            </div>
            <div className="progress-item">
              <label>Current Fitness:</label>
              <span>{status.training_progress.current_fitness?.toFixed(4)}</span>
            </div>
            <div className="progress-item">
              <label>Best Fitness:</label>
              <span className="highlight">{status.training_progress.best_fitness?.toFixed(4)}</span>
            </div>
          </div>
          <div className="progress-bar-container">
            <div
              className="progress-bar"
              style={{
                width: `${(status.training_progress.generation / status.training_progress.total_generations) * 100}%`
              }}
            ></div>
          </div>
        </section>
      )}

      {/* Evolution History Chart */}
      {history && history.history.length > 0 && (
        <section className="history-section">
          <h2>Evolution History</h2>
          <div className="history-chart">
            <div className="chart-container">
              {history.history.map((gen, idx) => (
                <div
                  key={idx}
                  className="chart-bar"
                  style={{
                    height: `${Math.max(5, (gen.best_fitness / Math.max(...history.history.map(h => h.best_fitness))) * 100)}%`,
                  }}
                  title={`Gen ${gen.generation}: Best=${gen.best_fitness.toFixed(4)}, Avg=${gen.avg_fitness.toFixed(4)}`}
                >
                  <span className="bar-label">{gen.generation}</span>
                </div>
              ))}
            </div>
            <div className="chart-legend">
              <span>Generation</span>
            </div>
          </div>
          <div className="history-stats">
            <div className="stat">
              <label>Generations Completed:</label>
              <span>{history.generations_completed}</span>
            </div>
            <div className="stat">
              <label>Best Sharpe Ratio:</label>
              <span>{history.best_chromosome?.sharpe_ratio?.toFixed(4) || 'N/A'}</span>
            </div>
            <div className="stat">
              <label>Best Return:</label>
              <span>{history.best_chromosome?.total_return ? (history.best_chromosome.total_return * 100).toFixed(2) + '%' : 'N/A'}</span>
            </div>
          </div>
        </section>
      )}

      {/* Best Chromosome Details */}
      {history?.best_chromosome && (
        <section className="chromosome-section">
          <h2>Best Evolved Configuration</h2>
          <div className="chromosome-grid">
            <div className="chromosome-group">
              <h3>Network Architecture</h3>
              <div className="param">
                <label>Hidden Size:</label>
                <span>{history.best_chromosome.hidden_size}</span>
              </div>
              <div className="param">
                <label>Num Layers:</label>
                <span>{history.best_chromosome.num_layers}</span>
              </div>
              <div className="param">
                <label>Dueling Architecture:</label>
                <span>{history.best_chromosome.use_dueling ? 'Yes' : 'No'}</span>
              </div>
            </div>
            <div className="chromosome-group">
              <h3>Training Hyperparameters</h3>
              <div className="param">
                <label>Learning Rate:</label>
                <span>{history.best_chromosome.learning_rate?.toExponential(2)}</span>
              </div>
              <div className="param">
                <label>Gamma:</label>
                <span>{history.best_chromosome.gamma?.toFixed(4)}</span>
              </div>
              <div className="param">
                <label>Batch Size:</label>
                <span>{history.best_chromosome.batch_size}</span>
              </div>
            </div>
            <div className="chromosome-group">
              <h3>Trading Parameters</h3>
              <div className="param">
                <label>Position Size:</label>
                <span>{(history.best_chromosome.position_size * 100).toFixed(1)}%</span>
              </div>
              <div className="param">
                <label>Stop Loss:</label>
                <span>{(history.best_chromosome.stop_loss * 100).toFixed(1)}%</span>
              </div>
              <div className="param">
                <label>Take Profit:</label>
                <span>{(history.best_chromosome.take_profit * 100).toFixed(1)}%</span>
              </div>
            </div>
            <div className="chromosome-group">
              <h3>Performance Metrics</h3>
              <div className="param">
                <label>Fitness:</label>
                <span className="highlight">{history.best_chromosome.fitness?.toFixed(4)}</span>
              </div>
              <div className="param">
                <label>Sharpe Ratio:</label>
                <span>{history.best_chromosome.sharpe_ratio?.toFixed(4)}</span>
              </div>
              <div className="param">
                <label>Win Rate:</label>
                <span>{(history.best_chromosome.win_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Trading Signal Test */}
      {status?.has_trained_agent && (
        <section className="signal-section">
          <h2>Test Trading Signal</h2>
          <p>Generate a trading signal using the trained agent with random market conditions.</p>
          <button className="btn-secondary" onClick={handleTestSignal}>
            Generate Signal
          </button>
          {testSignal && (
            <div className="signal-result">
              <div className={`signal-action signal-${testSignal.signal.action}`}>
                {testSignal.signal.action.toUpperCase()}
              </div>
              <div className="signal-details">
                <div className="detail">
                  <label>Confidence:</label>
                  <span>{(testSignal.signal.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="detail">
                  <label>Q-Values:</label>
                  <span>
                    Buy: {testSignal.signal.q_values.buy?.toFixed(3)},
                    Sell: {testSignal.signal.q_values.sell?.toFixed(3)},
                    Hold: {testSignal.signal.q_values.hold?.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {/* How It Works */}
      <section className="info-section">
        <h2>How GA+RL Works</h2>
        <div className="info-grid">
          <div className="info-card">
            <h3>1. Genetic Algorithm</h3>
            <p>
              Evolves a population of DQN configurations. Each chromosome encodes
              neural network architecture, training hyperparameters, and trading strategy parameters.
            </p>
          </div>
          <div className="info-card">
            <h3>2. DQN Training</h3>
            <p>
              Each chromosome is used to create and train a Deep Q-Network agent
              that learns to make buy/sell/hold decisions from market data.
            </p>
          </div>
          <div className="info-card">
            <h3>3. Fitness Evaluation</h3>
            <p>
              Agents are evaluated on Sharpe ratio, total return, max drawdown, and win rate.
              The best performers survive to create the next generation.
            </p>
          </div>
          <div className="info-card">
            <h3>4. Evolution</h3>
            <p>
              Tournament selection, crossover, and mutation create new generations.
              Over time, the algorithm discovers optimal configurations.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default GARLTraining;
