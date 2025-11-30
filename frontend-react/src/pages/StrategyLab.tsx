import React, { useState, useEffect, useCallback } from 'react';
import { apiService, Strategy, BacktestResult } from '../services/api';
import { wsService } from '../services/websocket';
import './StrategyLab.css';

const StrategyLab: React.FC = () => {
  // Form state
  const [strategyName, setStrategyName] = useState('');
  const [strategyDescription, setStrategyDescription] = useState('');
  const [strategySymbol, setStrategySymbol] = useState('AAPL');
  const [strategyParameters, setStrategyParameters] = useState<{[key: string]: any}>({
    lookback_period: 14,
    threshold: 0.02,
    max_positions: 5
  });

  // Data state
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtestStrategies, setBacktestStrategies] = useState<{name: string; type: string}[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isBacktesting, setIsBacktesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Backtest form state
  const [backtestSymbol, setBacktestSymbol] = useState('AAPL');
  const [backtestStartDate, setBacktestStartDate] = useState('2023-01-01');
  const [backtestEndDate, setBacktestEndDate] = useState('2024-01-01');
  const [backtestCapital, setBacktestCapital] = useState(10000);
  const [selectedBacktestStrategy, setSelectedBacktestStrategy] = useState('');

  // Auto-login with demo credentials
  useEffect(() => {
    const autoLogin = async () => {
      if (!apiService.isAuthenticated()) {
        try {
          await apiService.login('admin', 'admin');
          setIsLoggedIn(true);
        } catch (err) {
          console.error('Auto-login failed:', err);
          setError('Failed to authenticate. Please refresh.');
        }
      } else {
        setIsLoggedIn(true);
      }
    };
    autoLogin();
  }, []);

  // Load strategies from API
  const loadStrategies = useCallback(async () => {
    if (!isLoggedIn) return;

    try {
      setIsLoading(true);
      const response = await apiService.getStrategies();
      setStrategies(response.strategies || []);
    } catch (err) {
      console.error('Failed to load strategies:', err);
      setError('Failed to load strategies');
    } finally {
      setIsLoading(false);
    }
  }, [isLoggedIn]);

  // Load backtest strategies
  const loadBacktestStrategies = useCallback(async () => {
    if (!isLoggedIn) return;

    try {
      const response = await apiService.getBacktestStrategies();
      setBacktestStrategies(response.strategies || []);
      if (response.strategies?.length > 0) {
        setSelectedBacktestStrategy(response.strategies[0].name);
      }
    } catch (err) {
      console.error('Failed to load backtest strategies:', err);
    }
  }, [isLoggedIn]);

  useEffect(() => {
    loadStrategies();
    loadBacktestStrategies();
  }, [loadStrategies, loadBacktestStrategies]);

  // WebSocket connection
  useEffect(() => {
    wsService.connect().catch(console.error);

    const unsubscribe = wsService.onStrategyCreated((newStrategy) => {
      setStrategies(prev => {
        // Avoid duplicates
        if (prev.find(s => s.id === newStrategy.id)) {
          return prev;
        }
        return [newStrategy as unknown as Strategy, ...prev];
      });
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const handleParameterChange = (paramName: string, value: any) => {
    setStrategyParameters(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    setError(null);

    try {
      const newStrategy = await apiService.createStrategy({
        name: strategyName,
        description: strategyDescription,
        symbol: strategySymbol,
        parameters: strategyParameters,
        strategy_type: 'custom'
      });

      // Add to list if not already there (WebSocket might have added it)
      setStrategies(prev => {
        if (prev.find(s => s.id === newStrategy.id)) {
          return prev;
        }
        return [newStrategy, ...prev];
      });

      // Reset form
      setStrategyName('');
      setStrategyDescription('');
      setStrategyParameters({
        lookback_period: 14,
        threshold: 0.02,
        max_positions: 5
      });
    } catch (err: any) {
      setError(err.message || 'Failed to create strategy');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (strategyId: number) => {
    if (!window.confirm('Are you sure you want to delete this strategy?')) {
      return;
    }

    try {
      await apiService.deleteStrategy(strategyId);
      setStrategies(prev => prev.filter(s => s.id !== strategyId));
      if (selectedStrategy?.id === strategyId) {
        setSelectedStrategy(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to delete strategy');
    }
  };

  const handleRunBacktest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedBacktestStrategy) {
      setError('Please select a strategy');
      return;
    }

    setIsBacktesting(true);
    setError(null);
    setBacktestResult(null);

    try {
      const result = await apiService.runBacktest({
        strategy_name: selectedBacktestStrategy,
        symbol: backtestSymbol,
        start_date: new Date(backtestStartDate).toISOString(),
        end_date: new Date(backtestEndDate).toISOString(),
        initial_capital: backtestCapital
      });
      setBacktestResult(result);
    } catch (err: any) {
      setError(err.message || 'Backtest failed');
    } finally {
      setIsBacktesting(false);
    }
  };

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatCurrency = (value: number) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  if (!isLoggedIn) {
    return (
      <div className="strategy-lab">
        <div className="loading">Authenticating...</div>
      </div>
    );
  }

  return (
    <div className="strategy-lab">
      <div className="strategy-lab-header">
        <h2>Strategy Lab</h2>
        <p>Create, manage, and backtest trading strategies</p>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="strategy-lab-content">
        {/* Strategy Creation Section */}
        <div className="strategy-form-section">
          <h3>Create New Strategy</h3>
          <form onSubmit={handleSubmit} className="strategy-form">
            <div className="form-group">
              <label htmlFor="strategyName">Strategy Name:</label>
              <input
                type="text"
                id="strategyName"
                value={strategyName}
                onChange={(e) => setStrategyName(e.target.value)}
                placeholder="e.g., Momentum Trader"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="strategyDescription">Description:</label>
              <textarea
                id="strategyDescription"
                value={strategyDescription}
                onChange={(e) => setStrategyDescription(e.target.value)}
                placeholder="Describe your strategy..."
                rows={3}
              />
            </div>

            <div className="form-group">
              <label htmlFor="strategySymbol">Target Symbol:</label>
              <input
                type="text"
                id="strategySymbol"
                value={strategySymbol}
                onChange={(e) => setStrategySymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
              />
            </div>

            <div className="form-group">
              <label>Parameters:</label>
              <div className="parameters-grid">
                <div className="parameter-item">
                  <label htmlFor="lookbackPeriod">Lookback Period:</label>
                  <input
                    type="number"
                    id="lookbackPeriod"
                    value={strategyParameters.lookback_period || ''}
                    onChange={(e) => handleParameterChange('lookback_period', parseInt(e.target.value))}
                    min="1"
                    max="365"
                  />
                </div>
                <div className="parameter-item">
                  <label htmlFor="threshold">Threshold:</label>
                  <input
                    type="number"
                    id="threshold"
                    value={strategyParameters.threshold || ''}
                    onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
                    step="0.01"
                    min="0"
                  />
                </div>
                <div className="parameter-item">
                  <label htmlFor="maxPositions">Max Positions:</label>
                  <input
                    type="number"
                    id="maxPositions"
                    value={strategyParameters.max_positions || ''}
                    onChange={(e) => handleParameterChange('max_positions', parseInt(e.target.value))}
                    min="1"
                    max="20"
                  />
                </div>
              </div>
            </div>

            <button type="submit" disabled={isCreating} className="submit-btn">
              {isCreating ? 'Creating...' : 'Create Strategy'}
            </button>
          </form>
        </div>

        {/* Strategy List Section */}
        <div className="strategy-list-section">
          <h3>Your Strategies {isLoading && <span className="loading-indicator">(Loading...)</span>}</h3>
          <div className="strategies-grid">
            {strategies.length === 0 && !isLoading ? (
              <p className="no-strategies">No strategies yet. Create your first one!</p>
            ) : (
              strategies.map((strategy) => (
                <div
                  key={strategy.id}
                  className={`strategy-card ${selectedStrategy?.id === strategy.id ? 'selected' : ''}`}
                  onClick={() => setSelectedStrategy(strategy)}
                >
                  <div className="strategy-card-header">
                    <h4>{strategy.name}</h4>
                    <button
                      className="delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(strategy.id);
                      }}
                      title="Delete strategy"
                    >
                      X
                    </button>
                  </div>
                  <p className="strategy-description">{strategy.description || 'No description'}</p>
                  <div className="strategy-meta">
                    <span className="symbol-tag">{strategy.symbol || 'N/A'}</span>
                    <span className="created-at">
                      {strategy.created_at ? new Date(strategy.created_at).toLocaleDateString() : 'Unknown'}
                    </span>
                  </div>
                  {strategy.parameters && (
                    <div className="strategy-params">
                      {Object.entries(strategy.parameters).map(([key, value]) => (
                        <span key={key} className="param-tag">
                          {`${key}: ${String(value)}`}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Backtest Section */}
      <div className="backtest-section">
        <h3>Run Backtest</h3>
        <form onSubmit={handleRunBacktest} className="backtest-form">
          <div className="backtest-inputs">
            <div className="form-group">
              <label htmlFor="backtestStrategy">Strategy:</label>
              <select
                id="backtestStrategy"
                value={selectedBacktestStrategy}
                onChange={(e) => setSelectedBacktestStrategy(e.target.value)}
                required
              >
                {backtestStrategies.map((s) => (
                  <option key={s.name} value={s.name}>{s.name}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label htmlFor="backtestSymbol">Symbol:</label>
              <input
                type="text"
                id="backtestSymbol"
                value={backtestSymbol}
                onChange={(e) => setBacktestSymbol(e.target.value.toUpperCase())}
              />
            </div>
            <div className="form-group">
              <label htmlFor="backtestStart">Start Date:</label>
              <input
                type="date"
                id="backtestStart"
                value={backtestStartDate}
                onChange={(e) => setBacktestStartDate(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label htmlFor="backtestEnd">End Date:</label>
              <input
                type="date"
                id="backtestEnd"
                value={backtestEndDate}
                onChange={(e) => setBacktestEndDate(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label htmlFor="backtestCapital">Initial Capital:</label>
              <input
                type="number"
                id="backtestCapital"
                value={backtestCapital}
                onChange={(e) => setBacktestCapital(parseInt(e.target.value))}
                min="1000"
              />
            </div>
          </div>
          <button type="submit" disabled={isBacktesting} className="backtest-btn">
            {isBacktesting ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </form>

        {/* Backtest Results */}
        {backtestResult && (
          <div className="backtest-results">
            <h4>Backtest Results: {backtestResult.strategy_name}</h4>
            <div className="results-grid">
              <div className="result-card">
                <span className="result-label">Total Return</span>
                <span className={`result-value ${backtestResult.total_return >= 0 ? 'positive' : 'negative'}`}>
                  {formatPercent(backtestResult.total_return)}
                </span>
              </div>
              <div className="result-card">
                <span className="result-label">Sharpe Ratio</span>
                <span className="result-value">{backtestResult.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="result-card">
                <span className="result-label">Max Drawdown</span>
                <span className="result-value negative">{formatPercent(backtestResult.max_drawdown || 0)}</span>
              </div>
              <div className="result-card">
                <span className="result-label">Total Trades</span>
                <span className="result-value">{backtestResult.trade_count}</span>
              </div>
              <div className="result-card">
                <span className="result-label">Win Rate</span>
                <span className="result-value">{formatPercent(backtestResult.win_rate || 0)}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Strategy Details Modal */}
      {selectedStrategy && (
        <div className="strategy-details-section">
          <h3>Strategy Details</h3>
          <button className="close-btn" onClick={() => setSelectedStrategy(null)}>Close</button>
          <div className="strategy-details">
            <div className="detail-row">
              <label>Name:</label>
              <span>{selectedStrategy.name}</span>
            </div>
            <div className="detail-row">
              <label>Description:</label>
              <span>{selectedStrategy.description || 'No description'}</span>
            </div>
            <div className="detail-row">
              <label>Symbol:</label>
              <span>{selectedStrategy.symbol}</span>
            </div>
            <div className="detail-row">
              <label>Created:</label>
              <span>{selectedStrategy.created_at ? new Date(selectedStrategy.created_at).toLocaleString() : 'Unknown'}</span>
            </div>
            <div className="detail-row">
              <label>Parameters:</label>
              <div className="parameters-list">
                {selectedStrategy.parameters && Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                  <div key={key} className="param-item">
                    <span className="param-name">{key}:</span>
                    <span className="param-value">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyLab;
