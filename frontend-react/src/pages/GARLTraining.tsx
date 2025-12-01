import React, { useState, useEffect, useCallback, useRef } from 'react';
import apiService, {
  GARLStatus,
  GARLHistory,
  GARLSignal
} from '../services/api';
import { wsService } from '../services/websocket';
import './GARLTraining.css';

interface TrainingConfig {
  symbol: string;
  population_size: number;
  num_generations: number;
  start_date: string;
  end_date: string;
}

interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

interface TrainingMetrics {
  currentGeneration: number;
  totalGenerations: number;
  currentChromosome: number;
  totalChromosomes: number;
  bestFitness: number;
  avgFitness: number;
  worstFitness: number;
  elapsedTime: number;
  estimatedTimeRemaining: number;
  fitnessImprovement: number;
}

const GARLTraining: React.FC = () => {
  const [status, setStatus] = useState<GARLStatus | null>(null);
  const [history, setHistory] = useState<GARLHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const [config, setConfig] = useState<TrainingConfig>({
    symbol: 'AAPL',
    population_size: 20,
    num_generations: 50,
    start_date: '',
    end_date: ''
  });

  const [testSignal, setTestSignal] = useState<GARLSignal | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'config' | 'logs' | 'analysis'>('overview');
  const [autoScroll, setAutoScroll] = useState(true);
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const trainingStartTime = useRef<number | null>(null);

  // Add log entry
  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const entry: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      level,
      message
    };
    setLogs(prev => [...prev.slice(-199), entry]); // Keep last 200 logs
  }, []);

  // Auto-login with demo credentials
  useEffect(() => {
    const autoLogin = async () => {
      if (!apiService.isAuthenticated()) {
        try {
          await apiService.login('admin', 'admin');
          setIsLoggedIn(true);
          addLog('success', 'Successfully authenticated');
        } catch (err) {
          console.error('Auto-login failed:', err);
          setError('Failed to authenticate. Please refresh the page.');
          addLog('error', 'Authentication failed');
        }
      } else {
        setIsLoggedIn(true);
        addLog('info', 'Using existing authentication');
      }
    };
    autoLogin();
  }, [addLog]);

  // Fetch status
  const fetchStatus = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const data = await apiService.getGARLStatus();
      setStatus(data);

      // Update metrics from status
      if (data.training_progress) {
        const prog = data.training_progress;
        const elapsed = prog.elapsed_time || 0;
        const progress = prog.generation / prog.total_generations;
        const estimatedTotal = progress > 0 ? elapsed / progress : 0;

        setMetrics(prev => ({
          currentGeneration: prog.generation,
          totalGenerations: prog.total_generations,
          currentChromosome: prog.chromosome,
          totalChromosomes: prog.total_chromosomes,
          bestFitness: prog.best_fitness,
          avgFitness: prev?.avgFitness || 0,
          worstFitness: prev?.worstFitness || 0,
          elapsedTime: elapsed,
          estimatedTimeRemaining: Math.max(0, estimatedTotal - elapsed),
          fitnessImprovement: prev ? prog.best_fitness - prev.bestFitness : 0
        }));
      }

      setError(null);
    } catch (err) {
      if (err instanceof Error && !err.message.includes('Unauthorized')) {
        setError(err.message);
      }
    }
  }, [isLoggedIn]);

  // Fetch history
  const fetchHistory = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const data = await apiService.getGARLHistory();
      setHistory(data);

      // Update avg/worst fitness from history
      if (data.history.length > 0) {
        const latest = data.history[data.history.length - 1];
        setMetrics(prev => prev ? {
          ...prev,
          avgFitness: latest.avg_fitness,
          worstFitness: latest.worst_fitness
        } : null);
      }
    } catch {
      // History may not be available if not initialized
    }
  }, [isLoggedIn]);

  // Initialize system
  const handleInitialize = async () => {
    try {
      setError(null);
      addLog('info', `Initializing GA+RL system for ${config.symbol}...`);
      await apiService.initializeGARL({
        symbol: config.symbol,
        population_size: config.population_size,
        num_generations: config.num_generations
      });
      setSuccessMessage('GA+RL system initialized successfully!');
      addLog('success', `System initialized: ${config.population_size} population, ${config.num_generations} generations`);
      await fetchStatus();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to initialize';
      setError(msg);
      addLog('error', `Initialization failed: ${msg}`);
    }
  };

  // Start training
  const handleStartTraining = async () => {
    try {
      setError(null);
      trainingStartTime.current = Date.now();
      addLog('info', `Starting training for ${config.symbol}...`);
      await apiService.startGARLTraining({
        symbol: config.symbol,
        start_date: config.start_date || undefined,
        end_date: config.end_date || undefined
      });
      setSuccessMessage('Training started! Monitor progress below.');
      addLog('success', 'Training started successfully');
      await fetchStatus();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to start training';
      setError(msg);
      addLog('error', `Failed to start training: ${msg}`);
    }
  };

  // Stop training
  const handleStopTraining = async () => {
    try {
      setError(null);
      addLog('warning', 'Stopping training...');
      await apiService.stopGARLTraining();
      setSuccessMessage('Training stop signal sent.');
      addLog('info', 'Training will stop after current generation completes');
      await fetchStatus();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to stop training';
      setError(msg);
      addLog('error', `Failed to stop training: ${msg}`);
    }
  };

  // Reset system
  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to reset the GA+RL system? This will clear all training progress.')) {
      return;
    }
    try {
      addLog('warning', 'Resetting GA+RL system...');
      // Re-initialize will reset the system
      await apiService.initializeGARL({
        symbol: config.symbol,
        population_size: config.population_size,
        num_generations: config.num_generations
      });
      setHistory(null);
      setMetrics(null);
      setSuccessMessage('System reset successfully');
      addLog('success', 'System reset complete');
      await fetchStatus();
    } catch (err) {
      addLog('error', `Reset failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Test signal
  const handleTestSignal = async () => {
    try {
      setError(null);
      addLog('info', 'Generating test trading signal...');
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
      addLog('success', `Signal generated: ${signal.signal.action.toUpperCase()} (${(signal.signal.confidence * 100).toFixed(1)}% confidence)`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to get signal';
      setError(msg);
      addLog('error', `Signal generation failed: ${msg}`);
    }
  };

  // Clear logs
  const handleClearLogs = () => {
    setLogs([]);
    addLog('info', 'Logs cleared');
  };

  // Export logs
  const handleExportLogs = () => {
    const logText = logs.map(log =>
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] ${log.message}`
    ).join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `garl-logs-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    addLog('info', 'Logs exported');
  };

  // WebSocket event handlers
  useEffect(() => {
    if (!isLoggedIn) return;

    const handleProgress = (data: any) => {
      // Update metrics from progress
      setMetrics(prev => ({
        ...prev,
        currentGeneration: data.generation,
        totalGenerations: data.total_generations || config.num_generations,
        bestFitness: data.best_fitness || prev?.bestFitness || 0,
        avgFitness: data.avg_fitness || prev?.avgFitness || 0,
      } as TrainingMetrics));
      fetchHistory();
    };

    const handleComplete = (data: any) => {
      if (data.success) {
        addLog('success', `Training complete! Final Sharpe: ${data.results?.sharpe_ratio?.toFixed(4)}, Return: ${(data.results?.total_return * 100)?.toFixed(2)}%`);
      } else {
        addLog('error', 'Training completed with errors');
      }
      fetchStatus();
      fetchHistory();
    };

    const handleError = (data: any) => {
      addLog('error', `Training error: ${data.error}`);
      fetchStatus();
    };

    // Handle log messages from backend
    const handleLog = (data: any) => {
      const level = data.level as LogEntry['level'] || 'info';
      addLog(level, data.message);
    };

    // Subscribe to WebSocket events
    wsService.connect().catch(console.error);

    const socket = (wsService as any).socket;
    if (socket) {
      socket.on('ga_rl_progress', handleProgress);
      socket.on('ga_rl_complete', handleComplete);
      socket.on('ga_rl_error', handleError);
      socket.on('ga_rl_log', handleLog);
    }

    return () => {
      if (socket) {
        socket.off('ga_rl_progress', handleProgress);
        socket.off('ga_rl_complete', handleComplete);
        socket.off('ga_rl_error', handleError);
        socket.off('ga_rl_log', handleLog);
      }
    };
  }, [isLoggedIn, addLog, fetchStatus, fetchHistory, config.num_generations]);

  // Initial load - only after logged in
  useEffect(() => {
    if (!isLoggedIn) return;

    const load = async () => {
      setLoading(true);
      await fetchStatus();
      await fetchHistory();
      setLoading(false);
      addLog('info', 'GA+RL Training page loaded');
    };
    load();

    // Poll for updates
    const interval = setInterval(() => {
      fetchStatus();
      if (status?.is_training) {
        fetchHistory();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isLoggedIn, fetchStatus, fetchHistory, status?.is_training]);

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Clear messages after timeout
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => setSuccessMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  // Format time duration
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  if (!isLoggedIn) {
    return (
      <div className="garl-training">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Authenticating...</p>
        </div>
      </div>
    );
  }

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

  const progressPercent = metrics && metrics.totalGenerations > 0
    ? (metrics.currentGeneration / metrics.totalGenerations) * 100
    : 0;

  return (
    <div className="garl-training">
      <header className="page-header">
        <div className="header-content">
          <h1>GA+RL Training System</h1>
          <p className="description">
            Genetic Algorithm optimized Deep Q-Networks for automated trading
          </p>
        </div>
        <div className="header-actions">
          <span className={`system-badge ${status?.available ? 'available' : 'unavailable'}`}>
            {status?.available ? 'System Available' : 'System Unavailable'}
          </span>
          <span className={`system-badge ${status?.is_training ? 'training' : status?.initialized ? 'ready' : 'not-init'}`}>
            {status?.is_training ? 'Training Active' : status?.initialized ? 'Ready' : 'Not Initialized'}
          </span>
        </div>
      </header>

      {error && <div className="error-message">{error}</div>}
      {successMessage && <div className="success-message">{successMessage}</div>}

      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`tab-btn ${activeTab === 'config' ? 'active' : ''}`}
          onClick={() => setActiveTab('config')}
        >
          Configuration
        </button>
        <button
          className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          Logs ({logs.length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          Analysis
        </button>
      </nav>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="tab-content">
          {/* Control Panel */}
          <section className="control-panel">
            <h2>Control Panel</h2>
            <div className="control-buttons">
              {!status?.initialized && (
                <button className="btn-primary btn-large" onClick={handleInitialize}>
                  <span className="btn-icon">▶</span> Initialize System
                </button>
              )}
              {status?.initialized && !status?.is_training && (
                <>
                  <button className="btn-primary btn-large" onClick={handleStartTraining}>
                    <span className="btn-icon">▶</span> Start Training
                  </button>
                  <button className="btn-secondary" onClick={handleReset}>
                    <span className="btn-icon">↺</span> Reset
                  </button>
                </>
              )}
              {status?.is_training && (
                <>
                  <button className="btn-danger btn-large" onClick={handleStopTraining}>
                    <span className="btn-icon">■</span> Stop Training
                  </button>
                  <span className="training-indicator">
                    <span className="pulse-dot"></span> Training in progress...
                  </span>
                </>
              )}
              {status?.has_trained_agent && (
                <button className="btn-secondary" onClick={handleTestSignal}>
                  <span className="btn-icon">📊</span> Test Signal
                </button>
              )}
            </div>
          </section>

          {/* KPI Dashboard */}
          <section className="kpi-dashboard">
            <h2>Key Performance Indicators</h2>
            <div className="kpi-grid">
              <div className="kpi-card primary">
                <div className="kpi-value">{metrics?.bestFitness?.toFixed(4) || '—'}</div>
                <div className="kpi-label">Best Fitness</div>
                {metrics?.fitnessImprovement !== undefined && metrics.fitnessImprovement > 0 && (
                  <div className="kpi-change positive">+{metrics.fitnessImprovement.toFixed(4)}</div>
                )}
              </div>
              <div className="kpi-card">
                <div className="kpi-value">{metrics?.avgFitness?.toFixed(4) || '—'}</div>
                <div className="kpi-label">Average Fitness</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-value">{history?.best_chromosome?.sharpe_ratio?.toFixed(3) || '—'}</div>
                <div className="kpi-label">Sharpe Ratio</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-value">
                  {history?.best_chromosome?.total_return
                    ? `${(history.best_chromosome.total_return * 100).toFixed(2)}%`
                    : '—'}
                </div>
                <div className="kpi-label">Total Return</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-value">
                  {history?.best_chromosome?.win_rate
                    ? `${(history.best_chromosome.win_rate * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div className="kpi-label">Win Rate</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-value">
                  {history?.best_chromosome?.max_drawdown
                    ? `${(Math.abs(history.best_chromosome.max_drawdown) * 100).toFixed(2)}%`
                    : '—'}
                </div>
                <div className="kpi-label">Max Drawdown</div>
              </div>
            </div>
          </section>

          {/* Training Progress */}
          {(status?.is_training || metrics) && (
            <section className="progress-section">
              <h2>Training Progress</h2>
              <div className="progress-header">
                <span>Generation {metrics?.currentGeneration || 0} of {metrics?.totalGenerations || 0}</span>
                <span>{progressPercent.toFixed(1)}%</span>
              </div>
              <div className="progress-bar-container large">
                <div className="progress-bar" style={{ width: `${progressPercent}%` }}></div>
              </div>
              <div className="progress-details">
                <div className="progress-stat">
                  <span className="stat-label">Chromosome:</span>
                  <span className="stat-value">{metrics?.currentChromosome || 0} / {metrics?.totalChromosomes || 0}</span>
                </div>
                <div className="progress-stat">
                  <span className="stat-label">Elapsed:</span>
                  <span className="stat-value">{formatDuration(metrics?.elapsedTime || 0)}</span>
                </div>
                <div className="progress-stat">
                  <span className="stat-label">ETA:</span>
                  <span className="stat-value">{formatDuration(metrics?.estimatedTimeRemaining || 0)}</span>
                </div>
              </div>
            </section>
          )}

          {/* Evolution Chart */}
          {history && history.history.length > 0 && (
            <section className="chart-section">
              <h2>Fitness Evolution</h2>
              <div className="chart-wrapper">
                <div className="chart-y-axis">
                  <span>{Math.max(...history.history.map(h => h.best_fitness)).toFixed(2)}</span>
                  <span>{(Math.max(...history.history.map(h => h.best_fitness)) / 2).toFixed(2)}</span>
                  <span>0</span>
                </div>
                <div className="evolution-chart">
                  {history.history.map((gen, idx) => {
                    const maxFitness = Math.max(...history.history.map(h => h.best_fitness));
                    const bestHeight = (gen.best_fitness / maxFitness) * 100;
                    const avgHeight = (gen.avg_fitness / maxFitness) * 100;
                    return (
                      <div key={idx} className="chart-column">
                        <div
                          className="chart-bar best"
                          style={{ height: `${bestHeight}%` }}
                          title={`Gen ${gen.generation}: Best=${gen.best_fitness.toFixed(4)}`}
                        ></div>
                        <div
                          className="chart-bar avg"
                          style={{ height: `${avgHeight}%` }}
                          title={`Gen ${gen.generation}: Avg=${gen.avg_fitness.toFixed(4)}`}
                        ></div>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-color best"></span> Best Fitness</span>
                <span className="legend-item"><span className="legend-color avg"></span> Average Fitness</span>
              </div>
            </section>
          )}

          {/* Test Signal Result */}
          {testSignal && (
            <section className="signal-section">
              <h2>Trading Signal Test</h2>
              <div className="signal-result">
                <div className={`signal-action signal-${testSignal.signal.action}`}>
                  {testSignal.signal.action.toUpperCase()}
                </div>
                <div className="signal-details">
                  <div className="signal-confidence">
                    <div className="confidence-bar" style={{ width: `${testSignal.signal.confidence * 100}%` }}></div>
                    <span>{(testSignal.signal.confidence * 100).toFixed(1)}% Confidence</span>
                  </div>
                  <div className="q-values">
                    <div className="q-value">
                      <span className="q-label">Buy:</span>
                      <span className={`q-score ${testSignal.signal.action === 'buy' ? 'active' : ''}`}>
                        {testSignal.signal.q_values.buy?.toFixed(3)}
                      </span>
                    </div>
                    <div className="q-value">
                      <span className="q-label">Sell:</span>
                      <span className={`q-score ${testSignal.signal.action === 'sell' ? 'active' : ''}`}>
                        {testSignal.signal.q_values.sell?.toFixed(3)}
                      </span>
                    </div>
                    <div className="q-value">
                      <span className="q-label">Hold:</span>
                      <span className={`q-score ${testSignal.signal.action === 'hold' ? 'active' : ''}`}>
                        {testSignal.signal.q_values.hold?.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* Configuration Tab */}
      {activeTab === 'config' && (
        <div className="tab-content">
          <section className="config-section">
            <h2>Training Configuration</h2>
            <div className="config-form">
              <div className="form-row">
                <div className="form-group">
                  <label>Trading Symbol</label>
                  <input
                    type="text"
                    value={config.symbol}
                    onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
                    placeholder="AAPL"
                    disabled={status?.is_training}
                  />
                  <span className="form-hint">Stock ticker symbol for training data</span>
                </div>
                <div className="form-group">
                  <label>Population Size</label>
                  <input
                    type="number"
                    value={config.population_size}
                    onChange={(e) => setConfig({ ...config, population_size: parseInt(e.target.value) || 20 })}
                    min={5}
                    max={100}
                    disabled={status?.is_training}
                  />
                  <span className="form-hint">Number of chromosomes per generation (5-100)</span>
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Number of Generations</label>
                  <input
                    type="number"
                    value={config.num_generations}
                    onChange={(e) => setConfig({ ...config, num_generations: parseInt(e.target.value) || 50 })}
                    min={5}
                    max={200}
                    disabled={status?.is_training}
                  />
                  <span className="form-hint">Evolution iterations (5-200)</span>
                </div>
                <div className="form-group">
                  <label>Estimated Training Time</label>
                  <div className="estimate-display">
                    ~{Math.round(config.population_size * config.num_generations * 0.5 / 60)} minutes
                  </div>
                  <span className="form-hint">Approximate based on config</span>
                </div>
              </div>

              <div className="form-divider">
                <button
                  className="toggle-advanced"
                  onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                >
                  {showAdvancedConfig ? '▼' : '▶'} Advanced Options
                </button>
              </div>

              {showAdvancedConfig && (
                <div className="advanced-config">
                  <div className="form-row">
                    <div className="form-group">
                      <label>Start Date (optional)</label>
                      <input
                        type="date"
                        value={config.start_date}
                        onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                        disabled={status?.is_training}
                      />
                      <span className="form-hint">Filter training data start</span>
                    </div>
                    <div className="form-group">
                      <label>End Date (optional)</label>
                      <input
                        type="date"
                        value={config.end_date}
                        onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                        disabled={status?.is_training}
                      />
                      <span className="form-hint">Filter training data end</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Best Chromosome Configuration */}
          {history?.best_chromosome && (
            <section className="chromosome-section">
              <h2>Best Evolved Configuration</h2>
              <div className="chromosome-grid">
                <div className="chromosome-group">
                  <h3>Neural Network</h3>
                  <div className="param-list">
                    <div className="param">
                      <span className="param-label">Hidden Size</span>
                      <span className="param-value">{history.best_chromosome.hidden_size}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Layers</span>
                      <span className="param-value">{history.best_chromosome.num_layers}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Dueling</span>
                      <span className={`param-value ${history.best_chromosome.use_dueling ? 'enabled' : 'disabled'}`}>
                        {history.best_chromosome.use_dueling ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="chromosome-group">
                  <h3>Training</h3>
                  <div className="param-list">
                    <div className="param">
                      <span className="param-label">Learning Rate</span>
                      <span className="param-value">{history.best_chromosome.learning_rate?.toExponential(2)}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Gamma</span>
                      <span className="param-value">{history.best_chromosome.gamma?.toFixed(4)}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Batch Size</span>
                      <span className="param-value">{history.best_chromosome.batch_size}</span>
                    </div>
                  </div>
                </div>
                <div className="chromosome-group">
                  <h3>Trading Strategy</h3>
                  <div className="param-list">
                    <div className="param">
                      <span className="param-label">Position Size</span>
                      <span className="param-value">{(history.best_chromosome.position_size * 100).toFixed(1)}%</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Stop Loss</span>
                      <span className="param-value">{(history.best_chromosome.stop_loss * 100).toFixed(1)}%</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Take Profit</span>
                      <span className="param-value">{(history.best_chromosome.take_profit * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                <div className="chromosome-group highlight">
                  <h3>Performance</h3>
                  <div className="param-list">
                    <div className="param">
                      <span className="param-label">Fitness</span>
                      <span className="param-value highlight">{history.best_chromosome.fitness?.toFixed(4)}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Sharpe</span>
                      <span className="param-value">{history.best_chromosome.sharpe_ratio?.toFixed(4)}</span>
                    </div>
                    <div className="param">
                      <span className="param-label">Win Rate</span>
                      <span className="param-value">{(history.best_chromosome.win_rate * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      )}

      {/* Logs Tab */}
      {activeTab === 'logs' && (
        <div className="tab-content">
          <section className="logs-section">
            <div className="logs-header">
              <h2>Training Logs</h2>
              <div className="logs-controls">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                  />
                  Auto-scroll
                </label>
                <button className="btn-small" onClick={handleExportLogs}>Export</button>
                <button className="btn-small btn-danger" onClick={handleClearLogs}>Clear</button>
              </div>
            </div>
            <div className="logs-container">
              {logs.length === 0 ? (
                <div className="logs-empty">No log entries yet</div>
              ) : (
                logs.map(log => (
                  <div key={log.id} className={`log-entry log-${log.level}`}>
                    <span className="log-time">{log.timestamp.toLocaleTimeString()}</span>
                    <span className={`log-level ${log.level}`}>{log.level.toUpperCase()}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </section>
        </div>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && (
        <div className="tab-content">
          <section className="analysis-section">
            <h2>Evolution Analysis</h2>

            {history && history.history.length > 0 ? (
              <>
                {/* Statistics Table */}
                <div className="stats-table-container">
                  <table className="stats-table">
                    <thead>
                      <tr>
                        <th>Generation</th>
                        <th>Best Fitness</th>
                        <th>Avg Fitness</th>
                        <th>Worst Fitness</th>
                        <th>Improvement</th>
                        <th>Best Sharpe</th>
                        <th>Best Return</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.history.map((gen, idx) => {
                        const prevBest = idx > 0 ? history.history[idx - 1].best_fitness : gen.best_fitness;
                        const improvement = gen.best_fitness - prevBest;
                        return (
                          <tr key={idx}>
                            <td>{gen.generation}</td>
                            <td className="highlight">{gen.best_fitness.toFixed(4)}</td>
                            <td>{gen.avg_fitness.toFixed(4)}</td>
                            <td>{gen.worst_fitness.toFixed(4)}</td>
                            <td className={improvement >= 0 ? 'positive' : 'negative'}>
                              {improvement >= 0 ? '+' : ''}{improvement.toFixed(4)}
                            </td>
                            <td>{gen.best_sharpe?.toFixed(4) || '—'}</td>
                            <td>{gen.best_return ? `${(gen.best_return * 100).toFixed(2)}%` : '—'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Summary Stats */}
                <div className="summary-stats">
                  <div className="summary-card">
                    <h4>Total Improvement</h4>
                    <div className="summary-value">
                      {(history.history[history.history.length - 1].best_fitness - history.history[0].best_fitness).toFixed(4)}
                    </div>
                  </div>
                  <div className="summary-card">
                    <h4>Generations</h4>
                    <div className="summary-value">{history.generations_completed}</div>
                  </div>
                  <div className="summary-card">
                    <h4>Convergence</h4>
                    <div className="summary-value">
                      {history.history.length >= 3
                        ? Math.abs(history.history[history.history.length - 1].best_fitness -
                                  history.history[history.history.length - 3].best_fitness) < 0.01
                          ? 'Converged'
                          : 'In Progress'
                        : 'Too Early'}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="no-data-message">
                <p>No training history available yet.</p>
                <p>Start training to see evolution analysis.</p>
              </div>
            )}
          </section>

          {/* How It Works */}
          <section className="info-section">
            <h2>How GA+RL Works</h2>
            <div className="info-grid">
              <div className="info-card">
                <div className="info-icon">🧬</div>
                <h3>1. Genetic Algorithm</h3>
                <p>
                  Evolves a population of DQN configurations. Each chromosome encodes
                  neural network architecture, training hyperparameters, and trading strategy parameters.
                </p>
              </div>
              <div className="info-card">
                <div className="info-icon">🤖</div>
                <h3>2. DQN Training</h3>
                <p>
                  Each chromosome creates and trains a Deep Q-Network agent
                  that learns buy/sell/hold decisions from historical market data.
                </p>
              </div>
              <div className="info-card">
                <div className="info-icon">📊</div>
                <h3>3. Fitness Evaluation</h3>
                <p>
                  Agents are scored on Sharpe ratio (40%), total return (30%),
                  max drawdown (20%), and win rate (10%). Best performers survive.
                </p>
              </div>
              <div className="info-card">
                <div className="info-icon">🔄</div>
                <h3>4. Evolution</h3>
                <p>
                  Tournament selection, crossover, and mutation create new generations.
                  Elitism preserves top performers. Process repeats until convergence.
                </p>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
};

export default GARLTraining;
