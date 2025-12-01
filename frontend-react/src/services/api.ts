/**
 * API Service for Stock Market Lab
 * Handles all backend API communications
 */

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5100';

// Token management
let authToken: string | null = null;

export const setToken = (token: string) => {
  authToken = token;
  localStorage.setItem('auth_token', token);
};

export const getToken = (): string | null => {
  if (!authToken) {
    authToken = localStorage.getItem('auth_token');
  }
  return authToken;
};

export const clearToken = () => {
  authToken = null;
  localStorage.removeItem('auth_token');
};

// Helper function for authenticated requests
const authFetch = async (url: string, options: RequestInit = {}) => {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  // Merge existing headers if any
  if (options.headers) {
    const existingHeaders = options.headers as Record<string, string>;
    Object.assign(headers, existingHeaders);
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, { ...options, headers });

  if (response.status === 401) {
    clearToken();
    throw new Error('Unauthorized - please login again');
  }

  return response;
};

// Types
export interface LoginResponse {
  token: string;
  expires_in: number;
  user: string;
}

export interface Strategy {
  id: number;
  name: string;
  description: string;
  query: string;
  parameters: Record<string, any>;
  symbol: string;
  strategy_type: string;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface StrategyForm {
  name: string;
  description?: string;
  query?: string;
  parameters?: Record<string, any>;
  symbol?: string;
  strategy_type?: string;
}

export interface BacktestParams {
  strategy_name: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
}

export interface BacktestResult {
  strategy_name: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  trade_count: number;
  win_rate: number;
  equity_curve: number[];
  trades: any[];
}

export interface MarketDataPoint {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SymbolInfo {
  symbol: string;
  data_points: number;
  start_date: string;
  end_date: string;
}

export interface HealthStatus {
  status: string;
  timestamp: number;
  agents: {
    rl: boolean;
    lstm: boolean;
    news: boolean;
  };
  system: {
    uptime_seconds: number;
    memory_mb: number;
    cpu_percent: number;
    active_threads: number;
    connected_clients: number;
  };
  metrics: {
    total_requests: number;
    error_count: number;
    requests_per_second: number;
  };
  data_streaming: boolean;
}

// API Functions
export const apiService = {
  // Authentication
  async login(username: string, password: string): Promise<LoginResponse> {
    const response = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Login failed');
    }

    const data = await response.json();
    setToken(data.token);
    return data;
  },

  logout() {
    clearToken();
  },

  isAuthenticated(): boolean {
    return !!getToken();
  },

  // Health Check
  async getHealth(): Promise<HealthStatus> {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  },

  // Strategies
  async getStrategies(): Promise<{ strategies: Strategy[] }> {
    const response = await authFetch(`${API_URL}/strategies`);
    if (!response.ok) {
      throw new Error('Failed to fetch strategies');
    }
    return response.json();
  },

  async getStrategy(id: number): Promise<Strategy> {
    const response = await authFetch(`${API_URL}/strategies/${id}`);
    if (!response.ok) {
      throw new Error('Failed to fetch strategy');
    }
    return response.json();
  },

  async createStrategy(strategy: StrategyForm): Promise<Strategy> {
    const response = await authFetch(`${API_URL}/strategies`, {
      method: 'POST',
      body: JSON.stringify(strategy),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to create strategy');
    }
    return response.json();
  },

  async updateStrategy(id: number, strategy: Partial<StrategyForm>): Promise<Strategy> {
    const response = await authFetch(`${API_URL}/strategies/${id}`, {
      method: 'PUT',
      body: JSON.stringify(strategy),
    });
    if (!response.ok) {
      throw new Error('Failed to update strategy');
    }
    return response.json();
  },

  async deleteStrategy(id: number): Promise<void> {
    const response = await authFetch(`${API_URL}/strategies/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete strategy');
    }
  },

  // Backtesting
  async getBacktestStrategies(): Promise<{ strategies: { name: string; type: string }[] }> {
    const response = await authFetch(`${API_URL}/backtest/strategies`);
    if (!response.ok) {
      throw new Error('Failed to fetch backtest strategies');
    }
    return response.json();
  },

  async runBacktest(params: BacktestParams): Promise<BacktestResult> {
    const response = await authFetch(`${API_URL}/backtest/run`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Backtest failed');
    }
    return response.json();
  },

  async compareStrategies(params: {
    strategies: string[];
    symbol: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
  }): Promise<{ results: BacktestResult[] }> {
    const response = await authFetch(`${API_URL}/backtest/compare`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      throw new Error('Strategy comparison failed');
    }
    return response.json();
  },

  // Market Data
  async getAvailableSymbols(): Promise<{ symbols: SymbolInfo[]; count: number }> {
    const response = await authFetch(`${API_URL}/market-data/symbols`);
    if (!response.ok) {
      throw new Error('Failed to fetch symbols');
    }
    return response.json();
  },

  async getMarketData(
    symbol: string,
    startDate?: string,
    endDate?: string,
    limit?: number
  ): Promise<{ symbol: string; data: MarketDataPoint[]; count: number }> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (limit) params.append('limit', limit.toString());

    const url = `${API_URL}/market-data/${symbol}${params.toString() ? '?' + params.toString() : ''}`;
    const response = await authFetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch market data');
    }
    return response.json();
  },

  async getLatestMarketData(symbol: string): Promise<MarketDataPoint & { symbol: string }> {
    const response = await authFetch(`${API_URL}/market-data/${symbol}/latest`);
    if (!response.ok) {
      throw new Error('Failed to fetch latest market data');
    }
    return response.json();
  },

  async getDailyMarketData(
    symbol: string,
    startDate?: string,
    endDate?: string,
    limit?: number
  ): Promise<{ symbol: string; data: any[]; count: number }> {
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (limit) params.append('limit', limit.toString());

    const url = `${API_URL}/market-data/${symbol}/daily${params.toString() ? '?' + params.toString() : ''}`;
    const response = await authFetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch daily market data');
    }
    return response.json();
  },

  // Agent Operations
  async getAgentMetrics(): Promise<any> {
    const response = await authFetch(`${API_URL}/metrics`);
    if (!response.ok) {
      throw new Error('Failed to fetch metrics');
    }
    return response.json();
  },

  async overrideAgent(agentId: string, action: string, reason: string): Promise<any> {
    const response = await authFetch(`${API_URL}/override/${agentId}`, {
      method: 'POST',
      body: JSON.stringify({ override_action: action, reason }),
    });
    if (!response.ok) {
      throw new Error('Failed to override agent');
    }
    return response.json();
  },

  async toggleGuardrails(agentId: string, setting: string, enabled: boolean): Promise<any> {
    const response = await authFetch(`${API_URL}/guardrails/${agentId}/${setting}`, {
      method: 'PUT',
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) {
      throw new Error('Failed to toggle guardrails');
    }
    return response.json();
  },

  // LSTM Operations
  async getLSTMStatus(): Promise<any> {
    const response = await authFetch(`${API_URL}/lstm/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch LSTM status');
    }
    return response.json();
  },

  async retrainLSTM(): Promise<any> {
    const response = await authFetch(`${API_URL}/lstm/retrain`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to retrain LSTM');
    }
    return response.json();
  },

  // GA+RL Operations
  async getGARLStatus(): Promise<GARLStatus> {
    const response = await authFetch(`${API_URL}/ga-rl/status`);
    if (!response.ok) {
      throw new Error('Failed to fetch GA+RL status');
    }
    return response.json();
  },

  async initializeGARL(params: GARLInitParams): Promise<any> {
    const response = await authFetch(`${API_URL}/ga-rl/initialize`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to initialize GA+RL');
    }
    return response.json();
  },

  async startGARLTraining(params: GARLTrainParams): Promise<any> {
    const response = await authFetch(`${API_URL}/ga-rl/train`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to start GA+RL training');
    }
    return response.json();
  },

  async stopGARLTraining(): Promise<any> {
    const response = await authFetch(`${API_URL}/ga-rl/stop`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to stop GA+RL training');
    }
    return response.json();
  },

  async getGARLSignal(marketState: GARLMarketState): Promise<GARLSignal> {
    const response = await authFetch(`${API_URL}/ga-rl/signal`, {
      method: 'POST',
      body: JSON.stringify(marketState),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get GA+RL signal');
    }
    return response.json();
  },

  async getGARLHistory(): Promise<GARLHistory> {
    const response = await authFetch(`${API_URL}/ga-rl/history`);
    if (!response.ok) {
      throw new Error('Failed to fetch GA+RL history');
    }
    return response.json();
  },
};

// GA+RL Types
export interface GARLStatus {
  available: boolean;
  initialized: boolean;
  symbol?: string;
  is_training?: boolean;
  is_live_trading?: boolean;
  has_trained_agent?: boolean;
  training_progress?: {
    generation: number;
    total_generations: number;
    chromosome: number;
    total_chromosomes: number;
    current_fitness: number;
    best_fitness: number;
    elapsed_time: number;
  };
  current_chromosome?: GARLChromosome;
}

export interface GARLInitParams {
  symbol?: string;
  population_size?: number;
  num_generations?: number;
  initial_capital?: number;
}

export interface GARLTrainParams {
  symbol?: string;
  start_date?: string;
  end_date?: string;
  data_path?: string;
}

export interface GARLMarketState {
  price_change_1d?: number;
  price_change_5d?: number;
  price_change_20d?: number;
  volume_ratio?: number;
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  bb_position?: number;
  position?: number;
  portfolio_value_change?: number;
  time_in_position?: number;
}

export interface GARLSignal {
  signal: {
    action: 'buy' | 'sell' | 'hold';
    confidence: number;
    q_values: Record<string, number>;
    chromosome_id?: string;
  };
  market_state: GARLMarketState;
  timestamp: number;
}

export interface GARLChromosome {
  hidden_size: number;
  num_layers: number;
  use_dueling: boolean;
  learning_rate: number;
  gamma: number;
  epsilon_decay: number;
  batch_size: number;
  buffer_size: number;
  position_size: number;
  stop_loss: number;
  take_profit: number;
  fitness: number;
  sharpe_ratio: number;
  total_return: number;
  max_drawdown: number;
  win_rate: number;
  generation: number;
  chromosome_id: string;
}

export interface GARLHistory {
  history: Array<{
    generation: number;
    best_fitness: number;
    avg_fitness: number;
    worst_fitness: number;
    best_chromosome_id: string;
    best_sharpe: number;
    best_return: number;
  }>;
  generations_completed: number;
  best_chromosome: GARLChromosome | null;
}

export default apiService;
