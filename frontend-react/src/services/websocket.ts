/**
 * WebSocket Service for Stock Market Lab
 * Handles real-time communication with the backend
 */

import { io, Socket } from 'socket.io-client';

const WS_URL = process.env.REACT_APP_WS_URL || 'http://localhost:5100';

// Event types
export interface AgentDecisionEvent {
  agent_id: string;
  agent_type: string;
  timestamp: number;
  decision: {
    action: string;
    symbol: string;
    confidence: number;
    reason: string;
  };
}

export interface PricePredictionEvent {
  agent_id: string;
  agent_type: string;
  timestamp: number;
  prediction: {
    symbol: string;
    predicted_price: number;
    confidence: number;
    direction: 'up' | 'down' | 'neutral';
  };
}

export interface SentimentAnalysisEvent {
  agent_id: string;
  agent_type: string;
  timestamp: number;
  sentiment: {
    symbol: string;
    sentiment: 'positive' | 'negative' | 'neutral';
    confidence: number;
    topics: string[];
  };
}

export interface StrategyCreatedEvent {
  id: number;
  name: string;
  description: string;
  created_by: string;
  created_at: string;
}

export interface BacktestProgressEvent {
  strategy_id: number;
  progress: number;
  current_value: number;
}

export interface BacktestCompleteEvent {
  strategy_id: number;
  result: any;
}

type EventCallback<T> = (data: T) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private listeners: Map<string, Set<EventCallback<any>>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        // Wait for existing connection attempt
        const checkConnection = setInterval(() => {
          if (this.socket?.connected) {
            clearInterval(checkConnection);
            resolve();
          }
        }, 100);
        return;
      }

      this.isConnecting = true;

      this.socket = io(WS_URL, {
        transports: ['websocket', 'polling'],
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
        timeout: 10000,
      });

      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        resolve();
      });

      this.socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        this.emit('disconnect', { reason });
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.reconnectAttempts++;
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          this.isConnecting = false;
          reject(new Error('Failed to connect to WebSocket server'));
        }
      });

      // Register event handlers
      this.socket.on('agent_decision', (data: AgentDecisionEvent) => {
        this.emit('agent_decision', data);
      });

      this.socket.on('price_prediction', (data: PricePredictionEvent) => {
        this.emit('price_prediction', data);
      });

      this.socket.on('sentiment_analysis', (data: SentimentAnalysisEvent) => {
        this.emit('sentiment_analysis', data);
      });

      this.socket.on('strategy_created', (data: StrategyCreatedEvent) => {
        this.emit('strategy_created', data);
      });

      this.socket.on('agent_override', (data: any) => {
        this.emit('agent_override', data);
      });

      this.socket.on('backtest_progress', (data: BacktestProgressEvent) => {
        this.emit('backtest_progress', data);
      });

      this.socket.on('backtest_complete', (data: BacktestCompleteEvent) => {
        this.emit('backtest_complete', data);
      });
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.listeners.clear();
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  subscribe<T>(event: string, callback: EventCallback<T>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    // Return unsubscribe function
    return () => this.unsubscribe(event, callback);
  }

  unsubscribe<T>(event: string, callback: EventCallback<T>): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
    }
  }

  private emit(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  // Helper methods for specific event subscriptions
  onAgentDecision(callback: EventCallback<AgentDecisionEvent>): () => void {
    return this.subscribe('agent_decision', callback);
  }

  onPricePrediction(callback: EventCallback<PricePredictionEvent>): () => void {
    return this.subscribe('price_prediction', callback);
  }

  onSentimentAnalysis(callback: EventCallback<SentimentAnalysisEvent>): () => void {
    return this.subscribe('sentiment_analysis', callback);
  }

  onStrategyCreated(callback: EventCallback<StrategyCreatedEvent>): () => void {
    return this.subscribe('strategy_created', callback);
  }

  onBacktestProgress(callback: EventCallback<BacktestProgressEvent>): () => void {
    return this.subscribe('backtest_progress', callback);
  }

  onBacktestComplete(callback: EventCallback<BacktestCompleteEvent>): () => void {
    return this.subscribe('backtest_complete', callback);
  }

  onDisconnect(callback: EventCallback<{ reason: string }>): () => void {
    return this.subscribe('disconnect', callback);
  }
}

// Export singleton instance
export const wsService = new WebSocketService();
export default wsService;
