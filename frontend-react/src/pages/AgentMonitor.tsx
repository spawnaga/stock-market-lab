import React, { useState, useEffect, useCallback } from 'react';
import { apiService, HealthStatus } from '../services/api';
import { wsService, AgentDecisionEvent, PricePredictionEvent, SentimentAnalysisEvent } from '../services/websocket';
import './AgentMonitor.css';

interface AgentEvent {
  id: string;
  timestamp: Date;
  type: 'decision' | 'prediction' | 'sentiment';
  agentType: string;
  message: string;
  details: any;
}

interface AgentStatus {
  name: string;
  type: string;
  status: 'online' | 'offline' | 'error';
  lastActivity: Date | null;
  metrics: {
    decisionsCount?: number;
    predictionsCount?: number;
    accuracy?: number;
  };
}

const AgentMonitor: React.FC = () => {
  const [agents, setAgents] = useState<AgentStatus[]>([
    { name: 'Reinforcement Learning Agent', type: 'rl', status: 'offline', lastActivity: null, metrics: {} },
    { name: 'LSTM Price Predictor', type: 'lstm', status: 'offline', lastActivity: null, metrics: {} },
    { name: 'News/NLP Agent', type: 'news', status: 'offline', lastActivity: null, metrics: {} },
  ]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

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

  // Fetch health status
  const fetchHealth = useCallback(async () => {
    if (!isLoggedIn) return;

    try {
      const health = await apiService.getHealth();
      setHealthStatus(health);

      // Update agent statuses based on health response
      setAgents(prev => prev.map(agent => ({
        ...agent,
        status: health.agents[agent.type as keyof typeof health.agents] ? 'online' : 'offline'
      })));
    } catch (err) {
      console.error('Failed to fetch health:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isLoggedIn]);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  // WebSocket connection and event handling
  useEffect(() => {
    wsService.connect()
      .then(() => {
        setIsConnected(true);
      })
      .catch((err) => {
        console.error('WebSocket connection failed:', err);
        setIsConnected(false);
      });

    // Handle agent decisions
    const unsubDecision = wsService.onAgentDecision((data: AgentDecisionEvent) => {
      const newEvent: AgentEvent = {
        id: `decision-${Date.now()}`,
        timestamp: new Date(data.timestamp * 1000),
        type: 'decision',
        agentType: data.agent_type,
        message: `${data.agent_type.toUpperCase()} Agent: ${data.decision.action} ${data.decision.symbol} with ${(data.decision.confidence * 100).toFixed(0)}% confidence`,
        details: data.decision
      };
      setEvents(prev => [newEvent, ...prev].slice(0, 50)); // Keep last 50 events

      // Update agent last activity
      setAgents(prev => prev.map(agent =>
        agent.type === data.agent_type
          ? { ...agent, lastActivity: new Date(), status: 'online' }
          : agent
      ));
    });

    // Handle price predictions
    const unsubPrediction = wsService.onPricePrediction((data: PricePredictionEvent) => {
      const newEvent: AgentEvent = {
        id: `prediction-${Date.now()}`,
        timestamp: new Date(data.timestamp * 1000),
        type: 'prediction',
        agentType: data.agent_type,
        message: `LSTM: Predicted ${data.prediction.symbol} ${data.prediction.direction} to $${data.prediction.predicted_price.toFixed(2)}`,
        details: data.prediction
      };
      setEvents(prev => [newEvent, ...prev].slice(0, 50));

      setAgents(prev => prev.map(agent =>
        agent.type === data.agent_type
          ? { ...agent, lastActivity: new Date(), status: 'online' }
          : agent
      ));
    });

    // Handle sentiment analysis
    const unsubSentiment = wsService.onSentimentAnalysis((data: SentimentAnalysisEvent) => {
      const newEvent: AgentEvent = {
        id: `sentiment-${Date.now()}`,
        timestamp: new Date(data.timestamp * 1000),
        type: 'sentiment',
        agentType: data.agent_type,
        message: `News Agent: ${data.sentiment.sentiment} sentiment for ${data.sentiment.symbol} (${(data.sentiment.confidence * 100).toFixed(0)}% confidence)`,
        details: data.sentiment
      };
      setEvents(prev => [newEvent, ...prev].slice(0, 50));

      setAgents(prev => prev.map(agent =>
        agent.type === data.agent_type
          ? { ...agent, lastActivity: new Date(), status: 'online' }
          : agent
      ));
    });

    // Handle disconnection
    const unsubDisconnect = wsService.onDisconnect(() => {
      setIsConnected(false);
    });

    return () => {
      unsubDecision();
      unsubPrediction();
      unsubSentiment();
      unsubDisconnect();
    };
  }, []);

  const formatTime = (date: Date | null) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString();
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString();
  };

  const getEventTypeClass = (type: string) => {
    switch (type) {
      case 'decision': return 'event-decision';
      case 'prediction': return 'event-prediction';
      case 'sentiment': return 'event-sentiment';
      default: return '';
    }
  };

  const getEventTypeLabel = (type: string) => {
    switch (type) {
      case 'decision': return 'Decision';
      case 'prediction': return 'Prediction';
      case 'sentiment': return 'Analysis';
      default: return type;
    }
  };

  if (!isLoggedIn) {
    return (
      <div className="agent-monitor">
        <div className="loading">Authenticating...</div>
      </div>
    );
  }

  return (
    <div className="agent-monitor">
      <div className="agent-monitor-header">
        <h2>Agent Monitor</h2>
        <p>Real-time monitoring of AI agents</p>
        <div className="connection-status">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* System Metrics */}
      {healthStatus && (
        <div className="system-metrics">
          <h3>System Metrics</h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Uptime</span>
              <span className="metric-value">{Math.floor(healthStatus.system.uptime_seconds / 60)}m</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Memory</span>
              <span className="metric-value">{healthStatus.system.memory_mb.toFixed(0)} MB</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">CPU</span>
              <span className="metric-value">{healthStatus.system.cpu_percent.toFixed(1)}%</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Threads</span>
              <span className="metric-value">{healthStatus.system.active_threads}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Clients</span>
              <span className="metric-value">{healthStatus.system.connected_clients}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Req/s</span>
              <span className="metric-value">{healthStatus.metrics.requests_per_second.toFixed(1)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Agent Status Cards */}
      <div className="agent-status-grid">
        {isLoading ? (
          <div className="loading-indicator">Loading agent status...</div>
        ) : (
          agents.map((agent) => (
            <div key={agent.type} className={`agent-status-card ${agent.status}`}>
              <h3>{agent.name}</h3>
              <div className={`status-indicator ${agent.status}`}></div>
              <p>Status: <span className={agent.status}>{agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span></p>
              <p>Last Activity: {formatTime(agent.lastActivity)}</p>
            </div>
          ))
        )}
      </div>

      {/* Recent Events */}
      <div className="agent-events">
        <h3>Recent Events {events.length > 0 && <span className="event-count">({events.length})</span>}</h3>
        <div className="events-list">
          {events.length === 0 ? (
            <div className="no-events">
              <p>No events yet. Waiting for agent activity...</p>
              <p className="hint">Events will appear here when agents make decisions, predictions, or analyses.</p>
            </div>
          ) : (
            events.map((event) => (
              <div key={event.id} className={`event-item ${getEventTypeClass(event.type)}`}>
                <span className="event-time">{formatTimestamp(event.timestamp)}</span>
                <span className={`event-type ${event.type}`}>{getEventTypeLabel(event.type)}</span>
                <span className="event-message">{event.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Data Streaming Status */}
      {healthStatus && (
        <div className="streaming-status">
          <h3>Data Streaming</h3>
          <div className={`streaming-indicator ${healthStatus.data_streaming ? 'active' : 'inactive'}`}>
            <span className="streaming-dot"></span>
            {healthStatus.data_streaming ? 'Active - Receiving market data' : 'Inactive'}
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentMonitor;
