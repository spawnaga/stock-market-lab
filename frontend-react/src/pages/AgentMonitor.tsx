import React from 'react';
import './AgentMonitor.css';

const AgentMonitor: React.FC = () => {
  return (
    <div className="agent-monitor">
      <div className="agent-monitor-header">
        <h2>Agent Monitor</h2>
        <p>Real-time monitoring of AI agents</p>
      </div>
      
      <div className="agent-status-grid">
        <div className="agent-status-card">
          <h3>Reinforcement Learning Agent</h3>
          <div className="status-indicator online"></div>
          <p>Status: <span className="online">Online</span></p>
          <p>Last Activity: Just now</p>
        </div>
        
        <div className="agent-status-card">
          <h3>LSTM Price Predictor</h3>
          <div className="status-indicator online"></div>
          <p>Status: <span className="online">Online</span></p>
          <p>Last Activity: Just now</p>
        </div>
        
        <div className="agent-status-card">
          <h3>News/NLP Agent</h3>
          <div className="status-indicator online"></div>
          <p>Status: <span className="online">Online</span></p>
          <p>Last Activity: Just now</p>
        </div>
      </div>
      
      <div className="agent-events">
        <h3>Recent Events</h3>
        <div className="events-list">
          <div className="event-item">
            <span className="event-time">10:30:45</span>
            <span className="event-type">Decision</span>
            <span className="event-message">RL Agent: Buy AAPL with 85% confidence</span>
          </div>
          <div className="event-item">
            <span className="event-time">10:30:42</span>
            <span className="event-type">Prediction</span>
            <span className="event-message">LSTM: Predicted AAPL price $176.50</span>
          </div>
          <div className="event-item">
            <span className="event-time">10:30:38</span>
            <span className="event-type">Analysis</span>
            <span className="event-message">News Agent: Positive sentiment detected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentMonitor;