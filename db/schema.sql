-- PostgreSQL Schema for Stock Market Lab

-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    volume BIGINT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);

-- Create table for trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    side VARCHAR(4) NOT NULL, -- 'BUY' or 'SELL'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for trades
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);

-- Create table for agent decisions
CREATE TABLE IF NOT EXISTS agent_decisions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    action VARCHAR(20) NOT NULL,
    confidence NUMERIC(5, 4),
    reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for agent decisions
CREATE INDEX idx_agent_decisions_agent_id ON agent_decisions(agent_id);
CREATE INDEX idx_agent_decisions_timestamp ON agent_decisions(timestamp);

-- Create table for price predictions
CREATE TABLE IF NOT EXISTS price_predictions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    predicted_price NUMERIC(10, 2) NOT NULL,
    confidence NUMERIC(5, 4),
    direction VARCHAR(10), -- 'up', 'down', 'neutral'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for price predictions
CREATE INDEX idx_price_predictions_agent_id ON price_predictions(agent_id);
CREATE INDEX idx_price_predictions_symbol ON price_predictions(symbol);
CREATE INDEX idx_price_predictions_timestamp ON price_predictions(timestamp);

-- Create table for sentiment analysis
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    sentiment VARCHAR(20) NOT NULL, -- 'positive', 'negative', 'neutral'
    confidence NUMERIC(5, 4),
    topics TEXT, -- JSON array of topics
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sentiment analysis
CREATE INDEX idx_sentiment_agent_id ON sentiment_analysis(agent_id);
CREATE INDEX idx_sentiment_symbol ON sentiment_analysis(symbol);
CREATE INDEX idx_sentiment_timestamp ON sentiment_analysis(timestamp);

-- Create table for user strategies
CREATE TABLE IF NOT EXISTS user_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    query TEXT NOT NULL, -- Natural language query
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for strategy executions
CREATE TABLE IF NOT EXISTS strategy_executions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES user_strategies(id),
    status VARCHAR(20) NOT NULL, -- 'pending', 'running', 'completed', 'failed'
    result JSONB, -- Results of the execution
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create indexes for strategy executions
CREATE INDEX idx_strategy_executions_strategy_id ON strategy_executions(strategy_id);
CREATE INDEX idx_strategy_executions_status ON strategy_executions(status);