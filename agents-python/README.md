# Python Agents for Stock Market Analysis

This directory contains the Python agents that implement various AI/ML models for stock market analysis:
- Reinforcement Learning Agent
- LSTM Price Prediction Agent
- News/NLP Sentiment Analysis Agent
- Communication layer for WebSocket messaging
- Real Market Data Integration
- Enhanced Human-AI Collaboration Features
- Advanced Monitoring and Metrics Collection

## Security Features

The agents service implements several production-grade security features:

### Authentication
- JWT-based authentication for protected API endpoints
- Login endpoint to obtain access tokens
- Token validation and expiration handling

### Rate Limiting
- Per-IP rate limiting to prevent abuse
- Configurable limits for different endpoints
- 429 Too Many Requests responses when limits exceeded

### Logging
- Rotating file logs with automatic cleanup
- Structured logging for better monitoring
- Detailed error logging for debugging

### Graceful Shutdown
- Signal handling for SIGINT and SIGTERM
- Proper cleanup of agents and resources
- Smooth shutdown without data loss

## Market Data Integration

The system now supports integration with real market data providers:
- Polygon.io (primary implementation)
- Tradier (secondary implementation)

### Configuration

To use real market data, set the following environment variables:
```
MARKET_DATA_PROVIDER=polygon
MARKET_DATA_API_KEY=your_api_key_here
```

### Supported Features

1. **Real-time Price Data**: Fetches current market prices for stocks
2. **Historical Data**: Retrieves price history for technical analysis
3. **News Integration**: Gets recent news articles related to securities
4. **Data Streaming**: Continuously streams data to Redis for agent consumption

## Enhanced Human-AI Collaboration

The system now includes guardrails and override mechanisms to ensure safe human-AI interaction:

### Guardrails
- Confidence level capping to prevent overconfident decisions
- Extreme action mitigation (e.g., preventing "sell_all" actions with very high confidence)
- Decision validation and safety checks

### Override Mechanisms
- REST API endpoints for human override of agent decisions
- WebSocket events for real-time override notifications
- Guardrail toggling for specific agents

### API Endpoints

#### Override Agent Decisions
```
POST /override/{agent_id}
{
  "override_action": "hold|buy|sell",
  "reason": "Human override reason",
  "user": "username"
}
```

#### Toggle Guardrails
```
PUT /guardrails/{agent_id}/{enable|disable}
```

#### Authentication
```
POST /login
{
  "username": "your_username",
  "password": "your_password"
}
```

#### Monitoring
```
GET /health
GET /metrics (requires authentication)
GET /debug/agents (requires authentication)
GET /strategies (requires authentication)
POST /strategies (requires authentication)
```

## Enhanced Monitoring and Metrics

The system now includes comprehensive monitoring capabilities:

### Health Checks
- `/health` endpoint provides detailed system health information
- Checks Redis connectivity, agent status, and system resources
- Returns memory usage, CPU utilization, and uptime statistics

### Metrics Collection
- `/metrics` endpoint provides detailed system and agent metrics
- Tracks request counts, error rates, and performance timings
- Reports agent execution counts, error rates, and execution times
- System resource monitoring (memory, CPU, threads)

### Debugging Endpoints
- `/debug/agents` provides detailed internal agent state information
- Shows agent-specific metrics, running status, and guardrail settings
- Useful for troubleshooting and performance analysis

### Performance Tracking
- Request timing measurements for all endpoints
- Agent execution time tracking
- Memory usage monitoring
- Garbage collection awareness

## Backtesting Framework

The system now includes a comprehensive backtesting framework that allows users to test trading strategies against historical market data:

### Key Features
- Support for multiple trading strategies (SMA, LSTM-based)
- Historical data loading from Redis
- Comprehensive performance metrics calculation
- Strategy comparison capabilities
- Trade execution simulation
- Detailed reporting and visualization support

### API Endpoints

#### Get Available Strategies
```
GET /backtest/strategies
```

#### Run Single Backtest
```
POST /backtest/run
{
  "strategy_name": "SMA_10_30",
  "symbol": "AAPL",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_capital": 10000.0
}
```

#### Compare Strategies
```
POST /backtest/compare
{
  "strategy_names": ["SMA_10_30", "SMA_5_20"],
  "symbol": "AAPL",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_capital": 10000.0
}
```

### Performance Metrics
The framework calculates various performance metrics:
- Total Return
- Annualized Return
- Maximum Drawdown
- Sharpe Ratio
- Win/Loss Statistics
- Trade Count