# Python Agents for Stock Market Analysis

This directory contains the Python agents that implement various AI/ML models for stock market analysis:
- Reinforcement Learning Agent
- LSTM Price Prediction Agent
- News/NLP Sentiment Analysis Agent
- Communication layer for WebSocket messaging
- Real Market Data Integration
- Enhanced Human-AI Collaboration Features

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
GET /strategies (requires authentication)
POST /strategies (requires authentication)
```