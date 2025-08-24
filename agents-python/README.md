# Python Agents for Stock Market Analysis

This directory contains the Python agents that implement various AI/ML models for stock market analysis:
- Reinforcement Learning Agent
- LSTM Price Prediction Agent
- News/NLP Sentiment Analysis Agent
- Communication layer for WebSocket messaging
- Real Market Data Integration

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

### Usage

The market data is automatically streamed to Redis at regular intervals:
- Latest prices stored in `latest_market_data` key
- Historical data stored in `historical_prices_{symbol}` lists
- News articles stored in `news_articles` list