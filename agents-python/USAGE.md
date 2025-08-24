# Enhanced Stock Market Lab - Usage Guide

This document explains how to use the enhanced features of the stock market lab including:

1. OHLCV Data Integration
2. Interactive Brokers Integration
3. Sentiment Analysis from StockTwits and Twitter
4. Trade Execution System

## 1. OHLCV Data Integration

The C++ backend now provides real OHLCV data instead of dummy data. The data includes:

- Open price
- High price  
- Low price
- Close price
- Volume
- Timestamp

### Data Format
```json
{
  "symbol": "AAPL",
  "open": 175.23,
  "high": 176.45,
  "low": 174.89,
  "close": 175.98,
  "volume": 1234567,
  "timestamp": "1698765432123"
}
```

## 2. Interactive Brokers Integration

The system includes an Interactive Brokers agent that can process trade requests.

### Trade Execution Flow

1. Trades are submitted to Redis queue (`pending_trades`)
2. IB agent polls the queue and processes trades
3. Trades are executed via IB API (simulated in this demo)

### Sending Trades

Use the `trade_sender.py` script to send trades:

```bash
python trade_sender.py
```

Or programmatically:

```python
import redis
import json

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

trade = {
    "id": "unique-trade-id",
    "symbol": "AAPL",
    "quantity": 10,
    "action": "BUY",
    "price": 175.50,
    "account": "DU123456",
    "timestamp": "2023-10-27T10:30:00Z"
}

redis_client.lpush("pending_trades", json.dumps(trade))
```

## 3. Sentiment Analysis

Three sentiment analysis agents are now available:

### News/NLP Agent
Uses HuggingFace transformers for sentiment analysis of news articles.

### StockTwits Agent
Analyzes sentiment from StockTwits community posts.

### Twitter Agent
Analyzes sentiment from Twitter/X posts.

### Data Format
```json
{
  "overall_sentiment": "positive",
  "confidence": 0.85,
  "key_topics": ["earnings", "market"],
  "score": 0.75,
  "timestamp": "2023-10-27T10:30:00Z"
}
```

## 4. Running the Enhanced System

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available

### Quick Start
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Environment Variables for IB Integration
Create a `.env` file in the root directory:
```
IB_HOST=127.0.0.1
IB_PORT=4001
IB_CLIENT_ID=1001
STOCKTWITS_API_KEY=your_stocktwits_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
```

## 5. Testing the System

### Simulate News Feed
```bash
python news_simulator.py
```

### Send Sample Trades
```bash
python trade_sender.py
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 6. Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Client   │    │   User Client   │    │   User Client   │
│   (Browser)     │    │   (Mobile)      │    │   (Terminal)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Frontend      │    │   Frontend      │
│   (React/TS)    │    │   (React/TS)    │    │   Frontend      │
│                 │    │                 │    │   (CLI Tools)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   API Gateway   │    │   API Gateway   │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   C++ Backend   │    │   C++ Backend   │    │   C++ Backend   │
│   (Data Ingest) │    │   (Data Ingest) │    │   (Data Ingest) │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kafka Bus     │    │   Kafka Bus     │    │   Kafka Bus     │
│   (Messaging)   │    │   (Messaging)   │    │   (Messaging)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Python Agents │    │   Python Agents │    │   Python Agents │
│   (RL/LSTM/NLP) │    │   (RL/LSTM/NLP) │    │   (RL/LSTM/NLP) │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis Cache   │    │   PostgreSQL    │
│   (Storage)     │    │   (Cache)       │    │   (Storage)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IB Gateway    │    │   IB Gateway    │    │   IB Gateway    │
│   (Interactive) │    │   (Interactive) │    │   (Interactive) │
│   Connectors    │    │   Connectors    │    │   Connectors    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 7. Extending the System

### Adding New Data Sources
To add a new data source:
1. Create a new agent class inheriting from BaseAgent
2. Implement the `run()` method to fetch and process data
3. Add the agent to the initialization in `main.py`

### Adding New Sentiment Sources
To add a new sentiment source:
1. Create a new agent class inheriting from BaseAgent
2. Implement the `_fetch_sentiment()` method to get data from the source
3. Add appropriate WebSocket events for communication

### Adding New Trading Platforms
To add support for other trading platforms:
1. Create a new agent class for the platform
2. Implement connection and trade execution logic
3. Follow the same pattern as the IB agent