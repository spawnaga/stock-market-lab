# Backtesting Framework

The Backtesting Framework is a core component of the AI-Driven Multi-Agent Stock Market Lab that allows users to test trading strategies against historical market data before deploying them in live markets.

## Overview

The framework provides:
- Support for multiple trading strategies (currently SMA and LSTM-based)
- Historical data loading from Redis
- Comprehensive performance metrics calculation
- Strategy comparison capabilities
- Trade execution simulation
- Detailed reporting and visualization support

## Key Components

### 1. Trading Strategies

#### SimpleMovingAverageStrategy
A classic technical analysis strategy that uses moving average crossovers to generate buy/sell signals.

```python
strategy = SimpleMovingAverageStrategy("SMA_10_30", initial_capital=10000.0, short_window=10, long_window=30)
```

#### LSTMStrategy
A strategy that leverages predictions from the LSTM agent to make trading decisions.

```python
strategy = LSTMStrategy("LSTM_Prediction", initial_capital=10000.0, prediction_threshold=0.02)
```

### 2. Backtesting Engine

The `BacktestingEngine` class orchestrates the backtesting process:

```python
engine = BacktestingEngine(redis_client)
result = engine.run_backtest(strategy, "AAPL", start_date, end_date)
```

### 3. Performance Metrics

The framework calculates various performance metrics:
- Total Return
- Annualized Return
- Maximum Drawdown
- Sharpe Ratio
- Win/Loss Statistics
- Trade Count

## API Endpoints

### Get Available Strategies
```
GET /backtest/strategies
```

### Run Single Backtest
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

### Compare Strategies
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

## Usage Example

```python
from backtesting_framework import BacktestingEngine, create_default_strategies
import redis
from datetime import datetime

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize backtesting engine
engine = BacktestingEngine(redis_client)

# Get available strategies
strategies = create_default_strategies(10000.0)

# Run backtest for a specific strategy
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

result = engine.run_backtest(strategies[0], "AAPL", start_date, end_date)

print(f"Strategy: {result.strategy_name}")
print(f"Total Return: {result.total_return:.2%}")
print(f"Final Capital: ${result.final_capital:.2f}")
```

## Integration with Existing System

The backtesting framework integrates seamlessly with:
- Existing market data pipeline
- Redis data storage
- Authentication and authorization system
- Monitoring and logging infrastructure
- WebSocket communication for real-time updates

## Future Enhancements

1. **Advanced Strategies**: Add more sophisticated trading strategies (momentum, mean reversion, etc.)
2. **Risk Management**: Enhanced position sizing and risk controls
3. **Portfolio Optimization**: Multi-asset backtesting capabilities
4. **Visualization**: Charting and reporting features
5. **Parallel Processing**: Faster backtesting with multi-threading
6. **External Data Sources**: Integration with additional data providers
7. **Strategy Builder**: GUI for creating custom strategies