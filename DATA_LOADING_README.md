# OHLCV Data Loading Documentation

This document explains how to load your 17+ years of 1-minute OHLCV historical data into the AI-Driven Multi-Agent Stock Market Lab system.

## Data Format Requirements

Your CSV file should contain the following columns:
- `datetime` or `date`: Timestamp in ISO format (e.g., "2020-01-01 09:30:00" or "2020-01-01")
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price  
- `close`: Closing price
- `volume`: Trading volume

Example row:
```
2020-01-01 09:30:00,135.23,135.87,135.12,135.78,1250000
```

## Loading Data

### Using the Data Loading Script

```bash
# Basic usage
python data_loading_script.py --data-file /path/to/your/data.csv --symbol AAPL

# With custom Redis settings
python data_loading_script.py \
    --data-file /path/to/your/data.csv \
    --symbol AAPL \
    --redis-host localhost \
    --redis-port 6379
```

### Data Storage Structure

The data will be stored in Redis with the following structure:
- `historical_prices`: General list of all historical data points
- `historical_prices_{SYMBOL}`: Symbol-specific historical data (e.g., `historical_prices_AAPL`)

## Validation

After loading, you can validate the data:

```bash
python validate_data.py
```

This will show:
- Total number of data points loaded
- Sample data points with timestamps and prices
- Symbol-specific data availability

## Benefits of Your Data

With 17+ years of 1-minute data (2008-2025), you gain:

1. **Enhanced LSTM Training**: 17 years of high-frequency data for robust model training
2. **Comprehensive Backtesting**: Test strategies across multiple market cycles and conditions
3. **Risk Analysis**: Analyze volatility patterns over extended periods
4. **Performance Benchmarking**: Validate trading strategies against historical performance
5. **Market Condition Learning**: Models learn from various bull/bear markets, volatility regimes, and economic conditions

## Important Notes

1. **Data Volume**: 17 years of 1-minute data equals approximately 4 million data points
2. **Memory Usage**: Ensure your Redis instance has sufficient memory
3. **Processing Time**: Large datasets may take several minutes to load
4. **Data Quality**: The script handles missing values and parsing errors gracefully

## Next Steps

After loading your data:
1. Restart the Python agents service to pick up the new data
2. Run backtests to validate your strategies
3. Monitor LSTM agent performance with the new data
4. Consider running manual retraining of the LSTM model (`POST /lstm/retrain`)