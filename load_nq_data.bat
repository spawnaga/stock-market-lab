@echo off
REM Load NQ data into the running system

echo Loading NQ Futures Data...

REM Copy data file into container
docker cp "F:\Market Data\Extracted\Futures_Original_full_1min_continuous_adjusted\NQ.csv" stock-market-lab-python-agents-1:/app/nq_data.csv

REM Run data loading script inside container
docker exec stock-market-lab-python-agents-1 python3 -c "import pandas as pd; import redis; import json; df = pd.read_csv('/app/nq_data.csv', header=None, names=['datetime', 'open', 'high', 'low', 'close', 'volume']); print(f'Total records: {len(df):,}'); df = df.tail(50000); print(f'Using last {len(df):,} records'); r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True); r.delete('historical_prices'); r.delete('historical_prices_NQ'); [r.lpush('historical_prices', json.dumps({'timestamp': pd.Timestamp(row['datetime']).timestamp(), 'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']), 'close': float(row['close']), 'volume': int(row['volume']), 'symbol': 'NQ'})) for _, row in df.iterrows()]; print('Data loaded successfully')"

echo.
echo NQ data loaded successfully!
echo.
echo The LSTM agent will automatically start training on this data.
echo Monitor training: docker logs -f stock-market-lab-python-agents-1
