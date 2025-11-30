#!/bin/bash
# Load NQ data into the running system

echo "🚀 Loading NQ Futures Data..."

# Copy data file into container
docker cp "F:/Market Data/Extracted/Futures_Original_full_1min_continuous_adjusted/NQ.csv" stock-market-lab-python-agents-1:/app/nq_data.csv

# Run data loading script inside container
docker exec stock-market-lab-python-agents-1 python3 - <<'EOF'
import pandas as pd
import redis
import json
import sys

print("📊 Loading NQ data...")

# Read CSV
df = pd.read_csv('/app/nq_data.csv', header=None, 
                 names=['datetime', 'open', 'high', 'low', 'close', 'volume'])

print(f"   Total records: {len(df):,}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Sample last 50k records for faster processing
df = df.tail(50000)
print(f"   Using last {len(df):,} records")

# Connect to Redis
r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# Clear old data
r.delete('historical_prices')
r.delete('historical_prices_NQ')

# Load data
count = 0
for _, row in df.iterrows():
    data_point = {
        'timestamp': pd.Timestamp(row['datetime']).timestamp(),
        'open': float(row['open']),
        'high': float(row['high']),
        'low': float(row['low']),
        'close': float(row['close']),
        'volume': int(row['volume']),
        'symbol': 'NQ'
    }
    
    json_data = json.dumps(data_point)
    r.lpush('historical_prices', json_data)
    r.lpush('historical_prices_NQ', json_data)
    
    count += 1
    if count % 5000 == 0:
        print(f"   Loaded {count:,} records...")

print(f"✅ Loaded {count:,} NQ records to Redis")
EOF

echo ""
echo "✅ NQ data loaded successfully!"
echo ""
echo "Next: Train the model and test strategies"
echo "Run: python train_nq_strategy.py"
