from backtesting_framework import *
from datetime import datetime, timedelta
import redis

r = redis.Redis(host='redis', port=6379, db=0)
engine = BacktestingEngine(r)
strategies = create_default_strategies(50000)

end = datetime.now()
start = end - timedelta(days=7)

print('\n=== NQ Strategy Comparison ===\n')
results = engine.compare_strategies(strategies, 'NQ', start, end)

print(f"{'Strategy':<20} {'Return %':<12} {'Sharpe':<10} {'Trades':<10}")
print('='*55)

for name, r in results.items():
    if r:
        print(f"{name:<20} {r.total_return:>10.2f}% {r.sharpe_ratio:>9.2f} {r.trade_count:>9}")
    else:
        print(f"{name:<20} FAILED")

print('\n=== Summary ===')
print(f'Total NQ data points: {redis.Redis(host="redis", port=6379, db=0).llen("historical_prices")}')
print('LSTM agent is training continuously in the background')
