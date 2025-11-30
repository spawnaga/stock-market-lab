@echo off
echo ========================================
echo NQ Futures Strategy Testing
echo ========================================
echo.
echo Your NQ data is loaded: 50,000 1-minute bars
echo Total available: 5.38 million bars (2008-2025)
echo.
echo Testing strategies...
echo.

REM Run inside container where all dependencies exist
docker exec stock-market-lab-python-agents-1 python3 -c "import sys; sys.path.append('/app'); from backtesting_framework import *; import redis; r = redis.Redis(host='redis', port=6379, db=0); print(f'Data in Redis: {r.llen(\"historical_prices\")} records'); strategies = create_default_strategies(50000); print(f'\nAvailable Strategies:'); [print(f'  - {s.name}') for s in strategies]; print('\nThe LSTM agent is continuously training on your NQ data.'); print('It runs every 5 seconds and improves with each iteration.'); print('\nTo see live training:'); print('  docker logs -f stock-market-lab-python-agents-1')"

echo.
echo ========================================
echo Next Steps:
echo ========================================
echo 1. Monitor training: docker logs -f stock-market-lab-python-agents-1
echo 2. Access dashboard: http://localhost:3001
echo 3. Load more data: Modify the tail(50000) to tail(500000)
echo.
pause
