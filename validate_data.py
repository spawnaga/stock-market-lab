#!/usr/bin/env python3
"""
Validation script to verify that OHLCV data has been loaded correctly
into the Redis database for the stock market lab.
"""

import redis
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data_in_redis():
    """Validate that data has been loaded correctly in Redis."""
    try:
        # Connect to Redis
        r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        r.ping()
        logger.info("Connected to Redis successfully")
        
        # Check for historical prices
        historical_data = r.lrange("historical_prices", 0, 9)  # Get first 10 items
        
        if not historical_data:
            logger.warning("No historical data found in 'historical_prices'")
            return False
            
        logger.info(f"Found {len(historical_data)} data points in historical_prices")
        
        # Show sample data
        logger.info("Sample data points:")
        for i, data_point in enumerate(historical_data[:3]):
            try:
                parsed_data = json.loads(data_point)
                logger.info(f"  Point {i+1}: {parsed_data['timestamp']} - Close: ${parsed_data['close']}")
            except Exception as e:
                logger.warning(f"  Could not parse data point {i+1}: {e}")
                
        # Check for symbol-specific data
        symbols = r.keys("historical_prices_*")
        if symbols:
            logger.info(f"Found symbol-specific data for symbols: {[s.replace('historical_prices_', '') for s in symbols]}")
            for symbol_key in symbols[:3]:  # Show first 3 symbols
                symbol_data = r.lrange(symbol_key, 0, 4)  # Get first 5 items
                symbol_name = symbol_key.replace('historical_prices_', '')
                logger.info(f"  {symbol_name}: {len(symbol_data)} data points")
        else:
            logger.info("No symbol-specific data found")
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

def main():
    logger.info("Validating OHLCV data in Redis...")
    success = validate_data_in_redis()
    
    if success:
        logger.info("Data validation completed successfully!")
        return 0
    else:
        logger.error("Data validation failed!")
        return 1

if __name__ == "__main__":
    exit(main())