#!/usr/bin/env python3
"""
Utility script to demonstrate how to send trades to Interactive Brokers.
This would typically be called by the strategy lab or other components.
"""

import json
import redis
import time
import uuid
from datetime import datetime

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def send_trade(symbol, quantity, action, price, account="DU123456"):
    """
    Send a trade request to be processed by the Interactive Brokers agent.
    
    Args:
        symbol (str): Stock symbol
        quantity (int): Number of shares
        action (str): 'BUY' or 'SELL'
        price (float): Limit price
        account (str): IB account number
    """
    
    trade = {
        "id": str(uuid.uuid4()),
        "symbol": symbol,
        "quantity": quantity,
        "action": action.upper(),
        "price": price,
        "account": account,
        "timestamp": datetime.now().isoformat(),
        "status": "pending"
    }
    
    # Add to Redis queue for processing
    redis_client.lpush("pending_trades", json.dumps(trade))
    
    print(f"Trade sent: {trade}")
    return trade["id"]

def main():
    """Demonstrate sending sample trades."""
    print("Sending sample trades to Interactive Brokers...")
    
    # Send a few sample trades
    trade_ids = []
    
    # Buy AAPL
    trade_id = send_trade("AAPL", 10, "BUY", 175.50)
    trade_ids.append(trade_id)
    
    # Sell MSFT
    trade_id = send_trade("MSFT", 5, "SELL", 330.25)
    trade_ids.append(trade_id)
    
    # Buy GOOGL
    trade_id = send_trade("GOOGL", 3, "BUY", 135.75)
    trade_ids.append(trade_id)
    
    print(f"Trades queued with IDs: {trade_ids}")
    
    # Wait for processing
    print("Waiting for trades to be processed...")
    time.sleep(10)
    
    print("Done.")

if __name__ == "__main__":
    main()