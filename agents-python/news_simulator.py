#!/usr/bin/env python3
"""
Simulator for news articles to feed into sentiment analysis agents.
"""

import json
import redis
import time
import random
from datetime import datetime

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# Sample news articles
NEWS_ARTICLES = [
    "Apple Inc. reports record quarterly earnings beating analyst expectations by 15%",
    "Microsoft announces major AI partnership with leading tech company",
    "Google stock surges after breakthrough in quantum computing research",
    "Amazon faces regulatory scrutiny over marketplace practices",
    "Tesla announces new battery technology that could revolutionize EV industry",
    "Federal Reserve hints at potential interest rate cuts in upcoming meeting",
    "Oil prices spike amid geopolitical tensions in Middle East region",
    "Tech stocks rally as investor confidence improves following strong earnings season",
    "Company XYZ reports significant revenue growth in Q3 with 25% increase",
    "Market volatility increases as investors react to economic data releases"
]

def simulate_news_feed():
    """Simulate continuous news feed for sentiment analysis."""
    print("Starting news simulation...")
    
    while True:
        # Pick a random article
        article = random.choice(NEWS_ARTICLES)
        
        # Add to Redis list
        redis_client.lpush("news_articles", article)
        
        # Keep only last 10 articles
        redis_client.ltrim("news_articles", 0, 9)
        
        print(f"News article added: {article}")
        
        # Wait between articles
        time.sleep(random.randint(5, 15))

def main():
    """Main function to start news simulation."""
    try:
        simulate_news_feed()
    except KeyboardInterrupt:
        print("\nNews simulation stopped.")

if __name__ == "__main__":
    main()