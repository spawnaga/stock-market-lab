#!/usr/bin/env python3
"""
Market data integration module for connecting to real market data providers.
Supports Polygon.io, Tradier, and Schwab APIs.
"""

import os
import time
import json
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketDataConnector:
    """Base class for market data connectors."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('MARKET_DATA_API_KEY')
        self.base_url = ""
        self.session = requests.Session()
        
    def get_latest_prices(self, symbols: List[str]) -> Dict:
        """Get latest prices for given symbols."""
        raise NotImplementedError
        
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data for a symbol."""
        raise NotImplementedError
        
    def get_news(self, symbols: List[str], limit: int = 10) -> List[Dict]:
        """Get recent news for given symbols."""
        raise NotImplementedError

class PolygonConnector(MarketDataConnector):
    """Connector for Polygon.io API."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.polygon.io/v2"
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}'
        })
        
    def get_latest_prices(self, symbols: List[str]) -> Dict:
        """Get latest prices for given symbols from Polygon."""
        prices = {}
        try:
            # Get quotes for multiple symbols
            for symbol in symbols:
                url = f"{self.base_url}/quotes/{symbol}"
                params = {
                    'limit': 1,
                    'apiKey': self.api_key
                }
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data and len(data['results']) > 0:
                        quote = data['results'][0]
                        prices[symbol] = {
                            'symbol': symbol,
                            'price': quote.get('p', 0),
                            'size': quote.get('s', 0),
                            'timestamp': quote.get('t', 0),
                            'exchange': quote.get('x', '')
                        }
        except Exception as e:
            logger.error(f"Error getting prices from Polygon: {e}")
            
        return prices
        
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data for a symbol from Polygon."""
        historical_data = []
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apiKey': self.api_key,
                'unadjusted': 'true'
            }
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    for agg in data['results']:
                        historical_data.append({
                            'timestamp': agg.get('t'),
                            'open': agg.get('o'),
                            'high': agg.get('h'),
                            'low': agg.get('l'),
                            'close': agg.get('c'),
                            'volume': agg.get('v'),
                            'symbol': symbol
                        })
        except Exception as e:
            logger.error(f"Error getting historical data from Polygon: {e}")
            
        return historical_data
        
    def get_news(self, symbols: List[str], limit: int = 10) -> List[Dict]:
        """Get recent news for given symbols from Polygon."""
        news_articles = []
        try:
            # Polygon doesn't have a direct news API, so we'll simulate
            # In a real implementation, you'd use their news API or another source
            for symbol in symbols:
                # Simulated news data
                news_articles.extend([
                    {
                        'symbol': symbol,
                        'title': f"Breaking News: {symbol} announces quarterly earnings",
                        'summary': f"Company {symbol} reports strong quarterly results with 15% revenue growth.",
                        'url': f"https://example.com/news/{symbol}-earnings",
                        'published_at': datetime.now().isoformat(),
                        'source': 'Simulated Source'
                    },
                    {
                        'symbol': symbol,
                        'title': f"Market Outlook: {symbol} stock shows positive momentum",
                        'summary': f"Analysts are bullish on {symbol} stock following recent market trends.",
                        'url': f"https://example.com/news/{symbol}-outlook",
                        'published_at': datetime.now().isoformat(),
                        'source': 'Simulated Source'
                    }
                ])
        except Exception as e:
            logger.error(f"Error getting news from Polygon: {e}")
            
        return news_articles[:limit]

class TradierConnector(MarketDataConnector):
    """Connector for Tradier API."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://sandbox.tradier.com/v1"  # Using sandbox for demo
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        })
        
    def get_latest_prices(self, symbols: List[str]) -> Dict:
        """Get latest prices for given symbols from Tradier."""
        prices = {}
        try:
            for symbol in symbols:
                url = f"{self.base_url}/markets/quotes"
                params = {
                    'symbols': symbol,
                    'include_all': 'true'
                }
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'quotes' in data and 'quote' in data['quotes']:
                        quote = data['quotes']['quote']
                        prices[symbol] = {
                            'symbol': symbol,
                            'price': quote.get('last', 0),
                            'change': quote.get('change', 0),
                            'change_percent': quote.get('change_percentage', 0),
                            'timestamp': quote.get('timestamp', ''),
                            'volume': quote.get('volume', 0)
                        }
        except Exception as e:
            logger.error(f"Error getting prices from Tradier: {e}")
            
        return prices
        
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical price data for a symbol from Tradier."""
        historical_data = []
        try:
            url = f"{self.base_url}/markets/history"
            params = {
                'symbol': symbol,
                'interval': 'daily',
                'days': days
            }
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'history' in data and 'day' in data['history']:
                    for day in data['history']['day']:
                        historical_data.append({
                            'timestamp': day.get('date'),
                            'open': day.get('open', 0),
                            'high': day.get('high', 0),
                            'low': day.get('low', 0),
                            'close': day.get('close', 0),
                            'volume': day.get('volume', 0),
                            'symbol': symbol
                        })
        except Exception as e:
            logger.error(f"Error getting historical data from Tradier: {e}")
            
        return historical_data
        
    def get_news(self, symbols: List[str], limit: int = 10) -> List[Dict]:
        """Get recent news for given symbols from Tradier."""
        news_articles = []
        try:
            # Tradier doesn't have a direct news API, so we'll simulate
            for symbol in symbols:
                news_articles.extend([
                    {
                        'symbol': symbol,
                        'title': f"Market Update: {symbol} stock performance",
                        'summary': f"Recent market activity shows {symbol} stock moving positively.",
                        'url': f"https://example.com/news/{symbol}-update",
                        'published_at': datetime.now().isoformat(),
                        'source': 'Simulated Source'
                    }
                ])
        except Exception as e:
            logger.error(f"Error getting news from Tradier: {e}")
            
        return news_articles[:limit]

class MarketDataHandler:
    """Handles market data integration and distribution to agents."""
    
    def __init__(self, connector_type: str = 'polygon', api_key: str = None):
        self.connector_type = connector_type
        self.api_key = api_key
        self.connector = self._initialize_connector()
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default symbols
        self.logger = logging.getLogger(__name__)
        
    def _initialize_connector(self) -> MarketDataConnector:
        """Initialize the appropriate market data connector."""
        if self.connector_type.lower() == 'polygon':
            return PolygonConnector(self.api_key)
        elif self.connector_type.lower() == 'tradier':
            return TradierConnector(self.api_key)
        else:
            raise ValueError(f"Unsupported connector type: {self.connector_type}")
            
    def set_symbols(self, symbols: List[str]):
        """Set the list of symbols to track."""
        self.symbols = symbols
        
    def get_latest_market_data(self) -> Dict:
        """Get latest market data for all tracked symbols."""
        return self.connector.get_latest_prices(self.symbols)
        
    def get_historical_data(self, days: int = 30) -> Dict:
        """Get historical data for all tracked symbols."""
        historical_data = {}
        for symbol in self.symbols:
            historical_data[symbol] = self.connector.get_historical_data(symbol, days)
        return historical_data
        
    def get_news(self) -> List[Dict]:
        """Get recent news for all tracked symbols."""
        return self.connector.get_news(self.symbols)
        
    def start_data_streaming(self, redis_client, interval: int = 5):
        """Continuously stream market data to Redis."""
        self.logger.info(f"Starting market data streaming with {self.connector_type} connector")
        
        while True:
            try:
                # Get latest prices
                latest_prices = self.get_latest_market_data()
                if latest_prices:
                    # Store in Redis
                    redis_client.set("latest_market_data", json.dumps(latest_prices))
                    
                # Get historical data
                historical_data = self.get_historical_data(30)
                if historical_data:
                    # Store in Redis as list
                    for symbol, data in historical_data.items():
                        if data:
                            # Clear old data and store new data
                            redis_client.delete(f"historical_prices_{symbol}")
                            for item in data:
                                redis_client.lpush(f"historical_prices_{symbol}", json.dumps(item))
                                
                # Get news
                news_articles = self.get_news()
                if news_articles:
                    # Store news in Redis
                    redis_client.delete("news_articles")
                    for article in news_articles:
                        redis_client.lpush("news_articles", json.dumps(article))
                        
                self.logger.info(f"Streamed market data for {len(self.symbols)} symbols")
                
            except Exception as e:
                self.logger.error(f"Error in market data streaming: {e}")
                
            time.sleep(interval)