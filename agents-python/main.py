#!/usr/bin/env python3
"""
Main entry point for the Python agents service.
This service runs multiple AI agents that analyze market data and make trading decisions.
"""

import threading
import time
import json
import os
from datetime import datetime
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
import redis
import logging
import requests
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from logging.handlers import RotatingFileHandler

# Configure logging with rotation to prevent memory exhaustion
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup rotating file handler
    log_handler = RotatingFileHandler('logs/agents.log', maxBytes=1024*1024, backupCount=5)  # 1MB max, 5 backups
    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d - %(message)s'
    )
    log_handler.setFormatter(log_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Setup logging
logger = setup_logging()

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock-market-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis connection for real-time data
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# IB Gateway configuration (can be loaded from environment variables)
IB_CONFIG = {
    'host': os.getenv('IB_HOST', '127.0.0.1'),
    'port': int(os.getenv('IB_PORT', 4001)),
    'client_id': int(os.getenv('IB_CLIENT_ID', 1001))
}

# Log counter for memory management
log_counter = 0
log_reset_threshold = 10

def rotate_log_if_needed():
    """Rotate logs every 10 log entries to prevent memory exhaustion."""
    global log_counter
    log_counter += 1
    if log_counter >= log_reset_threshold:
        logger.info("Log rotation threshold reached, resetting counters")
        log_counter = 0

# In-memory storage for strategies (would be replaced with DB in production)
strategies = []


class BaseAgent:
    """Base class for all trading agents."""
    
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        rotate_log_if_needed()
        logger.info(f"Initializing {self.agent_type} agent {self.agent_id}")
        
    def start(self):
        """Start the agent."""
        self.running = True
        rotate_log_if_needed()
        logger.info(f"Starting {self.agent_type} agent {self.agent_id}")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        rotate_log_if_needed()
        logger.info(f"Stopping {self.agent_type} agent {self.agent_id}")

class RLAgent(BaseAgent):
    """Reinforcement Learning Agent for trading decisions."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "RL")
        self.model = None  # Placeholder for actual model
        
    def run(self):
        """Main execution loop for RL agent."""
        while self.running:
            # Simulate processing market data
            try:
                # Get latest market data from Redis
                market_data = redis_client.get("latest_market_data")
                if market_data:
                    data = json.loads(market_data)
                    # Process with RL model (placeholder)
                    decision = self._make_decision(data)
                    
                    # Publish decision to Kafka-like topic
                    socketio.emit('agent_decision', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'decision': decision,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in RL agent: {e}")
                
            time.sleep(2)  # Poll every 2 seconds
            
    def _make_decision(self, market_data):
        """Placeholder for actual RL decision making logic."""
        # This would normally use a trained RL model
        rotate_log_if_needed()
        logger.debug(f"Making RL decision for {market_data.get('symbol', 'unknown')}")
        return {
            "action": "hold",
            "confidence": 0.8,
            "reason": "Market conditions stable"
        }

class LSTMPricePredictor(BaseAgent):
    """LSTM-based price prediction agent."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "LSTM")
        self.model = None  # Placeholder for actual model
        
    def run(self):
        """Main execution loop for LSTM agent."""
        while self.running:
            try:
                # Get historical data
                historical_data = redis_client.lrange("historical_prices", 0, 99)
                if historical_data:
                    # Process with LSTM model (placeholder)
                    prediction = self._predict_price(historical_data)
                    
                    # Publish prediction to Kafka-like topic
                    socketio.emit('price_prediction', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'prediction': prediction,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in LSTM agent: {e}")
                
            time.sleep(5)  # Poll every 5 seconds
            
    def _predict_price(self, historical_data):
        """Placeholder for actual LSTM prediction logic."""
        # This would normally use a trained LSTM model
        rotate_log_if_needed()
        logger.debug(f"LSTM predicting on {len(historical_data)} data points")
        return {
            "predicted_price": 175.50,
            "confidence": 0.75,
            "direction": "up"
        }

class NewsSentimentAgent(BaseAgent):
    """NLP agent for analyzing news sentiment."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "News/NLP")
        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            rotate_log_if_needed()
            logger.info("Successfully initialized sentiment analysis pipeline")
        except Exception as e:
            rotate_log_if_needed()
            logger.warning(f"Could not initialize sentiment pipeline: {e}")
            self.sentiment_pipeline = None
            
    def run(self):
        """Main execution loop for sentiment agent."""
        while self.running:
            try:
                # Get recent news articles
                news_articles = redis_client.lrange("news_articles", 0, 9)
                if news_articles:
                    # Process with NLP model (placeholder)
                    sentiment = self._analyze_sentiment(news_articles)
                    
                    # Publish sentiment to Kafka-like topic
                    socketio.emit('sentiment_analysis', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'sentiment': sentiment,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in News agent: {e}")
                
            time.sleep(10)  # Poll every 10 seconds
            
    def _analyze_sentiment(self, news_articles):
        """Analyze sentiment from news articles."""
        rotate_log_if_needed()
        if not self.sentiment_pipeline:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.5,
                "key_topics": ["news", "market"],
                "score": 0.0
            }
            
        try:
            # Combine all articles for analysis
            combined_text = " ".join([article for article in news_articles if article])
            
            if not combined_text.strip():
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "key_topics": ["no_content"],
                    "score": 0.0
                }
                
            # Analyze sentiment
            results = self.sentiment_pipeline(combined_text[:512])  # Limit to 512 tokens
            
            # Process results
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 0.0
            
            for result in results:
                if result['label'] == 'LABEL_2':  # Positive
                    positive_score = result['score']
                elif result['label'] == 'LABEL_0':  # Negative
                    negative_score = result['score']
                else:  # Neutral
                    neutral_score = result['score']
                    
            # Determine overall sentiment
            if positive_score > negative_score and positive_score > neutral_score:
                overall_sentiment = "positive"
                score = positive_score
            elif negative_score > positive_score and negative_score > neutral_score:
                overall_sentiment = "negative"
                score = negative_score
            else:
                overall_sentiment = "neutral"
                score = neutral_score
                
            return {
                "overall_sentiment": overall_sentiment,
                "confidence": max(positive_score, negative_score, neutral_score),
                "key_topics": ["earnings", "market", "news"],
                "score": score
            }
            
        except Exception as e:
            rotate_log_if_needed()
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.5,
                "key_topics": ["error"],
                "score": 0.0
            }

class StockTwitsSentimentAgent(BaseAgent):
    """Agent for analyzing sentiment from StockTwits."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "StockTwits")
        self.api_key = os.getenv('STOCKTWITS_API_KEY', '')
        
    def run(self):
        """Main execution loop for StockTwits sentiment agent."""
        while self.running:
            try:
                # Simulate fetching StockTwits data
                sentiment = self._fetch_stocktwits_sentiment()
                
                if sentiment:
                    # Publish sentiment to Kafka-like topic
                    socketio.emit('stocktwits_sentiment', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'sentiment': sentiment,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in StockTwits agent: {e}")
                
            time.sleep(30)  # Poll every 30 seconds
            
    def _fetch_stocktwits_sentiment(self):
        """Fetch and analyze sentiment from StockTwits API."""
        # In a real implementation, this would make actual API calls
        # For now, we'll simulate some data
        try:
            rotate_log_if_needed()
            logger.debug("Fetching StockTwits sentiment data")
            # Simulated sentiment data
            sentiments = ['positive', 'negative', 'neutral']
            sentiment = np.random.choice(sentiments, p=[0.3, 0.2, 0.5])
            
            return {
                "overall_sentiment": sentiment,
                "confidence": round(np.random.uniform(0.6, 0.9), 2),
                "key_topics": ["stocks", "trading", "market"],
                "score": round(np.random.uniform(-1, 1), 2),
                "messages_count": np.random.randint(100, 1000),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            rotate_log_if_needed()
            logger.error(f"Error fetching StockTwits data: {e}")
            return None

class TwitterSentimentAgent(BaseAgent):
    """Agent for analyzing sentiment from Twitter/X."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "Twitter")
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        
    def run(self):
        """Main execution loop for Twitter sentiment agent."""
        while self.running:
            try:
                # Simulate fetching Twitter data
                sentiment = self._fetch_twitter_sentiment()
                
                if sentiment:
                    # Publish sentiment to Kafka-like topic
                    socketio.emit('twitter_sentiment', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'sentiment': sentiment,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in Twitter agent: {e}")
                
            time.sleep(30)  # Poll every 30 seconds
            
    def _fetch_twitter_sentiment(self):
        """Fetch and analyze sentiment from Twitter API."""
        # In a real implementation, this would make actual API calls
        # For now, we'll simulate some data
        try:
            rotate_log_if_needed()
            logger.debug("Fetching Twitter sentiment data")
            # Simulated sentiment data
            sentiments = ['positive', 'negative', 'neutral']
            sentiment = np.random.choice(sentiments, p=[0.35, 0.15, 0.5])
            
            return {
                "overall_sentiment": sentiment,
                "confidence": round(np.random.uniform(0.5, 0.85), 2),
                "key_topics": ["stocks", "market", "finance"],
                "score": round(np.random.uniform(-1, 1), 2),
                "tweets_count": np.random.randint(50, 500),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            rotate_log_if_needed()
            logger.error(f"Error fetching Twitter data: {e}")
            return None

class InteractiveBrokersAgent(BaseAgent):
    """Agent for interacting with Interactive Brokers."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "InteractiveBrokers")
        self.ib_connected = False
        self.ib_client = None
        
    def run(self):
        """Main execution loop for IB agent."""
        while self.running:
            try:
                # Check if we should attempt to connect
                if not self.ib_connected:
                    self._connect_to_ib()
                
                # Process any pending orders or trades
                if self.ib_connected:
                    self._process_trades()
                    
            except Exception as e:
                rotate_log_if_needed()
                logger.error(f"Error in IB agent: {e}")
                self.ib_connected = False  # Reset connection
                
            time.sleep(5)  # Poll every 5 seconds
            
    def _connect_to_ib(self):
        """Connect to Interactive Brokers."""
        try:
            rotate_log_if_needed()
            logger.info("Attempting to connect to Interactive Brokers...")
            
            # Simulate successful connection
            self.ib_connected = True
            rotate_log_if_needed()
            logger.info("Successfully connected to Interactive Brokers")
            
        except Exception as e:
            rotate_log_if_needed()
            logger.error(f"Failed to connect to Interactive Brokers: {e}")
            
    def _process_trades(self):
        """Process trades and orders."""
        try:
            # Get pending trades from Redis
            pending_trades = redis_client.lrange("pending_trades", 0, -1)
            
            for trade_json in pending_trades:
                trade = json.loads(trade_json)
                
                # In a real implementation, this would place the order via IB API
                rotate_log_if_needed()
                logger.info(f"Processing trade: {trade}")
                
                # Simulate trade execution
                trade_result = {
                    "trade_id": trade.get("id", "unknown"),
                    "status": "executed",
                    "execution_time": time.time(),
                    "filled_quantity": trade.get("quantity", 0),
                    "average_fill_price": trade.get("price", 0.0)
                }
                
                # Publish trade result
                socketio.emit('trade_execution', {
                    'agent_id': self.agent_id,
                    'agent_type': self.agent_type,
                    'result': trade_result,
                    'timestamp': time.time()
                })
                
                # Remove processed trade from queue
                redis_client.lrem("pending_trades", 1, trade_json)
                
        except Exception as e:
            rotate_log_if_needed()
            logger.error(f"Error processing trades: {e}")

# Initialize agents
rl_agent = RLAgent("rl-001")
lstm_agent = LSTMPricePredictor("lstm-001")
news_agent = NewsSentimentAgent("news-001")
stocktwits_agent = StockTwitsSentimentAgent("stocktwits-001")
twitter_agent = TwitterSentimentAgent("twitter-001")
ib_agent = InteractiveBrokersAgent("ib-001")

def start_agents():
    """Start all agents in separate threads."""
    rotate_log_if_needed()
    logger.info("Starting all agents...")
    
    # Start agents
    rl_thread = threading.Thread(target=rl_agent.run)
    lstm_thread = threading.Thread(target=lstm_agent.run)
    news_thread = threading.Thread(target=news_agent.run)
    stocktwits_thread = threading.Thread(target=stocktwits_agent.run)
    twitter_thread = threading.Thread(target=twitter_agent.run)
    ib_thread = threading.Thread(target=ib_agent.run)
    
    rl_thread.daemon = True
    lstm_thread.daemon = True
    news_thread.daemon = True
    stocktwits_thread.daemon = True
    twitter_thread.daemon = True
    ib_thread.daemon = True
    
    rl_thread.start()
    lstm_thread.start()
    news_thread.start()
    stocktwits_thread.start()
    twitter_thread.start()
    ib_thread.start()
    
    # Start agents in running state
    rl_agent.start()
    lstm_agent.start()
    news_agent.start()
    stocktwits_agent.start()
    twitter_agent.start()
    ib_agent.start()
    
    rotate_log_if_needed()
    logger.info("All agents started successfully")

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "agents": ["RL", "LSTM", "News/NLP", "StockTwits", "Twitter", "IB"]})

@app.route('/strategies', methods=['GET'])
def get_strategies():
    """Get all saved strategies."""
    return jsonify({"strategies": strategies})

@app.route('/strategies', methods=['POST'])
def create_strategy():
    """Create a new strategy."""
    data = request.get_json()
    
    if not data or 'name' not in data or 'description' not in data:
        return jsonify({"error": "Name and description are required"}), 400
    
    strategy = {
        "id": f"strat-{len(strategies) + 1}",
        "name": data['name'],
        "description": data['description'],
        "parameters": data.get('parameters', {}),
        "createdAt": time.time()
    }
    
    strategies.append(strategy)
    
    # Notify frontend about new strategy
    socketio.emit('strategy_created', strategy)
    
    return jsonify(strategy), 201

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connections."""
    rotate_log_if_needed()
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections."""
    rotate_log_if_needed()
    logger.info('Client disconnected')

if __name__ == '__main__':
    # Start agents
    start_agents()
    
    # Run the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)