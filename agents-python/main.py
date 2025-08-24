#!/usr/bin/env python3
"""
Main entry point for the Python agents service.
This service runs multiple AI agents that analyze market data and make trading decisions.
"""

import threading
import time
import json
import os
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock-market-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Redis connection for real-time data
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# In-memory storage for strategies (would be replaced with DB in production)
strategies = []

# Import market data handler
from market_data import MarketDataHandler

class BaseAgent:
    """Base class for all trading agents."""
    
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        self.guardrails_enabled = True  # Enable guardrails by default
        
    def start(self):
        """Start the agent."""
        self.running = True
        logger.info(f"Starting {self.agent_type} agent {self.agent_id}")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        logger.info(f"Stopping {self.agent_type} agent {self.agent_id}")
        
    def apply_guardrails(self, decision):
        """Apply safety checks and guardrails to agent decisions."""
        if not self.guardrails_enabled:
            return decision
            
        # Basic guardrail checks
        if 'action' in decision:
            # Prevent extreme actions that could be risky
            if decision['action'] in ['sell_all', 'buy_all'] and decision.get('confidence', 0) > 0.9:
                # Reduce confidence for extreme actions
                decision['confidence'] = min(decision.get('confidence', 1.0) * 0.7, 0.8)
                decision['reason'] += " (Guardrail: Reduced confidence for extreme action)"
                
            # Check for suspicious confidence levels
            if decision.get('confidence', 0) > 0.95:
                decision['confidence'] = 0.95
                decision['reason'] += " (Guardrail: Confidence capped at 95%)"
                
        return decision

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
                    
                    # Apply guardrails
                    decision = self.apply_guardrails(decision)
                    
                    # Publish decision to Kafka-like topic
                    socketio.emit('agent_decision', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'decision': decision,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in RL agent: {e}")
                
            time.sleep(2)  # Poll every 2 seconds
            
    def _make_decision(self, market_data):
        """Placeholder for actual RL decision making logic."""
        # This would normally use a trained RL model
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
                    
                    # Apply guardrails
                    prediction = self.apply_guardrails(prediction)
                    
                    # Publish prediction to Kafka-like topic
                    socketio.emit('price_prediction', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'prediction': prediction,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in LSTM agent: {e}")
                
            time.sleep(5)  # Poll every 5 seconds
            
    def _predict_price(self, historical_data):
        """Placeholder for actual LSTM prediction logic."""
        # This would normally use a trained LSTM model
        return {
            "predicted_price": 175.50,
            "confidence": 0.75,
            "direction": "up"
        }

class NewsSentimentAgent(BaseAgent):
    """NLP agent for analyzing news sentiment."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "News/NLP")
        
    def run(self):
        """Main execution loop for sentiment agent."""
        while self.running:
            try:
                # Get recent news articles
                news_articles = redis_client.lrange("news_articles", 0, 9)
                if news_articles:
                    # Process with NLP model (placeholder)
                    sentiment = self._analyze_sentiment(news_articles)
                    
                    # Apply guardrails
                    sentiment = self.apply_guardrails(sentiment)
                    
                    # Publish sentiment to Kafka-like topic
                    socketio.emit('sentiment_analysis', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'sentiment': sentiment,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Error in News agent: {e}")
                
            time.sleep(10)  # Poll every 10 seconds
            
    def _analyze_sentiment(self, news_articles):
        """Placeholder for actual sentiment analysis logic."""
        # This would normally use a transformer model for sentiment analysis
        return {
            "overall_sentiment": "positive",
            "confidence": 0.85,
            "key_topics": ["earnings", "market"]
        }

# Global variables for market data
market_data_handler = None
data_streaming_thread = None

# Initialize agents
rl_agent = RLAgent("rl-001")
lstm_agent = LSTMPricePredictor("lstm-001")
news_agent = NewsSentimentAgent("news-001")

def start_market_data_streaming():
    """Start streaming real market data."""
    global market_data_handler, data_streaming_thread
    
    # Get market data provider from environment or default to polygon
    provider = os.getenv('MARKET_DATA_PROVIDER', 'polygon')
    api_key = os.getenv('MARKET_DATA_API_KEY')
    
    try:
        market_data_handler = MarketDataHandler(provider, api_key)
        logger.info(f"Initialized {provider} market data handler")
        
        # Start data streaming in a separate thread
        data_streaming_thread = threading.Thread(
            target=market_data_handler.start_data_streaming,
            args=(redis_client,),
            daemon=True
        )
        data_streaming_thread.start()
        logger.info("Started market data streaming")
        
    except Exception as e:
        logger.error(f"Failed to initialize market data streaming: {e}")
        # Fall back to dummy data if real data fails
        logger.info("Falling back to dummy data simulation")

def start_agents():
    """Start all agents in separate threads."""
    logger.info("Starting all agents...")
    
    # Start agents
    rl_thread = threading.Thread(target=rl_agent.run)
    lstm_thread = threading.Thread(target=lstm_agent.run)
    news_thread = threading.Thread(target=news_agent.run)
    
    rl_thread.daemon = True
    lstm_thread.daemon = True
    news_thread.daemon = True
    
    rl_thread.start()
    lstm_thread.start()
    news_thread.start()
    
    # Start agents in running state
    rl_agent.start()
    lstm_agent.start()
    news_agent.start()
    
    logger.info("All agents started successfully")

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "agents": ["RL", "LSTM", "News/NLP"]})

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

@app.route('/override/<agent_id>', methods=['POST'])
def override_agent_decision(agent_id):
    """Allow human override of agent decisions."""
    data = request.get_json()
    
    if not data or 'override_action' not in data:
        return jsonify({"error": "Override action is required"}), 400
    
    # In a real implementation, this would validate the override and apply it
    override_data = {
        "agent_id": agent_id,
        "override_action": data['override_action'],
        "override_reason": data.get('reason', ''),
        "timestamp": time.time(),
        "user": data.get('user', 'human')
    }
    
    # Emit override event to frontend
    socketio.emit('agent_override', override_data)
    
    return jsonify({"status": "override applied", "data": override_data}), 200

@app.route('/guardrails/<agent_id>/<setting>', methods=['PUT'])
def toggle_guardrails(agent_id, setting):
    """Enable/disable guardrails for a specific agent."""
    try:
        # Find the agent and toggle guardrails
        # This is a simplified implementation - in practice, you'd have a registry of agents
        if setting.lower() in ['enable', 'disable']:
            # In a real implementation, we'd maintain a registry of agents
            # For now, we'll just return success
            return jsonify({
                "status": "guardrails toggled",
                "agent_id": agent_id,
                "setting": setting,
                "message": f"Guardrails {'enabled' if setting.lower() == 'enable' else 'disabled'} for agent {agent_id}"
            }), 200
        else:
            return jsonify({"error": "Invalid setting. Use 'enable' or 'disable'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connections."""
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections."""
    logger.info('Client disconnected')

if __name__ == '__main__':
    # Start market data streaming
    start_market_data_streaming()
    
    # Start agents
    start_agents()
    
    # Run the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)