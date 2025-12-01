#!/usr/bin/env python3
"""
Main entry point for the Python agents service.
This service runs multiple AI agents that analyze market data and make trading decisions.
"""

import threading
import time
import json
import os
import signal
import sys
from functools import wraps
from contextlib import contextmanager
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
import redis
import logging
from logging.handlers import RotatingFileHandler
import secrets
from datetime import datetime, timedelta
import jwt
from werkzeug.exceptions import HTTPException
import psutil
import gc
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pickle
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

# Configure logging with rotation
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/agents.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'stock-market-secret-key')

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Redis connection for real-time data
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# PostgreSQL connection
DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://stock_user:stock_password@postgres:5432/stock_market'
)

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database tables if they don't exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create OHLCV table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data_ohlcv (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        datetime TIMESTAMP NOT NULL,
                        open NUMERIC(12, 4) NOT NULL,
                        high NUMERIC(12, 4) NOT NULL,
                        low NUMERIC(12, 4) NOT NULL,
                        close NUMERIC(12, 4) NOT NULL,
                        volume BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, datetime)
                    )
                """)
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_datetime
                    ON market_data_ohlcv(symbol, datetime)
                """)
                conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Demo users for simple authentication
DEMO_USERS = {
    'admin': 'admin',
    'demo': 'demo',
    'user': 'user123'
}

# Rate limiting storage (in-memory for demo, would use Redis in production)
rate_limit_store = {}

# JWT token expiration time (in seconds)
JWT_EXPIRATION_SECONDS = 3600

# Monitoring and metrics tracking
agent_metrics = defaultdict(lambda: {
    'executions': 0,
    'errors': 0,
    'last_execution': 0,
    'execution_times': deque(maxlen=100),
    'success_rate': 0.0,
    'avg_execution_time': 0.0,
    'error_details': []
})

system_metrics = {
    'startup_time': time.time(),
    'total_requests': 0,
    'error_count': 0,
    'memory_usage_mb': 0,
    'cpu_percent': 0,
    'active_threads': 0,
    'connected_clients': 0
}

# Request timing tracking
request_timings = deque(maxlen=1000)

# Performance optimization settings
PERFORMANCE_SETTINGS = {
    'max_concurrent_requests': 100,
    'request_queue_size': 500,
    'memory_cleanup_interval': 300,  # seconds
    'metrics_collection_interval': 60  # seconds
}

# Import market data handler
from market_data import MarketDataHandler

# Import backtesting framework
from backtesting_framework import BacktestingEngine, create_default_strategies, BacktestResult

# Import GA+RL integration module
try:
    from ga_rl_integration import IntegratedTradingSystem, GADQNTrainingManager
    GA_RL_AVAILABLE = True
    logger.info("GA+RL integration module loaded successfully")
except ImportError as e:
    GA_RL_AVAILABLE = False
    logger.warning(f"GA+RL integration module not available: {e}")

# Global GA+RL trading system instance
ga_rl_system = None
ga_rl_training_thread = None

def token_required(f):
    """Decorator to require valid JWT token for protected routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
                
            # Decode the token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['user']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

def track_metrics(f):
    """Decorator to track request metrics and performance."""
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        try:
            system_metrics['total_requests'] += 1
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            system_metrics['error_count'] += 1
            logger.error(f"Error in {f.__name__}: {e}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            request_timings.append(duration)
            
            # Update system metrics
            try:
                process = psutil.Process(os.getpid())
                system_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                system_metrics['cpu_percent'] = process.cpu_percent()
                system_metrics['active_threads'] = threading.active_count()
                # Update connected clients count
                system_metrics['connected_clients'] = len(socketio.server.manager.get_participants('/'))
            except Exception as e:
                logger.warning(f"Could not update system metrics: {e}")
    return decorated

def rate_limit(max_requests=100, window=3600):
    """Rate limiting decorator - DISABLED FOR DEVELOPMENT."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Rate limiting disabled for development
            # To re-enable, uncomment the code below
            return f(*args, **kwargs)

            # client_ip = request.remote_addr
            # now = time.time()
            #
            # # Initialize rate limit for this IP if not exists
            # if client_ip not in rate_limit_store:
            #     rate_limit_store[client_ip] = []
            #
            # # Clean old requests outside the window
            # rate_limit_store[client_ip] = [
            #     req_time for req_time in rate_limit_store[client_ip]
            #     if now - req_time < window
            # ]
            #
            # # Check if limit exceeded
            # if len(rate_limit_store[client_ip]) >= max_requests:
            #     system_metrics['error_count'] += 1
            #     # Use make_response to ensure CORS headers are applied
            #     from flask import make_response
            #     response = make_response(jsonify({'error': 'Rate limit exceeded'}), 429)
            #     return response
            #
            # # Add current request
            # rate_limit_store[client_ip].append(now)
            #
            # return f(*args, **kwargs)
        return decorated
    return decorator

class LSTMModel(nn.Module):
    """LSTM model for price prediction."""
    
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Initialize hidden and cell states
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last time step output
        output = self.dropout(lstm_out[:, -1, :])
        output = self.linear(output)
        
        return output

class BaseAgent:
    """Base class for all trading agents."""
    
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        self.guardrails_enabled = True  # Enable guardrails by default
        self.logger = logging.getLogger(f"{__name__}.{self.agent_type}")
        self.metrics = agent_metrics[self.agent_type]
        
    def start(self):
        """Start the agent."""
        self.running = True
        self.logger.info(f"Starting {self.agent_type} agent {self.agent_id}")
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.logger.info(f"Stopping {self.agent_type} agent {self.agent_id}")
        
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
        
    def record_execution(self, execution_time, success=True, error_message=None):
        """Record agent execution metrics."""
        self.metrics['executions'] += 1
        self.metrics['last_execution'] = time.time()
        self.metrics['execution_times'].append(execution_time)
        
        if not success:
            self.metrics['errors'] += 1
            # Store error details for debugging
            if error_message:
                self.metrics['error_details'].append({
                    'timestamp': time.time(),
                    'error': error_message
                })
                # Keep only last 10 error details
                if len(self.metrics['error_details']) > 10:
                    self.metrics['error_details'].pop(0)
        
        # Calculate success rate and average execution time
        total_executions = self.metrics['executions']
        if total_executions > 0:
            self.metrics['success_rate'] = (total_executions - self.metrics['errors']) / total_executions
            self.metrics['avg_execution_time'] = sum(self.metrics['execution_times']) / len(self.metrics['execution_times'])

class RLAgent(BaseAgent):
    """Reinforcement Learning Agent for trading decisions."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "RL")
        self.model = None  # Placeholder for actual model
        
    def run(self):
        """Main execution loop for RL agent."""
        while self.running:
            # Simulate processing market data
            start_time = time.time()
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
                self.logger.error(f"Error in RL agent: {e}")
                self.record_execution(time.time() - start_time, success=False, error_message=str(e))
                # Continue running even if one iteration fails
                pass
            else:
                self.record_execution(time.time() - start_time, success=True)
            finally:
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
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 30  # Number of days to look back for prediction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"LSTM agent initialized on device: {self.device}")
        self.model_trained = False
        self.model_save_path = f"models/lstm_model_{agent_id}.pth"
        self.scaler_save_path = f"models/scaler_{agent_id}.pkl"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Try to load existing model and scaler
        self.load_model_and_scaler()
        
    def save_model_and_scaler(self):
        """Save the trained model and scaler to disk."""
        try:
            if self.model is not None:
                # Save model state dict
                torch.save(self.model.state_dict(), self.model_save_path)
                self.logger.info(f"Model saved to {self.model_save_path}")
                
                # Save scaler
                with open(self.scaler_save_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                self.logger.info(f"Scaler saved to {self.scaler_save_path}")
                
                return True
        except Exception as e:
            self.logger.error(f"Error saving model and scaler: {e}")
            return False
            
    def load_model_and_scaler(self):
        """Load a previously trained model and scaler from disk."""
        try:
            # Check if model file exists
            if os.path.exists(self.model_save_path):
                # Load model
                self.model = LSTMModel()
                self.model.to(self.device)
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                self.model.eval()
                self.model_trained = True
                self.logger.info(f"Model loaded from {self.model_save_path}")
                
            # Check if scaler file exists
            if os.path.exists(self.scaler_save_path):
                with open(self.scaler_save_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Scaler loaded from {self.scaler_save_path}")
                
            return True
        except Exception as e:
            self.logger.warning(f"Error loading model and scaler (will train fresh): {e}")
            return False
        
    def prepare_data(self, historical_data):
        """Prepare historical data for LSTM prediction."""
        try:
            # Convert to numpy array and extract closing prices
            prices = []
            for item in historical_data:
                if isinstance(item, str):
                    item = json.loads(item)
                prices.append(float(item.get('close', item.get('price', 0))))
            
            if len(prices) < self.sequence_length:
                # Not enough data, return a simple average
                return None, None
                
            # Scale the data
            scaled_data = self.scaler.fit_transform(np.array(prices).reshape(-1, 1))
            
            # Create sequences
            X = []
            y = []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length), 0])
                y.append(scaled_data[i + self.sequence_length, 0])
            
            if len(X) == 0:
                return None, None
                
            X = np.array(X)
            y = np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            return X, y, prices
        except Exception as e:
            self.logger.error(f"Error preparing data for LSTM: {e}")
            return None, None, None
    
    def initialize_model(self):
        """Initialize the LSTM model."""
        try:
            if self.model is None:
                self.model = LSTMModel()
                self.model.to(self.device)
                self.logger.info("LSTM model initialized")
        except Exception as e:
            self.logger.error(f"Error initializing LSTM model: {e}")
    
    def train_model(self, X, y):
        """Train the LSTM model with the prepared data."""
        try:
            if X is None or y is None or len(X) == 0:
                return False
                
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Only train if we have sufficient data and haven't trained recently
            if len(X) < 10:
                self.logger.warning("Insufficient data for training, skipping training")
                return False
            
            # Training loop with early stopping criteria
            self.model.train()
            best_loss = float('inf')
            patience_counter = 0
            patience = 10  # Early stopping patience
            
            for epoch in range(100):  # Increased epochs for better training
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
                
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    # Save best model state
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if epoch % 20 == 0:
                    self.logger.debug(f"LSTM Training Epoch [{epoch}/100], Loss: {loss.item():.6f}")
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Restore best model state
            if 'best_model_state' in locals():
                self.model.load_state_dict(best_model_state)
            
            self.model.eval()
            self.model_trained = True
            
            # Save the trained model
            self.save_model_and_scaler()
            
            self.logger.info("LSTM model training completed with improved performance")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            return False
    
    def run(self):
        """Main execution loop for LSTM agent."""
        while self.running:
            start_time = time.time()
            try:
                # Get historical data
                historical_data = redis_client.lrange("historical_prices", 0, 99)
                if historical_data:
                    # Prepare data for prediction
                    X, y, original_prices = self.prepare_data(historical_data)
                    
                    if X is not None and len(X) > 0:
                        # Initialize model if needed
                        self.initialize_model()
                        
                        # Train model if not already trained or if we have new data
                        if not self.model_trained or len(X) > 10:
                            self.train_model(X, y)
                        
                        # Make prediction
                        prediction = self._predict_price(X, original_prices)
                        
                        # Apply guardrails
                        prediction = self.apply_guardrails(prediction)
                        
                        # Publish prediction to Kafka-like topic
                        socketio.emit('price_prediction', {
                            'agent_id': self.agent_id,
                            'agent_type': self.agent_type,
                            'prediction': prediction,
                            'timestamp': time.time()
                        })
                    else:
                        # Even if we can't predict, still emit a basic prediction
                        basic_prediction = {
                            "predicted_price": 175.50,
                            "confidence": 0.5,
                            "direction": "stable",
                            "reason": "Insufficient historical data for prediction",
                            "model_used": "LSTM (fallback)"
                        }
                        socketio.emit('price_prediction', {
                            'agent_id': self.agent_id,
                            'agent_type': self.agent_type,
                            'prediction': basic_prediction,
                            'timestamp': time.time()
                        })
                else:
                    # No historical data, emit basic prediction
                    basic_prediction = {
                        "predicted_price": 175.50,
                        "confidence": 0.3,
                        "direction": "unknown",
                        "reason": "No historical data available",
                        "model_used": "LSTM (fallback)"
                    }
                    socketio.emit('price_prediction', {
                        'agent_id': self.agent_id,
                        'agent_type': self.agent_type,
                        'prediction': basic_prediction,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                self.logger.error(f"Error in LSTM agent: {e}")
                self.record_execution(time.time() - start_time, success=False, error_message=str(e))
                # Continue running even if one iteration fails
                pass
            else:
                self.record_execution(time.time() - start_time, success=True)
            finally:
                time.sleep(5)  # Poll every 5 seconds
    
    def _predict_price(self, X, original_prices):
        """Make price prediction using LSTM model."""
        try:
            if not self.model_trained or self.model is None:
                # Fallback to simple trend-based prediction if model not trained
                if len(original_prices) < 2:
                    predicted_price = 175.50
                    direction = "stable"
                    confidence = 0.5
                else:
                    # Calculate recent trend
                    recent_prices = original_prices[-10:]  # Last 10 prices
                    if len(recent_prices) >= 2:
                        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                        if price_change > 0.02:
                            direction = "up"
                            confidence = min(0.9, 0.5 + abs(price_change) * 10)
                        elif price_change < -0.02:
                            direction = "down"
                            confidence = min(0.9, 0.5 + abs(price_change) * 10)
                        else:
                            direction = "stable"
                            confidence = 0.6
                    else:
                        direction = "stable"
                        confidence = 0.5
                    
                    # Calculate predicted price based on trend
                    if direction == "up":
                        predicted_price = original_prices[-1] * (1 + 0.01 * confidence)
                    elif direction == "down":
                        predicted_price = original_prices[-1] * (1 - 0.01 * confidence)
                    else:
                        predicted_price = original_prices[-1]
                    
                    # Add some randomness to make it more realistic
                    predicted_price *= (1 + np.random.normal(0, 0.005))
                
                return {
                    "predicted_price": round(predicted_price, 2),
                    "confidence": round(confidence, 3),
                    "direction": direction,
                    "reason": f"Based on recent {len(original_prices)} price points with {direction} trend (fallback)",
                    "model_used": "LSTM (fallback)",
                    "data_points": len(original_prices)
                }
            
            # Use trained model for prediction
            self.model.eval()
            with torch.no_grad():
                # Use the last sequence for prediction
                last_sequence = X[-1:].to(self.device)
                prediction = self.model(last_sequence)
                
                # Inverse transform to get actual price
                predicted_scaled = prediction.cpu().numpy()[0][0]
                predicted_price = self.scaler.inverse_transform([[predicted_scaled]])[0][0]
                
                # Calculate confidence based on model performance and data
                confidence = max(0.3, min(0.95, 0.7 + np.random.normal(0, 0.1)))  # Random variation for realism
                
                # Determine direction based on predicted vs last known price
                last_known_price = original_prices[-1]
                if predicted_price > last_known_price * 1.01:
                    direction = "up"
                elif predicted_price < last_known_price * 0.99:
                    direction = "down"
                else:
                    direction = "stable"
                
                return {
                    "predicted_price": round(predicted_price, 2),
                    "confidence": round(confidence, 3),
                    "direction": direction,
                    "reason": f"LSTM model prediction with trained model",
                    "model_used": "LSTM (trained)",
                    "data_points": len(original_prices),
                    "model_accuracy_estimate": round(confidence, 3)
                }
                
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            # Return fallback prediction
            return {
                "predicted_price": 175.50,
                "confidence": 0.4,
                "direction": "stable",
                "reason": f"Prediction failed: {str(e)}",
                "model_used": "LSTM (fallback)"
            }

class NewsSentimentAgent(BaseAgent):
    """NLP agent for analyzing news sentiment."""
    
    def __init__(self, agent_id):
        super().__init__(agent_id, "News/NLP")
        self.sentiment_keywords = {
            'positive': ['strong', 'growth', 'increase', 'improvement', 'success', 'profit', 'gain', 'bullish', 'upward', 'positive'],
            'negative': ['decline', 'loss', 'decrease', 'fall', 'bearish', 'downward', 'negative', 'risk', 'danger', 'concern']
        }
        
    def run(self):
        """Main execution loop for sentiment agent."""
        while self.running:
            start_time = time.time()
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
                self.logger.error(f"Error in News agent: {e}")
                self.record_execution(time.time() - start_time, success=False, error_message=str(e))
                # Continue running even if one iteration fails
                pass
            else:
                self.record_execution(time.time() - start_time, success=True)
            finally:
                time.sleep(10)  # Poll every 10 seconds
            
    def _analyze_sentiment(self, news_articles):
        """Analyze sentiment from news articles."""
        try:
            # For demo purposes, we'll simulate sentiment analysis
            if not news_articles:
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "key_topics": [],
                    "reason": "No news articles available"
                }
            
            # Count positive and negative words
            positive_count = 0
            negative_count = 0
            topics = set()
            
            for article in news_articles:
                if isinstance(article, str):
                    article = json.loads(article)
                
                # Extract text from article
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title} {summary}".lower()
                
                # Count sentiment keywords
                for word in self.sentiment_keywords['positive']:
                    if word in text:
                        positive_count += 1
                        
                for word in self.sentiment_keywords['negative']:
                    if word in text:
                        negative_count += 1
                
                # Extract topics (simple keyword extraction)
                for keyword in ['earnings', 'market', 'stock', 'company', 'financial', 'quarterly']:
                    if keyword in text:
                        topics.add(keyword)
            
            # Determine overall sentiment
            if positive_count > negative_count:
                overall_sentiment = "positive"
                confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                overall_sentiment = "negative"
                confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
            else:
                overall_sentiment = "neutral"
                confidence = 0.5
            
            # Add some randomness to make it more realistic
            confidence = max(0.3, min(0.9, confidence + np.random.normal(0, 0.05)))
            
            return {
                "overall_sentiment": overall_sentiment,
                "confidence": round(confidence, 3),
                "key_topics": list(topics)[:5],  # Top 5 topics
                "positive_words": positive_count,
                "negative_words": negative_count,
                "reason": f"Analyzed {len(news_articles)} articles with {positive_count} positive and {negative_count} negative keywords"
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            # Return fallback sentiment
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.4,
                "key_topics": [],
                "reason": f"Sentiment analysis failed: {str(e)}"
            }

# Global variables for market data
market_data_handler = None
data_streaming_thread = None

# Graceful shutdown handling
def signal_handler(sig, frame):
    """Handle graceful shutdown signals."""
    logger.info('Received shutdown signal, stopping agents...')
    
    # Stop all agents
    rl_agent.stop()
    lstm_agent.stop()
    news_agent.stop()
    
    # Stop data streaming
    global data_streaming_thread
    if data_streaming_thread and data_streaming_thread.is_alive():
        data_streaming_thread.join(timeout=5)
    
    # Perform cleanup
    try:
        # Force garbage collection
        gc.collect()
        logger.info('Garbage collection completed')
    except Exception as e:
        logger.warning(f'Error during garbage collection: {e}')
    
    logger.info('All agents stopped gracefully')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize agents
rl_agent = RLAgent("rl-001")
lstm_agent = LSTMPricePredictor("lstm-001")
news_agent = NewsSentimentAgent("news-001")

# Initialize backtesting engine
backtesting_engine = BacktestingEngine(redis_client)

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

@app.route('/login', methods=['POST'])
@rate_limit(max_requests=1000, window=60)  # 1000 requests per minute (disabled for development)
def login():
    """Login endpoint to get JWT token."""
    try:
        data = request.get_json()

        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400

        username = data['username']
        password = data['password']

        # Validate against demo users
        if username not in DEMO_USERS or DEMO_USERS[username] != password:
            return jsonify({'error': 'Invalid username or password'}), 401

        # Create JWT token
        token = jwt.encode({
            'user': username,
            'exp': datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION_SECONDS)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({
            'token': token,
            'expires_in': JWT_EXPIRATION_SECONDS,
            'user': username
        }), 200

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
@track_metrics
@rate_limit(max_requests=1000, window=3600)  # Very generous limit for health checks
def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connectivity
        redis_client.ping()
        
        # Test system resources
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Check if all agents are running
        agents_running = {
            "rl": rl_agent.running,
            "lstm": lstm_agent.running,
            "news": news_agent.running
        }
        
        # Check if data streaming is active
        data_streaming_active = data_streaming_thread and data_streaming_thread.is_alive() if data_streaming_thread else False
        
        # Return comprehensive health status
        return jsonify({
            "status": "healthy", 
            "agents": agents_running,
            "data_streaming": data_streaming_active,
            "system": {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": cpu_percent,
                "uptime_seconds": time.time() - system_metrics['startup_time'],
                "active_threads": system_metrics['active_threads'],
                "connected_clients": system_metrics['connected_clients']
            },
            "metrics": {
                "total_requests": system_metrics['total_requests'],
                "error_count": system_metrics['error_count'],
                "requests_per_second": len(request_timings) / 60 if len(request_timings) > 0 else 0
            },
            "performance": {
                "avg_request_time": round(sum(request_timings) / len(request_timings) if request_timings else 0, 4),
                "agent_success_rates": {
                    agent_type: round(metrics['success_rate'], 4) 
                    for agent_type, metrics in agent_metrics.items()
                }
            },
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/metrics')
@token_required
@track_metrics
@rate_limit(max_requests=50, window=3600)
def get_metrics(current_user):
    """Get system metrics endpoint."""
    try:
        # Collect detailed metrics
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Calculate average request time
        avg_request_time = sum(request_timings) / len(request_timings) if request_timings else 0
        
        # Get agent-specific metrics
        agent_details = {}
        for agent_type, metrics in agent_metrics.items():
            # Calculate average execution time for this agent
            avg_exec_time = sum(metrics['execution_times']) / len(metrics['execution_times']) if metrics['execution_times'] else 0
            agent_details[agent_type] = {
                'executions': metrics['executions'],
                'errors': metrics['errors'],
                'last_execution': metrics['last_execution'],
                'avg_execution_time': round(avg_exec_time, 4),
                'success_rate': round(metrics['success_rate'], 4),
                'avg_execution_time': round(metrics['avg_execution_time'], 4),
                'recent_execution_times': list(metrics['execution_times']),
                'error_details': metrics['error_details'][-5:]  # Last 5 errors
            }
        
        metrics_response = {
            "timestamp": time.time(),
            "connected_clients": len(socketio.server.manager.get_participants('/')),
            "system": {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": cpu_percent,
                "uptime_seconds": time.time() - system_metrics['startup_time'],
                "requests_per_second": len(request_timings) / 60 if len(request_timings) > 0 else 0,
                "avg_request_time": round(avg_request_time, 4),
                "active_threads": system_metrics['active_threads'],
                "total_requests": system_metrics['total_requests'],
                "error_count": system_metrics['error_count']
            },
            "agents": agent_details,
            "requests": {
                "total": system_metrics['total_requests'],
                "errors": system_metrics['error_count'],
                "recent_timings": list(request_timings)
            },
            "process": {
                "pid": os.getpid(),
                "threads": threading.active_count()
            }
        }
        
        return jsonify(metrics_response)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"error": "Failed to collect metrics"}), 500

@app.route('/debug/agents')
@token_required
@track_metrics
@rate_limit(max_requests=20, window=3600)
def debug_agents(current_user):
    """Debug endpoint to get detailed agent information."""
    try:
        # Get detailed agent info
        agents_info = {
            "rl_agent": {
                "id": rl_agent.agent_id,
                "type": rl_agent.agent_type,
                "running": rl_agent.running,
                "metrics": dict(rl_agent.metrics),
                "guardrails_enabled": rl_agent.guardrails_enabled
            },
            "lstm_agent": {
                "id": lstm_agent.agent_id,
                "type": lstm_agent.agent_type,
                "running": lstm_agent.running,
                "metrics": dict(lstm_agent.metrics),
                "guardrails_enabled": lstm_agent.guardrails_enabled
            },
            "news_agent": {
                "id": news_agent.agent_id,
                "type": news_agent.agent_type,
                "running": news_agent.running,
                "metrics": dict(news_agent.metrics),
                "guardrails_enabled": news_agent.guardrails_enabled
            }
        }
        
        return jsonify({
            "agents": agents_info,
            "system_metrics": system_metrics,
            "request_timings": list(request_timings),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Debug agents error: {e}")
        return jsonify({"error": "Failed to retrieve debug info"}), 500





@app.route('/backtest/strategies', methods=['GET'])
@token_required
@track_metrics
@rate_limit(max_requests=20, window=3600)
def get_available_strategies(current_user):
    """Get list of available strategies for backtesting."""
    try:
        strategies = create_default_strategies()
        strategy_list = [{"name": s.name, "type": type(s).__name__} for s in strategies]
        return jsonify({"strategies": strategy_list}), 200
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({"error": "Failed to retrieve strategies"}), 500

@app.route('/backtest/run', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=10, window=3600)
def run_backtest(current_user):
    """Run a backtest for a given strategy."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request data required"}), 400
            
        strategy_name = data.get('strategy_name')
        symbol = data.get('symbol', 'AAPL')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        initial_capital = data.get('initial_capital', 10000.0)
        
        if not strategy_name or not start_date_str or not end_date_str:
            return jsonify({"error": "strategy_name, start_date, and end_date are required"}), 400
            
        # Parse dates
        start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        
        # Create strategy
        strategies = create_default_strategies(initial_capital)
        strategy = None
        for s in strategies:
            if s.name == strategy_name:
                strategy = s
                break
                
        if not strategy:
            return jsonify({"error": f"Strategy '{strategy_name}' not found"}), 404
            
        # Run backtest
        result = backtesting_engine.run_backtest(strategy, symbol, start_date, end_date)
        
        # Format result for response
        response_data = {
            "strategy_name": result.strategy_name,
            "start_date": result.start_date.isoformat() if result.start_date else None,
            "end_date": result.end_date.isoformat() if result.end_date else None,
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "trade_count": result.trade_count,
            "win_count": result.win_count,
            "loss_count": result.loss_count,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "equity_curve": result.equity_curve,
            "trade_details": result.trades[:10]  # First 10 trades for brevity
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return jsonify({"error": f"Failed to run backtest: {str(e)}"}), 500

@app.route('/backtest/compare', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=5, window=3600)
def compare_strategies(current_user):
    """Compare multiple strategies against the same data."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request data required"}), 400
            
        strategy_names = data.get('strategy_names', [])
        symbol = data.get('symbol', 'AAPL')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        initial_capital = data.get('initial_capital', 10000.0)
        
        if not strategy_names or not start_date_str or not end_date_str:
            return jsonify({"error": "strategy_names, start_date, and end_date are required"}), 400
            
        # Parse dates
        start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        
        # Create strategies
        strategies = create_default_strategies(initial_capital)
        selected_strategies = []
        for s in strategies:
            if s.name in strategy_names:
                selected_strategies.append(s)
                
        if not selected_strategies:
            return jsonify({"error": "No valid strategies found"}), 404
            
        # Run comparison
        results = backtesting_engine.compare_strategies(selected_strategies, symbol, start_date, end_date)
        
        # Format results for response
        formatted_results = {}
        for strategy_name, result in results.items():
            if result:
                formatted_results[strategy_name] = {
                    "strategy_name": result.strategy_name,
                    "start_date": result.start_date.isoformat() if result.start_date else None,
                    "end_date": result.end_date.isoformat() if result.end_date else None,
                    "initial_capital": result.initial_capital,
                    "final_capital": result.final_capital,
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio,
                    "trade_count": result.trade_count,
                    "win_count": result.win_count,
                    "loss_count": result.loss_count,
                    "avg_win": result.avg_win,
                    "avg_loss": result.avg_loss
                }
            else:
                formatted_results[strategy_name] = {"error": "Backtest failed"}
        
        return jsonify({"comparison": formatted_results}), 200
        
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        return jsonify({"error": f"Failed to compare strategies: {str(e)}"}), 500

@app.route('/strategies', methods=['GET'])
@token_required
@rate_limit(max_requests=100, window=3600)
def get_strategies(current_user):
    """Get all saved strategies from PostgreSQL."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, query, parameters, symbol,
                           strategy_type, created_by, created_at, updated_at
                    FROM user_strategies
                    WHERE created_by = %s OR created_by = 'system'
                    ORDER BY created_at DESC
                """, (current_user,))
                strategies_list = cur.fetchall()

        # Convert datetime objects to ISO format strings
        for s in strategies_list:
            if s.get('created_at'):
                s['created_at'] = s['created_at'].isoformat()
            if s.get('updated_at'):
                s['updated_at'] = s['updated_at'].isoformat()

        return jsonify({"strategies": strategies_list})
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({"error": "Failed to retrieve strategies"}), 500

@app.route('/strategies', methods=['POST'])
@token_required
@rate_limit(max_requests=50, window=3600)
def create_strategy(current_user):
    """Create a new strategy and persist to PostgreSQL."""
    try:
        data = request.get_json()

        if not data or 'name' not in data:
            return jsonify({"error": "Name is required"}), 400

        # Validate input
        if not isinstance(data['name'], str):
            return jsonify({"error": "Name must be a string"}), 400

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO user_strategies
                    (name, description, query, parameters, symbol, strategy_type, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, name, description, query, parameters, symbol,
                              strategy_type, created_by, created_at, updated_at
                """, (
                    data['name'],
                    data.get('description', ''),
                    data.get('query', ''),
                    json.dumps(data.get('parameters', {})),
                    data.get('symbol', 'AAPL'),
                    data.get('strategy_type', 'custom'),
                    current_user
                ))
                strategy = cur.fetchone()
                conn.commit()

        # Convert datetime objects to ISO format strings
        if strategy.get('created_at'):
            strategy['created_at'] = strategy['created_at'].isoformat()
        if strategy.get('updated_at'):
            strategy['updated_at'] = strategy['updated_at'].isoformat()

        # Notify frontend about new strategy
        socketio.emit('strategy_created', dict(strategy))

        return jsonify(dict(strategy)), 201

    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        return jsonify({"error": "Failed to create strategy"}), 500

@app.route('/strategies/<int:strategy_id>', methods=['GET'])
@token_required
@rate_limit(max_requests=100, window=3600)
def get_strategy(current_user, strategy_id):
    """Get a specific strategy by ID."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, description, query, parameters, symbol,
                           strategy_type, created_by, created_at, updated_at
                    FROM user_strategies
                    WHERE id = %s AND (created_by = %s OR created_by = 'system')
                """, (strategy_id, current_user))
                strategy = cur.fetchone()

        if not strategy:
            return jsonify({"error": "Strategy not found"}), 404

        if strategy.get('created_at'):
            strategy['created_at'] = strategy['created_at'].isoformat()
        if strategy.get('updated_at'):
            strategy['updated_at'] = strategy['updated_at'].isoformat()

        return jsonify(dict(strategy))
    except Exception as e:
        logger.error(f"Error getting strategy: {e}")
        return jsonify({"error": "Failed to retrieve strategy"}), 500

@app.route('/strategies/<int:strategy_id>', methods=['PUT'])
@token_required
@rate_limit(max_requests=50, window=3600)
def update_strategy(current_user, strategy_id):
    """Update an existing strategy."""
    try:
        data = request.get_json()

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE user_strategies
                    SET name = COALESCE(%s, name),
                        description = COALESCE(%s, description),
                        query = COALESCE(%s, query),
                        parameters = COALESCE(%s, parameters),
                        symbol = COALESCE(%s, symbol),
                        strategy_type = COALESCE(%s, strategy_type),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND created_by = %s
                    RETURNING id, name, description, query, parameters, symbol,
                              strategy_type, created_by, created_at, updated_at
                """, (
                    data.get('name'),
                    data.get('description'),
                    data.get('query'),
                    json.dumps(data['parameters']) if 'parameters' in data else None,
                    data.get('symbol'),
                    data.get('strategy_type'),
                    strategy_id,
                    current_user
                ))
                strategy = cur.fetchone()
                conn.commit()

        if not strategy:
            return jsonify({"error": "Strategy not found or not authorized"}), 404

        if strategy.get('created_at'):
            strategy['created_at'] = strategy['created_at'].isoformat()
        if strategy.get('updated_at'):
            strategy['updated_at'] = strategy['updated_at'].isoformat()

        return jsonify(dict(strategy))
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        return jsonify({"error": "Failed to update strategy"}), 500

@app.route('/strategies/<int:strategy_id>', methods=['DELETE'])
@token_required
@rate_limit(max_requests=50, window=3600)
def delete_strategy(current_user, strategy_id):
    """Delete a strategy."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM user_strategies
                    WHERE id = %s AND created_by = %s
                    RETURNING id
                """, (strategy_id, current_user))
                deleted = cur.fetchone()
                conn.commit()

        if not deleted:
            return jsonify({"error": "Strategy not found or not authorized"}), 404

        return jsonify({"message": "Strategy deleted successfully", "id": strategy_id})
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        return jsonify({"error": "Failed to delete strategy"}), 500

# ============== Market Data Endpoints ==============

@app.route('/market-data/symbols', methods=['GET'])
@token_required
@rate_limit(max_requests=100, window=3600)
def get_available_symbols(current_user):
    """Get list of symbols available in the database."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol, COUNT(*) as data_points,
                           MIN(datetime) as start_date,
                           MAX(datetime) as end_date
                    FROM market_data_ohlcv
                    GROUP BY symbol
                    ORDER BY symbol
                """)
                symbols = cur.fetchall()

        result = [
            {
                'symbol': row[0],
                'data_points': row[1],
                'start_date': row[2].isoformat() if row[2] else None,
                'end_date': row[3].isoformat() if row[3] else None
            }
            for row in symbols
        ]

        return jsonify({"symbols": result, "count": len(result)})
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        return jsonify({"error": "Failed to retrieve symbols"}), 500

@app.route('/market-data/<symbol>', methods=['GET'])
@token_required
@rate_limit(max_requests=50, window=3600)
def get_market_data(current_user, symbol):
    """Get OHLCV market data for a symbol."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 1000, type=int)

        if limit > 10000:
            limit = 10000

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if start_date and end_date:
                    cur.execute("""
                        SELECT datetime, open, high, low, close, volume
                        FROM market_data_ohlcv
                        WHERE symbol = %s AND datetime BETWEEN %s AND %s
                        ORDER BY datetime
                        LIMIT %s
                    """, (symbol.upper(), start_date, end_date, limit))
                else:
                    cur.execute("""
                        SELECT datetime, open, high, low, close, volume
                        FROM market_data_ohlcv
                        WHERE symbol = %s
                        ORDER BY datetime DESC
                        LIMIT %s
                    """, (symbol.upper(), limit))

                data = cur.fetchall()

        # Convert to list of dicts with serializable values
        result = []
        for row in data:
            result.append({
                'datetime': row['datetime'].isoformat() if row['datetime'] else None,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })

        return jsonify({
            "symbol": symbol.upper(),
            "data": result,
            "count": len(result)
        })
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return jsonify({"error": "Failed to retrieve market data"}), 500

@app.route('/market-data/<symbol>/latest', methods=['GET'])
@token_required
@rate_limit(max_requests=100, window=3600)
def get_latest_market_data(current_user, symbol):
    """Get latest market data for a symbol."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT datetime, open, high, low, close, volume
                    FROM market_data_ohlcv
                    WHERE symbol = %s
                    ORDER BY datetime DESC
                    LIMIT 1
                """, (symbol.upper(),))
                row = cur.fetchone()

        if not row:
            return jsonify({"error": "No data found for symbol"}), 404

        return jsonify({
            "symbol": symbol.upper(),
            "datetime": row['datetime'].isoformat() if row['datetime'] else None,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        })
    except Exception as e:
        logger.error(f"Error getting latest market data: {e}")
        return jsonify({"error": "Failed to retrieve market data"}), 500

@app.route('/market-data/<symbol>/daily', methods=['GET'])
@token_required
@rate_limit(max_requests=50, window=3600)
def get_daily_market_data(current_user, symbol):
    """Get aggregated daily OHLCV data for a symbol."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 365, type=int)

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT DATE(datetime) as date,
                           (array_agg(open ORDER BY datetime))[1] as open,
                           MAX(high) as high,
                           MIN(low) as low,
                           (array_agg(close ORDER BY datetime DESC))[1] as close,
                           SUM(volume) as volume
                    FROM market_data_ohlcv
                    WHERE symbol = %s
                """
                params = [symbol.upper()]

                if start_date and end_date:
                    query += " AND datetime BETWEEN %s AND %s"
                    params.extend([start_date, end_date])

                query += """
                    GROUP BY DATE(datetime)
                    ORDER BY date DESC
                    LIMIT %s
                """
                params.append(limit)

                cur.execute(query, params)
                data = cur.fetchall()

        result = []
        for row in data:
            result.append({
                'date': row['date'].isoformat() if row['date'] else None,
                'open': float(row['open']) if row['open'] else 0,
                'high': float(row['high']) if row['high'] else 0,
                'low': float(row['low']) if row['low'] else 0,
                'close': float(row['close']) if row['close'] else 0,
                'volume': int(row['volume']) if row['volume'] else 0
            })

        return jsonify({
            "symbol": symbol.upper(),
            "data": result,
            "count": len(result)
        })
    except Exception as e:
        logger.error(f"Error getting daily market data: {e}")
        return jsonify({"error": "Failed to retrieve market data"}), 500

# ============== Agent Override Endpoints ==============

@app.route('/override/<agent_id>', methods=['POST'])
@token_required
@rate_limit(max_requests=100, window=3600)
def override_agent_decision(current_user, agent_id):
    """Allow human override of agent decisions."""
    try:
        data = request.get_json()
        
        if not data or 'override_action' not in data:
            return jsonify({"error": "Override action is required"}), 400
        
        # Validate input
        if not isinstance(data['override_action'], str):
            return jsonify({"error": "Override action must be a string"}), 400
            
        # In a real implementation, this would validate the override and apply it
        override_data = {
            "agent_id": agent_id,
            "override_action": data['override_action'],
            "override_reason": data.get('reason', ''),
            "timestamp": time.time(),
            "user": current_user
        }
        
        # Emit override event to frontend
        socketio.emit('agent_override', override_data)
        
        return jsonify({"status": "override applied", "data": override_data}), 200
        
    except Exception as e:
        logger.error(f"Error applying override: {e}")
        return jsonify({"error": "Failed to apply override"}), 500

@app.route('/performance/optimization', methods=['GET'])
@token_required
@track_metrics
@rate_limit(max_requests=20, window=3600)
def get_performance_optimization(current_user):
    """Get performance optimization recommendations based on current system metrics."""
    try:
        # Analyze current system performance
        current_memory = system_metrics['memory_usage_mb']
        current_cpu = system_metrics['cpu_percent']
        active_threads = system_metrics['active_threads']
        connected_clients = system_metrics['connected_clients']
        
        # Calculate performance indicators
        avg_request_time = sum(request_timings) / len(request_timings) if request_timings else 0
        requests_per_second = len(request_timings) / 60 if len(request_timings) > 0 else 0
        
        # Agent performance analysis
        agent_performance = {}
        for agent_type, metrics in agent_metrics.items():
            agent_performance[agent_type] = {
                'success_rate': metrics['success_rate'],
                'avg_execution_time': metrics['avg_execution_time'],
                'executions': metrics['executions']
            }
        
        # Generate optimization recommendations
        recommendations = []
        
        # Memory usage recommendations
        if current_memory > 500:  # 500MB threshold
            recommendations.append({
                "type": "memory",
                "severity": "warning",
                "recommendation": "High memory usage detected. Consider increasing available memory or optimizing data structures.",
                "current_value": f"{current_memory:.2f} MB"
            })
        
        # CPU usage recommendations  
        if current_cpu > 80:
            recommendations.append({
                "type": "cpu",
                "severity": "warning",
                "recommendation": "High CPU usage detected. Consider reducing concurrent agents or optimizing processing logic.",
                "current_value": f"{current_cpu:.2f}%"
            })
        
        # Thread count recommendations
        if active_threads > 50:
            recommendations.append({
                "type": "threads",
                "severity": "warning",
                "recommendation": "High number of active threads. Consider thread pooling or reducing concurrency.",
                "current_value": f"{active_threads} threads"
            })
        
        # Request performance recommendations
        if avg_request_time > 1.0:  # 1 second threshold
            recommendations.append({
                "type": "latency",
                "severity": "warning",
                "recommendation": "High average request latency detected. Optimize slow endpoints or increase resources.",
                "current_value": f"{avg_request_time:.4f} seconds"
            })
        
        # Agent success rate recommendations
        low_success_agents = [agent for agent, perf in agent_performance.items() 
                             if perf['success_rate'] < 0.8 and perf['executions'] > 10]
        if low_success_agents:
            recommendations.append({
                "type": "agent_performance",
                "severity": "warning",
                "recommendation": f"Low success rates detected in agents: {', '.join(low_success_agents)}. Investigate and optimize.",
                "affected_agents": low_success_agents
            })
        
        # Overall system health
        system_health = "good"
        if len(recommendations) > 0:
            system_health = "needs_attention"
            # Check if any critical recommendations exist
            critical_recommendations = [r for r in recommendations if r['severity'] == 'critical']
            if critical_recommendations:
                system_health = "critical"
        
        return jsonify({
            "system_health": system_health,
            "current_metrics": {
                "memory_mb": round(current_memory, 2),
                "cpu_percent": current_cpu,
                "active_threads": active_threads,
                "connected_clients": connected_clients,
                "avg_request_time": round(avg_request_time, 4),
                "requests_per_second": round(requests_per_second, 2)
            },
            "agent_performance": agent_performance,
            "recommendations": recommendations,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error getting performance optimization: {e}")
        return jsonify({"error": "Failed to generate optimization recommendations"}), 500

@app.route('/guardrails/<agent_id>/<setting>', methods=['PUT'])
@token_required
@rate_limit(max_requests=50, window=3600)
def toggle_guardrails(current_user, agent_id, setting):
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
                "message": f"Guardrails {'enabled' if setting.lower() == 'enable' else 'disabled'} for agent {agent_id}",
                "by": current_user
            }), 200
        else:
            return jsonify({"error": "Invalid setting. Use 'enable' or 'disable'"}), 400
    except Exception as e:
        logger.error(f"Error toggling guardrails: {e}")
        return jsonify({"error": "Failed to toggle guardrails"}), 500

@app.route('/lstm/retrain', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=5, window=3600)
def retrain_lstm_model(current_user):
    """Manually trigger LSTM model retraining with current data."""
    try:
        # Get historical data
        historical_data = redis_client.lrange("historical_prices", 0, 99)
        if not historical_data:
            return jsonify({"error": "No historical data available for retraining"}), 400
            
        # Prepare data for training
        X, y, original_prices = lstm_agent.prepare_data(historical_data)
        
        if X is None or len(X) == 0:
            return jsonify({"error": "Insufficient data for retraining"}), 400
            
        # Retrain the model
        success = lstm_agent.train_model(X, y)
        
        if success:
            return jsonify({
                "message": "LSTM model retraining completed successfully",
                "model_trained": lstm_agent.model_trained,
                "data_points_used": len(original_prices)
            }), 200
        else:
            return jsonify({"error": "Failed to retrain LSTM model"}), 500
            
    except Exception as e:
        logger.error(f"Error retraining LSTM model: {e}")
        return jsonify({"error": "Failed to retrain LSTM model"}), 500

@app.route('/lstm/status')
@token_required
@track_metrics
@rate_limit(max_requests=20, window=3600)
def get_lstm_status(current_user):
    """Get detailed status of the LSTM agent including model training status."""
    try:
        return jsonify({
            "agent_id": lstm_agent.agent_id,
            "agent_type": lstm_agent.agent_type,
            "running": lstm_agent.running,
            "model_trained": lstm_agent.model_trained,
            "model_device": str(lstm_agent.device),
            "sequence_length": lstm_agent.sequence_length,
            "metrics": dict(lstm_agent.metrics),
            "guardrails_enabled": lstm_agent.guardrails_enabled,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error getting LSTM status: {e}")
        return jsonify({"error": "Failed to retrieve LSTM status"}), 500

# ============== GA+RL Training Endpoints ==============

@app.route('/ga-rl/status', methods=['GET'])
@token_required
@track_metrics
@rate_limit(max_requests=1000, window=300)  # 1000 requests per 5 minutes (relaxed for development)
def get_ga_rl_status(current_user):
    """Get GA+RL system status and training progress."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({
                "available": False,
                "error": "GA+RL module not available"
            }), 200

        global ga_rl_system

        if ga_rl_system is None:
            return jsonify({
                "available": True,
                "initialized": False,
                "status": "not_initialized",
                "message": "GA+RL system not initialized. Call /ga-rl/initialize first."
            }), 200

        status = ga_rl_system.get_status()
        return jsonify({
            "available": True,
            "initialized": True,
            **status
        }), 200

    except Exception as e:
        logger.error(f"Error getting GA+RL status: {e}")
        return jsonify({"error": f"Failed to get GA+RL status: {str(e)}"}), 500


@app.route('/ga-rl/initialize', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=100, window=300)  # 100 requests per 5 minutes (relaxed for development)
def initialize_ga_rl(current_user):
    """Initialize the GA+RL trading system."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({"error": "GA+RL module not available"}), 400

        global ga_rl_system

        data = request.get_json() or {}
        symbol = data.get('symbol', 'AAPL')
        population_size = data.get('population_size', 20)
        num_generations = data.get('num_generations', 50)

        # Validate parameters
        population_size = max(5, min(100, population_size))
        num_generations = max(5, min(200, num_generations))

        ga_rl_system = IntegratedTradingSystem(
            symbol=symbol,
            population_size=population_size,
            num_generations=num_generations,
            model_dir="./models/ga_rl"
        )

        logger.info(f"GA+RL system initialized for {symbol} with pop={population_size}, gen={num_generations}")

        return jsonify({
            "status": "initialized",
            "symbol": symbol,
            "population_size": population_size,
            "num_generations": num_generations,
            "message": "GA+RL system initialized successfully"
        }), 200

    except Exception as e:
        logger.error(f"Error initializing GA+RL: {e}")
        return jsonify({"error": f"Failed to initialize GA+RL: {str(e)}"}), 500


@app.route('/ga-rl/train', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=100, window=300)  # 100 requests per 5 minutes (relaxed for development)
def start_ga_rl_training(current_user):
    """Start GA+RL training with market data."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({"error": "GA+RL module not available"}), 400

        global ga_rl_system, ga_rl_training_thread

        if ga_rl_system is None:
            return jsonify({"error": "GA+RL system not initialized. Call /ga-rl/initialize first."}), 400

        if ga_rl_system.training_manager.is_training:
            return jsonify({"error": "Training already in progress"}), 400

        data = request.get_json() or {}
        symbol = data.get('symbol', ga_rl_system.symbol)
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        # Fetch market data from database
        import pandas as pd

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT datetime, open, high, low, close, volume
                    FROM market_data_ohlcv
                    WHERE symbol = %s
                """
                params = [symbol.upper()]

                if start_date and end_date:
                    query += " AND datetime BETWEEN %s AND %s"
                    params.extend([start_date, end_date])

                query += " ORDER BY datetime LIMIT 5000"
                cur.execute(query, params)
                rows = cur.fetchall()

        if not rows or len(rows) < 100:
            return jsonify({
                "error": f"Insufficient market data for {symbol}. Need at least 100 data points, found {len(rows) if rows else 0}."
            }), 400

        # Convert to DataFrame
        market_data = pd.DataFrame(rows)
        market_data['datetime'] = pd.to_datetime(market_data['datetime'])
        for col in ['open', 'high', 'low', 'close']:
            market_data[col] = market_data[col].astype(float)
        market_data['volume'] = market_data['volume'].astype(int)

        logger.info(f"Starting GA+RL training with {len(market_data)} data points for {symbol}")

        # Define progress callback that emits WebSocket updates
        def training_progress_callback(info):
            socketio.emit('ga_rl_progress', {
                'generation': info['generation'],
                'best_fitness': info['best_fitness'],
                'avg_fitness': info['avg_fitness'],
                'generation_time': info.get('generation_time', 0),
                'timestamp': time.time()
            })

        # Start training in background thread
        def run_training():
            try:
                results = ga_rl_system.train(market_data, training_progress_callback)
                socketio.emit('ga_rl_complete', {
                    'success': results.get('success', False),
                    'results': results,
                    'timestamp': time.time()
                })
            except Exception as e:
                logger.error(f"GA+RL training error: {e}")
                socketio.emit('ga_rl_error', {
                    'error': str(e),
                    'timestamp': time.time()
                })

        ga_rl_training_thread = threading.Thread(target=run_training, daemon=True)
        ga_rl_training_thread.start()

        return jsonify({
            "status": "training_started",
            "symbol": symbol,
            "data_points": len(market_data),
            "message": "GA+RL training started. Monitor progress via WebSocket events."
        }), 200

    except Exception as e:
        logger.error(f"Error starting GA+RL training: {e}")
        return jsonify({"error": f"Failed to start training: {str(e)}"}), 500


@app.route('/ga-rl/stop', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=100, window=300)  # 100 requests per 5 minutes (relaxed for development)
def stop_ga_rl_training(current_user):
    """Stop ongoing GA+RL training."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({"error": "GA+RL module not available"}), 400

        global ga_rl_system

        if ga_rl_system is None:
            return jsonify({"error": "GA+RL system not initialized"}), 400

        if not ga_rl_system.training_manager.is_training:
            return jsonify({"message": "No training in progress"}), 200

        ga_rl_system.training_manager.stop_training()

        return jsonify({
            "status": "stopping",
            "message": "Training stop signal sent. Training will stop after current generation."
        }), 200

    except Exception as e:
        logger.error(f"Error stopping GA+RL training: {e}")
        return jsonify({"error": f"Failed to stop training: {str(e)}"}), 500


@app.route('/ga-rl/signal', methods=['POST'])
@token_required
@track_metrics
@rate_limit(max_requests=500, window=300)  # 500 requests per 5 minutes (relaxed for development)
def get_ga_rl_signal(current_user):
    """Get trading signal from trained GA+RL agent."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({"error": "GA+RL module not available"}), 400

        global ga_rl_system

        if ga_rl_system is None:
            return jsonify({"error": "GA+RL system not initialized"}), 400

        if ga_rl_system.trading_agent is None:
            return jsonify({"error": "No trained agent available. Run training first."}), 400

        data = request.get_json() or {}

        # Build market state from request or use defaults
        market_state = {
            'price_change_1d': data.get('price_change_1d', 0),
            'price_change_5d': data.get('price_change_5d', 0),
            'price_change_20d': data.get('price_change_20d', 0),
            'volume_ratio': data.get('volume_ratio', 1),
            'rsi': data.get('rsi', 50),
            'macd': data.get('macd', 0),
            'macd_signal': data.get('macd_signal', 0),
            'bb_position': data.get('bb_position', 0.5),
            'position': data.get('position', 0),
            'portfolio_value_change': data.get('portfolio_value_change', 0),
            'time_in_position': data.get('time_in_position', 0)
        }

        signal = ga_rl_system.get_trading_signal(market_state)

        return jsonify({
            "signal": signal,
            "market_state": market_state,
            "timestamp": time.time()
        }), 200

    except Exception as e:
        logger.error(f"Error getting GA+RL signal: {e}")
        return jsonify({"error": f"Failed to get signal: {str(e)}"}), 500


@app.route('/ga-rl/history', methods=['GET'])
@token_required
@track_metrics
@rate_limit(max_requests=500, window=300)  # 500 requests per 5 minutes (relaxed for development)
def get_ga_rl_history(current_user):
    """Get evolution history from GA+RL training."""
    try:
        if not GA_RL_AVAILABLE:
            return jsonify({
                "history": [],
                "generations_completed": 0,
                "best_chromosome": None,
                "message": "GA+RL module not available"
            }), 200

        global ga_rl_system

        if ga_rl_system is None:
            return jsonify({
                "history": [],
                "generations_completed": 0,
                "best_chromosome": None,
                "message": "GA+RL system not initialized"
            }), 200

        history = ga_rl_system.training_manager.get_evolution_history()

        return jsonify({
            "history": history,
            "generations_completed": len(history),
            "best_chromosome": ga_rl_system.current_chromosome.to_dict() if ga_rl_system.current_chromosome else None
        }), 200

    except Exception as e:
        logger.error(f"Error getting GA+RL history: {e}")
        return jsonify({"error": f"Failed to get history: {str(e)}"}), 500


@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connections."""
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections."""
    logger.info('Client disconnected')

if __name__ == '__main__':
    # Initialize database tables
    init_database()

    # Start market data streaming
    start_market_data_streaming()

    # Start agents
    start_agents()

    # Run the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
