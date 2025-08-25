#!/usr/bin/env python3
"""
Backtesting Framework for the AI-Driven Multi-Agent Stock Market Lab.

This module provides a comprehensive backtesting framework that allows users
to test trading strategies against historical market data.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

# Try to import torch, but make it optional for environments without it
try:
    import torch
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for when torch is not available
    class MockTensor:
        pass
    class MockModule:
        pass
    torch = None

logger = logging.getLogger(__name__)

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.strategy_name = ""
        self.start_date = None
        self.end_date = None
        self.initial_capital = 0.0
        self.final_capital = 0.0
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.equity_curve = []
        self.trades = []
        self.metrics = {}

class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str, initial_capital: float = 10000.0):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # Number of shares held
        self.portfolio_value_history = []
        self.trades = []
        
    def should_buy(self, data: Dict) -> bool:
        """Determine if we should buy based on current data."""
        return False
        
    def should_sell(self, data: Dict) -> bool:
        """Determine if we should sell based on current data."""
        return False
        
    def calculate_position_size(self, data: Dict, risk_amount: float = 0.01) -> float:
        """Calculate position size based on risk management."""
        # Simple risk-based position sizing
        return min(risk_amount * self.current_capital / data.get('close', 1.0), 
                   self.current_capital / data.get('close', 1.0))

class SimpleMovingAverageStrategy(TradingStrategy):
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, name: str, initial_capital: float = 10000.0, short_window: int = 10, long_window: int = 30):
        super().__init__(name, initial_capital)
        self.short_window = short_window
        self.long_window = long_window
        self.short_ma = []
        self.long_ma = []
        
    def should_buy(self, data: Dict) -> bool:
        if len(self.short_ma) < self.short_window or len(self.long_ma) < self.long_window:
            return False
            
        # Buy when short MA crosses above long MA
        if (self.short_ma[-2] <= self.long_ma[-2] and 
            self.short_ma[-1] > self.long_ma[-1]):
            return True
        return False
        
    def should_sell(self, data: Dict) -> bool:
        if len(self.short_ma) < self.short_window or len(self.long_ma) < self.long_window:
            return False
            
        # Sell when short MA crosses below long MA
        if (self.short_ma[-2] >= self.long_ma[-2] and 
            self.short_ma[-1] < self.long_ma[-1]):
            return True
        return False
        
    def update_moving_averages(self, close_price: float):
        """Update moving averages with new price data."""
        self.short_ma.append(close_price)
        self.long_ma.append(close_price)
        
        # Keep only the required number of values
        if len(self.short_ma) > self.long_window:
            self.short_ma.pop(0)
        if len(self.long_ma) > self.long_window:
            self.long_ma.pop(0)

class LSTMStrategy(TradingStrategy):
    """LSTM-based strategy that uses predictions from LSTM agent."""
    
    def __init__(self, name: str, initial_capital: float = 10000.0, prediction_threshold: float = 0.02):
        super().__init__(name, initial_capital)
        self.prediction_threshold = prediction_threshold
        self.last_prediction = None
        
    def should_buy(self, data: Dict) -> bool:
        # Check if we have a good positive prediction
        if 'prediction' in data and data['prediction']:
            prediction = data['prediction']
            if (prediction.get('direction') == 'up' and 
                prediction.get('confidence', 0) > 0.7 and
                prediction.get('predicted_price', 0) > data.get('close', 0)):
                return True
        return False
        
    def should_sell(self, data: Dict) -> bool:
        # Check if we have a negative prediction
        if 'prediction' in data and data['prediction']:
            prediction = data['prediction']
            if (prediction.get('direction') == 'down' and 
                prediction.get('confidence', 0) > 0.7 and
                prediction.get('predicted_price', 0) < data.get('close', 0)):
                return True
        return False

class BacktestingEngine:
    """Main backtesting engine that runs strategies against historical data."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.results = []
        self.logger = logging.getLogger(__name__)
        
    def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Load historical data from Redis for backtesting."""
        try:
            # Get historical prices from Redis
            historical_data = self.redis_client.lrange(f"historical_prices_{symbol}", 0, -1)
            
            if not historical_data:
                # Fallback to generic historical data
                return self._generate_sample_data(symbol, start_date, end_date)
                
            # Parse the data
            parsed_data = []
            for item in historical_data:
                if isinstance(item, str):
                    item = json.loads(item)
                parsed_data.append(item)
                
            # Filter by date range if needed
            filtered_data = []
            for item in parsed_data:
                # Convert timestamp to datetime if needed
                if 'timestamp' in item:
                    try:
                        # Handle different timestamp formats
                        ts = item['timestamp']
                        if isinstance(ts, str):
                            # Try common timestamp formats
                            if 'T' in ts:
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            else:
                                dt = datetime.fromtimestamp(int(ts))
                        else:
                            dt = datetime.fromtimestamp(ts)
                            
                        if start_date <= dt <= end_date:
                            filtered_data.append(item)
                    except Exception:
                        # If we can't parse timestamp, include anyway
                        filtered_data.append(item)
                else:
                    filtered_data.append(item)
                    
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return self._generate_sample_data(symbol, start_date, end_date)
    
    def _generate_sample_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate sample historical data for testing."""
        # Generate sample data for backtesting
        sample_data = []
        current_date = start_date
        base_price = 100.0
        
        while current_date <= end_date:
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # Daily change ~2%
            base_price = max(base_price * (1 + price_change), 0.01)
            
            sample_data.append({
                'timestamp': current_date.isoformat(),
                'open': base_price * (1 - np.random.uniform(0, 0.01)),
                'high': base_price * (1 + np.random.uniform(0, 0.02)),
                'low': base_price * (1 - np.random.uniform(0, 0.02)),
                'close': base_price,
                'volume': int(np.random.uniform(1000000, 10000000)),
                'symbol': symbol
            })
            
            current_date += timedelta(days=1)
            
        return sample_data
    
    def run_backtest(self, strategy: TradingStrategy, symbol: str, 
                     start_date: datetime, end_date: datetime, 
                     verbose: bool = False) -> BacktestResult:
        """Run a backtest for the given strategy."""
        self.logger.info(f"Running backtest for {strategy.name} on {symbol}")
        
        # Load historical data
        historical_data = self.load_historical_data(symbol, start_date, end_date)
        if not historical_data:
            raise ValueError("No historical data available for backtesting")
            
        # Initialize result object
        result = BacktestResult()
        result.strategy_name = strategy.name
        result.start_date = start_date
        result.end_date = end_date
        result.initial_capital = strategy.initial_capital
        result.current_capital = strategy.initial_capital
        
        # Initialize portfolio tracking
        portfolio_values = [strategy.initial_capital]
        equity_curve = [strategy.initial_capital]
        
        # Process each data point
        for i, data_point in enumerate(historical_data):
            try:
                # Update strategy with current data
                if hasattr(strategy, 'update_moving_averages') and 'close' in data_point:
                    strategy.update_moving_averages(data_point['close'])
                
                # Check for buy/sell signals
                should_buy = strategy.should_buy(data_point)
                should_sell = strategy.should_sell(data_point)
                
                # Execute trades
                if should_buy and strategy.position == 0:
                    # Buy signal
                    position_size = strategy.calculate_position_size(data_point)
                    cost = position_size * data_point['close']
                    
                    if cost <= strategy.current_capital:
                        strategy.position = position_size
                        strategy.current_capital -= cost
                        
                        # Record trade
                        trade = {
                            'date': data_point.get('timestamp', ''),
                            'action': 'BUY',
                            'price': data_point['close'],
                            'quantity': position_size,
                            'cost': cost,
                            'portfolio_value': strategy.current_capital + (position_size * data_point['close'])
                        }
                        strategy.trades.append(trade)
                        
                elif should_sell and strategy.position > 0:
                    # Sell signal
                    revenue = strategy.position * data_point['close']
                    strategy.current_capital += revenue
                    
                    # Record trade
                    trade = {
                        'date': data_point.get('timestamp', ''),
                        'action': 'SELL',
                        'price': data_point['close'],
                        'quantity': strategy.position,
                        'revenue': revenue,
                        'portfolio_value': strategy.current_capital
                    }
                    strategy.trades.append(trade)
                    
                    strategy.position = 0
                
                # Calculate portfolio value at this point
                current_value = strategy.current_capital + (strategy.position * data_point.get('close', 0))
                portfolio_values.append(current_value)
                equity_curve.append(current_value)
                
                if verbose and i % 100 == 0:
                    self.logger.info(f"Processed {i} data points...")
                    
            except Exception as e:
                self.logger.error(f"Error processing data point {i}: {e}")
                continue
        
        # Final portfolio value
        final_value = strategy.current_capital + (strategy.position * historical_data[-1].get('close', 0))
        result.final_capital = final_value
        result.total_return = (final_value - strategy.initial_capital) / strategy.initial_capital
        
        # Calculate additional metrics
        result.equity_curve = equity_curve
        result.trades = strategy.trades
        result.trade_count = len(strategy.trades)
        
        # Calculate win/loss statistics
        wins = [trade for trade in strategy.trades if trade.get('revenue', 0) > trade.get('cost', 0)]
        losses = [trade for trade in strategy.trades if trade.get('revenue', 0) < trade.get('cost', 0)]
        result.win_count = len(wins)
        result.loss_count = len(losses)
        
        if wins:
            result.avg_win = sum([trade.get('revenue', 0) - trade.get('cost', 0) for trade in wins]) / len(wins)
        if losses:
            result.avg_loss = sum([trade.get('cost', 0) - trade.get('revenue', 0) for trade in losses]) / len(losses)
        
        # Calculate max drawdown
        if len(equity_curve) > 1:
            peak = equity_curve[0]
            max_drawdown = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            result.max_drawdown = max_drawdown
        
        # Calculate Sharpe ratio (assuming 0.02 annual risk-free rate)
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
            
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    result.sharpe_ratio = (mean_return - 0.02/252) / std_return  # Daily risk-free rate
        
        # Annualized return calculation
        days_held = (end_date - start_date).days
        if days_held > 0:
            result.annualized_return = ((final_value / strategy.initial_capital) ** (365.0 / days_held)) - 1
        
        # Store results
        self.results.append(result)
        
        self.logger.info(f"Backtest completed for {strategy.name}. Final capital: {final_value:.2f}")
        return result
    
    def compare_strategies(self, strategies: List[TradingStrategy], symbol: str, 
                          start_date: datetime, end_date: datetime) -> Dict[str, BacktestResult]:
        """Compare multiple strategies against the same data."""
        results = {}
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, symbol, start_date, end_date)
                results[strategy.name] = result
            except Exception as e:
                self.logger.error(f"Error running backtest for {strategy.name}: {e}")
                results[strategy.name] = None
        return results
    
    def export_results(self, result: BacktestResult, filepath: str):
        """Export backtest results to JSON file."""
        export_data = {
            'strategy_name': result.strategy_name,
            'start_date': result.start_date.isoformat() if result.start_date else None,
            'end_date': result.end_date.isoformat() if result.end_date else None,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'trade_count': result.trade_count,
            'win_count': result.win_count,
            'loss_count': result.loss_count,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'equity_curve': result.equity_curve,
            'trades': result.trades,
            'metrics': result.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filepath}")

def create_default_strategies(initial_capital: float = 10000.0) -> List[TradingStrategy]:
    """Create a set of default strategies for backtesting."""
    strategies = [
        SimpleMovingAverageStrategy("SMA_10_30", initial_capital, 10, 30),
        SimpleMovingAverageStrategy("SMA_5_20", initial_capital, 5, 20),
        LSTMStrategy("LSTM_Prediction", initial_capital, 0.02)
    ]
    return strategies

# Example usage function
def example_backtest():
    """Example of how to use the backtesting framework."""
    # This would be called from the main application
    print("Backtesting framework initialized")
    return True

if __name__ == "__main__":
    # This would be used for testing the framework
    print("Backtesting framework module loaded")