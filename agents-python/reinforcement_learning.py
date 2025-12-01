"""
Deep Q-Network (DQN) Reinforcement Learning Module for Trading

This module implements a DQN agent that learns to make trading decisions
(buy, sell, hold) based on market state observations.

Key components:
- Trading Environment: Simulates market with actions and rewards
- DQN Network: Neural network for Q-value approximation
- Experience Replay: Buffer for stable training
- Target Network: Separate network for stable Q-targets
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
import json
import os

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class TradingAction(IntEnum):
    """Trading actions the agent can take."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    state_size: int = 11
    action_size: int = 3
    hidden_size: int = 128
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 10
    tau: float = 0.001
    use_double_dqn: bool = True
    use_dueling: bool = True
    device: str = 'cpu'


@dataclass
class TradingState:
    """Represents the state of the trading environment."""
    # Price features
    price: float
    price_change_1: float  # 1-period price change
    price_change_5: float  # 5-period price change
    price_change_10: float  # 10-period price change

    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    bb_position: float  # Position within Bollinger Bands (-1 to 1)

    # Volume features
    volume_ratio: float  # Current volume / average volume

    # Portfolio state
    position: float  # Current position (0 = no position, 1 = long)
    unrealized_pnl: float  # Current unrealized P&L
    cash_ratio: float  # Cash / total portfolio value

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input."""
        return np.array([
            self.price_change_1,
            self.price_change_5,
            self.price_change_10,
            self.rsi / 100.0,  # Normalize RSI to [0, 1]
            self.macd,
            self.macd_signal,
            self.bb_position,
            self.volume_ratio,
            self.position,
            self.unrealized_pnl,
            self.cash_ratio
        ], dtype=np.float32)


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.

    Simulates a market where the agent can buy, sell, or hold.
    Provides observations and rewards based on trading performance.
    """

    def __init__(
        self,
        data: np.ndarray,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        """
        Args:
            data: OHLCV data as numpy array [open, high, low, close, volume]
            initial_capital: Starting capital
            transaction_cost: Cost per transaction as fraction of trade value
            max_position: Maximum position size (1.0 = 100% of capital)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # Precompute technical indicators
        self._precompute_indicators()

        # State variables (will be set on reset)
        self.current_step = 0
        self.capital = initial_capital
        self.position = 0  # Number of shares
        self.position_value = 0
        self.entry_price = 0
        self.done = False

        # Tracking
        self.equity_curve = []
        self.trades = []
        self.total_reward = 0

    def _precompute_indicators(self):
        """Precompute technical indicators for all data points."""
        closes = self.data[:, 3]
        volumes = self.data[:, 4]

        n = len(closes)

        # Initialize arrays
        self.rsi = np.zeros(n)
        self.macd = np.zeros(n)
        self.macd_signal = np.zeros(n)
        self.bb_upper = np.zeros(n)
        self.bb_lower = np.zeros(n)
        self.bb_middle = np.zeros(n)
        self.avg_volume = np.zeros(n)

        # RSI (14-period)
        rsi_period = 14
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        # Initial averages
        if len(gains) >= rsi_period:
            avg_gain[rsi_period] = np.mean(gains[:rsi_period])
            avg_loss[rsi_period] = np.mean(losses[:rsi_period])

            for i in range(rsi_period + 1, n):
                avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gains[i-1]) / rsi_period
                avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + losses[i-1]) / rsi_period

            rs = np.divide(avg_gain, avg_loss + 1e-10)
            self.rsi = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        self.macd = ema_12 - ema_26
        self.macd_signal = self._ema(self.macd, 9)

        # Bollinger Bands (20, 2)
        bb_period = 20
        for i in range(bb_period, n):
            window = closes[i-bb_period:i]
            self.bb_middle[i] = np.mean(window)
            std = np.std(window)
            self.bb_upper[i] = self.bb_middle[i] + 2 * std
            self.bb_lower[i] = self.bb_middle[i] - 2 * std

        # Average volume
        vol_period = 20
        for i in range(vol_period, n):
            self.avg_volume[i] = np.mean(volumes[i-vol_period:i])

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)

        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = 30  # Start after indicators are valid
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.done = False
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.total_reward = 0

        return self._get_state().to_array()

    def _get_state(self) -> TradingState:
        """Get the current state observation."""
        i = self.current_step
        closes = self.data[:, 3]
        volumes = self.data[:, 4]

        current_price = closes[i]

        # Price changes
        price_change_1 = (current_price - closes[i-1]) / closes[i-1] if i > 0 else 0
        price_change_5 = (current_price - closes[i-5]) / closes[i-5] if i >= 5 else 0
        price_change_10 = (current_price - closes[i-10]) / closes[i-10] if i >= 10 else 0

        # Bollinger Band position
        if self.bb_upper[i] != self.bb_lower[i]:
            bb_position = 2 * (current_price - self.bb_middle[i]) / (self.bb_upper[i] - self.bb_lower[i])
        else:
            bb_position = 0

        # Volume ratio
        volume_ratio = volumes[i] / (self.avg_volume[i] + 1e-10) if self.avg_volume[i] > 0 else 1

        # Portfolio state
        total_value = self.capital + self.position * current_price
        position_pct = (self.position * current_price) / total_value if total_value > 0 else 0

        unrealized_pnl = 0
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price

        cash_ratio = self.capital / total_value if total_value > 0 else 1

        return TradingState(
            price=current_price,
            price_change_1=price_change_1,
            price_change_5=price_change_5,
            price_change_10=price_change_10,
            rsi=self.rsi[i],
            macd=self.macd[i],
            macd_signal=self.macd_signal[i],
            bb_position=bb_position,
            volume_ratio=min(volume_ratio, 5.0),  # Cap at 5x
            position=position_pct,
            unrealized_pnl=unrealized_pnl,
            cash_ratio=cash_ratio
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return the new state, reward, done flag, and info.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            state: New state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        current_price = self.data[self.current_step, 3]
        prev_total_value = self.capital + self.position * current_price

        reward = 0
        trade_info = None

        # Execute action
        if action == TradingAction.BUY and self.position == 0:
            # Buy with all available capital (adjusted for position size)
            shares_to_buy = int((self.capital * self.max_position) / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.capital:
                    self.position = shares_to_buy
                    self.capital -= cost
                    self.entry_price = current_price
                    trade_info = {'action': 'BUY', 'price': current_price, 'shares': shares_to_buy}

        elif action == TradingAction.SELL and self.position > 0:
            # Sell all shares
            proceeds = self.position * current_price * (1 - self.transaction_cost)
            pnl = proceeds - (self.position * self.entry_price)
            pnl_pct = (current_price - self.entry_price) / self.entry_price

            self.capital += proceeds
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'shares': self.position
            })
            trade_info = {'action': 'SELL', 'price': current_price, 'pnl': pnl}

            self.position = 0
            self.entry_price = 0

        # Move to next step
        self.current_step += 1

        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Close any open position
            if self.position > 0:
                final_price = self.data[self.current_step, 3]
                self.capital += self.position * final_price * (1 - self.transaction_cost)
                self.position = 0

        # Calculate reward
        new_price = self.data[self.current_step, 3] if self.current_step < len(self.data) else current_price
        new_total_value = self.capital + self.position * new_price

        # Reward is the percentage change in portfolio value
        reward = (new_total_value - prev_total_value) / prev_total_value

        # Add small penalty for holding to encourage trading
        if action == TradingAction.HOLD:
            reward -= 0.0001

        # Bonus for profitable trades
        if trade_info and trade_info.get('action') == 'SELL':
            if trade_info['pnl'] > 0:
                reward += 0.01  # Bonus for profitable trade
            else:
                reward -= 0.005  # Small penalty for losing trade

        self.total_reward += reward
        self.equity_curve.append(new_total_value)

        # Get new state
        state = self._get_state() if not self.done else np.zeros(11, dtype=np.float32)

        info = {
            'total_value': new_total_value,
            'trade': trade_info,
            'step': self.current_step
        }

        return state.to_array() if not self.done else state, reward, self.done, info

    def get_metrics(self) -> Dict:
        """Get performance metrics for the episode."""
        if not self.equity_curve:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Total return
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        if self.trades:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'final_value': equity[-1],
            'total_reward': self.total_reward
        }


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture.

    Uses a simple feedforward network with optional dueling architecture.
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dims: List[int] = [128, 64, 32],
        dueling: bool = True
    ):
        super(DQNetwork, self).__init__()

        self.dueling = dueling

        # Shared feature layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.feature_layers = nn.Sequential(*layers)

        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )
        else:
            # Standard Q-value output
            self.q_layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values."""
        features = self.feature_layers(x)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine value and advantage (subtract mean advantage for identifiability)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_layer(features)

        return q_values


class DQNAgent:
    """
    Deep Q-Network agent for trading.

    Implements:
    - Experience replay
    - Target network with soft updates
    - Epsilon-greedy exploration
    - Double DQN (optional)
    """

    def __init__(
        self,
        config: DQNConfig = None,
        state_dim: int = 11,
        action_dim: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        tau: float = 0.001,
        double_dqn: bool = True,
        device: str = None
    ):
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
            state_dim = config.state_size
            action_dim = config.action_size
            learning_rate = config.learning_rate
            gamma = config.gamma
            epsilon_start = config.epsilon_start
            epsilon_end = config.epsilon_end
            epsilon_decay = config.epsilon_decay
            buffer_size = config.buffer_size
            batch_size = config.batch_size
            target_update_freq = config.target_update_freq
            tau = config.tau
            double_dqn = config.use_double_dqn
            device = config.device
        else:
            self.config = DQNConfig(
                state_size=state_dim,
                action_size=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                buffer_size=buffer_size,
                batch_size=batch_size,
                target_update_freq=target_update_freq,
                tau=tau,
                use_double_dqn=double_dqn,
                device=device or 'cpu'
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.double_dqn = double_dqn

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"DQN Agent initialized on device: {self.device}")

        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: Whether we're in training mode (enables exploration)

        Returns:
            Selected action (0=HOLD, 1=BUY, 2=SELL)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store an experience in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
                next_q = self.target_net(next_state_batch).gather(1, next_actions)
            else:
                next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]

            target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network (soft update)
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._soft_update_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def train_episode(self, env: TradingEnvironment) -> Dict:
        """
        Train for one episode.

        Returns:
            Episode metrics
        """
        state = env.reset()
        total_reward = 0
        losses = []

        while not env.done:
            # Select and execute action
            action = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Store experience
            self.store_experience(state, action, reward, next_state, done)

            # Train
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            state = next_state

        self.episode_rewards.append(total_reward)

        metrics = env.get_metrics()
        metrics['episode_reward'] = total_reward
        metrics['avg_loss'] = np.mean(losses) if losses else 0
        metrics['epsilon'] = self.epsilon

        return metrics

    def evaluate(self, env: TradingEnvironment) -> Dict:
        """
        Evaluate the agent without exploration.

        Returns:
            Performance metrics
        """
        state = env.reset()

        while not env.done:
            action = self.select_action(state, training=False)
            state, _, done, _ = env.step(action)

        return env.get_metrics()

    def save(self, filepath: str):
        """Save agent state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"Agent loaded from {filepath}")

    def get_q_values(self, state: np.ndarray) -> Dict[str, float]:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()

        return {
            'hold': float(q_values[0]),
            'buy': float(q_values[1]),
            'sell': float(q_values[2])
        }


class DQNTrainer:
    """
    Trainer class for DQN agent with GA integration.

    Can use GA-optimized hyperparameters for the DQN agent.
    """

    def __init__(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        initial_capital: float = 10000
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.initial_capital = initial_capital

        # Progress callbacks
        self.on_episode_complete = None
        self.on_training_complete = None

    def train(
        self,
        num_episodes: int = 500,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        early_stopping_patience: int = 50
    ) -> Tuple[DQNAgent, Dict]:
        """
        Train a DQN agent.

        Returns:
            Trained agent and training history
        """
        # Create environment and agent
        env = TradingEnvironment(self.train_data, self.initial_capital)
        agent = DQNAgent(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size
        )

        # Training history
        history = {
            'episode_rewards': [],
            'total_returns': [],
            'sharpe_ratios': [],
            'val_returns': [] if self.val_data is not None else None
        }

        best_return = float('-inf')
        patience_counter = 0

        for episode in range(num_episodes):
            # Train episode
            metrics = agent.train_episode(env)

            # Record metrics
            history['episode_rewards'].append(metrics['episode_reward'])
            history['total_returns'].append(metrics['total_return'])
            history['sharpe_ratios'].append(metrics['sharpe_ratio'])

            # Validation
            if self.val_data is not None and episode % 10 == 0:
                val_env = TradingEnvironment(self.val_data, self.initial_capital)
                val_metrics = agent.evaluate(val_env)
                history['val_returns'].append(val_metrics['total_return'])

                # Early stopping check
                if val_metrics['total_return'] > best_return:
                    best_return = val_metrics['total_return']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience // 10:
                    logger.info(f"Early stopping at episode {episode}")
                    break

            # Progress callback
            if self.on_episode_complete:
                self.on_episode_complete(episode, metrics)

            # Logging
            if episode % 50 == 0:
                logger.info(
                    f"Episode {episode}: Return={metrics['total_return']:.4f}, "
                    f"Sharpe={metrics['sharpe_ratio']:.2f}, Epsilon={agent.epsilon:.3f}"
                )

        if self.on_training_complete:
            self.on_training_complete(agent, history)

        return agent, history


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    np.random.seed(42)
    n_days = 1000
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))

    # Create OHLCV data
    data = np.zeros((n_days, 5))
    data[:, 3] = prices  # Close
    data[:, 0] = prices * 0.99  # Open
    data[:, 1] = prices * 1.01  # High
    data[:, 2] = prices * 0.98  # Low
    data[:, 4] = np.random.randint(1000, 10000, n_days)  # Volume

    # Split data
    train_size = int(0.8 * n_days)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # Train DQN agent
    trainer = DQNTrainer(train_data, val_data)
    agent, history = trainer.train(num_episodes=100)

    # Evaluate
    test_env = TradingEnvironment(val_data)
    metrics = agent.evaluate(test_env)

    print(f"\nTest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
