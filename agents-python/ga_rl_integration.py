"""
Genetic Algorithm + Reinforcement Learning Integration Module
============================================================
This module combines GA for hyperparameter optimization with DQN for trading decisions.

Two-level optimization approach:
1. GA evolves populations of DQN configurations (hyperparameters + strategy parameters)
2. Each DQN is trained and evaluated on trading performance
3. Best configurations survive and reproduce

This creates a meta-learning system where genetic evolution discovers
optimal neural network architectures and training configurations.
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import torch

from genetic_algorithm import (
    Gene, Chromosome, GeneticAlgorithm, FitnessEvaluator
)
from reinforcement_learning import (
    TradingEnvironment, DQNAgent, DQNTrainer, DQNConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global WebSocket emitter - set by main.py when training starts
_websocket_emitter = None


def set_websocket_emitter(emitter):
    """Set the WebSocket emitter function for real-time logging."""
    global _websocket_emitter
    _websocket_emitter = emitter


def emit_log(level: str, message: str):
    """Emit a log message via WebSocket if available, also log locally."""
    logger.info(message)
    if _websocket_emitter:
        try:
            import time
            _websocket_emitter('ga_rl_log', {
                'level': level,
                'message': message,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.warning(f"Failed to emit WebSocket log: {e}")


@dataclass
class GADQNChromosome:
    """
    Chromosome encoding both DQN hyperparameters and trading strategy parameters.
    This is evolved by the genetic algorithm to find optimal configurations.
    """
    # DQN Architecture genes
    hidden_size: int = 128
    num_layers: int = 2
    use_dueling: bool = True

    # Training hyperparameter genes
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_decay: float = 0.995
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 100
    tau: float = 0.005

    # Trading strategy genes
    position_size: float = 0.1  # Fraction of capital per trade
    stop_loss: float = 0.02  # Stop loss percentage
    take_profit: float = 0.04  # Take profit percentage
    max_positions: int = 3  # Maximum concurrent positions

    # Reward shaping genes
    reward_scale: float = 1.0
    risk_penalty: float = 0.1
    hold_penalty: float = 0.001

    # Fitness tracking
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Metadata
    generation: int = 0
    chromosome_id: str = ""

    def __post_init__(self):
        if not self.chromosome_id:
            self.chromosome_id = f"chr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(10000)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary for serialization."""
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'use_dueling': self.use_dueling,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'target_update_freq': self.target_update_freq,
            'tau': self.tau,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_positions': self.max_positions,
            'reward_scale': self.reward_scale,
            'risk_penalty': self.risk_penalty,
            'hold_penalty': self.hold_penalty,
            'fitness': self.fitness,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'generation': self.generation,
            'chromosome_id': self.chromosome_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GADQNChromosome':
        """Create chromosome from dictionary."""
        return cls(**data)

    def to_dqn_config(self) -> DQNConfig:
        """Convert chromosome to DQN configuration."""
        return DQNConfig(
            state_size=11,  # TradingState has 11 features
            action_size=3,  # Buy, Sell, Hold
            hidden_size=self.hidden_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=self.epsilon_decay,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq,
            tau=self.tau,
            use_double_dqn=True,
            use_dueling=self.use_dueling,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )


class GADQNPopulation:
    """
    Population of DQN chromosomes managed by genetic algorithm operations.
    """

    # Gene definitions with ranges for mutation/crossover
    GENE_DEFINITIONS = {
        'hidden_size': {'type': 'int', 'min': 32, 'max': 512, 'step': 32},
        'num_layers': {'type': 'int', 'min': 1, 'max': 4, 'step': 1},
        'use_dueling': {'type': 'bool'},
        'learning_rate': {'type': 'float', 'min': 0.00001, 'max': 0.01, 'log_scale': True},
        'gamma': {'type': 'float', 'min': 0.9, 'max': 0.999},
        'epsilon_decay': {'type': 'float', 'min': 0.99, 'max': 0.9999},
        'batch_size': {'type': 'int', 'min': 16, 'max': 128, 'step': 16},
        'buffer_size': {'type': 'int', 'min': 1000, 'max': 100000, 'step': 1000},
        'target_update_freq': {'type': 'int', 'min': 10, 'max': 500, 'step': 10},
        'tau': {'type': 'float', 'min': 0.001, 'max': 0.1},
        'position_size': {'type': 'float', 'min': 0.05, 'max': 0.3},
        'stop_loss': {'type': 'float', 'min': 0.01, 'max': 0.1},
        'take_profit': {'type': 'float', 'min': 0.02, 'max': 0.2},
        'max_positions': {'type': 'int', 'min': 1, 'max': 5, 'step': 1},
        'reward_scale': {'type': 'float', 'min': 0.1, 'max': 10.0},
        'risk_penalty': {'type': 'float', 'min': 0.0, 'max': 1.0},
        'hold_penalty': {'type': 'float', 'min': 0.0, 'max': 0.01},
    }

    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[GADQNChromosome] = []
        self.generation = 0
        self.best_chromosome: Optional[GADQNChromosome] = None
        self.history: List[Dict[str, Any]] = []

    def initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            self.population.append(chromosome)
        logger.info(f"Initialized population with {self.population_size} chromosomes")

    def _create_random_chromosome(self) -> GADQNChromosome:
        """Create a chromosome with random gene values within valid ranges."""
        genes = {}
        for gene_name, gene_def in self.GENE_DEFINITIONS.items():
            if gene_def['type'] == 'bool':
                genes[gene_name] = np.random.choice([True, False])
            elif gene_def['type'] == 'int':
                min_val = gene_def['min']
                max_val = gene_def['max']
                step = gene_def.get('step', 1)
                num_steps = (max_val - min_val) // step
                genes[gene_name] = min_val + np.random.randint(0, num_steps + 1) * step
            elif gene_def['type'] == 'float':
                min_val = gene_def['min']
                max_val = gene_def['max']
                if gene_def.get('log_scale', False):
                    genes[gene_name] = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                else:
                    genes[gene_name] = np.random.uniform(min_val, max_val)

        return GADQNChromosome(**genes, generation=self.generation)

    def mutate(self, chromosome: GADQNChromosome, mutation_rate: float = 0.1) -> GADQNChromosome:
        """Apply mutation to a chromosome."""
        genes = chromosome.to_dict()

        for gene_name, gene_def in self.GENE_DEFINITIONS.items():
            if np.random.random() < mutation_rate:
                if gene_def['type'] == 'bool':
                    genes[gene_name] = not genes[gene_name]
                elif gene_def['type'] == 'int':
                    min_val = gene_def['min']
                    max_val = gene_def['max']
                    step = gene_def.get('step', 1)
                    # Gaussian mutation
                    current = genes[gene_name]
                    mutation = int(np.random.normal(0, (max_val - min_val) / 6))
                    new_val = current + mutation * step
                    genes[gene_name] = max(min_val, min(max_val, new_val))
                elif gene_def['type'] == 'float':
                    min_val = gene_def['min']
                    max_val = gene_def['max']
                    current = genes[gene_name]
                    if gene_def.get('log_scale', False):
                        log_current = np.log(current)
                        log_min, log_max = np.log(min_val), np.log(max_val)
                        mutation = np.random.normal(0, (log_max - log_min) / 6)
                        new_log_val = log_current + mutation
                        genes[gene_name] = np.exp(max(log_min, min(log_max, new_log_val)))
                    else:
                        mutation = np.random.normal(0, (max_val - min_val) / 6)
                        genes[gene_name] = max(min_val, min(max_val, current + mutation))

        # Reset fitness for mutated chromosome
        genes['fitness'] = 0.0
        genes['sharpe_ratio'] = 0.0
        genes['total_return'] = 0.0
        genes['max_drawdown'] = 0.0
        genes['win_rate'] = 0.0
        genes['generation'] = self.generation
        genes['chromosome_id'] = f"chr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(10000)}"

        return GADQNChromosome.from_dict(genes)

    def crossover(self, parent1: GADQNChromosome, parent2: GADQNChromosome) -> Tuple[GADQNChromosome, GADQNChromosome]:
        """Perform uniform crossover between two parent chromosomes."""
        genes1 = parent1.to_dict()
        genes2 = parent2.to_dict()

        child1_genes = {}
        child2_genes = {}

        for gene_name in self.GENE_DEFINITIONS.keys():
            if np.random.random() < 0.5:
                child1_genes[gene_name] = genes1[gene_name]
                child2_genes[gene_name] = genes2[gene_name]
            else:
                child1_genes[gene_name] = genes2[gene_name]
                child2_genes[gene_name] = genes1[gene_name]

        # Set metadata for children
        for child_genes in [child1_genes, child2_genes]:
            child_genes['fitness'] = 0.0
            child_genes['sharpe_ratio'] = 0.0
            child_genes['total_return'] = 0.0
            child_genes['max_drawdown'] = 0.0
            child_genes['win_rate'] = 0.0
            child_genes['generation'] = self.generation
            child_genes['chromosome_id'] = f"chr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(10000)}"

        return GADQNChromosome.from_dict(child1_genes), GADQNChromosome.from_dict(child2_genes)

    def tournament_selection(self, tournament_size: int = 3) -> GADQNChromosome:
        """Select a chromosome using tournament selection."""
        tournament = np.random.choice(len(self.population), size=min(tournament_size, len(self.population)), replace=False)
        tournament_chromosomes = [self.population[i] for i in tournament]
        return max(tournament_chromosomes, key=lambda c: c.fitness)

    def evolve(self, elitism: int = 2, mutation_rate: float = 0.1, crossover_rate: float = 0.8) -> None:
        """
        Evolve the population to create the next generation.

        Args:
            elitism: Number of best chromosomes to keep unchanged
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of performing crossover
        """
        # Sort population by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Track best chromosome
        if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = self.population[0]

        # Record generation statistics
        fitnesses = [c.fitness for c in self.population]
        self.history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'best_chromosome_id': self.population[0].chromosome_id,
            'best_sharpe': self.population[0].sharpe_ratio,
            'best_return': self.population[0].total_return
        })

        # Create new population
        new_population = []

        # Elitism: keep best chromosomes
        for i in range(min(elitism, len(self.population))):
            elite = GADQNChromosome.from_dict(self.population[i].to_dict())
            elite.generation = self.generation + 1
            new_population.append(elite)

        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            if np.random.random() < crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = GADQNChromosome.from_dict(parent1.to_dict())
                child2 = GADQNChromosome.from_dict(parent2.to_dict())
                child1.generation = self.generation + 1
                child2.generation = self.generation + 1

            # Apply mutation
            child1 = self.mutate(child1, mutation_rate)
            child2 = self.mutate(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.generation += 1

        logger.info(f"Generation {self.generation}: Best fitness = {self.history[-1]['best_fitness']:.4f}, "
                   f"Avg fitness = {self.history[-1]['avg_fitness']:.4f}")


class GADQNTrainingManager:
    """
    Manages the training process for GA-optimized DQN agents.
    Handles parallel training, evaluation, and evolution cycles.
    """

    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 50,
        training_episodes: int = 100,
        evaluation_episodes: int = 20,
        symbol: str = "AAPL",
        initial_capital: float = 100000
    ):
        self.population = GADQNPopulation(population_size)
        self.num_generations = num_generations
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        self.symbol = symbol
        self.initial_capital = initial_capital

        # Training state
        self.is_training = False
        self.current_generation = 0
        self.current_chromosome_idx = 0
        self.training_progress: Dict[str, Any] = {}
        self.best_agents: List[Tuple[GADQNChromosome, DQNAgent]] = []

        # Data storage
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None

        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def load_market_data(self, data: pd.DataFrame, train_ratio: float = 0.8) -> None:
        """
        Load and split market data for training and validation.

        Args:
            data: DataFrame with OHLCV data
            train_ratio: Fraction of data to use for training
        """
        split_idx = int(len(data) * train_ratio)
        self.training_data = data.iloc[:split_idx].copy()
        self.validation_data = data.iloc[split_idx:].copy()
        logger.info(f"Loaded {len(self.training_data)} training samples and "
                   f"{len(self.validation_data)} validation samples")

    def evaluate_chromosome(
        self,
        chromosome: GADQNChromosome,
        data: pd.DataFrame,
        num_episodes: int
    ) -> Dict[str, float]:
        """
        Train and evaluate a DQN agent with the given chromosome configuration.

        Returns:
            Dictionary with fitness metrics
        """
        # Create environment and agent from chromosome
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data[['open', 'high', 'low', 'close', 'volume']].values
        else:
            data_array = data

        env = TradingEnvironment(
            data=data_array,
            initial_capital=self.initial_capital,
            transaction_cost=0.001
        )

        config = chromosome.to_dqn_config()
        agent = DQNAgent(config)

        # Train the agent directly using train_episode
        for episode in range(num_episodes):
            env.reset()
            agent.train_episode(env)

        # Evaluate on same data to compute fitness
        total_rewards = []
        total_returns = []

        for _ in range(self.evaluation_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            total_returns.append(info.get('total_return', 0))

        # Compute fitness metrics
        avg_reward = np.mean(total_rewards)
        avg_return = np.mean(total_returns)

        # Compute Sharpe ratio from returns
        if len(total_returns) > 1 and np.std(total_returns) > 0:
            sharpe = np.mean(total_returns) / np.std(total_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Compute max drawdown (simplified)
        max_drawdown = min(total_returns) if total_returns else 0

        # Compute win rate
        win_rate = sum(1 for r in total_returns if r > 0) / len(total_returns) if total_returns else 0

        # Combined fitness function
        # Emphasizes Sharpe ratio but considers returns and drawdown
        fitness = (
            sharpe * 0.4 +
            avg_return * 100 * 0.3 +  # Scale returns
            (1 - abs(max_drawdown)) * 0.2 +  # Penalize drawdown
            win_rate * 0.1
        )

        return {
            'fitness': fitness,
            'sharpe_ratio': sharpe,
            'total_return': avg_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_reward': avg_reward
        }

    def run_evolution(self, callback: Optional[Callable[[Dict], None]] = None) -> GADQNChromosome:
        """
        Run the complete GA-DQN evolution process.

        Args:
            callback: Optional callback function called after each generation
                     with progress information

        Returns:
            The best chromosome found
        """
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_market_data first.")

        self.is_training = True
        self._stop_event.clear()

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

        emit_log('info', "=" * 50)
        emit_log('info', "🧬 STARTING GA+RL EVOLUTION")
        emit_log('info', f"   Device: {device.upper()} ({device_name})")
        emit_log('info', f"   Population: {self.population.population_size} chromosomes")
        emit_log('info', f"   Generations: {self.num_generations}")
        emit_log('info', f"   Training episodes per chromosome: {self.training_episodes}")
        emit_log('info', f"   Training data: {len(self.training_data)} samples")
        emit_log('info', "=" * 50)

        # Initialize population
        emit_log('info', "🔄 Creating initial random population...")
        self.population.initialize_population()
        emit_log('success', f"   ✓ Created {len(self.population.population)} chromosomes with random genes")

        try:
            for generation in range(self.num_generations):
                if self._stop_event.is_set():
                    emit_log('warning', "⚠️ Training stopped by user")
                    break

                self.current_generation = generation
                generation_start = time.time()

                emit_log('info', "")
                emit_log('info', f"📊 GENERATION {generation + 1}/{self.num_generations}")
                emit_log('info', "-" * 40)

                # Evaluate each chromosome in the population
                for idx, chromosome in enumerate(self.population.population):
                    if self._stop_event.is_set():
                        break

                    self.current_chromosome_idx = idx
                    chr_start = time.time()

                    emit_log('info', f"   🧬 [{idx + 1}/{len(self.population.population)}] Training chromosome...")
                    emit_log('info', f"      Config: hidden={chromosome.hidden_size}, lr={chromosome.learning_rate:.6f}, batch={chromosome.batch_size}")

                    # Evaluate chromosome
                    metrics = self.evaluate_chromosome(
                        chromosome,
                        self.training_data,
                        self.training_episodes
                    )

                    # Update chromosome with fitness
                    chromosome.fitness = metrics['fitness']
                    chromosome.sharpe_ratio = metrics['sharpe_ratio']
                    chromosome.total_return = metrics['total_return']
                    chromosome.max_drawdown = metrics['max_drawdown']
                    chromosome.win_rate = metrics['win_rate']

                    chr_time = time.time() - chr_start
                    emit_log('success', f"      ✓ Results: fitness={chromosome.fitness:.4f}, sharpe={chromosome.sharpe_ratio:.4f}, return={chromosome.total_return*100:.2f}% ({chr_time:.1f}s)")

                    # Update progress
                    with self._lock:
                        self.training_progress = {
                            'generation': generation + 1,
                            'total_generations': self.num_generations,
                            'chromosome': idx + 1,
                            'total_chromosomes': len(self.population.population),
                            'current_fitness': chromosome.fitness,
                            'best_fitness': self.population.best_chromosome.fitness if self.population.best_chromosome else 0,
                            'elapsed_time': time.time() - generation_start
                        }

                # Evolve population
                emit_log('info', "   🔀 Evolving population (selection → crossover → mutation)...")
                self.population.evolve()

                generation_time = time.time() - generation_start
                best_fitness = self.population.best_chromosome.fitness if self.population.best_chromosome else 0
                avg_fitness = np.mean([c.fitness for c in self.population.population])

                emit_log('success', f"   ✅ Generation {generation+1} complete!")
                emit_log('info', f"      Best fitness: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Time: {generation_time:.1f}s")

                # Call callback if provided
                if callback:
                    callback({
                        'generation': generation + 1,
                        'best_fitness': self.population.history[-1]['best_fitness'],
                        'avg_fitness': self.population.history[-1]['avg_fitness'],
                        'best_chromosome': self.population.best_chromosome.to_dict() if self.population.best_chromosome else None,
                        'generation_time': generation_time
                    })

            # Final validation of best chromosome
            if self.population.best_chromosome and self.validation_data is not None:
                emit_log('info', "")
                emit_log('info', "📋 FINAL VALIDATION")
                emit_log('info', "-" * 40)
                emit_log('info', "   Testing best chromosome on validation data...")
                validation_metrics = self.evaluate_chromosome(
                    self.population.best_chromosome,
                    self.validation_data,
                    self.evaluation_episodes * 2
                )
                emit_log('success', f"   ✓ Validation Sharpe: {validation_metrics['sharpe_ratio']:.4f}")
                emit_log('success', f"   ✓ Validation Return: {validation_metrics['total_return']*100:.2f}%")
                emit_log('success', f"   ✓ Validation Win Rate: {validation_metrics['win_rate']*100:.2f}%")

            emit_log('info', "")
            emit_log('success', "🎉 GA+RL EVOLUTION COMPLETE!")
            return self.population.best_chromosome

        finally:
            self.is_training = False

    def stop_training(self) -> None:
        """Signal the training process to stop."""
        self._stop_event.set()

    def get_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        with self._lock:
            return self.training_progress.copy()

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get the complete evolution history."""
        return self.population.history

    def save_best_agent(self, path: str) -> None:
        """Save the best chromosome and its configuration."""
        if self.population.best_chromosome is None:
            raise ValueError("No best chromosome available")

        save_data = {
            'chromosome': self.population.best_chromosome.to_dict(),
            'history': self.population.history,
            'training_config': {
                'num_generations': self.num_generations,
                'population_size': self.population.population_size,
                'training_episodes': self.training_episodes,
                'evaluation_episodes': self.evaluation_episodes,
                'symbol': self.symbol
            }
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Saved best agent configuration to {path}")

    def load_best_agent(self, path: str) -> GADQNChromosome:
        """Load a previously saved best chromosome."""
        with open(path, 'r') as f:
            save_data = json.load(f)

        chromosome = GADQNChromosome.from_dict(save_data['chromosome'])
        self.population.best_chromosome = chromosome
        self.population.history = save_data.get('history', [])

        logger.info(f"Loaded agent configuration from {path}")
        return chromosome

    def create_trading_agent(self, chromosome: Optional[GADQNChromosome] = None) -> DQNAgent:
        """
        Create a DQN agent from a chromosome configuration.
        If no chromosome provided, uses the best one from evolution.
        """
        if chromosome is None:
            chromosome = self.population.best_chromosome

        if chromosome is None:
            raise ValueError("No chromosome available")

        config = chromosome.to_dqn_config()
        return DQNAgent(config)


class IntegratedTradingSystem:
    """
    Complete trading system integrating GA-optimized DQN with live trading capabilities.
    This is the main entry point for the GA+RL trading system.
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        population_size: int = 20,
        num_generations: int = 50,
        model_dir: str = "./models",
        initial_capital: float = 100000
    ):
        self.symbol = symbol
        self.model_dir = model_dir
        self.initial_capital = initial_capital
        os.makedirs(model_dir, exist_ok=True)

        # Initialize training manager
        self.training_manager = GADQNTrainingManager(
            population_size=population_size,
            num_generations=num_generations,
            symbol=symbol,
            initial_capital=initial_capital
        )

        # Trading agent (initialized after training)
        self.trading_agent: Optional[DQNAgent] = None
        self.current_chromosome: Optional[GADQNChromosome] = None

        # Trading state
        self.is_live_trading = False
        self.positions: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []

    def train(
        self,
        market_data: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the trading system using GA-optimized DQN.

        Args:
            market_data: Historical OHLCV data
            progress_callback: Optional callback for progress updates

        Returns:
            Training results summary
        """
        logger.info(f"Starting GA-DQN training for {self.symbol}")

        # Load data
        self.training_manager.load_market_data(market_data)

        # Run evolution
        best_chromosome = self.training_manager.run_evolution(progress_callback)

        if best_chromosome:
            self.current_chromosome = best_chromosome
            self.trading_agent = self.training_manager.create_trading_agent(best_chromosome)

            # Save the best configuration
            save_path = os.path.join(self.model_dir, f"{self.symbol}_ga_dqn_config.json")
            self.training_manager.save_best_agent(save_path)

            return {
                'success': True,
                'best_fitness': best_chromosome.fitness,
                'sharpe_ratio': best_chromosome.sharpe_ratio,
                'total_return': best_chromosome.total_return,
                'max_drawdown': best_chromosome.max_drawdown,
                'win_rate': best_chromosome.win_rate,
                'generations_completed': self.training_manager.current_generation,
                'chromosome_config': best_chromosome.to_dict()
            }

        return {'success': False, 'error': 'Training failed to produce valid results'}

    def get_trading_signal(self, market_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Get a trading signal based on current market state.

        Args:
            market_state: Dictionary with current market indicators

        Returns:
            Trading signal with action and confidence
        """
        if self.trading_agent is None:
            return {'action': 'hold', 'confidence': 0, 'reason': 'No trained agent available'}

        # Convert market state to numpy array
        state = np.array([
            market_state.get('price_change_1d', 0),
            market_state.get('price_change_5d', 0),
            market_state.get('price_change_20d', 0),
            market_state.get('volume_ratio', 1),
            market_state.get('rsi', 50),
            market_state.get('macd', 0),
            market_state.get('macd_signal', 0),
            market_state.get('bb_position', 0.5),
            market_state.get('position', 0),
            market_state.get('portfolio_value_change', 0),
            market_state.get('time_in_position', 0)
        ], dtype=np.float32)

        # Get action and Q-values from agent
        action = self.trading_agent.select_action(state, training=False)

        # Get Q-values for confidence estimation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.trading_agent.config.device)
            q_values = self.trading_agent.policy_net(state_tensor).cpu().numpy()[0]

        # Compute confidence based on Q-value differences
        q_softmax = np.exp(q_values) / np.sum(np.exp(q_values))
        confidence = float(q_softmax[action])

        action_names = ['buy', 'sell', 'hold']

        return {
            'action': action_names[action],
            'confidence': confidence,
            'q_values': {name: float(q) for name, q in zip(action_names, q_values)},
            'chromosome_id': self.current_chromosome.chromosome_id if self.current_chromosome else None
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'symbol': self.symbol,
            'is_training': self.training_manager.is_training,
            'is_live_trading': self.is_live_trading,
            'has_trained_agent': self.trading_agent is not None,
            'training_progress': self.training_manager.get_progress(),
            'current_chromosome': self.current_chromosome.to_dict() if self.current_chromosome else None,
            'positions': self.positions,
            'total_trades': len(self.trade_history)
        }


# Convenience function for quick training
def quick_train(
    data: pd.DataFrame,
    symbol: str = "AAPL",
    population_size: int = 10,
    num_generations: int = 10,
    verbose: bool = True
) -> IntegratedTradingSystem:
    """
    Quick training function for testing and development.
    Uses smaller parameters for faster iteration.
    """
    system = IntegratedTradingSystem(
        symbol=symbol,
        population_size=population_size,
        num_generations=num_generations
    )

    def progress_callback(info):
        if verbose:
            print(f"Generation {info['generation']}: "
                  f"Best fitness = {info['best_fitness']:.4f}, "
                  f"Avg fitness = {info['avg_fitness']:.4f}")

    results = system.train(data, progress_callback)

    if verbose:
        print("\nTraining Results:")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"  Total Return: {results.get('total_return', 0)*100:.2f}%")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate: {results.get('win_rate', 0)*100:.2f}%")

    return system


if __name__ == "__main__":
    # Example usage
    print("GA-DQN Integration Module")
    print("=" * 50)

    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')

    # Generate synthetic price data with trend and volatility
    price = 100
    prices = [price]
    for _ in range(499):
        change = np.random.normal(0.0002, 0.02)  # Slight upward drift
        price *= (1 + change)
        prices.append(price)

    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices]
    })

    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data['datetime'].min()} to {sample_data['datetime'].max()}")

    # Quick test with small parameters
    print("\nRunning quick training test...")
    system = quick_train(
        sample_data,
        symbol="TEST",
        population_size=5,
        num_generations=3,
        verbose=True
    )

    # Test trading signal
    print("\nTesting trading signal generation...")
    test_state = {
        'price_change_1d': 0.01,
        'price_change_5d': 0.03,
        'price_change_20d': 0.05,
        'volume_ratio': 1.2,
        'rsi': 55,
        'macd': 0.5,
        'macd_signal': 0.3,
        'bb_position': 0.6,
        'position': 0,
        'portfolio_value_change': 0,
        'time_in_position': 0
    }

    signal = system.get_trading_signal(test_state)
    print(f"Trading Signal: {signal['action']} (confidence: {signal['confidence']:.2%})")
    print(f"Q-values: {signal['q_values']}")
