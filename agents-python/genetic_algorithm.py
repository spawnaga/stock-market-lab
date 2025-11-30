"""
Genetic Algorithm Module for Trading Strategy Optimization

This module implements a genetic algorithm that evolves trading strategy parameters.
Each chromosome represents a trading strategy with parameters like:
- Technical indicator periods (MA, RSI, MACD, etc.)
- Entry/exit thresholds
- Position sizing rules
- Risk management parameters

The GA uses tournament selection, uniform crossover, and Gaussian mutation.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import json
import logging
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


@dataclass
class Gene:
    """Represents a single gene (parameter) in the chromosome."""
    name: str
    value: float
    min_value: float
    max_value: float
    is_integer: bool = False

    def mutate(self, mutation_strength: float = 0.1) -> 'Gene':
        """Apply Gaussian mutation to the gene."""
        range_size = self.max_value - self.min_value
        mutation = np.random.normal(0, mutation_strength * range_size)
        new_value = self.value + mutation
        new_value = np.clip(new_value, self.min_value, self.max_value)

        if self.is_integer:
            new_value = int(round(new_value))

        return Gene(
            name=self.name,
            value=new_value,
            min_value=self.min_value,
            max_value=self.max_value,
            is_integer=self.is_integer
        )

    def randomize(self) -> 'Gene':
        """Create a random value for this gene."""
        if self.is_integer:
            new_value = random.randint(int(self.min_value), int(self.max_value))
        else:
            new_value = random.uniform(self.min_value, self.max_value)

        return Gene(
            name=self.name,
            value=new_value,
            min_value=self.min_value,
            max_value=self.max_value,
            is_integer=self.is_integer
        )


@dataclass
class Chromosome:
    """
    Represents a complete trading strategy as a chromosome.

    The chromosome encodes all parameters needed to define a trading strategy:
    - Technical indicator parameters
    - Entry/exit signals
    - Risk management rules
    """
    genes: List[Gene]
    fitness: float = 0.0
    generation: int = 0
    strategy_type: str = "technical"

    def __post_init__(self):
        self._gene_dict = {gene.name: gene for gene in self.genes}

    def get_gene(self, name: str) -> Optional[Gene]:
        """Get a gene by name."""
        return self._gene_dict.get(name)

    def get_value(self, name: str) -> Optional[float]:
        """Get a gene's value by name."""
        gene = self.get_gene(name)
        return gene.value if gene else None

    def to_dict(self) -> Dict:
        """Convert chromosome to dictionary for serialization."""
        return {
            'genes': {gene.name: gene.value for gene in self.genes},
            'fitness': self.fitness,
            'generation': self.generation,
            'strategy_type': self.strategy_type
        }

    @classmethod
    def from_dict(cls, data: Dict, gene_templates: List[Gene]) -> 'Chromosome':
        """Create chromosome from dictionary."""
        genes = []
        for template in gene_templates:
            value = data['genes'].get(template.name, template.value)
            genes.append(Gene(
                name=template.name,
                value=value,
                min_value=template.min_value,
                max_value=template.max_value,
                is_integer=template.is_integer
            ))
        return cls(
            genes=genes,
            fitness=data.get('fitness', 0.0),
            generation=data.get('generation', 0),
            strategy_type=data.get('strategy_type', 'technical')
        )

    def copy(self) -> 'Chromosome':
        """Create a deep copy of the chromosome."""
        return Chromosome(
            genes=[Gene(g.name, g.value, g.min_value, g.max_value, g.is_integer) for g in self.genes],
            fitness=self.fitness,
            generation=self.generation,
            strategy_type=self.strategy_type
        )


class TradingStrategyChromosome:
    """
    Factory for creating trading strategy chromosomes.

    Defines the gene templates for different strategy types.
    """

    @staticmethod
    def create_technical_strategy() -> Chromosome:
        """Create a chromosome for a technical analysis strategy."""
        genes = [
            # Moving Average parameters
            Gene("fast_ma_period", 10, 5, 50, is_integer=True),
            Gene("slow_ma_period", 30, 20, 200, is_integer=True),

            # RSI parameters
            Gene("rsi_period", 14, 5, 30, is_integer=True),
            Gene("rsi_oversold", 30, 10, 40),
            Gene("rsi_overbought", 70, 60, 90),

            # MACD parameters
            Gene("macd_fast", 12, 5, 20, is_integer=True),
            Gene("macd_slow", 26, 15, 40, is_integer=True),
            Gene("macd_signal", 9, 5, 15, is_integer=True),

            # Bollinger Bands
            Gene("bb_period", 20, 10, 50, is_integer=True),
            Gene("bb_std", 2.0, 1.0, 3.0),

            # Entry/Exit thresholds
            Gene("entry_threshold", 0.02, 0.001, 0.1),
            Gene("exit_threshold", 0.015, 0.001, 0.1),
            Gene("stop_loss", 0.05, 0.01, 0.15),
            Gene("take_profit", 0.1, 0.02, 0.3),

            # Position sizing
            Gene("position_size_pct", 0.1, 0.01, 0.5),
            Gene("max_positions", 5, 1, 20, is_integer=True),

            # Signal weights (how much each indicator contributes)
            Gene("ma_weight", 0.3, 0.0, 1.0),
            Gene("rsi_weight", 0.25, 0.0, 1.0),
            Gene("macd_weight", 0.25, 0.0, 1.0),
            Gene("bb_weight", 0.2, 0.0, 1.0),
        ]

        # Randomize initial values
        genes = [g.randomize() for g in genes]

        return Chromosome(genes=genes, strategy_type="technical")

    @staticmethod
    def create_momentum_strategy() -> Chromosome:
        """Create a chromosome for a momentum-based strategy."""
        genes = [
            # Momentum parameters
            Gene("momentum_period", 14, 5, 50, is_integer=True),
            Gene("momentum_threshold", 0.02, 0.001, 0.1),

            # Volume parameters
            Gene("volume_ma_period", 20, 5, 50, is_integer=True),
            Gene("volume_spike_threshold", 1.5, 1.1, 3.0),

            # Trend strength
            Gene("adx_period", 14, 7, 30, is_integer=True),
            Gene("adx_threshold", 25, 15, 40),

            # Entry/Exit
            Gene("entry_threshold", 0.02, 0.001, 0.1),
            Gene("stop_loss", 0.05, 0.01, 0.15),
            Gene("take_profit", 0.1, 0.02, 0.3),
            Gene("trailing_stop", 0.03, 0.01, 0.1),

            # Position sizing
            Gene("position_size_pct", 0.15, 0.01, 0.5),
            Gene("max_positions", 3, 1, 10, is_integer=True),
        ]

        genes = [g.randomize() for g in genes]
        return Chromosome(genes=genes, strategy_type="momentum")

    @staticmethod
    def create_mean_reversion_strategy() -> Chromosome:
        """Create a chromosome for a mean reversion strategy."""
        genes = [
            # Mean reversion parameters
            Gene("lookback_period", 20, 5, 100, is_integer=True),
            Gene("z_score_entry", 2.0, 1.0, 3.0),
            Gene("z_score_exit", 0.5, 0.0, 1.5),

            # Volatility filter
            Gene("volatility_period", 20, 10, 50, is_integer=True),
            Gene("volatility_threshold", 0.02, 0.005, 0.1),

            # Entry/Exit
            Gene("stop_loss", 0.08, 0.02, 0.2),
            Gene("take_profit", 0.06, 0.02, 0.15),

            # Position sizing
            Gene("position_size_pct", 0.2, 0.05, 0.5),
            Gene("max_positions", 4, 1, 10, is_integer=True),
        ]

        genes = [g.randomize() for g in genes]
        return Chromosome(genes=genes, strategy_type="mean_reversion")


class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving trading strategies.

    Uses tournament selection, uniform crossover, and Gaussian mutation
    to evolve a population of trading strategies.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 5,
        elitism_count: int = 2,
        mutation_strength: float = 0.1
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.mutation_strength = mutation_strength

        self.population: List[Chromosome] = []
        self.generation = 0
        self.best_chromosome: Optional[Chromosome] = None
        self.fitness_history: List[Dict] = []

        # Callbacks for progress reporting
        self.on_generation_complete: Optional[Callable] = None
        self.on_fitness_evaluated: Optional[Callable] = None

    def initialize_population(self, strategy_type: str = "technical") -> None:
        """Initialize the population with random chromosomes."""
        self.population = []

        factory_map = {
            "technical": TradingStrategyChromosome.create_technical_strategy,
            "momentum": TradingStrategyChromosome.create_momentum_strategy,
            "mean_reversion": TradingStrategyChromosome.create_mean_reversion_strategy
        }

        factory = factory_map.get(strategy_type, TradingStrategyChromosome.create_technical_strategy)

        for _ in range(self.population_size):
            chromosome = factory()
            chromosome.generation = self.generation
            self.population.append(chromosome)

        logger.info(f"Initialized population with {self.population_size} {strategy_type} strategies")

    def tournament_selection(self) -> Chromosome:
        """Select a chromosome using tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        winner = max(tournament, key=lambda c: c.fitness)
        return winner.copy()

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform uniform crossover between two parents.
        Each gene has a 50% chance of coming from either parent.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_genes = []
        child2_genes = []

        for g1, g2 in zip(parent1.genes, parent2.genes):
            if random.random() < 0.5:
                child1_genes.append(Gene(g1.name, g1.value, g1.min_value, g1.max_value, g1.is_integer))
                child2_genes.append(Gene(g2.name, g2.value, g2.min_value, g2.max_value, g2.is_integer))
            else:
                child1_genes.append(Gene(g2.name, g2.value, g2.min_value, g2.max_value, g2.is_integer))
                child2_genes.append(Gene(g1.name, g1.value, g1.min_value, g1.max_value, g1.is_integer))

        child1 = Chromosome(
            genes=child1_genes,
            generation=self.generation + 1,
            strategy_type=parent1.strategy_type
        )
        child2 = Chromosome(
            genes=child2_genes,
            generation=self.generation + 1,
            strategy_type=parent1.strategy_type
        )

        return child1, child2

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation to a chromosome."""
        mutated_genes = []

        for gene in chromosome.genes:
            if random.random() < self.mutation_rate:
                mutated_genes.append(gene.mutate(self.mutation_strength))
            else:
                mutated_genes.append(Gene(
                    gene.name, gene.value, gene.min_value, gene.max_value, gene.is_integer
                ))

        return Chromosome(
            genes=mutated_genes,
            generation=chromosome.generation,
            strategy_type=chromosome.strategy_type
        )

    def evolve(self, fitness_function: Callable[[Chromosome], float]) -> Chromosome:
        """
        Evolve the population for one generation.

        Args:
            fitness_function: Function that evaluates a chromosome and returns fitness score

        Returns:
            The best chromosome from this generation
        """
        # Evaluate fitness for all chromosomes
        for chromosome in self.population:
            if chromosome.fitness == 0.0:  # Only evaluate if not already evaluated
                chromosome.fitness = fitness_function(chromosome)
                if self.on_fitness_evaluated:
                    self.on_fitness_evaluated(chromosome)

        # Sort by fitness (descending)
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update best chromosome
        if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = self.population[0].copy()

        # Record fitness history
        fitness_stats = {
            'generation': self.generation,
            'best_fitness': self.population[0].fitness,
            'avg_fitness': np.mean([c.fitness for c in self.population]),
            'worst_fitness': self.population[-1].fitness,
            'best_chromosome': self.population[0].to_dict()
        }
        self.fitness_history.append(fitness_stats)

        # Create new population
        new_population = []

        # Elitism: keep the best chromosomes
        for i in range(min(self.elitism_count, len(self.population))):
            elite = self.population[i].copy()
            elite.generation = self.generation + 1
            new_population.append(elite)

        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.generation += 1

        if self.on_generation_complete:
            self.on_generation_complete(fitness_stats)

        logger.info(f"Generation {self.generation}: Best={fitness_stats['best_fitness']:.4f}, "
                   f"Avg={fitness_stats['avg_fitness']:.4f}")

        return self.best_chromosome

    def run(
        self,
        fitness_function: Callable[[Chromosome], float],
        num_generations: int = 100,
        early_stopping_generations: int = 20,
        target_fitness: Optional[float] = None
    ) -> Chromosome:
        """
        Run the genetic algorithm for multiple generations.

        Args:
            fitness_function: Function to evaluate chromosome fitness
            num_generations: Maximum number of generations
            early_stopping_generations: Stop if no improvement for this many generations
            target_fitness: Stop if this fitness is achieved

        Returns:
            The best chromosome found
        """
        generations_without_improvement = 0
        best_fitness_so_far = float('-inf')

        for gen in range(num_generations):
            best = self.evolve(fitness_function)

            # Check for improvement
            if best.fitness > best_fitness_so_far:
                best_fitness_so_far = best.fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Check early stopping
            if generations_without_improvement >= early_stopping_generations:
                logger.info(f"Early stopping: No improvement for {early_stopping_generations} generations")
                break

            # Check target fitness
            if target_fitness is not None and best.fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} achieved")
                break

        return self.best_chromosome

    def get_top_chromosomes(self, n: int = 10) -> List[Chromosome]:
        """Get the top n chromosomes from the current population."""
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        return sorted_pop[:n]

    def save_state(self, filepath: str) -> None:
        """Save the current state of the GA to a file."""
        state = {
            'generation': self.generation,
            'population': [c.to_dict() for c in self.population],
            'best_chromosome': self.best_chromosome.to_dict() if self.best_chromosome else None,
            'fitness_history': self.fitness_history,
            'config': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'tournament_size': self.tournament_size,
                'elitism_count': self.elitism_count,
                'mutation_strength': self.mutation_strength
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"GA state saved to {filepath}")

    def load_state(self, filepath: str, gene_templates: List[Gene]) -> None:
        """Load GA state from a file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.generation = state['generation']
        self.population = [
            Chromosome.from_dict(c, gene_templates) for c in state['population']
        ]
        if state['best_chromosome']:
            self.best_chromosome = Chromosome.from_dict(state['best_chromosome'], gene_templates)
        self.fitness_history = state['fitness_history']

        config = state['config']
        self.population_size = config['population_size']
        self.mutation_rate = config['mutation_rate']
        self.crossover_rate = config['crossover_rate']
        self.tournament_size = config['tournament_size']
        self.elitism_count = config['elitism_count']
        self.mutation_strength = config['mutation_strength']

        logger.info(f"GA state loaded from {filepath}")


class FitnessEvaluator:
    """
    Evaluates the fitness of a trading strategy chromosome
    by running backtests against historical data.
    """

    def __init__(self, historical_data: np.ndarray, initial_capital: float = 10000):
        """
        Args:
            historical_data: OHLCV data as numpy array with columns [open, high, low, close, volume]
            initial_capital: Starting capital for backtests
        """
        self.historical_data = historical_data
        self.initial_capital = initial_capital

    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Evaluate a chromosome by running a backtest.

        Returns a fitness score that considers:
        - Total return
        - Sharpe ratio
        - Max drawdown (penalty)
        - Number of trades (penalty for too few or too many)
        """
        try:
            # Run backtest with chromosome parameters
            results = self._run_backtest(chromosome)

            # Calculate fitness components
            total_return = results['total_return']
            sharpe_ratio = results['sharpe_ratio']
            max_drawdown = results['max_drawdown']
            num_trades = results['num_trades']
            win_rate = results['win_rate']

            # Fitness function (higher is better)
            # Reward: returns and sharpe ratio
            # Penalty: drawdown and extreme trade counts

            fitness = 0.0

            # Return component (40% weight)
            fitness += 0.4 * np.tanh(total_return * 5)  # Normalize with tanh

            # Sharpe ratio component (30% weight)
            fitness += 0.3 * np.tanh(sharpe_ratio)

            # Drawdown penalty (20% weight)
            drawdown_penalty = max(0, max_drawdown - 0.1) * 2  # Penalize drawdown > 10%
            fitness -= 0.2 * drawdown_penalty

            # Trade frequency component (10% weight)
            # Penalize too few trades (< 10) or too many (> 500)
            if num_trades < 10:
                fitness -= 0.1 * (10 - num_trades) / 10
            elif num_trades > 500:
                fitness -= 0.1 * min(1, (num_trades - 500) / 500)
            else:
                fitness += 0.1 * win_rate  # Reward good win rate

            return max(0, fitness)  # Ensure non-negative fitness

        except Exception as e:
            logger.warning(f"Error evaluating chromosome: {e}")
            return 0.0

    def _run_backtest(self, chromosome: Chromosome) -> Dict:
        """
        Run a simplified backtest using chromosome parameters.

        This is a basic implementation - the full backtesting_framework.py
        can be used for more comprehensive backtests.
        """
        data = self.historical_data
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [capital]

        # Get strategy parameters from chromosome
        fast_ma = int(chromosome.get_value('fast_ma_period') or 10)
        slow_ma = int(chromosome.get_value('slow_ma_period') or 30)
        stop_loss = chromosome.get_value('stop_loss') or 0.05
        take_profit = chromosome.get_value('take_profit') or 0.1
        position_size_pct = chromosome.get_value('position_size_pct') or 0.1

        # Calculate moving averages
        closes = data[:, 3]  # Close prices

        if len(closes) < slow_ma + 1:
            return self._empty_results()

        # Simple moving averages
        fast_ma_values = np.convolve(closes, np.ones(fast_ma)/fast_ma, mode='valid')
        slow_ma_values = np.convolve(closes, np.ones(slow_ma)/slow_ma, mode='valid')

        # Align arrays
        start_idx = slow_ma - fast_ma
        fast_ma_values = fast_ma_values[start_idx:]

        # Ensure same length
        min_len = min(len(fast_ma_values), len(slow_ma_values))
        fast_ma_values = fast_ma_values[:min_len]
        slow_ma_values = slow_ma_values[:min_len]

        # Get corresponding prices
        price_start = slow_ma - 1
        prices = closes[price_start:price_start + min_len]

        for i in range(1, len(prices)):
            current_price = prices[i]

            # Trading signals
            if position == 0:
                # Entry signal: fast MA crosses above slow MA
                if fast_ma_values[i] > slow_ma_values[i] and fast_ma_values[i-1] <= slow_ma_values[i-1]:
                    # Buy
                    shares = int((capital * position_size_pct) / current_price)
                    if shares > 0:
                        position = shares
                        entry_price = current_price
                        capital -= shares * current_price
            else:
                # Check exit conditions
                pnl_pct = (current_price - entry_price) / entry_price

                # Exit signal: stop loss, take profit, or fast MA crosses below slow MA
                should_exit = (
                    pnl_pct <= -stop_loss or  # Stop loss
                    pnl_pct >= take_profit or  # Take profit
                    (fast_ma_values[i] < slow_ma_values[i] and fast_ma_values[i-1] >= slow_ma_values[i-1])  # MA crossover
                )

                if should_exit:
                    # Sell
                    capital += position * current_price
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'shares': position
                    })
                    position = 0
                    entry_price = 0

            # Update equity curve
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)

        # Close any open position at the end
        if position > 0:
            final_price = prices[-1]
            capital += position * final_price
            pnl_pct = (final_price - entry_price) / entry_price
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl_pct': pnl_pct,
                'shares': position
            })

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        if trades:
            winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'final_capital': capital,
            'equity_curve': equity_curve,
            'trades': trades
        }

    def _empty_results(self) -> Dict:
        """Return empty results for invalid data."""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0,
            'win_rate': 0,
            'final_capital': self.initial_capital,
            'equity_curve': [self.initial_capital],
            'trades': []
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    np.random.seed(42)
    n_days = 500
    prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.02))

    # Create OHLCV data
    data = np.zeros((n_days, 5))
    data[:, 3] = prices  # Close
    data[:, 0] = prices * 0.99  # Open
    data[:, 1] = prices * 1.01  # High
    data[:, 2] = prices * 0.98  # Low
    data[:, 4] = np.random.randint(1000, 10000, n_days)  # Volume

    # Create fitness evaluator
    evaluator = FitnessEvaluator(data, initial_capital=10000)

    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=5,
        elitism_count=2
    )

    ga.initialize_population(strategy_type="technical")

    # Run for 20 generations
    best = ga.run(
        fitness_function=evaluator.evaluate,
        num_generations=20,
        early_stopping_generations=10
    )

    print(f"\nBest Strategy Found:")
    print(f"Fitness: {best.fitness:.4f}")
    print(f"Parameters: {best.to_dict()['genes']}")
