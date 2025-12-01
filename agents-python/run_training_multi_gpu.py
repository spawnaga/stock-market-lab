#!/usr/bin/env python3
"""
Multi-GPU GA+RL Training Script
Uses the existing ga_rl_integration module with parallel chromosome evaluation
"""

import torch
import torch.multiprocessing as mp
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import time
import logging

# Add work directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("Multi-GPU GA+RL Trading Strategy Optimization")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    print("=" * 60)


def generate_synthetic_data(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic market data for training."""
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='1min')
    price = 100.0
    prices = [price]

    for _ in range(num_samples - 1):
        change = np.random.normal(0.00005, 0.003)
        price *= (1 + change)
        prices.append(price)

    prices = np.array(prices)

    return pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, num_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, num_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, num_samples))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, num_samples)
    })


def train_chromosome_worker(gpu_id: int, chromosome_configs: List[Dict],
                           training_data_dict: Dict, validation_data_dict: Dict,
                           episodes: int, result_queue: mp.Queue):
    """Worker process to train chromosomes on a specific GPU."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)  # Device 0 in this process's view

        # Reconstruct DataFrames
        training_data = pd.DataFrame(training_data_dict)
        validation_data = pd.DataFrame(validation_data_dict)

        # Import core components directly
        from ga_rl_integration import GADQNChromosome, DQNAgent, TradingEnvironment

        for config in chromosome_configs:
            try:
                # Create chromosome with specified genes
                chromosome = GADQNChromosome()
                chromosome.genes = {
                    'hidden_size': config.get('hidden_size', 256),
                    'num_layers': 2,
                    'use_dueling': True,
                    'learning_rate': config.get('learning_rate', 0.0001),
                    'gamma': config.get('gamma', 0.99),
                    'epsilon_decay': config.get('epsilon_decay', 0.995),
                    'batch_size': config.get('batch_size', 32),
                    'buffer_size': 10000,
                    'target_update_freq': 100,
                    'tau': 0.005,
                    'position_size': 0.1,
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'max_positions': 1,
                    'reward_scale': 1.0,
                    'risk_penalty': 0.1,
                    'hold_penalty': 0.001,
                }

                # Convert data to numpy for environment
                data_array = training_data[['open', 'high', 'low', 'close', 'volume']].values

                # Create environment
                env = TradingEnvironment(
                    data=data_array,
                    initial_capital=100000.0,
                    transaction_cost=0.001
                )

                # Create agent from chromosome config
                dqn_config = chromosome.to_dqn_config()
                agent = DQNAgent(dqn_config)

                # Train the agent
                for episode in range(episodes):
                    env.reset()
                    agent.train_episode(env)

                # Evaluate final performance
                eval_rewards = []
                for _ in range(5):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent.select_action(state, training=False)
                        state, reward, done, info = env.step(action)
                        episode_reward += reward
                    eval_rewards.append(episode_reward)

                fitness = np.mean(eval_rewards)

                result_queue.put({
                    'gpu_id': gpu_id,
                    'chromosome_idx': config['idx'],
                    'fitness': fitness,
                    'config': config,
                    'status': 'success'
                })

                print(f"  GPU {gpu_id}: Chromosome {config['idx']} fitness = {fitness:.4f}", flush=True)

            except Exception as e:
                import traceback
                result_queue.put({
                    'gpu_id': gpu_id,
                    'chromosome_idx': config['idx'],
                    'fitness': float('-inf'),
                    'error': f"{str(e)}\n{traceback.format_exc()}",
                    'status': 'failed'
                })
                print(f"  GPU {gpu_id}: Chromosome {config['idx']} FAILED - {e}", flush=True)

    except Exception as e:
        import traceback
        print(f"Worker GPU {gpu_id} failed: {e}\n{traceback.format_exc()}", flush=True)


class MultiGPUGATrainer:
    """GA trainer that distributes chromosomes across multiple GPUs."""

    def __init__(self, num_gpus: int, population_size: int = 16,
                 num_generations: int = 5, episodes_per_chromosome: int = 50):
        self.num_gpus = num_gpus
        self.population_size = population_size
        self.num_generations = num_generations
        self.episodes_per_chromosome = episodes_per_chromosome
        self.population = []
        self.best_fitness = float('-inf')
        self.best_chromosome = None

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for i in range(self.population_size):
            chromosome = {
                'idx': i,
                'symbol': 'SYNTH',
                'hidden_size': int(np.random.choice([128, 256, 320, 512])),
                'learning_rate': float(np.random.choice([0.0001, 0.0005, 0.001, 0.00001])),
                'batch_size': int(np.random.choice([16, 32, 64])),
                'gamma': float(np.random.uniform(0.9, 0.999)),
                'epsilon_decay': float(np.random.uniform(0.99, 0.999)),
                'fitness': None
            }
            self.population.append(chromosome)
        print(f"Initialized population with {self.population_size} chromosomes")

    def evaluate_population_parallel(self, training_data: pd.DataFrame,
                                    validation_data: pd.DataFrame):
        """Evaluate all chromosomes in parallel across GPUs."""
        # Convert DataFrames to dicts for pickling
        training_dict = training_data.to_dict('list')
        validation_dict = validation_data.to_dict('list')

        result_queue = mp.Queue()

        # Distribute chromosomes to GPUs
        gpu_assignments = [[] for _ in range(self.num_gpus)]
        for i, chrom in enumerate(self.population):
            gpu_id = i % self.num_gpus
            gpu_assignments[gpu_id].append(chrom)

        # Start worker processes
        processes = []
        for gpu_id in range(self.num_gpus):
            if gpu_assignments[gpu_id]:
                p = mp.Process(
                    target=train_chromosome_worker,
                    args=(gpu_id, gpu_assignments[gpu_id], training_dict,
                          validation_dict, self.episodes_per_chromosome, result_queue)
                )
                processes.append(p)
                p.start()

        # Wait for all to complete
        for p in processes:
            p.join(timeout=600)

        # Collect results
        results_count = 0
        while results_count < self.population_size:
            try:
                result = result_queue.get(timeout=5)
                idx = result['chromosome_idx']
                if result['status'] == 'success':
                    self.population[idx]['fitness'] = result['fitness']
                else:
                    self.population[idx]['fitness'] = float('-inf')
                results_count += 1
            except:
                break

    def selection(self) -> List[Dict]:
        """Select top performers."""
        sorted_pop = sorted(self.population,
                           key=lambda x: x['fitness'] if x['fitness'] is not None else float('-inf'),
                           reverse=True)
        return sorted_pop[:self.population_size // 2]

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Create child from two parents."""
        child = {'idx': 0, 'symbol': 'SYNTH'}
        for key in ['hidden_size', 'learning_rate', 'batch_size', 'gamma', 'epsilon_decay']:
            child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
        child['fitness'] = None
        return child

    def mutate(self, chromosome: Dict, rate: float = 0.2) -> Dict:
        """Apply mutations."""
        if np.random.random() < rate:
            chromosome['hidden_size'] = int(np.random.choice([128, 256, 320, 512]))
        if np.random.random() < rate:
            chromosome['learning_rate'] = float(np.random.choice([0.0001, 0.0005, 0.001, 0.00001]))
        if np.random.random() < rate:
            chromosome['batch_size'] = int(np.random.choice([16, 32, 64]))
        if np.random.random() < rate:
            chromosome['gamma'] = float(np.random.uniform(0.9, 0.999))
        if np.random.random() < rate:
            chromosome['epsilon_decay'] = float(np.random.uniform(0.99, 0.999))
        return chromosome

    def evolve(self):
        """Create next generation."""
        parents = self.selection()
        new_population = parents[:2]  # Elitism

        while len(new_population) < self.population_size:
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child = self.crossover(parents[p1], parents[p2])
            child = self.mutate(child)
            new_population.append(child)

        for i, chrom in enumerate(new_population):
            chrom['idx'] = i

        self.population = new_population

    def train(self, training_data: pd.DataFrame, validation_data: pd.DataFrame):
        """Run full evolution."""
        print(f"\n{'='*60}")
        print(f"STARTING MULTI-GPU GA+RL EVOLUTION")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Population: {self.population_size}")
        print(f"  Generations: {self.num_generations}")
        print(f"  Episodes/chromosome: {self.episodes_per_chromosome}")
        print(f"{'='*60}\n")

        self.initialize_population()

        for gen in range(self.num_generations):
            print(f"\n{'='*60}")
            print(f"GENERATION {gen + 1}/{self.num_generations}")
            print(f"{'='*60}")

            start = time.time()
            self.evaluate_population_parallel(training_data, validation_data)
            elapsed = time.time() - start

            valid = [c['fitness'] for c in self.population if c['fitness'] and c['fitness'] != float('-inf')]
            if valid:
                gen_best = max(valid)
                gen_avg = np.mean(valid)

                if gen_best > self.best_fitness:
                    self.best_fitness = gen_best
                    self.best_chromosome = next(c for c in self.population if c['fitness'] == gen_best).copy()

                print(f"\nGeneration {gen+1} Summary:")
                print(f"  Best: {gen_best:.4f}, Avg: {gen_avg:.4f}")
                print(f"  Overall best: {self.best_fitness:.4f}")
                print(f"  Time: {elapsed:.1f}s")

            if gen < self.num_generations - 1:
                self.evolve()

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Best fitness: {self.best_fitness:.4f}")
        if self.best_chromosome:
            print(f"Best config: hidden={self.best_chromosome['hidden_size']}, "
                  f"lr={self.best_chromosome['learning_rate']}, "
                  f"batch={self.best_chromosome['batch_size']}")
        print(f"{'='*60}")

        return self.best_chromosome, self.best_fitness


def main():
    mp.set_start_method('spawn', force=True)

    print_gpu_info()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("ERROR: No GPUs!")
        return 1

    print(f"\nUsing {num_gpus} GPUs for parallel training")

    # Generate data
    print("\nGenerating market data...")
    data = generate_synthetic_data(num_samples=10000)
    split = int(len(data) * 0.8)
    training_data = data.iloc[:split].reset_index(drop=True)
    validation_data = data.iloc[split:].reset_index(drop=True)
    print(f"Training: {len(training_data)}, Validation: {len(validation_data)}")

    # Train
    trainer = MultiGPUGATrainer(
        num_gpus=num_gpus,
        population_size=num_gpus * 4,  # 4 chromosomes per GPU
        num_generations=5,
        episodes_per_chromosome=30
    )

    try:
        best, fitness = trainer.train(training_data, validation_data)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())