"""
Distributed GA+RL Training Script
=================================
Main entry point for distributed training across multiple machines and GPUs.

Usage:
    Master node (Windows):
        python distributed_ga_rl.py --is_master --data_path "F:/Market Data/Extracted"

    Worker node (Linux):
        python distributed_ga_rl.py --rank 1 --data_path /mnt/market_data
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_rl_integration import (
    GADQNChromosome, GADQNPopulation, GADQNTrainingManager,
    set_websocket_emitter, emit_log
)
from reinforcement_learning import TradingEnvironment, DQNAgent
from market_data_loader import load_market_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Rank %(rank)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class DistributedGARLTrainer:
    """
    Distributed trainer for GA+RL that parallelizes chromosome evaluation.

    Training flow:
    1. Master initializes population
    2. Population is broadcast to all workers
    3. Each worker evaluates its assigned chromosomes
    4. Fitness scores are gathered at master
    5. Master performs evolution (selection, crossover, mutation)
    6. Repeat for all generations
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        is_master: bool,
        data_path: str,
        master_addr: str = "192.168.1.195",
        master_port: int = 29500,
        population_size: int = 20,
        num_generations: int = 50,
        training_episodes: int = 100,
        initial_capital: float = 100000
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_master = is_master
        self.data_path = data_path
        self.master_addr = master_addr
        self.master_port = master_port

        # Training parameters
        self.population_size = population_size
        self.num_generations = num_generations
        self.training_episodes = training_episodes
        self.initial_capital = initial_capital

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
            device_name = torch.cuda.get_device_name(self.device)
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        # Add rank to log format
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.rank = rank
            return record

        logging.setLogRecordFactory(record_factory)

        logger.info(f"Initialized on device: {device_name}")

        # Population manager (only used fully on master)
        self.population = GADQNPopulation(population_size)

        # Training data
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None

    def init_distributed(self):
        """Initialize PyTorch distributed training."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_RANK'] = str(self.local_rank)

        # Use GLOO for cross-platform (Windows + Linux)
        # Change to 'nccl' if both machines are Linux
        backend = 'gloo'

        logger.info(f"Initializing process group: {self.master_addr}:{self.master_port}")

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size
        )

        logger.info("Process group initialized successfully")

    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed")

    def load_data(self, symbol: str = "AAPL", train_ratio: float = 0.8):
        """Load market data for training."""
        logger.info(f"Loading market data from {self.data_path}")

        try:
            # Try to load using market_data_loader
            data = load_market_data(
                symbol=symbol,
                data_path=self.data_path
            )
        except Exception as e:
            logger.warning(f"Could not load via market_data_loader: {e}")
            # Fallback: look for CSV files directly
            import glob
            csv_files = glob.glob(os.path.join(self.data_path, "**", f"*{symbol}*.csv"), recursive=True)
            if csv_files:
                data = pd.read_csv(csv_files[0])
            else:
                # Use synthetic data for testing
                logger.warning("No data found, using synthetic data")
                data = self._generate_synthetic_data()

        # Split into train/validation
        split_idx = int(len(data) * train_ratio)
        self.training_data = data.iloc[:split_idx].copy()
        self.validation_data = data.iloc[split_idx:].copy()

        logger.info(f"Loaded {len(self.training_data)} training samples, "
                   f"{len(self.validation_data)} validation samples")

    def _generate_synthetic_data(self, num_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        np.random.seed(42 + self.rank)

        dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='1min')
        price = 100.0
        prices = [price]

        for _ in range(num_samples - 1):
            change = np.random.normal(0.0001, 0.005)
            price *= (1 + change)
            prices.append(price)

        prices = np.array(prices)

        return pd.DataFrame({
            'datetime': dates,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, num_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, num_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, num_samples))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, num_samples)
        })

    def broadcast_object(self, obj: Any) -> Any:
        """Broadcast an object from master to all workers."""
        if self.is_master:
            data = json.dumps(obj).encode()
            size_tensor = torch.tensor([len(data)], dtype=torch.long, device='cpu')
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device='cpu')

        dist.broadcast(size_tensor, src=0)

        if self.is_master:
            data_tensor = torch.ByteTensor(list(data))
        else:
            data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8)

        dist.broadcast(data_tensor, src=0)

        if not self.is_master:
            data = bytes(data_tensor.tolist()).decode()
            return json.loads(data)
        return obj

    def gather_objects(self, local_obj: Any) -> List[Any]:
        """Gather objects from all workers to master."""
        local_data = json.dumps(local_obj).encode()

        # Gather sizes first
        local_size = torch.tensor([len(local_data)], dtype=torch.long, device='cpu')
        if self.is_master:
            all_sizes = [torch.zeros(1, dtype=torch.long) for _ in range(self.world_size)]
        else:
            all_sizes = None

        dist.gather(local_size, gather_list=all_sizes, dst=0)

        # Gather data
        if self.is_master:
            all_objects = []
            # Master's own data
            all_objects.append(local_obj)

            # Receive from workers
            for r in range(1, self.world_size):
                size = all_sizes[r].item()
                data_tensor = torch.zeros(size, dtype=torch.uint8)
                dist.recv(data_tensor, src=r)
                data = bytes(data_tensor.tolist()).decode()
                all_objects.append(json.loads(data))

            return all_objects
        else:
            # Send to master
            data_tensor = torch.ByteTensor(list(local_data))
            dist.send(data_tensor, dst=0)
            return []

    def evaluate_chromosomes(
        self,
        chromosomes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of chromosomes on this worker's GPU.

        Returns:
            List of chromosome dicts with updated fitness values
        """
        results = []

        for chr_dict in chromosomes:
            chromosome = GADQNChromosome.from_dict(chr_dict)

            # Create environment
            data_array = self.training_data[['open', 'high', 'low', 'close', 'volume']].values
            env = TradingEnvironment(
                data=data_array,
                initial_capital=self.initial_capital,
                transaction_cost=0.001
            )

            # Create agent from chromosome
            config = chromosome.to_dqn_config()
            config.device = str(self.device)
            agent = DQNAgent(config)

            # Train
            for _ in range(self.training_episodes):
                env.reset()
                agent.train_episode(env)

            # Evaluate
            total_returns = []
            for _ in range(10):  # Evaluation episodes
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state, training=False)
                    state, _, done, info = env.step(action)
                total_returns.append(info.get('total_return', 0))

            # Compute fitness metrics
            avg_return = np.mean(total_returns)
            if len(total_returns) > 1 and np.std(total_returns) > 0:
                sharpe = np.mean(total_returns) / np.std(total_returns) * np.sqrt(252)
            else:
                sharpe = 0.0

            max_drawdown = min(total_returns) if total_returns else 0
            win_rate = sum(1 for r in total_returns if r > 0) / len(total_returns)

            # Combined fitness
            fitness = (
                sharpe * 0.4 +
                avg_return * 100 * 0.3 +
                (1 - abs(max_drawdown)) * 0.2 +
                win_rate * 0.1
            )

            # Update chromosome
            chromosome.fitness = fitness
            chromosome.sharpe_ratio = sharpe
            chromosome.total_return = avg_return
            chromosome.max_drawdown = max_drawdown
            chromosome.win_rate = win_rate

            results.append(chromosome.to_dict())

            logger.info(f"Evaluated {chromosome.chromosome_id}: "
                       f"fitness={fitness:.4f}, sharpe={sharpe:.4f}")

        return results

    def run_distributed_evolution(self) -> Optional[Dict[str, Any]]:
        """
        Run the full distributed evolution process.

        Returns:
            Best chromosome configuration found
        """
        start_time = time.time()

        if self.is_master:
            emit_log('info', "=" * 60)
            emit_log('info', "DISTRIBUTED GA+RL EVOLUTION")
            emit_log('info', f"World size: {self.world_size} GPUs")
            emit_log('info', f"Population: {self.population_size}")
            emit_log('info', f"Generations: {self.num_generations}")
            emit_log('info', "=" * 60)

            # Initialize population on master
            self.population.initialize_population()

        # Synchronize before starting
        dist.barrier()

        for generation in range(self.num_generations):
            gen_start = time.time()

            if self.is_master:
                emit_log('info', f"\n{'='*40}")
                emit_log('info', f"GENERATION {generation + 1}/{self.num_generations}")
                emit_log('info', f"{'='*40}")

                # Convert population to dicts
                population_dicts = [c.to_dict() for c in self.population.population]
            else:
                population_dicts = None

            # Broadcast population to all workers
            population_dicts = self.broadcast_object(
                population_dicts if self.is_master else None
            )

            # Assign chromosomes to this worker
            # Simple round-robin assignment
            my_chromosomes = [
                population_dicts[i]
                for i in range(len(population_dicts))
                if i % self.world_size == self.rank
            ]

            logger.info(f"Evaluating {len(my_chromosomes)} chromosomes")

            # Evaluate assigned chromosomes
            evaluated = self.evaluate_chromosomes(my_chromosomes)

            # Gather results at master
            all_evaluated = self.gather_objects(evaluated)

            if self.is_master:
                # Flatten and update population
                all_chromosomes = []
                for worker_results in all_evaluated:
                    all_chromosomes.extend(worker_results)

                # Update population with evaluated chromosomes
                self.population.population = [
                    GADQNChromosome.from_dict(c) for c in all_chromosomes
                ]

                # Perform evolution
                self.population.evolve()

                gen_time = time.time() - gen_start
                best = self.population.best_chromosome

                emit_log('success', f"Generation {generation + 1} complete ({gen_time:.1f}s)")
                emit_log('info', f"  Best fitness: {best.fitness:.4f}")
                emit_log('info', f"  Best sharpe: {best.sharpe_ratio:.4f}")
                emit_log('info', f"  Best return: {best.total_return*100:.2f}%")

            # Synchronize before next generation
            dist.barrier()

        # Final results
        if self.is_master:
            total_time = time.time() - start_time
            emit_log('success', "\n" + "=" * 60)
            emit_log('success', "DISTRIBUTED TRAINING COMPLETE")
            emit_log('info', f"Total time: {total_time:.1f}s")
            emit_log('info', f"Best chromosome: {self.population.best_chromosome.chromosome_id}")
            emit_log('success', "=" * 60)

            return self.population.best_chromosome.to_dict()

        return None


def launch_all_local_gpus(args):
    """
    Launch workers for all local GPUs.
    Used when running on the worker node with multiple GPUs.
    """
    import subprocess
    import multiprocessing

    num_gpus = torch.cuda.device_count()
    logger.info(f"Launching {num_gpus} workers for local GPUs")

    processes = []

    # Determine starting rank (master is 0, so workers start at 1)
    start_rank = 1

    for local_rank in range(num_gpus):
        rank = start_rank + local_rank

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        cmd = [
            sys.executable,
            __file__,
            '--master_addr', args.master_addr,
            '--master_port', str(args.master_port),
            '--world_size', str(args.world_size),
            '--rank', str(rank),
            '--local_rank', '0',  # Each process sees only one GPU
            '--data_path', args.data_path,
        ]

        if args.population_size:
            cmd.extend(['--population_size', str(args.population_size)])
        if args.num_generations:
            cmd.extend(['--num_generations', str(args.num_generations)])
        if args.training_episodes:
            cmd.extend(['--training_episodes', str(args.training_episodes)])

        logger.info(f"Launching worker rank {rank} on GPU {local_rank}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    # Wait for all processes
    for proc in processes:
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Distributed GA+RL Training")

    # Distributed training args
    parser.add_argument('--master_addr', type=str, default='192.168.1.195',
                       help='Master node address')
    parser.add_argument('--master_port', type=int, default=29500,
                       help='Master node port')
    parser.add_argument('--world_size', type=int, default=5,
                       help='Total number of GPUs')
    parser.add_argument('--rank', type=int, default=0,
                       help='Global rank of this process')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local GPU rank')
    parser.add_argument('--is_master', action='store_true',
                       help='This is the master node')

    # Data args
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to market data')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to train on')

    # Training args
    parser.add_argument('--population_size', type=int, default=20,
                       help='GA population size')
    parser.add_argument('--num_generations', type=int, default=50,
                       help='Number of GA generations')
    parser.add_argument('--training_episodes', type=int, default=100,
                       help='RL training episodes per chromosome')
    parser.add_argument('--initial_capital', type=float, default=100000,
                       help='Initial trading capital')

    # Special modes
    parser.add_argument('--launch_all_local_gpus', action='store_true',
                       help='Launch workers for all local GPUs')

    args = parser.parse_args()

    # Handle multi-GPU launch on worker
    if args.launch_all_local_gpus:
        launch_all_local_gpus(args)
        return

    # Determine if this is master
    is_master = args.is_master or args.rank == 0

    # Create trainer
    trainer = DistributedGARLTrainer(
        rank=args.rank,
        world_size=args.world_size,
        local_rank=args.local_rank,
        is_master=is_master,
        data_path=args.data_path,
        master_addr=args.master_addr,
        master_port=args.master_port,
        population_size=args.population_size,
        num_generations=args.num_generations,
        training_episodes=args.training_episodes,
        initial_capital=args.initial_capital
    )

    try:
        # Initialize distributed training
        trainer.init_distributed()

        # Load data
        trainer.load_data(symbol=args.symbol)

        # Run training
        result = trainer.run_distributed_evolution()

        if result and is_master:
            # Save best chromosome
            output_path = f"best_chromosome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved best chromosome to {output_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()