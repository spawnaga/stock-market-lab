#!/usr/bin/env python3
"""
GA+RL Training Script
Run this on the remote Linux server with RTX 3090 GPUs
"""

import torch
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("GA+RL Trading Strategy Optimization")
print("=" * 60)
print(f"Start time: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

print("=" * 60)

# Add work directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_rl_integration import IntegratedTradingSystem

def generate_synthetic_data(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic market data for training."""
    np.random.seed(seed)

    dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='1min')
    price = 100.0
    prices = [price]

    for _ in range(num_samples - 1):
        # Random walk with slight upward drift
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


def main():
    print("\nGenerating synthetic market data...")
    data = generate_synthetic_data(num_samples=10000)
    print(f"Generated {len(data)} price bars")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    print("\nInitializing GA+RL training system...")
    system = IntegratedTradingSystem(
        population_size=10,      # Small for testing
        num_generations=5,       # Small for testing
        training_episodes=20,    # Episodes per chromosome
        initial_capital=100000.0
    )

    print("\nStarting optimization...")
    print("This will evolve trading strategies using genetic algorithms")
    print("and train each strategy using deep reinforcement learning.")
    print("-" * 60)

    try:
        system.run_full_optimization(data)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())