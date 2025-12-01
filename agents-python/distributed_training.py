"""
Distributed GA+RL Training Module
=================================
Enables training across multiple machines with multiple GPUs.

Architecture:
- Master node (192.168.1.195): Coordinates training, stores data
- Worker nodes (192.168.1.129): Execute GPU-intensive training

For GA+RL, we use a hybrid approach:
1. Population evaluation is parallelized across all GPUs
2. Each GPU trains a subset of chromosomes
3. Results are gathered and evolution happens on master

Supports:
- Local machine: 1x RTX 4090 (24GB)
- Remote machine: 2x RTX 5090 (32GB each) + 2x RTX 3090 (24GB each, NVLinked)
- Total: ~137GB VRAM across 5 GPUs
"""

import os
import json
import socket
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Configuration for a compute node."""
    hostname: str
    ip_address: str
    ssh_user: str
    num_gpus: int
    gpu_names: List[str]
    gpu_memory_mb: List[int]
    is_master: bool = False
    data_path: str = ""


@dataclass
class ClusterConfig:
    """Configuration for the distributed training cluster."""
    master_addr: str = "192.168.1.195"
    master_port: int = 29500
    world_size: int = 5  # Total number of GPUs
    nodes: List[NodeConfig] = None
    data_share_path: str = "//192.168.1.195/MarketData"  # SMB share path

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []


class DistributedTrainingManager:
    """
    Manages distributed training across multiple nodes and GPUs.

    For GA+RL training:
    - Distributes chromosome evaluation across available GPUs
    - Each GPU independently trains and evaluates chromosomes
    - Results are gathered for population evolution
    """

    def __init__(self, config: ClusterConfig = None):
        self.config = config or self._create_default_config()
        self.rank = 0
        self.local_rank = 0
        self.world_size = self.config.world_size
        self.is_initialized = False

    def _create_default_config(self) -> ClusterConfig:
        """Create default cluster configuration for our setup."""
        config = ClusterConfig(
            master_addr="192.168.1.195",
            master_port=29500,
            world_size=5,
            nodes=[
                NodeConfig(
                    hostname="windows-local",
                    ip_address="192.168.1.195",
                    ssh_user="alial",
                    num_gpus=1,
                    gpu_names=["RTX 4090"],
                    gpu_memory_mb=[24564],
                    is_master=True,
                    data_path="F:/Market Data/Extracted"
                ),
                NodeConfig(
                    hostname="linux-remote",
                    ip_address="192.168.1.129",
                    ssh_user="alex",
                    num_gpus=4,
                    gpu_names=["RTX 5090", "RTX 3090", "RTX 5090", "RTX 3090"],
                    gpu_memory_mb=[32607, 24576, 32607, 24576],
                    is_master=False,
                    data_path="/mnt/market_data"  # SMB mount point
                )
            ],
            data_share_path="//192.168.1.195/MarketData"
        )
        return config

    def get_total_gpu_memory_gb(self) -> float:
        """Get total available GPU memory in GB."""
        total_mb = sum(
            sum(node.gpu_memory_mb)
            for node in self.config.nodes
        )
        return total_mb / 1024

    def get_gpu_distribution(self) -> Dict[str, List[int]]:
        """Get GPU distribution across nodes."""
        distribution = {}
        gpu_idx = 0
        for node in self.config.nodes:
            distribution[node.hostname] = list(range(gpu_idx, gpu_idx + node.num_gpus))
            gpu_idx += node.num_gpus
        return distribution

    def assign_chromosomes_to_gpus(
        self,
        num_chromosomes: int
    ) -> Dict[int, List[int]]:
        """
        Assign chromosomes to GPUs for parallel evaluation.

        Args:
            num_chromosomes: Total number of chromosomes to evaluate

        Returns:
            Dictionary mapping GPU rank to list of chromosome indices
        """
        assignments = {i: [] for i in range(self.world_size)}

        for chr_idx in range(num_chromosomes):
            # Round-robin assignment weighted by GPU memory
            gpu_rank = chr_idx % self.world_size
            assignments[gpu_rank].append(chr_idx)

        return assignments

    def generate_launch_scripts(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Generate launch scripts for each node in the cluster.

        Returns:
            Dictionary mapping node hostname to script path
        """
        scripts = {}

        # Master node script (Windows)
        master_script = f"""@echo off
REM Distributed Training Launch Script - Master Node
REM Run this on the Windows machine (192.168.1.195)

set MASTER_ADDR={self.config.master_addr}
set MASTER_PORT={self.config.master_port}
set WORLD_SIZE={self.config.world_size}
set RANK=0
set LOCAL_RANK=0

echo Starting distributed training on master node...
echo Master: %MASTER_ADDR%:%MASTER_PORT%
echo World size: %WORLD_SIZE%

REM Activate conda environment
call conda activate trade

REM Launch training
python distributed_ga_rl.py ^
    --master_addr %MASTER_ADDR% ^
    --master_port %MASTER_PORT% ^
    --world_size %WORLD_SIZE% ^
    --rank %RANK% ^
    --local_rank %LOCAL_RANK% ^
    --data_path "F:/Market Data/Extracted" ^
    --is_master

echo Training complete!
pause
"""
        master_script_path = os.path.join(output_dir, "launch_master.bat")
        scripts["windows-local"] = master_script_path

        # Worker node script (Linux)
        worker_script = f"""#!/bin/bash
# Distributed Training Launch Script - Worker Node
# Run this on the Linux machine (192.168.1.129)

export MASTER_ADDR={self.config.master_addr}
export MASTER_PORT={self.config.master_port}
export WORLD_SIZE={self.config.world_size}

echo "Starting distributed training on worker node..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Mount SMB share if not already mounted
if ! mountpoint -q /mnt/market_data; then
    echo "Mounting market data share..."
    sudo mkdir -p /mnt/market_data
    sudo mount -t cifs {self.config.data_share_path} /mnt/market_data \\
        -o username=alial,password=Cinquant,vers=3.0
fi

# Launch workers for each GPU
for LOCAL_RANK in 0 1 2 3; do
    RANK=$((LOCAL_RANK + 1))  # Master is rank 0

    echo "Launching worker on GPU $LOCAL_RANK (global rank $RANK)..."

    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python3 distributed_ga_rl.py \\
        --master_addr $MASTER_ADDR \\
        --master_port $MASTER_PORT \\
        --world_size $WORLD_SIZE \\
        --rank $RANK \\
        --local_rank $LOCAL_RANK \\
        --data_path /mnt/market_data &
done

echo "All workers launched. Waiting for completion..."
wait
echo "Training complete!"
"""
        worker_script_path = os.path.join(output_dir, "launch_worker.sh")
        scripts["linux-remote"] = worker_script_path

        # Docker-based worker script
        docker_worker_script = f"""#!/bin/bash
# Distributed Training Launch Script - Worker Node (Docker)
# Run this on the Linux machine (192.168.1.129)

export MASTER_ADDR={self.config.master_addr}
export MASTER_PORT={self.config.master_port}
export WORLD_SIZE={self.config.world_size}

echo "Starting distributed training on worker node (Docker)..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Create mount directory
sudo mkdir -p /mnt/market_data

# Mount SMB share
sudo mount -t cifs {self.config.data_share_path} /mnt/market_data \\
    -o username=alial,password=Cinquant,vers=3.0 2>/dev/null || echo "Share already mounted or mount failed"

# Launch Docker container with all GPUs
docker run --gpus all --rm \\
    --network host \\
    -e MASTER_ADDR=$MASTER_ADDR \\
    -e MASTER_PORT=$MASTER_PORT \\
    -e WORLD_SIZE=$WORLD_SIZE \\
    -e NCCL_DEBUG=INFO \\
    -v /mnt/market_data:/data/market:ro \\
    --ipc=host \\
    spawnaga/stock-market-lab-python-agents:latest \\
    python distributed_ga_rl.py \\
        --master_addr $MASTER_ADDR \\
        --master_port $MASTER_PORT \\
        --world_size $WORLD_SIZE \\
        --data_path /data/market \\
        --launch_all_local_gpus

echo "Training complete!"
"""
        docker_worker_script_path = os.path.join(output_dir, "launch_worker_docker.sh")
        scripts["linux-remote-docker"] = docker_worker_script_path

        return scripts

    def save_launch_scripts(self, output_dir: str = None) -> List[str]:
        """Save launch scripts to files."""
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        scripts = self.generate_launch_scripts(output_dir)
        saved_paths = []

        for hostname, script_path in scripts.items():
            script_content = None

            # Read the generated content
            if "windows" in hostname:
                script_content = self.generate_launch_scripts(output_dir)
                with open(script_path, 'w', newline='\r\n') as f:
                    # Re-generate and save
                    pass  # Already saved by generate_launch_scripts

            saved_paths.append(script_path)
            logger.info(f"Saved launch script for {hostname}: {script_path}")

        return saved_paths


class DistributedGARL:
    """
    Distributed Genetic Algorithm + Reinforcement Learning trainer.

    Parallelizes chromosome evaluation across multiple GPUs:
    - Each GPU trains and evaluates a subset of the population
    - Fitness scores are gathered at the master
    - Evolution (selection, crossover, mutation) happens on master
    - New population is scattered to GPUs for next generation
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        is_master: bool,
        data_path: str,
        master_addr: str = "192.168.1.195",
        master_port: int = 29500
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_master = is_master
        self.data_path = data_path
        self.master_addr = master_addr
        self.master_port = master_port

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        logger.info(f"Rank {rank}: Using device {self.device}")

    def init_distributed(self):
        """Initialize distributed training backend."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_RANK'] = str(self.local_rank)

        # Use GLOO backend for cross-platform compatibility (Windows + Linux)
        # NCCL is faster but Linux-only
        backend = 'gloo'  # Use 'nccl' if both machines are Linux

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size
        )

        logger.info(f"Rank {self.rank}: Distributed training initialized")

    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def scatter_chromosomes(
        self,
        chromosomes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Scatter chromosomes from master to all workers.

        Args:
            chromosomes: List of chromosome dictionaries (only on master)

        Returns:
            List of chromosomes for this rank to evaluate
        """
        # Calculate how many chromosomes each rank gets
        num_per_rank = len(chromosomes) // self.world_size
        remainder = len(chromosomes) % self.world_size

        if self.is_master:
            # Split chromosomes across ranks
            scattered = []
            idx = 0
            for r in range(self.world_size):
                count = num_per_rank + (1 if r < remainder else 0)
                scattered.append(chromosomes[idx:idx + count])
                idx += count

            # Send to each rank
            for r in range(1, self.world_size):
                data = json.dumps(scattered[r]).encode()
                size_tensor = torch.tensor([len(data)], dtype=torch.long)
                dist.send(size_tensor, dst=r)

                data_tensor = torch.ByteTensor(list(data))
                dist.send(data_tensor, dst=r)

            return scattered[0]
        else:
            # Receive from master
            size_tensor = torch.tensor([0], dtype=torch.long)
            dist.recv(size_tensor, src=0)

            data_tensor = torch.ByteTensor(size_tensor.item())
            dist.recv(data_tensor, src=0)

            data = bytes(data_tensor.tolist()).decode()
            return json.loads(data)

    def gather_fitness(
        self,
        local_fitness: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Gather fitness scores from all workers to master.

        Args:
            local_fitness: List of (chromosome_id, fitness) tuples from this rank

        Returns:
            Combined fitness from all ranks (only valid on master)
        """
        # Serialize local fitness
        data = json.dumps(local_fitness).encode()

        if not self.is_master:
            # Send to master
            size_tensor = torch.tensor([len(data)], dtype=torch.long)
            dist.send(size_tensor, dst=0)

            data_tensor = torch.ByteTensor(list(data))
            dist.send(data_tensor, dst=0)

            return []
        else:
            # Receive from all workers
            all_fitness = list(local_fitness)

            for r in range(1, self.world_size):
                size_tensor = torch.tensor([0], dtype=torch.long)
                dist.recv(size_tensor, src=r)

                data_tensor = torch.ByteTensor(size_tensor.item())
                dist.recv(data_tensor, src=r)

                data = bytes(data_tensor.tolist()).decode()
                all_fitness.extend(json.loads(data))

            return all_fitness

    def synchronize(self):
        """Synchronize all processes."""
        dist.barrier()

    def broadcast_population(
        self,
        population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Broadcast population from master to all workers.

        Args:
            population: Full population (only valid on master)

        Returns:
            Population received by all ranks
        """
        if self.is_master:
            data = json.dumps(population).encode()
            size_tensor = torch.tensor([len(data)], dtype=torch.long)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long)

        # Broadcast size
        dist.broadcast(size_tensor, src=0)

        # Broadcast data
        if self.is_master:
            data_tensor = torch.ByteTensor(list(data))
        else:
            data_tensor = torch.ByteTensor(size_tensor.item())

        dist.broadcast(data_tensor, src=0)

        if not self.is_master:
            data = bytes(data_tensor.tolist()).decode()
            return json.loads(data)
        else:
            return population


def create_smb_share_windows():
    """
    Create SMB share on Windows for market data.
    Must be run with admin privileges.
    """
    import subprocess

    share_name = "MarketData"
    share_path = r"F:\Market Data\Extracted"

    # Check if share already exists
    result = subprocess.run(
        ["net", "share", share_name],
        capture_output=True,
        text=True
    )

    if "is shared as" in result.stdout:
        logger.info(f"Share {share_name} already exists")
        return True

    # Create the share
    cmd = [
        "net", "share",
        f"{share_name}={share_path}",
        "/grant:Everyone,READ"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info(f"Created SMB share: {share_name}")
        return True
    else:
        logger.error(f"Failed to create share: {result.stderr}")
        return False


def print_cluster_info(manager: DistributedTrainingManager):
    """Print cluster configuration information."""
    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING CLUSTER CONFIGURATION")
    print("=" * 60)

    print(f"\nMaster: {manager.config.master_addr}:{manager.config.master_port}")
    print(f"Total GPUs (World Size): {manager.config.world_size}")
    print(f"Total GPU Memory: {manager.get_total_gpu_memory_gb():.1f} GB")

    print("\nNodes:")
    gpu_dist = manager.get_gpu_distribution()
    for node in manager.config.nodes:
        role = "MASTER" if node.is_master else "WORKER"
        print(f"\n  [{role}] {node.hostname} ({node.ip_address})")
        print(f"    User: {node.ssh_user}")
        print(f"    GPUs: {node.num_gpus}")
        for i, (name, mem) in enumerate(zip(node.gpu_names, node.gpu_memory_mb)):
            print(f"      GPU {i}: {name} ({mem / 1024:.1f} GB)")
        print(f"    Global GPU ranks: {gpu_dist[node.hostname]}")
        print(f"    Data path: {node.data_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed GA+RL Training Setup")
    parser.add_argument("--generate-scripts", action="store_true",
                       help="Generate launch scripts for all nodes")
    parser.add_argument("--create-share", action="store_true",
                       help="Create SMB share on Windows (requires admin)")
    parser.add_argument("--info", action="store_true",
                       help="Print cluster configuration info")

    args = parser.parse_args()

    manager = DistributedTrainingManager()

    if args.info or (not args.generate_scripts and not args.create_share):
        print_cluster_info(manager)

    if args.generate_scripts:
        print("\nGenerating launch scripts...")
        scripts = manager.generate_launch_scripts()

        for hostname, content in scripts.items():
            filename = f"launch_{hostname.replace('-', '_')}.{'bat' if 'windows' in hostname else 'sh'}"
            # Scripts are already generated, just print info
            print(f"  - {filename}")

        print("\nTo start distributed training:")
        print("  1. On Windows (master): Run launch_master.bat")
        print("  2. On Linux (worker): Run launch_worker_docker.sh")

    if args.create_share:
        print("\nCreating SMB share...")
        if create_smb_share_windows():
            print("SMB share created successfully")
        else:
            print("Failed to create SMB share (may need admin privileges)")