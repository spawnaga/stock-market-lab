#!/bin/bash
# Distributed Training Launch Script - Worker Node (Docker)
# Run this on the Linux machine (192.168.1.129)
# Usage: ./launch_worker.sh

set -e

MASTER_ADDR="192.168.1.195"
MASTER_PORT="29500"
WORLD_SIZE="5"
SMB_USER="alial"
SMB_PASS="Cinquant"
MOUNT_POINT="/mnt/market_data"
# Use administrative share F$ - requires admin credentials
SMB_SHARE="F\$"  # Administrative share for F: drive

echo "================================================"
echo "Distributed GA+RL Training - WORKER NODE"
echo "================================================"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE GPUs"
echo "================================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Mount SMB share for market data
echo "Setting up market data access..."
sudo mkdir -p "$MOUNT_POINT"

# Check if already mounted
if mountpoint -q "$MOUNT_POINT"; then
    echo "Market data share already mounted at $MOUNT_POINT"
else
    echo "Mounting market data share..."
    # Install cifs-utils if not present
    if ! command -v mount.cifs &> /dev/null; then
        echo "Installing cifs-utils..."
        sudo apt-get update && sudo apt-get install -y cifs-utils
    fi

    # Try different share paths
    # First try MarketData share, then administrative F$ share
    sudo mount -t cifs "//$MASTER_ADDR/MarketData" "$MOUNT_POINT" \
        -o username="$SMB_USER",password="$SMB_PASS",vers=3.0,ro 2>/dev/null || \
    sudo mount -t cifs "//$MASTER_ADDR/F\$/Market Data/Extracted" "$MOUNT_POINT" \
        -o username="$SMB_USER",password="$SMB_PASS",vers=3.0,ro 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "Market data share mounted successfully"
    else
        echo "Warning: Could not mount market data share"
        echo "Training will use synthetic data"
    fi
fi

# Verify mount
echo ""
echo "Market data contents:"
ls -la "$MOUNT_POINT" 2>/dev/null || echo "Mount point not accessible"

echo ""
echo "Starting Docker container with all GPUs..."
echo ""

# Get the distributed training scripts from the master
SCRIPT_DIR="$HOME/distributed_scripts"
mkdir -p "$SCRIPT_DIR"

echo "Downloading distributed training scripts from master..."
scp -o StrictHostKeyChecking=no alial@$MASTER_ADDR:"/c/Users/alial/PycharmProjects/stock-market-lab/agents-python/distributed_*.py" "$SCRIPT_DIR/" 2>/dev/null || \
    echo "Could not download scripts - they may already exist or will use synthetic data"

# Launch Docker container with all GPUs
# Mount both market data and the distributed training scripts
docker run --gpus all --rm -it \
    --network host \
    --ipc=host \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e MASTER_PORT="$MASTER_PORT" \
    -e WORLD_SIZE="$WORLD_SIZE" \
    -e NCCL_DEBUG=INFO \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -v "$MOUNT_POINT":/data/market:ro \
    -v "$SCRIPT_DIR":/app/distributed:ro \
    spawnaga/stock-market-lab-python-agents:latest \
    bash -c "cp /app/distributed/*.py . 2>/dev/null || true && python distributed_ga_rl.py \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --world_size $WORLD_SIZE \
        --data_path /data/market \
        --launch_all_local_gpus \
        --population_size 20 \
        --num_generations 50 \
        --training_episodes 100"

echo ""
echo "================================================"
echo "Training complete!"
echo "================================================"