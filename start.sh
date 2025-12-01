#!/bin/bash

# Stock Market Lab - Quick Start Script
echo "Starting Stock Market Lab..."

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Default values
ENABLE_GPU=${ENABLE_GPU:-true}
FRONTEND_PORT=${FRONTEND_PORT:-3001}
BACKEND_API_PORT=${BACKEND_API_PORT:-5100}
CPP_BACKEND_PORT=${CPP_BACKEND_PORT:-8080}
GPU_SERVICE_PORT=${GPU_SERVICE_PORT:-8081}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
REDIS_PORT=${REDIS_PORT:-7379}
KAFKA_PORT=${KAFKA_PORT:-9092}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "============================================"
echo "Configuration:"
echo "  Market Data Path: ${MARKET_DATA_PATH:-./data/market}"
echo "  GPU Enabled: $ENABLE_GPU"
echo "============================================"

# Determine compose files based on GPU setting
COMPOSE_FILES="-f docker-compose.yml"

if [ "$ENABLE_GPU" = "true" ]; then
    echo "Checking for NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_NAME" ]; then
            echo "GPU detected: $GPU_NAME"
            COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"
        else
            echo "WARNING: nvidia-smi available but no GPU found. Running in CPU mode."
        fi
    else
        echo "WARNING: nvidia-smi not found. Running in CPU mode."
    fi
else
    echo "GPU disabled by configuration. Running in CPU mode."
fi

# Build and start all services
echo "Building and starting all services..."
docker compose $COMPOSE_FILES up --build "$@"

# If running in detached mode, show status
if [[ "$*" == *"-d"* ]]; then
    echo "Waiting for services to start..."
    sleep 30

    echo "Service Status:"
    docker compose ps

    echo ""
    echo "Access URLs:"
    echo "   Frontend:        http://localhost:$FRONTEND_PORT"
    echo "   Backend API:     http://localhost:$BACKEND_API_PORT"
    echo "   C++ Backend:     http://localhost:$CPP_BACKEND_PORT"
    echo "   GPU Service:     http://localhost:$GPU_SERVICE_PORT"
    echo "   Database:        localhost:$POSTGRES_PORT"
    echo "   Redis:           localhost:$REDIS_PORT"
    echo "   Kafka:           localhost:$KAFKA_PORT"
    echo ""
    echo "Use 'docker compose logs -f' to view logs"
    echo "Use 'docker compose down' to stop all services"
    echo ""
    echo "Stock Market Lab is ready!"
fi
